import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from FlagEmbedding import FlagModel

# [NEW]
from tqdm import tqdm


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    a = l2norm(anchor)
    p = l2norm(positive)
    logits = (a @ p.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def multi_positive_nce(
    anchor: torch.Tensor,
    candidates: torch.Tensor,              # (B,1+S,D)
    temperature: float = 0.07,
    pos_weight: Optional[torch.Tensor] = None,  # (B,1+S)
) -> torch.Tensor:
    B, M, D = candidates.shape
    a = l2norm(anchor)
    bank = l2norm(candidates.reshape(B * M, D))

    logits = (a @ bank.t()) / temperature

    pos_mask = torch.zeros(B, B * M, device=anchor.device, dtype=torch.bool)
    for i in range(B):
        start = i * M
        pos_mask[i, start:start + M] = True

    if pos_weight is None:
        pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
        num = torch.logsumexp(pos_logits, dim=1)
    else:
        w = pos_weight.to(anchor.device)
        w = torch.clamp(w, min=1e-8)
        logw = torch.log(w).reshape(B, M)
        num_list = []
        for i in range(B):
            sl = logits[i, i*M:(i+1)*M] + logw[i]
            num_list.append(torch.logsumexp(sl, dim=0))
        num = torch.stack(num_list, dim=0)

    den = torch.logsumexp(logits, dim=1)
    loss = -(num - den).mean()
    return loss


@torch.no_grad()
def retrieval_metrics(query_emb: torch.Tensor, sql_emb: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    q = l2norm(query_emb)
    s = l2norm(sql_emb)
    sim = q @ s.t()
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    target = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    pos = (sorted_idx == target).nonzero(as_tuple=False)
    ranks = pos[:, 1] + 1

    out = {}
    for k in ks:
        out[f"R@{k}"] = (ranks <= k).float().mean().item()
    out["MRR"] = (1.0 / ranks.float()).mean().item()
    return out


# -------------------------
# Models
# -------------------------
class SafeMHA(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, q, k, v, key_padding_mask: Optional[torch.Tensor] = None):
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                out = torch.zeros_like(q)
                valid = ~all_masked
                if valid.any():
                    o, _ = self.mha(q[valid], k[valid], v[valid], key_padding_mask=key_padding_mask[valid])
                    out[valid] = o
                return out
        out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask)
        return out


class SchemaAwareModel(nn.Module):
    def __init__(self, dim: int = 1024, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.table_col_attn = SafeMHA(dim, heads, dropout)
        self.q_table_attn = SafeMHA(dim, heads, dropout)

        self.proj_q = nn.Linear(dim, dim)
        self.proj_t = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, table, col, table_mask, col_mask):
        B, T, D = table.shape

        table_p = self.drop(self.proj_t(table))
        col_p = self.drop(self.proj_c(col))

        enhanced_tables = []
        for i in range(T):
            ti = table_p[:, i:i+1, :]
            ci = col_p[:, i, :, :]
            ci_pad = ~col_mask[:, i, :]

            if col_mask[:, i, :].any():
                attn = self.table_col_attn(ti, ci, ci, key_padding_mask=ci_pad)
                ti2 = self.ln1(ti + attn)
                enhanced_tables.append(ti2)
            else:
                enhanced_tables.append(ti)

        enhanced_tables = torch.cat(enhanced_tables, dim=1)

        q1 = self.drop(self.proj_q(q)).unsqueeze(1)
        t_pad = ~table_mask

        if table_mask.any():
            attn2 = self.q_table_attn(q1, enhanced_tables, enhanced_tables, key_padding_mask=t_pad)
            q2 = self.ln2(q1 + attn2).squeeze(1)
        else:
            q2 = q1.squeeze(1)

        return self.out(q2)


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, s):
        g = self.gate(torch.cat([q, s], dim=-1))
        h = self.ln(q + g * s)
        return self.proj(h)


# -------------------------
# Dataset + Cache
# -------------------------
class EmbedCache:
    """Global text->embedding cache to avoid repeated FlagModel.encode."""
    def __init__(self, flag: FlagModel):
        self.flag = flag
        self.cache: Dict[str, torch.Tensor] = {}

    def enc(self, text: str) -> torch.Tensor:
        if text in self.cache:
            return self.cache[text]
        v = torch.tensor(self.flag.encode(text), dtype=torch.float32)
        self.cache[text] = v
        return v


def schema_to_tensors(
    cache: EmbedCache,
    schema: Dict[str, Any],
    dim: int,
    max_tables: int,
    max_cols: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    table = torch.zeros(max_tables, dim, dtype=torch.float32)
    col = torch.zeros(max_tables, max_cols, dim, dtype=torch.float32)
    table_mask = torch.zeros(max_tables, dtype=torch.bool)
    col_mask = torch.zeros(max_tables, max_cols, dtype=torch.bool)

    if not schema or "tables" not in schema:
        return table, col, table_mask, col_mask

    tables = (schema.get("tables", []) or [])[:max_tables]
    cols_dict = schema.get("columns", {}) or {}

    for i, tname in enumerate(tables):
        table_mask[i] = True
        table[i] = cache.enc(f"Table: {tname}")
        cols = (cols_dict.get(tname, []) or [])[:max_cols]
        for j, cname in enumerate(cols):
            col_mask[i, j] = True
            col[i, j] = cache.enc(f"Column: {cname} in {tname}")

    return table, col, table_mask, col_mask


# [NEW] --------- pre-encode + disk cache ----------
def default_cache_path(json_path: str, suffix: str) -> str:
    return json_path + f".{suffix}.pt"


def preencode_stage1_json(
    json_path: str,
    save_pt: str,
    cache: EmbedCache,
    dim: int,
    max_tables: int,
    max_cols: int,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for it in tqdm(data, desc=f"[Pre-encode Stage1] {os.path.basename(json_path)}"):
        q = it["question"]
        sql = it["query"]
        schema = it.get("schema", {})

        q_emb = cache.enc(q)
        sql_emb = cache.enc(sql)
        table, col, tmask, cmask = schema_to_tensors(cache, schema, dim, max_tables, max_cols)

        out.append({
            "q_emb": q_emb,
            "sql_emb": sql_emb,
            "table": table,
            "col": col,
            "tmask": tmask,
            "cmask": cmask,
        })

    os.makedirs(os.path.dirname(save_pt) or ".", exist_ok=True)
    torch.save(out, save_pt)
    print(f"[Cache Saved] {save_pt}  (#items={len(out)})")


def preencode_stage2_json(
    json_path: str,
    save_pt: str,
    cache: EmbedCache,
    dim: int,
    max_tables: int,
    max_cols: int,
    max_sim: int,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for it in tqdm(data, desc=f"[Pre-encode Stage2] {os.path.basename(json_path)}"):
        q = it["question"]
        sql = it["query"]
        schema = it.get("schema", {})

        q_emb = cache.enc(q)
        sql_emb = cache.enc(sql)
        table, col, tmask, cmask = schema_to_tensors(cache, schema, dim, max_tables, max_cols)

        sims = it.get("similar", []) or []
        sims = sims[:max_sim]
        while len(sims) < max_sim:
            sims.append({"question": q, "query": sql, "schema": schema})

        sim_q_emb, sim_sql_emb = [], []
        sim_table, sim_col, sim_tmask, sim_cmask = [], [], [], []

        for s in sims:
            sq = s["question"]
            ssql = s.get("query", sql)
            sschema = s.get("schema", schema)

            sim_q_emb.append(cache.enc(sq))
            sim_sql_emb.append(cache.enc(ssql))
            st, sc, stm, scm = schema_to_tensors(cache, sschema, dim, max_tables, max_cols)
            sim_table.append(st)
            sim_col.append(sc)
            sim_tmask.append(stm)
            sim_cmask.append(scm)

        out.append({
            "q_emb": q_emb,
            "sql_emb": sql_emb,
            "table": table,
            "col": col,
            "tmask": tmask,
            "cmask": cmask,

            "sim_q_emb": torch.stack(sim_q_emb, dim=0),
            "sim_sql_emb": torch.stack(sim_sql_emb, dim=0),
            "sim_table": torch.stack(sim_table, dim=0),
            "sim_col": torch.stack(sim_col, dim=0),
            "sim_tmask": torch.stack(sim_tmask, dim=0),
            "sim_cmask": torch.stack(sim_cmask, dim=0),
        })

    os.makedirs(os.path.dirname(save_pt) or ".", exist_ok=True)
    torch.save(out, save_pt)
    print(f"[Cache Saved] {save_pt}  (#items={len(out)})")


class PTListDataset(Dataset):
    """[NEW] Dataset backed by torch.load(list_of_dicts)."""
    def __init__(self, pt_path: str):
        self.data = torch.load(pt_path)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx: int): return self.data[idx]


# -------------------------
# Collate
# -------------------------
def collate_stage1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "q_emb": torch.stack([b["q_emb"] for b in batch], dim=0),
        "sql_emb": torch.stack([b["sql_emb"] for b in batch], dim=0),
        "table": torch.stack([b["table"] for b in batch], dim=0),
        "col": torch.stack([b["col"] for b in batch], dim=0),
        "tmask": torch.stack([b["tmask"] for b in batch], dim=0),
        "cmask": torch.stack([b["cmask"] for b in batch], dim=0),
    }


def collate_stage2(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "q_emb": torch.stack([b["q_emb"] for b in batch], dim=0),
        "sql_emb": torch.stack([b["sql_emb"] for b in batch], dim=0),
        "table": torch.stack([b["table"] for b in batch], dim=0),
        "col": torch.stack([b["col"] for b in batch], dim=0),
        "tmask": torch.stack([b["tmask"] for b in batch], dim=0),
        "cmask": torch.stack([b["cmask"] for b in batch], dim=0),

        "sim_q_emb": torch.stack([b["sim_q_emb"] for b in batch], dim=0),
        "sim_sql_emb": torch.stack([b["sim_sql_emb"] for b in batch], dim=0),
        "sim_table": torch.stack([b["sim_table"] for b in batch], dim=0),
        "sim_col": torch.stack([b["sim_col"] for b in batch], dim=0),
        "sim_tmask": torch.stack([b["sim_tmask"] for b in batch], dim=0),
        "sim_cmask": torch.stack([b["sim_cmask"] for b in batch], dim=0),
    }


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class Config:
    stage1_data: str
    stage2_data: str
    dev_data: Optional[str]
    flag_path: str

    dim: int = 1024
    heads: int = 8
    dropout: float = 0.1
    max_tables: int = 10
    max_cols: int = 20
    max_sim: int = 3

    batch: int = 8
    stage1_epochs: int = 10
    stage2_epochs: int = 5
    lr1: float = 5e-5
    lr2: float = 5e-5
    temperature: float = 0.07

    w_sim: float = 0.3
    w_multi_pos: float = 0.2

    device: str = "cuda"
    seed: int = 42
    out_dir: str = "./ckpt"


@torch.no_grad()
def embed_all_dev(
    loader: DataLoader,
    stage1: SchemaAwareModel,
    stage2: Optional[GatedFusion],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    stage1.eval()
    if stage2 is not None:
        stage2.eval()

    q_list, sql_list = [], []
    for b in loader:
        q = b["q_emb"].to(device)
        sql = b["sql_emb"].to(device)
        table = b["table"].to(device)
        col = b["col"].to(device)
        tmask = b["tmask"].to(device)
        cmask = b["cmask"].to(device)

        s = stage1(q, table, col, tmask, cmask)
        q_final = s if stage2 is None else stage2(q, s)

        q_list.append(q_final)
        sql_list.append(sql)

    return torch.cat(q_list, dim=0), torch.cat(sql_list, dim=0)


def train_stage1(cfg: Config, train_loader: DataLoader, dev_loader: Optional[DataLoader], device: torch.device):
    stage1 = SchemaAwareModel(cfg.dim, cfg.heads, cfg.dropout).to(device)
    opt = torch.optim.AdamW(stage1.parameters(), lr=cfg.lr1, weight_decay=1e-5)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "stage1_best.pt")
    best = -1.0

    for ep in range(1, cfg.stage1_epochs + 1):
        stage1.train()
        total_loss, steps = 0.0, 0

        for b in tqdm(train_loader, desc=f"[Stage1 Train] ep={ep}", leave=False):
            q = b["q_emb"].to(device)
            sql = b["sql_emb"].to(device)
            table = b["table"].to(device)
            col = b["col"].to(device)
            tmask = b["tmask"].to(device)
            cmask = b["cmask"].to(device)

            s = stage1(q, table, col, tmask, cmask)
            loss = info_nce(s, sql, cfg.temperature)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1

        msg = f"[Stage1] ep={ep} loss={total_loss/max(steps,1):.4f}"

        if dev_loader is not None:
            q_dev, sql_dev = embed_all_dev(dev_loader, stage1, None, device)
            m = retrieval_metrics(q_dev, sql_dev, ks=(1, 5, 10))
            msg += f" | dev R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} MRR={m['MRR']:.3f}"
            score = m["R@10"]
            if score > best:
                best = score
                torch.save({"stage1": stage1.state_dict(), "cfg": cfg.__dict__}, best_path)
                msg += "  <-- saved"

        print(msg)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        stage1.load_state_dict(ckpt["stage1"])
    return stage1


def train_stage2(cfg: Config, train_loader: DataLoader, dev_loader: Optional[DataLoader], device: torch.device, stage1: SchemaAwareModel):
    stage2 = GatedFusion(cfg.dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(stage2.parameters(), lr=cfg.lr2, weight_decay=1e-5)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "stage2_best.pt")
    best = -1.0

    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False

    for ep in range(1, cfg.stage2_epochs + 1):
        stage2.train()
        total_loss, steps = 0.0, 0

        for b in tqdm(train_loader, desc=f"[Stage2 Train] ep={ep}", leave=False):
            q = b["q_emb"].to(device)
            sql = b["sql_emb"].to(device)
            table = b["table"].to(device)
            col = b["col"].to(device)
            tmask = b["tmask"].to(device)
            cmask = b["cmask"].to(device)

            sim_q = b["sim_q_emb"].to(device)
            sim_sql = b["sim_sql_emb"].to(device)
            sim_table = b["sim_table"].to(device)
            sim_col = b["sim_col"].to(device)
            sim_tmask = b["sim_tmask"].to(device)
            sim_cmask = b["sim_cmask"].to(device)

            B, S, D = sim_q.shape

            with torch.no_grad():
                s_main = stage1(q, table, col, tmask, cmask)

                sim_q_flat = sim_q.reshape(B*S, D)
                sim_table_flat = sim_table.reshape(B*S, cfg.max_tables, D)
                sim_col_flat = sim_col.reshape(B*S, cfg.max_tables, cfg.max_cols, D)
                sim_tmask_flat = sim_tmask.reshape(B*S, cfg.max_tables)
                sim_cmask_flat = sim_cmask.reshape(B*S, cfg.max_tables, cfg.max_cols)
                s_sim_flat = stage1(sim_q_flat, sim_table_flat, sim_col_flat, sim_tmask_flat, sim_cmask_flat)
                s_sim = s_sim_flat.reshape(B, S, D)

            q_final = stage2(q, s_main)
            sim_final = stage2(sim_q.reshape(B*S, D), s_sim.reshape(B*S, D)).reshape(B, S, D)

            loss_main = info_nce(q_final, sql, cfg.temperature)

            qn = l2norm(q_final).unsqueeze(1)
            sn = l2norm(sim_final)
            loss_sim = (1.0 - (qn * sn).sum(dim=-1)).mean()

            candidates = torch.cat([sql.unsqueeze(1), sim_sql], dim=1)
            w = torch.ones(B, 1 + S, device=device)
            w[:, 1:] = 0.5
            loss_mpos = multi_positive_nce(q_final, candidates, cfg.temperature, pos_weight=w)

            loss = loss_main + cfg.w_sim * loss_sim + cfg.w_multi_pos * loss_mpos

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1

        msg = f"[Stage2] ep={ep} loss={total_loss/max(steps,1):.4f}"

        if dev_loader is not None:
            q_dev, sql_dev = embed_all_dev(dev_loader, stage1, stage2, device)
            m = retrieval_metrics(q_dev, sql_dev, ks=(1, 5, 10))
            msg += f" | dev R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} MRR={m['MRR']:.3f}"
            score = m["R@10"]
            if score > best:
                best = score
                torch.save({"stage2": stage2.state_dict(), "cfg": cfg.__dict__}, best_path)
                msg += "  <-- saved"

        print(msg)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        stage2.load_state_dict(ckpt["stage2"])
    return stage2


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_data", required=True)
    ap.add_argument("--stage2_data", required=True)
    ap.add_argument("--dev_data", default=None)

    ap.add_argument("--flag", required=True)
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_tables", type=int, default=10)
    ap.add_argument("--max_cols", type=int, default=20)
    ap.add_argument("--max_sim", type=int, default=3)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stage1_epochs", type=int, default=10)
    ap.add_argument("--stage2_epochs", type=int, default=5)
    ap.add_argument("--lr1", type=float, default=5e-5)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--temp", type=float, default=0.07)

    ap.add_argument("--w_sim", type=float, default=0.3)
    ap.add_argument("--w_multi_pos", type=float, default=0.2)

    ap.add_argument("--cache_dir", default=None, help="[NEW] cache dir for pre-encoded .pt (optional)")

    args = ap.parse_args()

    cfg = Config(
        stage1_data=args.stage1_data,
        stage2_data=args.stage2_data,
        dev_data=args.dev_data,
        flag_path=args.flag,
        dim=args.dim,
        heads=args.heads,
        dropout=args.dropout,
        max_tables=args.max_tables,
        max_cols=args.max_cols,
        max_sim=args.max_sim,
        batch=args.batch,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lr1=args.lr1,
        lr2=args.lr2,
        temperature=args.temp,
        w_sim=args.w_sim,
        w_multi_pos=args.w_multi_pos,
        device=args.device,
        seed=args.seed,
        out_dir=args.out,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading FlagModel:", cfg.flag_path)
    flag = FlagModel(cfg.flag_path, use_fp16=(device.type == "cuda"))

    # [NEW] Warmup to verify model works (and show where it's stuck)
    print("[Warmup] encoding one sentence ...")
    _ = flag.encode("warmup")
    print("[Warmup] done.")

    cache = EmbedCache(flag)

    # [NEW] Decide cache paths
    def make_cache_path(json_path: str, kind: str) -> str:
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            base = os.path.basename(json_path)
            return os.path.join(args.cache_dir, base + f".{kind}.pt")
        return default_cache_path(json_path, kind)

    stage1_pt = make_cache_path(cfg.stage1_data, "stage1")
    stage2_pt = make_cache_path(cfg.stage2_data, "stage2")
    dev_pt = make_cache_path(cfg.dev_data, "dev") if cfg.dev_data else None

    # [NEW] Pre-encode with tqdm if missing
    if not os.path.exists(stage1_pt):
        preencode_stage1_json(cfg.stage1_data, stage1_pt, cache, cfg.dim, cfg.max_tables, cfg.max_cols)
    else:
        print("[Cache Found]", stage1_pt)

    if not os.path.exists(stage2_pt):
        preencode_stage2_json(cfg.stage2_data, stage2_pt, cache, cfg.dim, cfg.max_tables, cfg.max_cols, cfg.max_sim)
    else:
        print("[Cache Found]", stage2_pt)

    if cfg.dev_data:
        if not os.path.exists(dev_pt):
            preencode_stage1_json(cfg.dev_data, dev_pt, cache, cfg.dim, cfg.max_tables, cfg.max_cols)
        else:
            print("[Cache Found]", dev_pt)

    # [NEW] Load datasets from pt (fast)
    ds1 = PTListDataset(stage1_pt)
    ds2 = PTListDataset(stage2_pt)
    dev_ds = PTListDataset(dev_pt) if cfg.dev_data else None

    dl1 = DataLoader(ds1, batch_size=cfg.batch, shuffle=True, num_workers=0, collate_fn=collate_stage1)
    dl2 = DataLoader(ds2, batch_size=cfg.batch, shuffle=True, num_workers=0, collate_fn=collate_stage2)
    dev_dl = DataLoader(dev_ds, batch_size=cfg.batch, shuffle=False, num_workers=0, collate_fn=collate_stage1) if dev_ds else None

    stage1 = train_stage1(cfg, dl1, dev_dl, device)
    stage2 = train_stage2(cfg, dl2, dev_dl, device, stage1)

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"stage1": stage1.state_dict(), "cfg": cfg.__dict__}, os.path.join(cfg.out_dir, "stage1_final.pt"))
    torch.save({"stage2": stage2.state_dict(), "cfg": cfg.__dict__}, os.path.join(cfg.out_dir, "stage2_final.pt"))
    print("\nSaved:", cfg.out_dir)

    if dev_dl is not None:
        q_dev, sql_dev = embed_all_dev(dev_dl, stage1, stage2, device)
        m = retrieval_metrics(q_dev, sql_dev, ks=(1, 5, 10, 20))
        print("\n[Final Dev Metrics]")
        for k, v in m.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
