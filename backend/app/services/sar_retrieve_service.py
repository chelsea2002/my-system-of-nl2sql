import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from FlagEmbedding import FlagModel

from app.core.config import settings
from app.services.train_SAR import SchemaAwareModel, GatedFusion, schema_to_tensors

_EXAMPLES: Optional[List[Dict[str, Any]]] = None
_EXAMPLE_EMBS: Optional[torch.Tensor] = None

_FLAG: Optional[FlagModel] = None
_STAGE1: Optional[SchemaAwareModel] = None
_STAGE2: Optional[GatedFusion] = None
_CFG: Optional[Dict[str, Any]] = None

_LOADED: Dict[Tuple[str, str], bool] = {}  # (db_id, device) -> True


def _backend_root() -> str:
    return os.path.abspath(os.path.join(settings.WORKSPACE_DIR, ".."))


def _sar_assets_dir() -> str:
    return os.path.join(_backend_root(), "assets", "sar")


def _workspace_sar_dir(db_id: str) -> str:
    return os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR, db_id)


def _examples_path(db_id: str) -> str:
    return os.path.join(_workspace_sar_dir(db_id), "supervised_data.json")


def _embs_path_candidates(db_id: str) -> List[str]:
    """
    ✅ 两条路径都支持：
      1) backend/workspace/sar/{db_id}/index.pt   (你现在就有)
      2) backend/workspace/sar/{db_id}/example_embs.pt
    都没有才现场生成，并保存到 example_embs.pt（不覆盖 index.pt）
    """
    d = _workspace_sar_dir(db_id)
    return [
        os.path.join(d, "index.pt"),
        os.path.join(d, "example_embs.pt"),
    ]


def _save_embs_path(db_id: str) -> str:
    # 现场生成时，统一落到 example_embs.pt
    return os.path.join(_workspace_sar_dir(db_id), "example_embs.pt")


def _get_flag_model(device: str = "cpu") -> FlagModel:
    global _FLAG
    if _FLAG is not None:
        return _FLAG

    model_path = os.path.join(_sar_assets_dir(), "bge-large-en-v1.5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[SAR] FlagModel path not found: {model_path}")

    _FLAG = FlagModel(model_path, use_fp16=("cuda" in device))
    return _FLAG


def _load_sar_models(device: str = "cpu") -> None:
    global _STAGE1, _STAGE2, _CFG
    if _STAGE1 is not None and _STAGE2 is not None and _CFG is not None:
        return

    ckpt_dir = os.path.join(_sar_assets_dir(), "ckpt")
    s1 = os.path.join(ckpt_dir, "stage1_best.pt")
    s2 = os.path.join(ckpt_dir, "stage2_best.pt")

    if not os.path.exists(s1):
        raise FileNotFoundError(f"[SAR] stage1 ckpt not found: {s1}")
    if not os.path.exists(s2):
        raise FileNotFoundError(f"[SAR] stage2 ckpt not found: {s2}")

    ckpt1 = torch.load(s1, map_location=device)
    ckpt2 = torch.load(s2, map_location=device)

    cfg = ckpt1.get("cfg")
    if not isinstance(cfg, dict):
        raise RuntimeError("[SAR] stage1 ckpt missing cfg dict")
    _CFG = cfg

    stage1 = SchemaAwareModel(dim=cfg["dim"], heads=cfg["heads"], dropout=cfg["dropout"]).to(device)
    stage1.load_state_dict(ckpt1["stage1"])
    stage1.eval()

    stage2 = GatedFusion(dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    stage2.load_state_dict(ckpt2["stage2"])
    stage2.eval()

    _STAGE1 = stage1
    _STAGE2 = stage2


def _safe_schema(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict) and "tables" in s and "columns" in s:
        return s
    return {"tables": [], "columns": {}}


def _get_sql_field(item: Dict[str, Any]) -> str:
    """
    ✅ 兼容 supervised_data 里 SQL 字段可能叫 sql 或 query
    """
    v = item.get("query")
    if v:
        return v
    v = item.get("sql")
    if v:
        return v
    # 再兜底
    for k in ["SQL", "Query", "rag_sql", "gold_sql"]:
        if item.get(k):
            return item[k]
    return ""


class _FlagCache:
    def __init__(self, flag_model: FlagModel, device: str = "cpu"):
        self.flag = flag_model
        self.device = device
        self._mem = {}  # text -> tensor

    def enc(self, text: str) -> torch.Tensor:
        # 训练代码期望返回 shape (dim,) 的 tensor
        if text in self._mem:
            return self._mem[text]
        v = self.flag.encode(text)
        t = torch.tensor(v, dtype=torch.float32, device=self.device)
        self._mem[text] = t
        return t

@torch.no_grad()
def _encode_query_with_schema(question: str, schema: dict, device: str = "cpu") -> torch.Tensor:
    assert _FLAG is not None, "_FLAG not loaded"
    assert _STAGE1 is not None and _STAGE2 is not None and _CFG is not None, "SAR models not loaded"

    q_emb = torch.tensor(_FLAG.encode(question), dtype=torch.float32, device=device).unsqueeze(0)

    schema = _safe_schema(schema)
    cache = _FlagCache(_FLAG, device=device) 
    table, col, tmask, cmask = schema_to_tensors(
        cache=cache,
        schema=schema,
        dim=_CFG["dim"],
        max_tables=_CFG["max_tables"],
        max_cols=_CFG["max_cols"],
    )

    table = table.unsqueeze(0).to(device)
    col = col.unsqueeze(0).to(device)
    tmask = tmask.unsqueeze(0).to(device)
    cmask = cmask.unsqueeze(0).to(device)

    s = _STAGE1(q_emb, table, col, tmask, cmask)
    q_final = _STAGE2(q_emb, s)
    q_final = F.normalize(q_final, dim=-1)
    return q_final.squeeze(0)


@torch.no_grad()
def _build_example_embs_in_memory(db_id: str, device: str = "cpu") -> torch.Tensor:
    """
    ✅ 当两个 pt 都没有时：用 supervised_data.json 现场构建并保存到 example_embs.pt
    """
    assert _EXAMPLES is not None

    embs: List[torch.Tensor] = []
    for item in _EXAMPLES:
        q = item.get("question", "") or ""
        s = _safe_schema(item.get("schema"))
        emb = _encode_query_with_schema(q, s, device=device)
        embs.append(emb)

    mat = torch.stack(embs, dim=0)  # (N,D)

    os.makedirs(_workspace_sar_dir(db_id), exist_ok=True)
    torch.save(mat.cpu(), _save_embs_path(db_id))  # 保存 cpu 版本
    return mat


def _load_embs_any(db_id: str, device: str) -> torch.Tensor:
    """
    ✅ 优先加载 index.pt，再加载 example_embs.pt，都没有则生成
    """
    for p in _embs_path_candidates(db_id):
        if os.path.exists(p):
            t = torch.load(p, map_location=device)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(f"[SAR] embedding file is not a Tensor: {p}")
            return t.to(device)

    # 都没有：生成并保存
    return _build_example_embs_in_memory(db_id, device=device).to(device)


def _load_once(db_id: str, device: str = "cpu") -> None:
    global _EXAMPLES, _EXAMPLE_EMBS

    key = (db_id, device)
    if _LOADED.get(key):
        return

    _get_flag_model(device=device)
    _load_sar_models(device=device)

    p = _examples_path(db_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[SAR] supervised_data.json not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        _EXAMPLES = json.load(f)
    if not isinstance(_EXAMPLES, list):
        raise RuntimeError("[SAR] supervised_data.json must be a list")

    _EXAMPLE_EMBS = _load_embs_any(db_id=db_id, device=device)

    # ✅ 对齐：确保 embeddings 数量 >= examples 数量（不一致就截断到最小，避免索引越界）
    n_ex = len(_EXAMPLES)
    n_emb = int(_EXAMPLE_EMBS.shape[0])
    if n_ex != n_emb:
        m = min(n_ex, n_emb)
        _EXAMPLES = _EXAMPLES[:m]
        _EXAMPLE_EMBS = _EXAMPLE_EMBS[:m]

    _LOADED[key] = True


@torch.no_grad()
def sar_retrieve_topk(
    db_id: str,
    question: str,
    schema: dict,
    k: int = 5,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    _load_once(db_id=db_id, device=device)

    assert _EXAMPLES is not None
    assert _EXAMPLE_EMBS is not None

    q = _encode_query_with_schema(question, schema, device=device)  # (D,)
    embs = F.normalize(_EXAMPLE_EMBS, dim=-1)                      # (N,D)
    scores = torch.mv(embs, q)                                     # (N,)

    k = max(1, min(int(k), int(scores.shape[0])))
    vals, idx = torch.topk(scores, k=k)

    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        item = _EXAMPLES[i]
        out.append({
            "score": float(v),
            "question": item.get("question", ""),
            # ✅ 统一对外叫 query，但内部兼容 sql/query
            "query": _get_sql_field(item),
            "schema": item.get("schema", None),
            "db_id": item.get("db_id", None),
            "sample_id": item.get("sample_id", None),
        })
    return out
