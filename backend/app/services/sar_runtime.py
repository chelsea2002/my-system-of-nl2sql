import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from FlagEmbedding import FlagModel

# ===== 训练定义（必须与训练时一致）=====
from app.services.train_SAR import SchemaAwareModel, GatedFusion, schema_to_tensors


class _EncCache:
    """
    schema_to_tensors 里会调用 cache.enc(text)，你之前传 None 会炸。
    这里用 FlagModel.encode 做一个最小可用 cache。
    """
    def __init__(self, flag_model: FlagModel):
        self.flag = flag_model

    def enc(self, text: str) -> torch.Tensor:
        # FlagModel.encode -> numpy
        v = self.flag.encode(text)
        # 转 torch (CPU)
        return torch.tensor(v, dtype=torch.float32)


class SARRuntime:
    """
    FastAPI 常驻 SAR runtime：
    - 启动时只 load 模型（bge + stage1/2）
    - 每个 db_id 的 examples/index 按需 lazy load + cache
    """

    def __init__(
        self,
        assets_dir: str,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ) -> None:
        self.assets_dir = os.path.abspath(assets_dir)

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_fp16 = use_fp16

        # --------- backend root / workspace root 推导 ----------
        # assets_dir = .../backend/assets/sar
        # backend_root = .../backend
        self.backend_root = os.path.abspath(os.path.join(self.assets_dir, "..", ".."))
        self.workspace_root = os.path.join(self.backend_root, "workspace")
        self.workspace_sar_root = os.path.join(self.workspace_root, "sar")

        # --------- model paths (必须存在) ----------
        self.bge_dir = os.path.join(self.assets_dir, "bge-large-en-v1.5")
        self.ckpt_stage1 = os.path.join(self.assets_dir, "ckpt", "stage1_best.pt")
        self.ckpt_stage2 = os.path.join(self.assets_dir, "ckpt", "stage2_best.pt")

        self._check_model_paths()

        # --------- load shared models ----------
        self.flag_model = FlagModel(self.bge_dir, use_fp16=(use_fp16 and "cuda" in self.device))
        self.cache = _EncCache(self.flag_model)

        self.stage1, self.stage2, self.cfg = self._load_sar_models(
            self.ckpt_stage1, self.ckpt_stage2, device=self.device
        )

        # --------- per-db cache ----------
        # key=(db_id) -> (examples_list, example_embs_tensor_normed)
        self._db_cache: Dict[str, Tuple[List[Dict[str, Any]], torch.Tensor]] = {}

    # ------------------------
    # Paths
    # ------------------------
    def _check_model_paths(self) -> None:
        must_exist = [self.bge_dir, self.ckpt_stage1, self.ckpt_stage2]
        for p in must_exist:
            if not os.path.exists(p):
                raise FileNotFoundError(f"[SARRuntime] Missing required model asset: {p}")

    def _workspace_db_dir(self, db_id: str) -> str:
        return os.path.join(self.workspace_sar_root, db_id)

    def _resolve_examples_and_index_paths(self, db_id: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        返回：
          examples_path: json（必需：workspace 优先，否则 assets）
          index_pt_path: 优先 index.pt，其次 example_embs.pt（若都无则 None）
          save_index_path: 若需要生成 embeddings 保存到哪里（优先 workspace/index.pt，否则 assets/example_embs.pt）
        """
        # workspace
        ws_dir = self._workspace_db_dir(db_id)
        ws_examples = os.path.join(ws_dir, "supervised_data.json")
        ws_index_pt = os.path.join(ws_dir, "index.pt")
        ws_example_embs_pt = os.path.join(ws_dir, "example_embs.pt")

        # assets fallback
        assets_examples = os.path.join(self.assets_dir, "examples.json")
        assets_example_embs = os.path.join(self.assets_dir, "example_embs.pt")

        # choose examples json
        if os.path.exists(ws_examples):
            examples_path = ws_examples
        elif os.path.exists(assets_examples):
            examples_path = assets_examples
        else:
            # 两边都没有
            examples_path = ws_examples  # 把期望路径带出来，方便报错
            raise FileNotFoundError(
                f"[SARRuntime] Missing examples json. Tried:\n- {ws_examples}\n- {assets_examples}"
            )

        # choose index pt (load)
        index_pt_path = None
        if os.path.exists(ws_index_pt):
            index_pt_path = ws_index_pt
        elif os.path.exists(ws_example_embs_pt):
            index_pt_path = ws_example_embs_pt
        elif os.path.exists(assets_example_embs):
            index_pt_path = assets_example_embs

        # choose save path for generated embeddings
        # 优先写回 workspace/index.pt（你想要的）
        save_index_path = None
        if os.path.exists(ws_dir) or True:
            os.makedirs(ws_dir, exist_ok=True)
            save_index_path = ws_index_pt
        else:
            save_index_path = assets_example_embs

        return examples_path, index_pt_path, save_index_path

    # ------------------------
    # Model load
    # ------------------------
    def _load_sar_models(self, stage1_ckpt: str, stage2_ckpt: str, device: str):
        ckpt1 = torch.load(stage1_ckpt, map_location=device)
        ckpt2 = torch.load(stage2_ckpt, map_location=device)

        cfg = ckpt1["cfg"]

        stage1 = SchemaAwareModel(
            dim=cfg["dim"],
            heads=cfg["heads"],
            dropout=cfg["dropout"],
        ).to(device)
        stage1.load_state_dict(ckpt1["stage1"])
        stage1.eval()

        stage2 = GatedFusion(
            dim=cfg["dim"],
            dropout=cfg["dropout"],
        ).to(device)
        stage2.load_state_dict(ckpt2["stage2"])
        stage2.eval()

        return stage1, stage2, cfg

    # ------------------------
    # Encoding
    # ------------------------
    @torch.no_grad()
    def encode_query_with_schema(self, question: str, schema: Dict[str, Any]) -> torch.Tensor:
        """
        question + schema -> (D,)
        """
        q_np = self.flag_model.encode(question)
        q_emb = torch.tensor(q_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,D)

        # schema -> tensors（⚠️ train_SAR.schema_to_tensors 会用 cache.enc）
        table, col, tmask, cmask = schema_to_tensors(
            cache=self.cache,   # ✅ 关键：不能是 None
            schema=schema,
            dim=self.cfg["dim"],
            max_tables=self.cfg["max_tables"],
            max_cols=self.cfg["max_cols"],
        )

        table = table.unsqueeze(0).to(self.device)
        col = col.unsqueeze(0).to(self.device)
        tmask = tmask.unsqueeze(0).to(self.device)
        cmask = cmask.unsqueeze(0).to(self.device)

        s = self.stage1(q_emb, table, col, tmask, cmask)
        q_final = self.stage2(q_emb, s)
        q_final = F.normalize(q_final, dim=-1)  # (1,D)
        return q_final.squeeze(0)  # (D,)

    # ------------------------
    # Load per-db examples & embs
    # ------------------------
    def _load_db_once(self, db_id: str) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        if db_id in self._db_cache:
            return self._db_cache[db_id]

        examples_path, index_pt_path, save_index_path = self._resolve_examples_and_index_paths(db_id)

        examples: List[Dict[str, Any]] = json.load(open(examples_path, "r", encoding="utf-8"))
        if not isinstance(examples, list):
            raise TypeError(f"[SARRuntime] examples json must be a list, got {type(examples)}: {examples_path}")

        # load embeddings if exists
        embs: Optional[torch.Tensor] = None
        if index_pt_path and os.path.exists(index_pt_path):
            embs = torch.load(index_pt_path, map_location=self.device)
            if not isinstance(embs, torch.Tensor):
                raise TypeError(f"[SARRuntime] index pt must be torch.Tensor: {index_pt_path}")
        else:
            # build embeddings on the fly
            embs = self._build_and_save_example_embs(examples, save_index_path)

        # normalize once (加速)
        embs = F.normalize(embs, dim=-1)

        # sanity check dim
        dim = embs.shape[-1]
        if dim != self.cfg["dim"]:
            raise ValueError(f"[SARRuntime] Embedding dim mismatch: embs dim={dim}, cfg dim={self.cfg['dim']}")

        self._db_cache[db_id] = (examples, embs)
        return examples, embs

    @torch.no_grad()
    def _build_and_save_example_embs(self, examples: List[Dict[str, Any]], save_path: str) -> torch.Tensor:
        """
        如果 workspace 没有 index.pt/example_embs.pt，则用 SAR 编码生成并保存。
        """
        vecs: List[torch.Tensor] = []

        for item in examples:
            q = (item.get("question") or item.get("nlq") or "").strip()
            if not q:
                # 没有 question 的样本直接跳过，用 0 向量占位也行；这里用 0 向量避免 shape 问题
                vecs.append(torch.zeros(self.cfg["dim"], device=self.device))
                continue

            schema = item.get("schema") or item.get("schema_dict") or {}
            if not isinstance(schema, dict):
                schema = {}

            v = self.encode_query_with_schema(q, schema)  # (D,)
            vecs.append(v)

        embs = torch.stack(vecs, dim=0)  # (N,D)

        # 保存到 workspace/index.pt
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(embs.detach().cpu(), save_path)
        except Exception as e:
            # 保存失败不影响检索，只打日志
            print(f"[SARRuntime] warn: failed to save index to {save_path}: {e}")

        return embs

    # ------------------------
    # Retrieve
    # ------------------------
    @torch.no_grad()
    def retrieve(self, db_id: str, question: str, schema: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        return: [{score, question, sql/query, schema?}, ...]
        """
        if not question or not question.strip():
            return []

        examples, example_embs = self._load_db_once(db_id)

        k = max(1, int(k))
        k = min(k, len(examples)) if examples else 0
        if k == 0:
            return []

        q_emb = self.encode_query_with_schema(question, schema)  # (D,)

        # cosine: (N,D) dot (D,) -> (N,)
        scores = torch.mv(example_embs, q_emb)  # already normalized
        vals, idx = torch.topk(scores, k=k)

        results: List[Dict[str, Any]] = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            item = examples[i]

            # ✅ SQL 字段两手准备：sql / query
            sql_text = item.get("sql", None)
            if not sql_text:
                sql_text = item.get("query", "")

            results.append(
                {
                    "score": float(v),
                    "question": item.get("question", ""),
                    "sql": sql_text,
                    "schema": item.get("schema", None),
                    "db_id": item.get("db_id", db_id),
                }
            )
        return results
