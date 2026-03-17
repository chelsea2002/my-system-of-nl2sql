import os, json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from FlagEmbedding import FlagModel

from app.core.config import settings

def _sar_dir(db_id: str) -> str:
    d = os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR, db_id)
    os.makedirs(d, exist_ok=True)
    return d

def _supervised_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "supervised_data.json")

def _index_meta_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "index.json")

def _index_pt_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "index.pt")

def _id_map_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "id_map.json")

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _load_supervised(db_id: str) -> List[Dict[str, Any]]:
    p = _supervised_path(db_id)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(db_id: str, meta: Dict[str, Any]) -> None:
    with open(_index_meta_path(db_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def get_index_status(db_id: str) -> Dict[str, Any]:
    p = _index_meta_path(db_id)
    if not os.path.exists(p):
        return {
            "db_id": db_id,
            "status": "NOT_BUILT",
            "total_embeddings": 0,
            "updated_at": "-",
            "last_build_mode": "-",
        }
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 兜底字段
    return {
        "db_id": db_id,
        "status": meta.get("status", "UNKNOWN"),
        "total_embeddings": meta.get("total_embeddings", 0),
        "updated_at": meta.get("updated_at", "-"),
        "last_build_mode": meta.get("last_build_mode", "-"),
        "embedding_model": meta.get("embedding_model", ""),
        "dim": meta.get("dim", None),
    }

def _get_flag_model() -> FlagModel:
    # sar_index_service.py 在 backend/app/services/
    # backend 根目录 = 上上级的上一级：services -> app -> backend
    backend_dir = Path(__file__).resolve().parents[2]  # .../backend
    model_path = backend_dir / "assets" / "sar" / "bge-large-en-v1.5"

    if not model_path.exists():
        raise FileNotFoundError(f"SAR embedding model not found: {model_path}")

    return FlagModel(str(model_path), use_fp16=True)

def build_index(db_id: str, mode: str = "incremental") -> Dict[str, Any]:
    data = _load_supervised(db_id)
    if not data:
        meta = {
            "db_id": db_id,
            "status": "EMPTY",
            "total_embeddings": 0,
            "updated_at": _now_str(),
            "last_build_mode": mode,
            "embedding_model": "bge-large-en-v1.5",
            "dim": 0,
        }
        _save_meta(db_id, meta)
        return meta

    flag = _get_flag_model()

    # 你说 examples.json 里 SQL 字段叫 query；但 supervised_data 里你现在是 sql
    # 这里兼容两种
    texts = []
    ids = []
    for item in data:
        q = (item.get("question") or "").strip()
        sql = (item.get("query") or item.get("sql") or "").strip()
        # 用 question + sql 作为 example 表示（简单但好用）
        txt = f"Q: {q}\nSQL: {sql}"
        texts.append(txt)
        ids.append(item.get("sample_id") or f"row_{len(ids)}")

    # encode -> numpy (N, D)
    embs = flag.encode(texts)
    embs = torch.tensor(embs, dtype=torch.float32)

    # normalize，后面检索直接 dot 就是 cosine
    embs = F.normalize(embs, dim=-1)

    # 写文件
    sar_dir = _sar_dir(db_id)
    torch.save(embs, _index_pt_path(db_id))
    with open(_id_map_path(db_id), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    meta = {
        "db_id": db_id,
        "status": "READY",
        "total_embeddings": int(embs.shape[0]),
        "updated_at": _now_str(),
        "last_build_mode": mode,
        "embedding_model": "bge-large-en-v1.5",
        "dim": int(embs.shape[1]),
        "index_file": "index.pt",
        "id_map_file": "id_map.json",
    }
    _save_meta(db_id, meta)
    return meta

def retrieve_topk(db_id: str, question: str, k: int = 5) -> List[Dict[str, Any]]:
    # 必须 index.pt 存在
    if not os.path.exists(_index_pt_path(db_id)) or not os.path.exists(_id_map_path(db_id)):
        return []

    # load
    example_embs = torch.load(_index_pt_path(db_id), map_location="cpu")  # (N,D)
    with open(_id_map_path(db_id), "r", encoding="utf-8") as f:
        ids = json.load(f)

    # load supervised meta
    data = _load_supervised(db_id)
    by_id = { (x.get("sample_id") or ""): x for x in data }

    flag = _get_flag_model()
    q_emb = torch.tensor(flag.encode([question])[0], dtype=torch.float32)
    q_emb = F.normalize(q_emb, dim=-1)

    scores = torch.mv(example_embs, q_emb)  # (N,)
    k = min(k, scores.shape[0])
    vals, idx = torch.topk(scores, k=k)

    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        sid = ids[i] if i < len(ids) else ""
        item = by_id.get(sid, {})
        out.append({
            "score": float(v),
            "question": item.get("question", ""),
            "query": item.get("query") or item.get("sql") or "",   # ✅ 返回字段用 query
            "schema": item.get("schema", None),
            "sample_id": sid,
        })
    return out
