# app/services/sar_service.py
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from app.core.config import settings

def _sar_dir(db_id: str) -> str:
    d = os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR, db_id)
    os.makedirs(d, exist_ok=True)
    return d

def _supervised_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "supervised_data.json")

def _load_supervised(db_id: str) -> List[Dict[str, Any]]:
    p = _supervised_path(db_id)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_supervised(db_id: str, data: List[Dict[str, Any]]) -> None:
    p = _supervised_path(db_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _new_sample_id(db_id: str) -> str:
    return f"s_{db_id}_{int(datetime.now().timestamp()*1000)}"

def add_sample(
    db_id: str,
    question: str,
    sql: str,
    schema: Optional[Dict[str, Any]],
    source: str,
    verified: bool
) -> Dict[str, Any]:
    data = _load_supervised(db_id)
    key = f"{question}|||{sql}"
    existed = set(f"{x.get('question','')}|||{x.get('sql','')}" for x in data)
    if key in existed:
        raise ValueError("DUPLICATE")

    sid = _new_sample_id(db_id)
    item = {
        "sample_id": sid,
        "db_id": db_id,
        "question": question,
        "sql": sql,
        "schema": schema,
        "source": source,
        "verified": verified,
        "created_at": _now_str(),
    }
    data.append(item)
    _save_supervised(db_id, data)

    dataset = {
        "dataset_id": f"ds_{db_id}",
        "db_id": db_id,
        "version": len(data),
        "total_samples": len(data),
        "updated_at": _now_str(),
    }
    return {"sample_id": sid, "dataset": dataset}

def import_supervised_file(
    db_id: str,
    file_text: str,
    mode: str = "merge",
    source: str = "import"
) -> Dict[str, Any]:
    """
    支持：
    - JSON array: [{question, query/sql, schema?, verified?}, ...]
    - JSONL: 每行一个 JSON
    兼容 SQL 字段：query / sql
    """
    text = (file_text or "").strip()
    if not text:
        raise ValueError("EMPTY_FILE")

    # 1) parse
    try:
        if text.startswith("["):
            records = json.loads(text)
            if not isinstance(records, list):
                raise ValueError
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception:
        raise ValueError("PARSE_ERROR")

    # 2) load existing
    data = [] if mode == "replace" else _load_supervised(db_id)
    existed = set(f"{x.get('question','')}|||{x.get('sql','')}" for x in data)

    imported_total = len(records)
    imported_valid = 0
    imported_invalid = 0
    added = 0

    now = _now_str()

    for r in records:
        if not isinstance(r, dict):
            imported_invalid += 1
            continue

        question = (r.get("question") or r.get("rag_text") or "").strip()
        sql = (r.get("sql") or r.get("query") or r.get("rag_sql") or "").strip()  # ✅ 兼容 query
        schema = r.get("schema", None)
        verified = bool(r.get("verified", True))

        if not question or not sql:
            imported_invalid += 1
            continue

        imported_valid += 1
        key = f"{question}|||{sql}"
        if key in existed:
            continue

        item = {
            "sample_id": r.get("sample_id") or _new_sample_id(db_id),
            "db_id": db_id,
            "question": question,
            "sql": sql,
            "schema": schema,
            "source": r.get("source") or source,
            "verified": verified,
            "created_at": r.get("created_at") or now,
        }
        data.append(item)
        existed.add(key)
        added += 1

    _save_supervised(db_id, data)

    dataset = {
        "dataset_id": f"ds_{db_id}",
        "db_id": db_id,
        "version": len(data),
        "total_samples": len(data),
        "updated_at": _now_str(),
    }

    return {
        "imported_total": imported_total,
        "imported_valid": imported_valid,
        "imported_invalid": imported_invalid,
        "added": added,
        "dataset": dataset,
    }

def _index_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "index.json")

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_index_status(db_id: str):
    p = _index_path(db_id)
    if not os.path.exists(p):
        return {
            "status": "NOT_BUILT",
            "total_embeddings": 0,
            "updated_at": "-",
            "last_build_mode": "-",
        }
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兜底补字段（避免前端空）
    return {
        "status": data.get("status", "READY"),
        "total_embeddings": int(data.get("total_embeddings", 0)),
        "updated_at": data.get("updated_at", "-"),
        "last_build_mode": data.get("last_build_mode", "-"),
    }

def build_index(db_id: str, mode: str = "incremental") -> Dict[str, Any]:
    """
    最小可用版：
    - 不做真实 embedding，只用 supervised_data 的样本数作为 total_embeddings
    - 写入 index.json，供 /sar/index/status 查询
    """
    if mode not in ("incremental", "full"):
        raise ValueError("BAD_MODE")

    data = _load_supervised(db_id)
    total = len(data)

    idx = {
        "status": "READY" if total > 0 else "EMPTY",
        "total_embeddings": total,
        "updated_at": _now_str(),
        "last_build_mode": mode,
    }

    os.makedirs(_sar_dir(db_id), exist_ok=True)
    with open(_index_path(db_id), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    return idx

def list_samples(
    db_id: str,
    q: str = "",
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    data = _load_supervised(db_id)

    # 倒序：最新在前
    data = list(reversed(data))

    q = (q or "").strip()
    if q:
        def hit(x: Dict[str, Any]) -> bool:
            return (q in (x.get("question") or "")) or (q in (x.get("sql") or ""))
        data = [x for x in data if hit(x)]

    total = len(data)
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    start = (page - 1) * page_size
    items = data[start:start + page_size]

    # 前端字段兼容：确保有 sample_id/sql/question/created_at/verified/source
    norm_items = []
    for x in items:
        norm_items.append({
            "sample_id": x.get("sample_id"),
            "db_id": x.get("db_id", db_id),
            "question": x.get("question", ""),
            "sql": x.get("sql", ""),  # ✅ 你落库字段是 sql
            "schema": x.get("schema", None),
            "source": x.get("source", "history"),
            "verified": bool(x.get("verified", True)),
            "created_at": x.get("created_at", ""),
        })

    return {"items": norm_items, "total": total}


def delete_sample(db_id: str, sample_id: str) -> Dict[str, Any]:
    data = _load_supervised(db_id)
    before = len(data)
    data = [x for x in data if x.get("sample_id") != sample_id]
    after = len(data)

    if after == before:
        # 没找到：返回 ok=false 更友好，也可以抛 404
        return {"ok": False, "deleted": 0}

    _save_supervised(db_id, data)

    # 顺手更新 dataset 信息（简单版）
    dataset = {
        "dataset_id": f"ds_{db_id}",
        "db_id": db_id,
        "version": len(data),  # 简化：用样本数当版本
        "total_samples": len(data),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"ok": True, "deleted": 1, "dataset": dataset}

from datetime import datetime

def get_dataset_summary(db_id: str) -> Dict[str, Any]:
    data = _load_supervised(db_id)
    total = len(data)

    # 最近更新时间：取最后一条的 created_at；没有就 '-'
    updated_at = "-"
    if total > 0:
        # 尽量取最后写入的那条（你的 add_sample 是 append）
        last = data[-1]
        updated_at = last.get("created_at") or "-"

    dataset = {
        "dataset_id": f"ds_{db_id}",
        "db_id": db_id,
        "version": total,          # 简化：用样本数当版本
        "total_samples": total,
        "updated_at": updated_at,
    }

    # 先给 index 一个占位（你后面会实现 /sar/index/status/build）
    index = {
        "status": "NOT_BUILT",
        "total_embeddings": 0,
        "updated_at": "-",
        "last_build_mode": "-",
    }

    return {"db_id": db_id, "dataset": dataset, "index": index}
