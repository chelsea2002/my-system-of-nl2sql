import json
import os
import time
from typing import Any, Dict, List, Optional

from app.services.schema_service import load_schema_dict
from app.core.config import settings

def _workspace_sar_dir(db_id: str) -> str:
    return os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR, db_id)

def _supervised_path(db_id: str) -> str:
    return os.path.join(_workspace_sar_dir(db_id), "supervised_data.json")

def _history_path(db_id: str) -> str:
    # 你也可以统一放一个 history.jsonl，这里按 db 分开更直观
    return os.path.join(settings.WORKSPACE_DIR, "history", f"{db_id}.jsonl")

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_history_record(db_id: str, rec: Dict[str, Any]) -> None:
    path = _history_path(db_id)
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def upsert_supervised_sample(
    db_id: str,
    question: str,
    sql: str,
    schema_snapshot: Dict[str, Any],
) -> bool:
    """
    写入 supervised_data.json
    - 若已存在完全相同 (question, sql)，则不重复写入
    - 返回：是否真的新增了样本
    """
    path = _supervised_path(db_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    # 去重：question+sql
    qn = (question or "").strip()
    sn = (sql or "").strip()
    for item in data:
        if (item.get("question", "").strip() == qn) and (item.get("query", item.get("sql", "")).strip() == sn):
            return False

    data.append({
        "db_id": db_id,
        "question": qn,
        # 兼容两种字段：你之前提到 supervised 可能叫 query 或 sql
        "query": sn,
        "sql": sn,
        "schema": schema_snapshot,
        "source": "feedback",
        "created_at": int(time.time() * 1000),
    })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True

def mark_sar_index_dirty(db_id: str) -> None:
    """
    最轻量的“触发增量索引”方式：
    写一个 dirty 标记，前端或定时任务/下一次 build 时发现 dirty 就重建。
    你也可以在这里直接调用 build 索引的函数。
    """
    sar_dir = _workspace_sar_dir(db_id)
    os.makedirs(sar_dir, exist_ok=True)
    dirty_path = os.path.join(sar_dir, ".index_dirty")
    with open(dirty_path, "w", encoding="utf-8") as f:
        f.write(str(int(time.time() * 1000)))
