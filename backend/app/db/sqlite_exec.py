import sqlite3
import time
from typing import Any, Dict, List, Tuple, Optional
from app.core.config import settings

def _add_limit_if_needed(sql: str, limit: int) -> str:
    s = (sql or "").strip().rstrip(";")
    if not s:
        return s
    # 若已有 LIMIT 则不加；若是 EXPLAIN/PRAGMA 等不加
    upper = s.upper()
    if "LIMIT" in upper or upper.startswith("EXPLAIN") or upper.startswith("PRAGMA"):
        return s
    return f"{s} LIMIT {limit}"

def execute_sql(db_path: str, sql: str, limit: int = None) -> Tuple[bool, int, List[str], List[List[Any]], Optional[str]]:
    if limit is None:
        limit = settings.DEFAULT_QUERY_LIMIT

    start = time.time()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        safe_sql = _add_limit_if_needed(sql, limit)
        cur.execute(safe_sql)
        rows = cur.fetchall()

        cols = rows[0].keys() if rows else []
        data = [list(r) for r in rows]

        conn.close()
        ms = int((time.time() - start) * 1000)
        return True, ms, list(cols), data, None
    except Exception as e:
        ms = int((time.time() - start) * 1000)
        try:
            conn.close()
        except:
            pass
        return False, ms, [], [], str(e)
