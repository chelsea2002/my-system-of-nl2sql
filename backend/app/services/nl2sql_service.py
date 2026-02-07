from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import sqlite3
import re

from app.services.chart_suggest_service import suggest_charts


# -------------------------
# Mock: /nl2sql/generate 时先跑通前端
# -------------------------
def generate_candidates_mock(db_id: str, question: str) -> Dict[str, Any]:
    # TODO: 后续在这里接：SchemaLinker + SAR + LLM + ParetoOptimal
    candidates = [
        {
            "id": "c1",
            "sql": "SELECT * FROM customers LIMIT 10;",
            "scores": {
                "executability": {"ok": True, "error": None},
                "schema": {"score": 0.55, "used": ["customers"], "missing": []},
                "examples": {"score": 0.30, "nearest": "SELECT * FROM customers LIMIT 5;", "distance": 0.70},
                "final_score": 0.42,
            },
        },
        {
            "id": "c2",
            "sql": "SELECT country, COUNT(*) AS cnt FROM customers GROUP BY country ORDER BY cnt DESC LIMIT 10;",
            "scores": {
                "executability": {"ok": True, "error": None},
                "schema": {"score": 0.82, "used": ["customers", "country"], "missing": []},
                "examples": {"score": 0.58, "nearest": "SELECT x, COUNT(*) FROM t GROUP BY x;", "distance": 0.42},
                "final_score": 0.70,
            },
        },
        {
            "id": "c3",
            "sql": "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 10;",
            "scores": {
                "executability": {"ok": True, "error": None},
                "schema": {"score": 0.76, "used": ["orders", "customer_id", "amount"], "missing": []},
                "examples": {"score": 0.46, "nearest": "SELECT id, SUM(x) FROM y GROUP BY id;", "distance": 0.54},
                "final_score": 0.61,
            },
        },
    ]

    return {
        "request_id": f"req_{int(time.time() * 1000)}",
        "candidates": candidates,
        "selected_sql": candidates[1]["sql"],
    }


# -------------------------
# Execute: /nl2sql/execute 用
# -------------------------

_SELECT_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

def _is_select_sql(sql: str) -> bool:
    return bool(_SELECT_RE.search(sql or ""))


def execute(
    db_path: str,
    sql: str,
    max_rows: int = 2000,
) -> Dict[str, Any]:
    """
    执行 SQL，返回：
      - ok / error
      - columns / rows / row_count
      - elapsed_ms
      - charts_suggest（仅 ok 且有 rows 时）
    """
    t0 = time.time()
    conn: Optional[sqlite3.Connection] = None

    # 基本保护：空 SQL
    if not sql or not str(sql).strip():
        return {
            "ok": False,
            "sql": sql or "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "elapsed_ms": 0,
            "error": "Empty SQL",
            "charts_suggest": [],
        }

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(sql)

        columns: List[str] = []
        rows: List[List[Any]] = []
        row_count: int = 0

        # SELECT / WITH：有结果集
        if cur.description is not None:
            columns = [d[0] for d in cur.description]

            fetched = cur.fetchmany(max_rows)
            rows = [list(r) for r in fetched]
            row_count = len(rows)

        else:
            # 非 SELECT：commit 并返回 rowcount
            conn.commit()
            rc = cur.rowcount
            row_count = int(rc) if rc is not None and rc != -1 else 0

        elapsed_ms = int((time.time() - t0) * 1000)

        # ✅ charts suggest：仅当有 rows & columns
        charts = []
        if columns and rows:
            try:
                charts = suggest_charts(columns, rows)
            except Exception:
                charts = []

        return {
            "ok": True,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "elapsed_ms": elapsed_ms,
            "error": None,
            "charts_suggest": charts,
        }

    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "ok": False,
            "sql": sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "elapsed_ms": elapsed_ms,
            "error": str(e),
            "charts_suggest": [],
        }

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
