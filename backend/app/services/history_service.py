import os
import json
import time
import sqlite3
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from app.core.config import settings
from app.services.schema_service import load_schema_text
from app.services.sar_index_service import build_index


# ----------------------------
# history db path
# ----------------------------
def _history_db_path() -> str:
    p1 = os.path.join(settings.WORKSPACE_DIR, "history.sqlite")
    p2 = os.path.join(settings.WORKSPACE_DIR, "history", "history.sqlite")
    if os.path.exists(p1):
        return p1
    return p2  # fallback


def _conn():
    db = _history_db_path()
    os.makedirs(os.path.dirname(db), exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_columns(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(history)")
    cols = {r[1] for r in cur.fetchall()}

    def add(name, ddl):
        if name not in cols:
            cur.execute(f"ALTER TABLE history ADD COLUMN {ddl}")

    add("gen_ms", "gen_ms INTEGER NOT NULL DEFAULT 0")
    add("exec_ms", "exec_ms INTEGER NOT NULL DEFAULT 0")
    add("total_ms", "total_ms INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def _ensure_table():
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_id TEXT NOT NULL,
                question TEXT NOT NULL,
                selected_sql TEXT NOT NULL,
                ok INTEGER NOT NULL DEFAULT 0,
                corrected_sql TEXT DEFAULT '',
                schema_snapshot TEXT DEFAULT '',
                gen_ms INTEGER NOT NULL DEFAULT 0,
                exec_ms INTEGER NOT NULL DEFAULT 0,
                total_ms INTEGER NOT NULL DEFAULT 0,
                imported_to_sar INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
            """
        )
        _ensure_columns(conn)
        conn.commit()
    finally:
        conn.close()


# ============================================================
# ✅ add_feedback：兼容两种调用
# 1) add_feedback(db_id=..., question=..., selected_sql=..., ok=..., corrected_sql=...)
# 2) add_feedback(payload_dict)
# 返回：history_id (int)
# ============================================================
def add_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload is None:
        payload = {}

    _ensure_table()

    db_id = (payload.get("db_id") or "").strip()
    question = (payload.get("question") or "").strip()
    selected_sql = (payload.get("selected_sql") or "").strip()
    ok = 1 if bool(payload.get("ok", False)) else 0
    corrected_sql = (payload.get("corrected_sql") or "").strip()

    gen_ms = int(payload.get("gen_ms") or 0)
    exec_ms = int(payload.get("exec_ms") or 0)
    total_ms = int(payload.get("total_ms") or (gen_ms + exec_ms))

    if not db_id or not question or not selected_sql:
        raise ValueError("missing required fields: db_id/question/selected_sql")

    schema_snapshot = ""
    try:
        schema_snapshot = load_schema_text(db_id)
    except Exception:
        pass

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO history (
                db_id, question, selected_sql, ok, corrected_sql,
                schema_snapshot, imported_to_sar, created_at,
                gen_ms, exec_ms, total_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                db_id, question, selected_sql, ok, corrected_sql,
                schema_snapshot, int(time.time()),
                gen_ms, exec_ms, total_ms
            ),
        )
        conn.commit()
        return {"ok": True, "id": int(cur.lastrowid)}
    finally:
        conn.close()



# 如果你 API 层希望返回 {ok,id}，用这个包装一下
def add_feedback_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    hid = add_feedback(payload=payload)
    return {"ok": True, "id": hid}


# ----------------------------
# list_history: 分页列表
# ----------------------------
def list_history(
    db_id: str = "",
    q: str = "",
    page: int = 1,
    page_size: int = 10,
) -> Dict[str, Any]:
    _ensure_table()

    page = max(1, int(page or 1))
    page_size = min(200, max(1, int(page_size or 10)))
    offset = (page - 1) * page_size

    where = []
    args: List[Any] = []
    if db_id:
        where.append("db_id = ?")
        args.append(db_id)
    if q:
        where.append("(question LIKE ? OR selected_sql LIKE ?)")
        args.extend([f"%{q}%", f"%{q}%"])

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = _conn()
    try:
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(1) AS cnt FROM history {where_sql}", args)
        total = int(cur.fetchone()["cnt"])

        cur.execute(
            f"""
            SELECT id, db_id, question, selected_sql AS sql, ok, corrected_sql,
                   imported_to_sar, created_at
            FROM history
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            args + [page_size, offset],
        )
        items = [dict(r) for r in cur.fetchall()]

        return {
            "ok": True,
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }
    finally:
        conn.close()


# ----------------------------
# get_detail: 单条详情
# ----------------------------
def get_detail(history_id: int) -> Dict[str, Any]:
    _ensure_table()

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM history WHERE id = ?", (int(history_id),))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": f"history id not found: {history_id}"}
        item = dict(row)
        if "sql" not in item:
            item["sql"] = item.get("selected_sql", "")
        return {"ok": True, "item": item}
    finally:
        conn.close()


# ----------------------------
# promote: 加入样本库 + 触发 SAR 索引
# ----------------------------
def _sar_dir(db_id: str) -> str:
    d = os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR, db_id)
    os.makedirs(d, exist_ok=True)
    return d


def _supervised_path(db_id: str) -> str:
    return os.path.join(_sar_dir(db_id), "supervised_data.json")


def _ensure_supervised_file(db_id: str) -> str:
    p = _supervised_path(db_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    return p


def promote_history_to_sar(history_id: int) -> Dict[str, Any]:
    _ensure_table()

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM history WHERE id = ?", (int(history_id),))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": f"history id not found: {history_id}"}

        item = dict(row)

        if int(item.get("ok", 0)) != 1:
            return {"ok": False, "error": "This history item is not ok=true, cannot promote."}

        if int(item.get("imported_to_sar", 0)) == 1:
            return {"ok": True, "history_id": history_id, "imported": True, "message": "already imported"}

        db_id = (item.get("db_id") or "").strip()
        question = (item.get("question") or "").strip()
        sql_text = (item.get("selected_sql") or "").strip()

        if not db_id or not question or not sql_text:
            return {"ok": False, "error": "history row missing required fields: db_id/question/selected_sql"}

        schema_snapshot = (item.get("schema_snapshot") or "").strip()
        if not schema_snapshot:
            try:
                schema_snapshot = load_schema_text(db_id)
            except Exception:
                schema_snapshot = ""

        # 写入 supervised_data.json（写双字段兼容）
        sup_path = _ensure_supervised_file(db_id)
        try:
            with open(sup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

        sample_id = f"h_{history_id}_{int(time.time())}"
        sample = {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": question,
            "query": sql_text,                 # ✅ SAR 返回 query
            "sql": sql_text,                   # ✅ 兼容其它模块
            "schema_snapshot": schema_snapshot,
            "source": "history_promote",
            "created_at": int(time.time()),
        }
        data.append(sample)

        with open(sup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 标记 imported_to_sar
        cur.execute("UPDATE history SET imported_to_sar = 1 WHERE id = ?", (int(history_id),))
        conn.commit()

        # 触发索引构建
        meta = build_index(db_id=db_id, mode="incremental")

        return {
            "ok": True,
            "history_id": history_id,
            "imported": True,
            "sample_id": sample_id,
            "sar_index_meta": meta,
        }
    finally:
        conn.close()



def get_history_stats(db_id: str = "") -> Dict[str, Any]:
    _ensure_table()

    # 今天 00:00 的 epoch 秒
    now = datetime.now()
    today0 = datetime(now.year, now.month, now.day)
    today_ts = int(today0.timestamp())

    conn = _conn()
    try:
        cur = conn.cursor()

        # 1) 先探测一下有没有 total_ms 列（避免你还没迁移就报错）
        cols = {r["name"] for r in cur.execute("PRAGMA table_info(history)").fetchall()}
        has_total_ms = "total_ms" in cols

        where = []
        args: List[Any] = []
        if db_id:
            where.append("db_id = ?")
            args.append(db_id)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # 2) 总量 / 成功量 / 今日量
        cur.execute(
            f"""
            SELECT
              COUNT(1) AS total,
              SUM(CASE WHEN ok=1 THEN 1 ELSE 0 END) AS ok_cnt,
              SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS today_cnt
            FROM history
            {where_sql}
            """,
            [today_ts] + args,
        )
        row = cur.fetchone()
        total = int(row["total"] or 0)
        ok_cnt = int(row["ok_cnt"] or 0)
        today_cnt = int(row["today_cnt"] or 0)

        success_rate = (ok_cnt / total) if total > 0 else None

        # 3) 平均耗时（如果你还没有 total_ms 字段，就返回 None）
        avg_latency = None
        if has_total_ms:
            cur.execute(f"SELECT AVG(total_ms) AS avg_ms FROM history {where_sql}", args)
            r2 = cur.fetchone()
            if r2 and r2["avg_ms"] is not None:
                avg_latency = int(round(float(r2["avg_ms"])))   # ✅ 强制变整数
        cur.execute(f"SELECT total_ms FROM history {where_sql} LIMIT 10", args)
        samples = [r["total_ms"] for r in cur.fetchall()]
        print(f"Sample total_ms values: {samples}")

        return {
            "ok": True,
            "db_id": db_id,
            "totalQueries": total,
            "todayQueries": today_cnt,
            "successRate": success_rate,
            "avgLatencyMs": avg_latency,
        }
    finally:
        conn.close()

