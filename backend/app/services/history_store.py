import os, sqlite3, time, uuid
from typing import Optional, Dict, Any, List, Tuple

def _now_ms() -> int:
    return int(time.time() * 1000)

def _gen_id() -> str:
    return f"h_{_now_ms()}_{uuid.uuid4().hex[:8]}"

class HistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
              id TEXT PRIMARY KEY,
              db_id TEXT NOT NULL,
              question TEXT NOT NULL,
              selected_sql TEXT NOT NULL,
              ok_flag INTEGER NOT NULL,
              corrected_sql TEXT DEFAULT '',
              created_at INTEGER NOT NULL,
              imported_to_sar INTEGER NOT NULL DEFAULT 0,
              schema_snapshot_path TEXT DEFAULT ''
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_dbid_time ON history(db_id, created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_ok ON history(ok_flag);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_imported ON history(imported_to_sar);")

    def add_feedback(
        self,
        db_id: str,
        question: str,
        selected_sql: str,
        ok: bool,
        corrected_sql: str = "",
        schema_snapshot_path: str = "",
    ) -> str:
        hid = _gen_id()
        with self._conn() as conn:
            conn.execute("""
              INSERT INTO history(id, db_id, question, selected_sql, ok_flag, corrected_sql, created_at, imported_to_sar, schema_snapshot_path)
              VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                hid, db_id, question, selected_sql, 1 if ok else 0,
                corrected_sql or "", _now_ms(), 0, schema_snapshot_path or ""
            ))
        return hid

    def list(
        self,
        db_id: Optional[str] = None,
        q: Optional[str] = None,
        ok: Optional[bool] = None,
        imported: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        where = []
        args: List[Any] = []

        if db_id:
            where.append("db_id = ?")
            args.append(db_id)

        if ok is not None:
            where.append("ok_flag = ?")
            args.append(1 if ok else 0)

        if imported is not None:
            where.append("imported_to_sar = ?")
            args.append(1 if imported else 0)

        if q:
            where.append("(question LIKE ? OR selected_sql LIKE ? OR corrected_sql LIKE ?)")
            like = f"%{q}%"
            args.extend([like, like, like])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        offset = max(0, (page - 1) * page_size)

        with self._conn() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM history {where_sql}", args).fetchone()[0]
            rows = conn.execute(f"""
              SELECT id, db_id, question, selected_sql, ok_flag, corrected_sql, created_at, imported_to_sar, schema_snapshot_path
              FROM history
              {where_sql}
              ORDER BY created_at DESC
              LIMIT ? OFFSET ?
            """, args + [page_size, offset]).fetchall()

        items = []
        for r in rows:
            items.append({
                "id": r[0],
                "db_id": r[1],
                "question": r[2],
                "selected_sql": r[3],
                "ok_flag": bool(r[4]),
                "corrected_sql": r[5] or "",
                "created_at": int(r[6]),
                "imported_to_sar": bool(r[7]),
                "schema_snapshot_path": r[8] or "",
            })
        return total, items

    def mark_imported(self, history_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE history SET imported_to_sar = 1 WHERE id = ?", (history_id,))
