import os
import re
import json
import sqlite3
from typing import Any, Dict, List, Tuple
from app.core.config import settings
from app.services.function import extract_db_samples_enriched_bm25
# -----------------------------
# 你脚本里的：替换 Examples
# -----------------------------
def replace_schema_examples(schema_text: str, new_examples_text: str) -> str:
    new_examples: Dict[str, Dict[str, List[str]]] = {}
    current_table = None

    for line in new_examples_text.strip().split("\n"):
        if line.startswith("## ") and "table samples:" in line:
            current_table = line.split()[1]
            new_examples[current_table] = {}
        elif line.startswith("# Example values for "):
            if current_table:
                parts = line.split("'.")
                if len(parts) >= 2:
                    table_name = parts[0].split("'")[1]
                    column_name = parts[1].split("'")[1]

                    if "column: [" not in line:
                        continue
                    values_str = line.split("column: [")[1].split("]")[0]

                    values = []
                    for val in values_str.split(", "):
                        cleaned_val = val.strip("'")
                        values.append(cleaned_val)

                    new_examples.setdefault(table_name, {})
                    new_examples[table_name][column_name] = values

    lines = schema_text.split("\n")
    updated_lines = []
    current_table = None

    for line in lines:
        if line.startswith("# Table: "):
            current_table = line.split("# Table: ")[1].strip()
            updated_lines.append(line)
        elif "Examples:" in line and current_table in new_examples:
            m = re.search(r"\((\w+)\s*:", line)
            if not m:
                updated_lines.append(line)
                continue

            column_name = m.group(1)
            if column_name in new_examples[current_table]:
                new_examples_str = str(new_examples[current_table][column_name]).replace("'NULL'", "NULL")
                before_examples = line.split("Examples: ")[0]
                updated_line = f"{before_examples}Examples: {new_examples_str})"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)

# -----------------------------
# sqlite -> schema text
# -----------------------------
def normalize_type(sqlite_type: str) -> str:
    if not sqlite_type:
        return "TEXT"
    t = sqlite_type.upper()
    if "INT" in t:
        return "INT"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t or "VARCHAR" in t:
        return "TEXT"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    if "BOOL" in t:
        return "BOOL"
    if "DATE" in t or "TIME" in t:
        return "TEXT"
    return "TEXT"

def safe_format_examples(values: List[Any]) -> str:
    out = []
    for v in values:
        if v is None:
            out.append("NULL")
        else:
            s = str(v).replace("\n", " ").replace("\r", " ").strip()
            out.append(s)
    return "[" + ", ".join(out) + "]"

def get_table_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]

def get_table_columns(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str, bool]]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}');")
    cols = []
    for _, name, col_type, _, _, pk in cur.fetchall():
        cols.append((name, normalize_type(col_type), pk == 1))
    return cols

def get_column_examples(conn: sqlite3.Connection, table: str, col: str, k: int = 2) -> List[Any]:
    cur = conn.cursor()
    sql = f"""
    SELECT DISTINCT "{col}"
    FROM "{table}"
    WHERE "{col}" IS NOT NULL
    LIMIT {k};
    """
    try:
        cur.execute(sql)
        vals = [r[0] for r in cur.fetchall()]
        while len(vals) < k:
            vals.append(None)
        return vals
    except Exception:
        return [None] * k

def get_foreign_keys(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA foreign_key_list('{table}');")
        fks = []
        for _, _, ref_table, from_col, to_col, *_ in cur.fetchall():
            fks.append(f"{table}.{from_col}={ref_table}.{to_col}")
        return fks
    except Exception:
        return []

def build_database_text(db_id: str, db_path: str, k_examples: int = 2) -> str:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        tables = get_table_names(conn)
        lines = []
        lines.append(f"【DB_ID】 {db_id}")
        lines.append("【Schema】")

        all_fks: List[str] = []

        for table in tables:
            cols = get_table_columns(conn, table)

            lines.append(f"# Table: {table}")
            lines.append("[")
            for idx, (col, ctype, is_pk) in enumerate(cols):
                ex_vals = get_column_examples(conn, table, col, k=k_examples)
                ex_str = safe_format_examples(ex_vals)

                pk_str = ", Primary Key" if is_pk else ""
                comma = "," if idx < len(cols) - 1 else ""
                lines.append(f"({col}:{ctype}{pk_str}, Examples: {ex_str}){comma}")
            lines.append("]")

            all_fks.extend(get_foreign_keys(conn, table))

        lines.append("【Foreign keys】")
        seen = set()
        for fk in all_fks:
            if fk not in seen:
                lines.append(fk)
                seen.add(fk)

        return "\n".join(lines) + "\n"
    finally:
        conn.close()

def get_schema_dict(db_path: str) -> Dict[str, Any]:
    """给前端/其他模块用的结构化 schema：tables/columns/fks"""
    conn = sqlite3.connect(db_path)
    try:
        tables = get_table_names(conn)
        columns = {}
        fks = []
        for t in tables:
            cols = get_table_columns(conn, t)
            columns[t] = [c[0] for c in cols]
            fks.extend(get_foreign_keys(conn, t))
        return {"tables": tables, "columns": columns, "foreign_keys": fks}
    finally:
        conn.close()

# -----------------------------
# 简化版 enrich（你之后替换成 BM25）
# 约定输出文本格式和你 replace_schema_examples 兼容
# -----------------------------
def extract_db_samples_enriched_simple(
    question: str,
    db_path: str,
    schema_dict: Dict[str, Any],
    evidence: str,
    k: int = 2
) -> str:
    """
    输出示例文本（兼容 replace_schema_examples）：
    ## <table> table samples:
    # Example values for '<table>'.'<column>' column: [a, b]
    """
    q = (question or "").lower()
    conn = sqlite3.connect(db_path)
    try:
        lines = []
        for table in schema_dict.get("tables", []):
            # 简单策略：如果问题里命中 table 名，则优先 enrich 这个表
            hit = table.lower() in q
            # 若没有命中，则只 enrich 少数表（避免太慢）
            if not hit and len(lines) > 0:
                continue

            lines.append(f"## {table} table samples:")
            for col in schema_dict.get("columns", {}).get(table, [])[: min(12, len(schema_dict["columns"][table]))]:
                try:
                    cur = conn.cursor()
                    cur.execute(
                        f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {k};'
                    )
                    vals = [r[0] for r in cur.fetchall()]
                    # pad
                    while len(vals) < k:
                        vals.append(None)
                    # stringify
                    out = []
                    for v in vals:
                        if v is None:
                            out.append("NULL")
                        else:
                            out.append(str(v).replace("\n", " ").replace("\r", " ").strip())
                    lines.append(f"# Example values for '{table}'.'{col}' column: [{', '.join(out)}]")
                except Exception:
                    lines.append(f"# Example values for '{table}'.'{col}' column: [NULL, NULL]")

        if not lines:
            # fallback：随便 enrich 第一张表
            tables = schema_dict.get("tables", [])
            if tables:
                t = tables[0]
                lines.append(f"## {t} table samples:")
                for col in schema_dict.get("columns", {}).get(t, [])[:3]:
                    lines.append(f"# Example values for '{t}'.'{col}' column: [NULL, NULL]")

        return "\n".join(lines)
    finally:
        conn.close()

def _schema_dir(db_id: str) -> str:
    return os.path.join(settings.WORKSPACE_DIR, settings.SCHEMA_DIR, db_id)

def save_schema_files(db_id: str, schema_text: str, schema_dict: Dict[str, Any]) -> None:
    os.makedirs(_schema_dir(db_id), exist_ok=True)
    with open(os.path.join(_schema_dir(db_id), "schema.txt"), "w", encoding="utf-8") as f:
        f.write(schema_text)
    with open(os.path.join(_schema_dir(db_id), "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, ensure_ascii=False, indent=2)

def load_schema_text(db_id: str) -> str:
    path = os.path.join(_schema_dir(db_id), "schema.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("schema.txt not found, please call /schema/parse first")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_schema_dict(db_id: str) -> Dict[str, Any]:
    path = os.path.join(_schema_dir(db_id), "schema.json")
    if not os.path.exists(path):
        raise FileNotFoundError("schema.json not found, please call /schema/parse first")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_and_persist_schema(db_id: str, db_path: str, k_examples: int = 2) -> Dict[str, Any]:
    schema_text = build_database_text(db_id, db_path, k_examples=k_examples)  # ✅ 用传入的 k
    schema_dict = get_schema_dict(db_path)
    save_schema_files(db_id, schema_text, schema_dict)

    tables = len(schema_dict.get("tables", []))
    cols = sum(len(v) for v in schema_dict.get("columns", {}).values())
    fks = len(schema_dict.get("foreign_keys", []))

    return {
        "db_id": db_id,
        "tables": tables,
        "columns": cols,
        "foreign_keys": fks,
        "schema_text": schema_text,
        "schema_dict": schema_dict,
    }

def enrich_schema(db_id: str, db_path: str, question: str, evidence: str = "", k: int = 2) -> Dict[str, Any]:
    schema_text = load_schema_text(db_id)
    schema_dict = load_schema_dict(db_id)
    sample_text = extract_db_samples_enriched_bm25(question, db_path, schema_dict, evidence, k)
    enriched = replace_schema_examples(schema_text, sample_text)
    return {"schema_text_enriched": enriched, "sample_text": sample_text}
