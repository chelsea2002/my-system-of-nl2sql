import re
import time
import logging
from typing import Any, Dict, List, Tuple, Optional

from app.db.registry import get_db_path
from app.services.schema_service import load_schema_text, load_schema_dict
from app.services.sar_retrieve_service import sar_retrieve_topk
from app.services.llm import LLM_generation
from app.services.po import ParetoOptimal

logger = logging.getLogger(__name__)

generation_prompt = """
You are an NL2SQL expert
I will give you the database structure, the most likely table columns. You can use this information to perform the NL2SQL task.
Please read and understand the database schema carefully, and generate an executable SQL. The generated SQL is protected by ```sql and ```.
""".strip()


# -------------------------
# Utils
# -------------------------
def prompt_maker(question: str, database_text: str, schema_links: List[str], examples: List[str]) -> str:
    # 你也可以把 examples 拼进去给 LLM 参考（可选）
    # 这里先不拼太长，避免 prompt 过大
    return (
        f"\n### Question: {question}\n"
        f"### Database: {database_text}\n"
        f"### Possible schemas: {schema_links}\n"
    )


def extract_sql_content(text: str) -> str:
    start_index = text.find("```sql")
    if start_index == -1:
        return ""
    end_index = text.find("```", start_index + 6)
    if end_index == -1:
        return ""
    return text[start_index + 6 : end_index].replace("\n", " ").strip()


def normalize_schema_links(x: Any) -> List[str]:
    """统一 schema_links 输入类型"""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return []


def schema_links_to_set(schema_links: Any) -> set:
    """
    PO 里 evaluate_schema_conformity 用的集合：
    - 把 table.col 拆成 {table, col}
    - 保留原 token
    """
    schema_set = set()
    links = normalize_schema_links(schema_links)
    for link in links:
        if "." in link:
            a, b = link.split(".", 1)
            if a.strip():
                schema_set.add(a.strip().lower())
            if b.strip():
                schema_set.add(b.strip().lower())
        if link.strip():
            schema_set.add(link.strip().lower())
    return schema_set


def extract_example_sqls(top_k_matches: Any) -> List[str]:
    """supervised_data 里 SQL 字段可能叫 sql 或 query，兼容"""
    out: List[str] = []
    if not isinstance(top_k_matches, list):
        return out
    for m in top_k_matches:
        if isinstance(m, dict):
            for k in ["sql", "query", "SQL", "Query"]:
                if k in m and m[k]:
                    out.append(str(m[k]))
                    break
        elif isinstance(m, str) and m.strip():
            out.append(m.strip())
    return out


def dedup_sqls(sqls: List[str]) -> List[str]:
    """忽略大小写/多空格去重"""
    out: List[str] = []
    seen = set()
    for s in sqls:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _pick_nearest_example(po: ParetoOptimal, sql: str, examples: List[str]) -> Tuple[Optional[str], float, float]:
    """
    给前端展示 nearest / distance：
    - nearest: 最相似的 example SQL
    - score: 相似度（0~1）
    - distance: 1-score
    """
    if not examples:
        return None, 0.0, 1.0

    # 用 PO 内部的 hybrid 思路：先 feature，再必要时 AST
    # 这里复用 po.feature_extractor / po.ast_processor
    try:
        sql_feat = po.feature_extractor.extract_features(sql)
        best_ex = None
        best_score = -1.0

        # 先做 feature similarity
        feats = []
        for ex in examples:
            feats.append(po.feature_extractor.extract_features(ex))

        # 粗筛
        sims = []
        for ex_feat in feats:
            sims.append(po.feature_extractor.cosine_similarity(sql_feat, ex_feat))

        # 选 top M 做 AST 精算（避免慢）
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[: min(5, len(sims))]

        for i in top_idx:
            feat_sim = sims[i]
            ex = examples[i]

            # 高相似再做 AST
            if feat_sim > 0.5:
                sql_ast = po.ast_processor.parse_sql_to_ast(sql)
                ex_ast = po.ast_processor.parse_sql_to_ast(ex)
                dist = po.ast_processor.calculate_edit_distance(sql_ast, ex_ast)
                ast_sim = max(0.0, 1.0 - dist)
                final_sim = 0.3 * feat_sim + 0.7 * ast_sim
            else:
                final_sim = feat_sim

            if final_sim > best_score:
                best_score = final_sim
                best_ex = ex

        best_score = max(0.0, min(1.0, float(best_score)))
        return best_ex, best_score, 1.0 - best_score

    except Exception:
        # 兜底
        return None, 0.0, 1.0


# -------------------------
# Core: LLM -> candidates -> PO scores -> selected
# -------------------------
def _llm_candidates(question: str, schema_text: str, schema_links: List[str], example_sqls: List[str]) -> List[str]:
    prompt = prompt_maker(question, schema_text, schema_links, example_sqls)

    responses = LLM_generation(generation_prompt, prompt)

    candidate_sqls: List[str] = []
    for res in responses:
        sql = extract_sql_content(res)
        if sql:
            candidate_sqls.append(sql)

    return dedup_sqls(candidate_sqls)


def _score_candidates(
    po: ParetoOptimal,
    candidate_sqls: List[str],
    schema_links_set: set,
    example_sqls: List[str],
) -> List[Dict[str, Any]]:
    """
    返回前端 candidates 结构（含 scores 全量字段）
    """
    scored: List[Dict[str, Any]] = []

    for i, sql in enumerate(candidate_sqls, start=1):
        # executability
        exec_detail = po.evaluate_executability_detailed(sql)
        exec_ok = bool(exec_detail.is_executable)
        exec_err = "" if exec_ok else (exec_detail.error_message or "")

        # schema
        schema_score = safe_float(po.evaluate_schema_conformity(sql, schema_links_set), 0.0)
        # used / missing：用 PO 的抽取（你的 PO 有 _extract_schema_from_sql）
        try:
            used_set = set()
            if hasattr(po, "_extract_schema_from_sql"):
                used_set = set(getattr(po, "_extract_schema_from_sql")(sql) or [])
            used_list = sorted({str(x) for x in used_set if str(x).strip()})
        except Exception:
            used_list = []

        missing_list = sorted({x for x in schema_links_set if x not in set(u.lower() for u in used_list)})

        # examples
        ex_nearest, ex_score, ex_dist = _pick_nearest_example(po, sql, example_sqls)

        # final_score：你可以按策略配权重
        # 注意：executability 是硬门槛，执行不了直接 0
        if not exec_ok:
            final_score = 0.0
        else:
            # balanced：schema + examples 平均
            final_score = 0.5 * schema_score + 0.5 * ex_score

        scored.append(
            {
                "id": f"c{i}",
                "sql": sql,
                "scores": {
                    "executability": {
                        "ok": exec_ok,
                        "error": exec_err,
                    },
                    "schema": {
                        "score": float(schema_score),
                        "used": used_list,         # 前端展示用
                        "missing": missing_list,   # 前端展示用
                    },
                    "examples": {
                        "score": float(ex_score),
                        "nearest": ex_nearest,
                        "distance": float(ex_dist),
                    },
                    "final_score": float(final_score),
                },
            }
        )

    # 给前端排序展示：final_score 从高到低
    scored.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
    return scored


def generate_for_api(
    db_id: str,
    question: str,
    options: Dict[str, Any],
    schema_linker=None,
) -> Dict[str, Any]:
    """
    POST /nl2sql/generate
    """
    schema_text = load_schema_text(db_id)
    schema_dict = load_schema_dict(db_id)

    db_path = get_db_path(db_id)
    if not db_path:
        raise ValueError(f"db not found: db_id={db_id}")

    # options
    evidence = str(options.get("evidence", "") or "")
    topk = int(options.get("sar_topk", 5))
    selection_strategy = str(options.get("selection_strategy", "balanced"))
    enable_schema_link = bool(options.get("enable_schema_link", True))

    # 0) schema links（优先外部传入，否则在线预测）
    schema_links = normalize_schema_links(options.get("schema_links", []))
    schema_links_pred: List[str] = []
    schema_links_fixed: List[str] = []

    if (not schema_links) and enable_schema_link and schema_linker is not None:
        try:
            raw_pred = schema_linker.predict_links(
                question=question,
                evidence=evidence,
                database_text=schema_text,
                max_new_tokens=int(options.get("schema_link_max_tokens", 512)),
                retry=int(options.get("schema_link_retry", 3)),
            )
            schema_links_pred = normalize_schema_links(raw_pred)

            fixed = schema_linker.fix_links(schema_links_pred, schema_text)
            schema_links_fixed = normalize_schema_links(fixed)

            schema_links = schema_links_fixed
        except Exception as e:
            logger.exception(f"[schema_link] failed db_id={db_id}: {e}")
            schema_links = []
            schema_links_pred = []
            schema_links_fixed = []
    else:
        schema_links_fixed = schema_links

    schema_links_set = schema_links_to_set(schema_links)

    # 1) SAR top-k 示例
    top_k_matches = sar_retrieve_topk(
        db_id=db_id,
        question=question,
        schema=schema_dict,
        k=topk,
    )
    example_sqls = extract_example_sqls(top_k_matches)

    # 2) LLM 候选
    candidate_sqls = _llm_candidates(question, schema_text, schema_links, example_sqls)

    # 如果 LLM 没产出候选，兜底返回空
    if not candidate_sqls:
        return {
            "request_id": f"req_{int(time.time() * 1000)}",
            "candidates": [],
            "selected_sql": "",
            "debug": {
                "schema_links_pred": schema_links_pred,
                "schema_links_fixed": schema_links_fixed,
                "sar_topk_matches": top_k_matches[:topk],
                "db_path": db_path,
                "note": "LLM returned no SQL candidates",
            },
        }

    # 3) PO：评估 + 选择
    po = ParetoOptimal(database_path=db_path)

    # 先打分，给前端完整展示
    candidates = _score_candidates(po, candidate_sqls, schema_links_set, example_sqls)

    # 再用 PO 的 select_final_sql 选最终（确保与你的 PO 行为一致）
    selected_sql = po.select_final_sql(
        candidates=[c["sql"] for c in candidates],
        schema_links=schema_links_set,
        examples=example_sqls,
        selection_strategy=selection_strategy,
    ).strip()

    # 让 selected_sql 在 candidates 中 final_score 更高（方便前端高亮）
    if selected_sql:
        norm_sel = re.sub(r"\s+", " ", selected_sql.strip().lower())
        bumped = False
        for c in candidates:
            norm_c = re.sub(r"\s+", " ", c["sql"].strip().lower())
            if norm_c == norm_sel:
                c["scores"]["final_score"] = float(max(c["scores"]["final_score"], 1.0))
                bumped = True
                break
        if bumped:
            candidates.sort(key=lambda x: x["scores"]["final_score"], reverse=True)

    return {
        "request_id": f"req_{int(time.time() * 1000)}",
        "candidates": candidates[:10],  # 最多展示 10 条
        "selected_sql": selected_sql or candidates[0]["sql"],
        "debug": {
            "schema_links_pred": schema_links_pred,
            "schema_links_fixed": schema_links_fixed,
            "sar_topk_matches": top_k_matches[:topk],
            "db_path": db_path,
            "strategy": selection_strategy,
            "schema_links_set_size": len(schema_links_set),
            "examples_cnt": len(example_sqls),
        },
    }
