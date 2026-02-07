import re
from typing import Any, Dict, List, Optional, Tuple

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

def _is_number(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str) and _NUMERIC_RE.match(x.strip()):
        return True
    return False

def _is_time_like(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        s = x.strip()
        # 够用的轻量判断：YYYY-MM-DD / YYYY/MM/DD / YYYY-MM / YYYY
        return bool(re.match(r"^\d{4}([-/]\d{1,2}([-/]\d{1,2})?)?$", s))
    return False

def _col_stats(col_idx: int, rows: List[List[Any]]) -> Dict[str, Any]:
    vals = [r[col_idx] for r in rows if col_idx < len(r)]
    nonnull = [v for v in vals if v is not None]
    n = len(nonnull)
    if n == 0:
        return {"nonnull": 0, "numeric_ratio": 0.0, "time_ratio": 0.0, "distinct": 0}
    numeric = sum(1 for v in nonnull if _is_number(v))
    time_like = sum(1 for v in nonnull if _is_time_like(v))
    distinct = len(set(str(v) for v in nonnull))
    return {
        "nonnull": n,
        "numeric_ratio": numeric / n,
        "time_ratio": time_like / n,
        "distinct": distinct,
    }

def suggest_charts(columns: List[str], rows: List[List[Any]], max_suggest: int = 3) -> List[Dict[str, Any]]:
    if not columns or not rows:
        return []

    # 只看前 N 行做推断，避免大数据慢
    sample = rows[:500]

    stats = [ _col_stats(i, sample) for i in range(len(columns)) ]

    numeric_cols = [i for i, st in enumerate(stats) if st["numeric_ratio"] >= 0.9]
    time_cols = [i for i, st in enumerate(stats) if st["time_ratio"] >= 0.6]
    cat_cols = [i for i, st in enumerate(stats) if i not in numeric_cols and st["distinct"] <= min(50, len(sample))]

    out: List[Dict[str, Any]] = []

    def add(item: Dict[str, Any]):
        if len(out) < max_suggest:
            out.append(item)

    # 1) 两列： (time|category) + numeric  => line/bar
    if len(columns) == 2 and numeric_cols:
        y = numeric_cols[0]
        x = 0 if y == 1 else 1

        if x in time_cols:
            add({
                "type": "line",
                "title": f"{columns[y]} by {columns[x]}",
                "x": {"field": columns[x], "type": "time"},
                "y": {"field": columns[y], "type": "value"},
                "series": None,
                "reason": "时间列 + 数值列，适合折线图"
            })
        else:
            # 如果类别较少，可以饼图，否则柱状图
            if stats[x]["distinct"] <= 12:
                add({
                    "type": "pie",
                    "title": f"{columns[y]} share by {columns[x]}",
                    "name": {"field": columns[x]},
                    "value": {"field": columns[y]},
                    "reason": "类别较少 + 数值列，适合饼图"
                })
            add({
                "type": "bar",
                "title": f"{columns[y]} by {columns[x]}",
                "x": {"field": columns[x], "type": "category"},
                "y": {"field": columns[y], "type": "value"},
                "series": None,
                "reason": "类别列 + 数值列，适合柱状图"
            })

    # 2) 三列：x + series + y（常见：year, category, value）
    if len(columns) >= 3:
        # 找一个 numeric 当 y
        if numeric_cols:
            y = numeric_cols[0]
            # x 优先 time，其次 category
            x = time_cols[0] if time_cols else (cat_cols[0] if cat_cols else None)
            # series 选一个类别列且不同于 x/y
            series = None
            for i in cat_cols:
                if i != x and i != y:
                    series = i
                    break

            if x is not None and series is not None:
                add({
                    "type": "bar",
                    "title": f"{columns[y]} by {columns[x]} (grouped by {columns[series]})",
                    "x": {"field": columns[x], "type": "time" if x in time_cols else "category"},
                    "y": {"field": columns[y], "type": "value"},
                    "series": {"field": columns[series]},
                    "reason": "x + series + y 结构，适合分组柱状/折线"
                })
                if x in time_cols:
                    add({
                        "type": "line",
                        "title": f"{columns[y]} trend by {columns[x]} (by {columns[series]})",
                        "x": {"field": columns[x], "type": "time"},
                        "y": {"field": columns[y], "type": "value"},
                        "series": {"field": columns[series]},
                        "reason": "时间 x + series + y，适合多折线"
                    })

    # 3) 兜底：有数值列就给个 bar
    if not out and numeric_cols:
        y = numeric_cols[0]
        # x 选 distinct 不太大的列
        x = cat_cols[0] if cat_cols else (0 if y != 0 else 1 if len(columns) > 1 else None)
        if x is not None:
            add({
                "type": "bar",
                "title": f"{columns[y]} by {columns[x]}",
                "x": {"field": columns[x], "type": "category"},
                "y": {"field": columns[y], "type": "value"},
                "series": None,
                "reason": "兜底推荐柱状图"
            })

    return out
