from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .ai_base import csv_columns, run_with_retry
from .plan_cache import PlanCache, cache_key
from . import run_state


def _pd():
    import pandas
    return pandas


SUPPORTED_OPS = {
    "select", "rename", "filter", "limit",
    "sort", "group_by", "deduplicate", "fill_null", "type_cast", "join", "rolling",
}

ALLOWED_AGG_FUNCS = {"sum", "mean", "min", "max", "count", "median", "std", "first", "last"}
ALLOWED_ROLL_AGGS = {"mean", "sum", "min", "max", "std"}
ALLOWED_CAST_TYPES = {"int", "float", "str", "bool", "datetime"}
ALLOWED_JOIN_TYPES = {"inner", "left", "right", "outer"}
ALLOWED_FILL_STRATEGIES = {"mean", "median", "mode", "forward", "backward"}


@dataclass
class AIContext:
    source_path: str
    source_type: str
    columns: List[str]
    sample_rows: List[Dict[str, Any]]


def _read_source(source_type: str, path: str) -> pd.DataFrame:
    pd = _pd()
    if source_type == "csv":
        return pd.read_csv(path)
    elif source_type == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def _validate_plan(plan: Dict[str, Any], columns: List[str]) -> None:
    source = plan.get("source", {})
    if source.get("type") not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported source.type: {source.get('type')}. Must be 'csv' or 'parquet'.")
    if "path" not in source:
        raise ValueError("Plan must include source.path")

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")

    known_columns = list(columns)

    for step in steps:
        op = step.get("op")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op: {op}. Supported: {sorted(SUPPORTED_OPS)}")

        if op == "select":
            cols = step.get("columns")
            if not isinstance(cols, list) or not cols:
                raise ValueError("select.columns must be a non-empty list")
            bad = [c for c in cols if c not in known_columns]
            if bad:
                raise ValueError(f"select references unknown columns: {bad}")
            known_columns = list(cols)

        elif op == "rename":
            mapping = step.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("rename.mapping must be a non-empty object")
            bad = [k for k in mapping.keys() if k not in known_columns]
            if bad:
                raise ValueError(f"rename references unknown columns: {bad}")
            for old, new in mapping.items():
                idx = known_columns.index(old)
                known_columns[idx] = new

        elif op == "filter":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"filter.col references unknown column: {col}")
            if step.get("operator") not in {"==", "!=", ">", "<", ">=", "<="}:
                raise ValueError(f"filter.operator unsupported: {step.get('operator')}")
            if "value" not in step:
                raise ValueError("filter.value is required")

        elif op == "limit":
            n = step.get("n")
            if not isinstance(n, int) or n < 1 or n > 10000:
                raise ValueError("limit.n must be an int between 1 and 10000")

        elif op == "sort":
            by = step.get("by")
            if not isinstance(by, list) or not by:
                raise ValueError("sort.by must be a non-empty list")
            bad = [c for c in by if c not in known_columns]
            if bad:
                raise ValueError(f"sort.by references unknown columns: {bad}")

        elif op == "group_by":
            by = step.get("by")
            agg = step.get("agg")
            if not isinstance(by, list) or not by:
                raise ValueError("group_by.by must be a non-empty list")
            bad = [c for c in by if c not in known_columns]
            if bad:
                raise ValueError(f"group_by.by references unknown columns: {bad}")
            if not isinstance(agg, dict) or not agg:
                raise ValueError("group_by.agg must be a non-empty object")
            for col, func in agg.items():
                if col not in known_columns:
                    raise ValueError(f"group_by.agg references unknown column: {col}")
                if func not in ALLOWED_AGG_FUNCS:
                    raise ValueError(f"group_by.agg function unsupported: {func}. Supported: {sorted(ALLOWED_AGG_FUNCS)}")
            known_columns = list(by) + list(agg.keys())

        elif op == "deduplicate":
            cols = step.get("columns")
            if cols is not None:
                if not isinstance(cols, list):
                    raise ValueError("deduplicate.columns must be a list")
                bad = [c for c in cols if c not in known_columns]
                if bad:
                    raise ValueError(f"deduplicate references unknown columns: {bad}")
            keep = step.get("keep", "first")
            if keep not in {"first", "last"}:
                raise ValueError(f"deduplicate.keep must be 'first' or 'last', got: {keep}")

        elif op == "fill_null":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"fill_null.col references unknown column: {col}")
            has_value = "value" in step
            has_strategy = "strategy" in step
            if not has_value and not has_strategy:
                raise ValueError("fill_null requires either 'value' or 'strategy'")
            if has_strategy and step["strategy"] not in ALLOWED_FILL_STRATEGIES:
                raise ValueError(f"fill_null.strategy unsupported: {step['strategy']}. Supported: {sorted(ALLOWED_FILL_STRATEGIES)}")

        elif op == "type_cast":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"type_cast.col references unknown column: {col}")
            to = step.get("to")
            if to not in ALLOWED_CAST_TYPES:
                raise ValueError(f"type_cast.to unsupported: {to}. Supported: {sorted(ALLOWED_CAST_TYPES)}")

        elif op == "join":
            on = step.get("on")
            if not on:
                raise ValueError("join.on is required")
            if isinstance(on, str):
                if on not in known_columns:
                    raise ValueError(f"join.on references unknown column: {on}")
            elif isinstance(on, list):
                bad = [c for c in on if c not in known_columns]
                if bad:
                    raise ValueError(f"join.on references unknown columns: {bad}")
            how = step.get("how", "inner")
            if how not in ALLOWED_JOIN_TYPES:
                raise ValueError(f"join.how unsupported: {how}. Supported: {sorted(ALLOWED_JOIN_TYPES)}")
            right = step.get("right_path")
            if not right:
                raise ValueError("join.right_path is required")

        elif op == "rolling":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"rolling.col references unknown column: {col}")
            window = step.get("window")
            if not isinstance(window, int) or window < 1:
                raise ValueError("rolling.window must be a positive integer")
            agg = step.get("agg")
            if agg not in ALLOWED_ROLL_AGGS:
                raise ValueError(f"rolling.agg must be one of {sorted(ALLOWED_ROLL_AGGS)}, got: {agg}")
            into = step.get("into")
            if not into:
                raise ValueError("rolling.into is required — specify the output column name")
            if into not in known_columns:
                known_columns.append(into)


def _execute_plan(plan: Dict[str, Any]) -> pd.DataFrame:
    pd = _pd()
    source = plan["source"]
    df = _read_source(source["type"], source["path"])

    for step in plan.get("steps", []):
        op = step["op"]

        if op == "select":
            df = df[step["columns"]]

        elif op == "rename":
            df = df.rename(columns=step["mapping"])

        elif op == "limit":
            df = df.head(step["n"])

        elif op == "filter":
            col, operator, value = step["col"], step["operator"], step["value"]
            if isinstance(value, str):
                try:
                    value = float(value) if "." in value else int(value)
                except Exception:
                    pass
            ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
            df = df[getattr(df[col], ops[operator])(value)]

        elif op == "sort":
            ascending = step.get("ascending", True)
            if isinstance(ascending, bool):
                ascending = [ascending] * len(step["by"])
            df = df.sort_values(by=step["by"], ascending=ascending)

        elif op == "group_by":
            agg_map = step["agg"]
            df = df.groupby(step["by"], as_index=False).agg(agg_map)

        elif op == "deduplicate":
            cols = step.get("columns")
            keep = step.get("keep", "first")
            df = df.drop_duplicates(subset=cols, keep=keep)

        elif op == "fill_null":
            col = step["col"]
            if "value" in step:
                df[col] = df[col].fillna(step["value"])
            else:
                strategy = step["strategy"]
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else None)
                elif strategy == "forward":
                    df[col] = df[col].ffill()
                elif strategy == "backward":
                    df[col] = df[col].bfill()

        elif op == "type_cast":
            col = step["col"]
            to = step["to"]
            if to == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif to == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif to == "str":
                df[col] = df[col].astype(str)
            elif to == "bool":
                df[col] = df[col].astype(bool)
            elif to == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")

        elif op == "join":
            right_df = pd.read_csv(step["right_path"])
            on = step["on"]
            how = step.get("how", "inner")
            df = df.merge(right_df, on=on, how=how)

        elif op == "rolling":
            col = step["col"]
            window = step["window"]
            agg = step["agg"]
            into = step["into"]
            min_periods = step.get("min_periods", 1)
            df[into] = getattr(df[col].rolling(window, min_periods=min_periods), agg)()

    return df


def expect_columns(*cols: str) -> Callable[[pd.DataFrame], None]:
    def _check(df: pd.DataFrame) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")
    return _check


def expect_non_empty(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("DataFrame is empty")


def ai_dataframe(
    *,
    csv_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
    llm_caller: Callable[[str, str], str],
    post_conditions: Optional[List[Callable[[pd.DataFrame], None]]] = None,
    max_attempts: int = 3,
    compile_only: bool = False,
    plan_cache: Optional[PlanCache] = None,
):
    """
    Decorator that turns a docstring into an LLM-compiled dataframe plan.
    Returns a real pandas DataFrame, or the validated plan dict if compile_only=True.

    Provide exactly one of ``csv_path`` or ``parquet_path``.
    ``llm_caller`` must be a ``(system: str, user: str) -> str`` callable.
    """
    if csv_path and parquet_path:
        raise ValueError("Provide exactly one of csv_path or parquet_path, not both.")
    if not csv_path and not parquet_path:
        raise ValueError("Provide exactly one of csv_path or parquet_path.")

    source_path = csv_path or parquet_path
    source_type = "csv" if csv_path else "parquet"
    post_conditions = post_conditions or []

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nl_spec = (func.__doc__ or "").strip()
            if not nl_spec:
                raise ValueError("AI dataframe function must have a docstring with instructions.")

            run_state.last_run_from_cache = False

            if plan_cache and not compile_only:
                columns_for_key = csv_columns(source_path) if source_type == "csv" else None
                if columns_for_key is not None:
                    key = cache_key(nl_spec, {
                        "source_path": source_path,
                        "source_type": source_type,
                        "columns": columns_for_key,
                        "runtime_params": sorted(kwargs.items()),
                    })
                    cached = plan_cache.get(key)
                    if cached is not None:
                        cached.setdefault("source", {})
                        cached["source"]["type"] = source_type
                        cached["source"]["path"] = source_path
                        _validate_plan(cached, columns_for_key)
                        run_state.last_run_from_cache = True
                        df = _execute_plan(cached)
                        for check in post_conditions:
                            check(df)
                        return df

            sample_df = _read_source(source_type, source_path).head(5)
            ctx = AIContext(
                source_path=source_path,
                source_type=source_type,
                columns=list(sample_df.columns),
                sample_rows=sample_df.to_dict(orient="records"),
            )

            system_prompt = (
                "You are a strict compiler that converts natural language dataframe instructions into a JSON plan.\n"
                "Return ONLY valid JSON. No prose.\n\n"
                "Supported ops: select, rename, filter, limit, sort, group_by, deduplicate, fill_null, type_cast, join, rolling.\n\n"
                "Plan schema:\n"
                "{\n"
                f'  "source": {{"type":"{source_type}","path":"..."}},\n'
                '  "steps": [\n'
                '    {"op":"select","columns":[...]},\n'
                '    {"op":"rename","mapping":{"old":"new"}},\n'
                '    {"op":"filter","col":"...","operator":"==|!=|>|<|>=|<=","value":...},\n'
                '    {"op":"limit","n":10},\n'
                '    {"op":"sort","by":["col"],"ascending":true},\n'
                '    {"op":"group_by","by":["col"],"agg":{"amount":"sum","id":"count"}},\n'
                '    {"op":"deduplicate","columns":["col"],"keep":"first|last"},\n'
                '    {"op":"fill_null","col":"...","value":0},\n'
                '    {"op":"fill_null","col":"...","strategy":"mean|median|mode|forward|backward"},\n'
                '    {"op":"type_cast","col":"...","to":"int|float|str|bool|datetime"},\n'
                '    {"op":"join","right_path":"other.csv","on":"col_or_list","how":"inner|left|right|outer"},\n'
                '    {"op":"rolling","col":"amount","window":7,"agg":"mean|sum|min|max|std","into":"new_col","min_periods":1}\n'
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- group_by.agg functions: sum, mean, min, max, count, median, std, first, last.\n"
                "- rolling.into is required — it names the new output column.\n"
                "- Never reference columns not present in the provided schema.\n"
                "- Use the exact source type and path provided in the dataset context.\n"
                "- If runtime_params are provided, use their values directly in the plan (e.g. as filter values)."
            )

            base_user: Dict[str, Any] = {
                "request": nl_spec,
                "dataset": {
                    "path": ctx.source_path,
                    "type": ctx.source_type,
                    "columns": ctx.columns,
                    "sample_rows": ctx.sample_rows,
                },
            }
            if kwargs:
                base_user["runtime_params"] = kwargs

            def build_payload(last_error: Optional[str]) -> str:
                payload = dict(base_user)
                if last_error:
                    payload["previous_error"] = last_error
                    payload["instruction"] = "Fix the JSON plan to satisfy schema + validations."
                return json.dumps(payload, indent=2)

            def process_result(plan: Dict[str, Any]) -> Any:
                plan.setdefault("source", {})
                plan["source"]["type"] = source_type
                plan["source"]["path"] = source_path

                _validate_plan(plan, ctx.columns)
                if compile_only:
                    return plan
                df = _execute_plan(plan)
                for check in post_conditions:
                    check(df)
                return df

            if plan_cache and not compile_only:
                key = cache_key(nl_spec, {
                    "source_path": source_path,
                    "source_type": source_type,
                    "columns": ctx.columns,
                    "runtime_params": sorted(kwargs.items()),
                })
                out_plan: Dict[str, Any] = {}
                result = run_with_retry(
                    llm_caller=llm_caller,
                    system_prompt=system_prompt,
                    build_payload=build_payload,
                    process_result=process_result,
                    max_attempts=max_attempts,
                    out_plan=out_plan,
                )
                if out_plan:
                    plan_cache.set(key, out_plan["plan"])
                return result

            return run_with_retry(
                llm_caller=llm_caller,
                system_prompt=system_prompt,
                build_payload=build_payload,
                process_result=process_result,
                max_attempts=max_attempts,
            )

        return wrapper

    return decorator
