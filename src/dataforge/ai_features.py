from __future__ import annotations

import json
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .ai_base import csv_columns, run_with_retry
from .plan_cache import PlanCache, cache_key
from . import run_state


def _pd():
    import pandas
    return pandas


def _np():
    import numpy
    return numpy


ALLOWED_OPS = {"filter", "filter_not_null", "derive", "bucketize", "drop", "normalize", "one_hot_encode"}
ALLOWED_NORMALIZE_METHODS = {"minmax", "zscore"}


def _validate_plan(plan: Dict[str, Any], columns: List[str]) -> None:
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")

    known_columns = list(columns)

    for step in steps:
        op = step.get("op")
        if op not in ALLOWED_OPS:
            raise ValueError(f"Unsupported op: {op}")

        if op == "filter":
            if step.get("col") not in known_columns:
                raise ValueError(f"filter.col references unknown column: {step.get('col')}")
            if step.get("operator") not in {"==", "!=", ">", "<", ">=", "<="}:
                raise ValueError(f"filter.operator unsupported: {step.get('operator')}")
            if "value" not in step:
                raise ValueError("filter.value is required")

        elif op == "filter_not_null":
            if step.get("col") not in known_columns:
                raise ValueError(f"filter_not_null.col references unknown column: {step.get('col')}")

        elif op == "derive":
            if not step.get("col"):
                raise ValueError("derive.col is required")
            expr = step.get("expr", "")
            if not expr:
                raise ValueError("derive.expr is required")
            safe_expr = re.sub(r"[a-zA-Z_]\w*", "", expr)
            safe_expr = re.sub(r"[\d.]+", "", safe_expr)
            remaining = safe_expr.strip().replace(" ", "")
            allowed_chars = set("+-*/()")
            if not all(c in allowed_chars for c in remaining):
                raise ValueError(f"derive.expr contains unsafe characters: {expr}")
            identifiers = re.findall(r"[a-zA-Z_]\w*", expr)
            bad = [i for i in identifiers if i not in known_columns]
            if bad:
                raise ValueError(f"derive.expr references unknown columns: {bad}")
            new_col = step["col"]
            if new_col not in known_columns:
                known_columns.append(new_col)

        elif op == "bucketize":
            if step.get("col") not in known_columns:
                raise ValueError(f"bucketize.col references unknown column: {step.get('col')}")
            if not step.get("into"):
                raise ValueError("bucketize.into is required")
            bins = step.get("bins")
            labels = step.get("labels")
            if not isinstance(bins, list) or len(bins) < 1:
                raise ValueError("bucketize.bins must be a non-empty list of numeric boundaries")
            if not isinstance(labels, list) or len(labels) != len(bins) + 1:
                raise ValueError("bucketize.labels must have exactly len(bins)+1 entries")
            into = step["into"]
            if into not in known_columns:
                known_columns.append(into)

        elif op == "drop":
            cols = step.get("columns")
            if not isinstance(cols, list) or not cols:
                raise ValueError("drop.columns must be a non-empty list")
            bad = [c for c in cols if c not in known_columns]
            if bad:
                raise ValueError(f"drop references unknown columns: {bad}")
            for c in cols:
                known_columns.remove(c)

        elif op == "normalize":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"normalize.col references unknown column: {col}")
            method = step.get("method")
            if method not in ALLOWED_NORMALIZE_METHODS:
                raise ValueError(f"normalize.method must be one of {sorted(ALLOWED_NORMALIZE_METHODS)}, got: {method}")
            into = step.get("into", col)
            if into not in known_columns:
                known_columns.append(into)

        elif op == "one_hot_encode":
            col = step.get("col")
            if col not in known_columns:
                raise ValueError(f"one_hot_encode.col references unknown column: {col}")
            drop_original = step.get("drop_original", True)
            if drop_original:
                known_columns.remove(col)


def _execute_plan(plan: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    pd = _pd()
    np = _np()
    for step in plan.get("steps", []):
        op = step["op"]

        if op == "filter":
            col, operator, value = step["col"], step["operator"], step["value"]
            if isinstance(value, str):
                try:
                    value = float(value) if "." in value else int(value)
                except Exception:
                    pass
            ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
            df = df[getattr(df[col], ops[operator])(value)]

        elif op == "filter_not_null":
            df = df[df[step["col"]].notna()]

        elif op == "derive":
            col_name = step["col"]
            expr = step["expr"]
            tokens = re.findall(r"[a-zA-Z_]\w*|[\d.]+|[+\-*/()]", expr)
            pd_expr_parts = []
            for token in tokens:
                if re.match(r"^[a-zA-Z_]\w*$", token) and token in df.columns:
                    pd_expr_parts.append(f"df['{token}']")
                else:
                    pd_expr_parts.append(token)
            pd_expr = " ".join(pd_expr_parts)
            df[col_name] = eval(pd_expr, {"__builtins__": {}, "df": df})  # noqa: S307

        elif op == "bucketize":
            bins = [-np.inf] + step["bins"] + [np.inf]
            df[step["into"]] = pd.cut(
                df[step["col"]], bins=bins, labels=step["labels"], right=False,
            )

        elif op == "drop":
            df = df.drop(columns=step["columns"])

        elif op == "normalize":
            col = step["col"]
            into = step.get("into", col)
            method = step["method"]
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df[into] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                df[into] = (df[col] - mean) / std if std != 0 else 0.0

        elif op == "one_hot_encode":
            col = step["col"]
            drop_original = step.get("drop_original", True)
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df = df.drop(columns=[col])

    return df


def expect_feature_types(type_map: Dict[str, str]) -> Callable[[pd.DataFrame], None]:
    """Check that columns exist and have the expected dtype family."""
    dtype_families = {
        "float": "float", "int": "int", "string": "object",
        "category": "category", "object": "object",
    }

    def _check(df: pd.DataFrame) -> None:
        for col, expected in type_map.items():
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' not found. Got: {list(df.columns)}")
            actual = str(df[col].dtype)
            family = dtype_families.get(expected, expected)
            if family not in actual:
                raise ValueError(f"Column '{col}' expected dtype '{expected}', got '{actual}'")
    return _check


def expect_columns(*cols: str) -> Callable[[pd.DataFrame], None]:
    def _check(df: pd.DataFrame) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")
    return _check


def ai_features(
    *,
    csv_path: str,
    llm_caller: Callable[[str, str], str],
    post_conditions: Optional[List[Callable[[pd.DataFrame], None]]] = None,
    max_attempts: int = 3,
    compile_only: bool = False,
    plan_cache: Optional[PlanCache] = None,
):
    """
    Decorator that turns a docstring into an LLM-compiled feature engineering plan.
    Returns a pandas DataFrame with the new features applied, or the validated plan
    dict if compile_only=True.
    """
    post_conditions = post_conditions or []

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nl_spec = (func.__doc__ or "").strip()
            if not nl_spec:
                raise ValueError("AI features function must have a docstring with instructions.")

            run_state.last_run_from_cache = False

            if plan_cache and not compile_only:
                columns_for_key = csv_columns(csv_path)
                key = cache_key(nl_spec, {
                    "csv_path": csv_path,
                    "columns": columns_for_key,
                    "runtime_params": sorted(kwargs.items()),
                })
                cached = plan_cache.get(key)
                if cached is not None:
                    _validate_plan(cached, columns_for_key)
                    run_state.last_run_from_cache = True
                    df = _pd().read_csv(csv_path)
                    df = _execute_plan(cached, df)
                    for check in post_conditions:
                        check(df)
                    return df

            sample_df = _pd().read_csv(csv_path).head(5)
            columns = list(sample_df.columns)
            dtypes = {col: str(sample_df[col].dtype) for col in columns}

            system_prompt = (
                "You are a strict compiler that converts natural language feature engineering "
                "instructions into a JSON plan.\n"
                "Return ONLY valid JSON. No prose.\n"
                "Allowed ops: filter, filter_not_null, derive, bucketize, drop, normalize, one_hot_encode.\n\n"
                "Schema:\n"
                "{\n"
                '  "steps": [\n'
                '    {"op":"filter","col":"...","operator":"==|!=|>|<|>=|<=","value":...},\n'
                '    {"op":"filter_not_null","col":"..."},\n'
                '    {"op":"derive","col":"new_col_name","expr":"col_a / col_b"},\n'
                '    {"op":"bucketize","col":"...","into":"new_col_name",'
                '"bins":[25,35,45,55],"labels":["<25","25-34","35-44","45-54","55+"]},\n'
                '    {"op":"drop","columns":["col1","col2"]},\n'
                '    {"op":"normalize","col":"...","method":"minmax|zscore","into":"optional_new_col"},\n'
                '    {"op":"one_hot_encode","col":"...","drop_original":true}\n'
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- derive.expr may only use column names, numeric literals, and +,-,*,/ operators.\n"
                "- bucketize.labels must have exactly len(bins)+1 entries.\n"
                "- normalize.into is optional; omit to overwrite the source column in place.\n"
                "- Never reference columns not present in the provided schema.\n"
                "- If runtime_params are provided, use their values directly in the plan (e.g. as filter values)."
            )

            base_user: Dict[str, Any] = {
                "request": nl_spec,
                "dataset": {
                    "path": csv_path,
                    "columns": columns,
                    "dtypes": dtypes,
                    "sample_rows": sample_df.to_dict(orient="records"),
                },
            }
            if kwargs:
                base_user["runtime_params"] = kwargs

            def build_payload(last_error: Optional[str]) -> str:
                payload = dict(base_user)
                if last_error:
                    payload["previous_error"] = last_error
                    payload["instruction"] = "Fix the JSON plan to satisfy the validation rules."
                return json.dumps(payload, indent=2)

            def process_result(plan: Dict[str, Any]) -> Any:
                _validate_plan(plan, columns)
                if compile_only:
                    return plan
                df = _pd().read_csv(csv_path)
                df = _execute_plan(plan, df)
                for check in post_conditions:
                    check(df)
                return df

            if plan_cache and not compile_only:
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
