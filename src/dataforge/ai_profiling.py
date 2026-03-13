from __future__ import annotations

import json
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


ALLOWED_METRICS = {"row_count", "null_counts", "cardinality", "numeric_stats", "outliers"}


def _validate_plan(plan: Dict[str, Any], columns: List[str]) -> None:
    metrics = plan.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")

    for m in metrics:
        metric = m.get("metric")
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")

        if metric in {"cardinality", "numeric_stats", "outliers"}:
            cols = m.get("columns")
            if not isinstance(cols, list) or not cols:
                raise ValueError(f"{metric}.columns must be a non-empty list")
            bad = [c for c in cols if c not in columns]
            if bad:
                raise ValueError(f"{metric} references unknown columns: {bad}")

        if metric == "outliers":
            method = m.get("method", "iqr")
            if method not in {"iqr", "zscore"}:
                raise ValueError(f"outliers.method must be 'iqr' or 'zscore', got: {method}")


def _execute_profile(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    pd = _pd()
    result: Dict[str, Any] = {}

    for m in plan["metrics"]:
        metric = m["metric"]

        if metric == "row_count":
            result["row_count"] = len(df)

        elif metric == "null_counts":
            result["null_counts"] = df.isnull().sum().to_dict()

        elif metric == "cardinality":
            result["cardinality"] = {
                col: int(df[col].nunique()) for col in m["columns"]
            }

        elif metric == "numeric_stats":
            stats = {}
            for col in m["columns"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        "min": _safe_num(df[col].min()),
                        "max": _safe_num(df[col].max()),
                        "mean": _safe_num(df[col].mean()),
                        "median": _safe_num(df[col].median()),
                        "std": _safe_num(df[col].std()),
                    }
                else:
                    stats[col] = {"error": f"Column '{col}' is not numeric (dtype: {df[col].dtype})"}
            result["numeric_stats"] = stats

        elif metric == "outliers":
            method = m.get("method", "iqr")
            outlier_info = {}
            for col in m["columns"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    outlier_info[col] = {"error": f"Column '{col}' is not numeric"}
                    continue

                series = df[col].dropna()
                if method == "iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (series < lower) | (series > upper)
                elif method == "zscore":
                    mean = series.mean()
                    std = series.std()
                    if std == 0:
                        outlier_mask = pd.Series(False, index=series.index)
                    else:
                        z = ((series - mean) / std).abs()
                        outlier_mask = z > 3

                outlier_info[col] = {
                    "method": method,
                    "count": int(outlier_mask.sum()),
                    "values": series[outlier_mask].tolist() if outlier_mask.sum() > 0 else [],
                }
            result["outliers"] = outlier_info

    return result


def _safe_num(val: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    np = _np()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    return val


def ai_profile(
    *,
    csv_path: str,
    llm_caller: Callable[[str, str], str],
    max_attempts: int = 3,
    compile_only: bool = False,
    plan_cache: Optional[PlanCache] = None,
):
    """
    Decorator that turns a docstring into an LLM-compiled profiling plan.
    Returns a structured profiling report dict, or the validated plan dict if compile_only=True.
    """

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nl_spec = (func.__doc__ or "").strip()
            if not nl_spec:
                raise ValueError("AI profile function must have a docstring with instructions.")

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
                    return _execute_profile(cached, df)

            sample_df = _pd().read_csv(csv_path).head(5)
            columns = list(sample_df.columns)
            dtypes = {col: str(sample_df[col].dtype) for col in columns}

            system_prompt = (
                "You are a strict compiler that converts natural language data profiling "
                "instructions into a JSON plan.\n"
                "Return ONLY valid JSON. No prose.\n"
                "Allowed metrics: row_count, null_counts, cardinality, numeric_stats, outliers.\n\n"
                "Schema:\n"
                "{\n"
                '  "metrics": [\n'
                '    {"metric":"row_count"},\n'
                '    {"metric":"null_counts"},\n'
                '    {"metric":"cardinality","columns":["col1","col2"]},\n'
                '    {"metric":"numeric_stats","columns":["col1"]},\n'
                '    {"metric":"outliers","columns":["col1"],"method":"iqr"}\n'
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- outliers.method must be 'iqr' or 'zscore'.\n"
                "- Only reference columns present in the provided schema.\n"
                "- numeric_stats and outliers are only meaningful for numeric columns."
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
                return _execute_profile(plan, df)

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
