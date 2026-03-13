from __future__ import annotations

import json
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


ALLOWED_RULES = {"unique", "not_null", "non_negative", "in_set", "null_rate"}


def _validate_plan(plan: Dict[str, Any], columns: List[str]) -> None:
    checks = plan.get("checks", [])
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a non-empty list")

    for check in checks:
        rule = check.get("rule")
        if rule not in ALLOWED_RULES:
            raise ValueError(f"Unsupported rule: {rule}")

        col = check.get("col")
        if col not in columns:
            raise ValueError(f"Rule '{rule}' references unknown column: {col}")

        if rule == "in_set":
            values = check.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError("in_set.values must be a non-empty list")

        if rule == "null_rate":
            max_rate = check.get("max_rate")
            if not isinstance(max_rate, (int, float)) or max_rate < 0 or max_rate > 1:
                raise ValueError("null_rate.max_rate must be a number between 0 and 1")


def _execute_checks(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    results = []

    for check in plan["checks"]:
        rule = check["rule"]
        col = check["col"]
        entry: Dict[str, Any] = {"rule": rule, "col": col}

        if rule == "unique":
            duplicates = df[col].duplicated().sum()
            entry["status"] = "PASS" if duplicates == 0 else "FAIL"
            entry["duplicates"] = int(duplicates)

        elif rule == "not_null":
            null_count = df[col].isnull().sum()
            entry["status"] = "PASS" if null_count == 0 else "FAIL"
            entry["null_count"] = int(null_count)

        elif rule == "non_negative":
            negative_count = (df[col] < 0).sum()
            entry["status"] = "PASS" if negative_count == 0 else "FAIL"
            entry["failed_rows"] = int(negative_count)

        elif rule == "in_set":
            allowed = set(check["values"])
            bad = df[~df[col].isin(allowed) & df[col].notna()]
            entry["status"] = "PASS" if len(bad) == 0 else "FAIL"
            entry["failed_rows"] = len(bad)
            if len(bad) > 0:
                entry["unexpected_values"] = sorted(bad[col].unique().tolist())

        elif rule == "null_rate":
            max_rate = check["max_rate"]
            actual_rate = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
            entry["status"] = "PASS" if actual_rate <= max_rate else "FAIL"
            entry["actual_rate"] = round(actual_rate, 6)
            entry["max_rate"] = max_rate

        results.append(entry)

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    return {
        "dataset": plan.get("dataset", "unknown"),
        "checks": results,
        "summary": {"pass": pass_count, "fail": fail_count},
    }


def ai_quality(
    *,
    csv_path: str,
    llm_caller: Callable[[str, str], str],
    max_attempts: int = 3,
    compile_only: bool = False,
    plan_cache: Optional[PlanCache] = None,
):
    """
    Decorator that turns a docstring into an LLM-compiled data quality check plan.
    Returns a structured quality report dict, or the validated plan dict if compile_only=True.
    """

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nl_spec = (func.__doc__ or "").strip()
            if not nl_spec:
                raise ValueError("AI quality function must have a docstring with instructions.")

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
                    return _execute_checks(cached, df)

            sample_df = _pd().read_csv(csv_path).head(5)
            columns = list(sample_df.columns)
            dtypes = {col: str(sample_df[col].dtype) for col in columns}

            system_prompt = (
                "You are a strict compiler that converts natural language data quality rules "
                "into a JSON plan.\n"
                "Return ONLY valid JSON. No prose.\n"
                "Allowed rules: unique, not_null, non_negative, in_set, null_rate.\n\n"
                "Schema:\n"
                "{\n"
                '  "dataset": "name_of_dataset",\n'
                '  "checks": [\n'
                '    {"rule":"unique","col":"..."},\n'
                '    {"rule":"not_null","col":"..."},\n'
                '    {"rule":"non_negative","col":"..."},\n'
                '    {"rule":"in_set","col":"...","values":["A","B","C"]},\n'
                '    {"rule":"null_rate","col":"...","max_rate":0.001}\n'
                "  ]\n"
                "}\n\n"
                "Never reference columns not present in the provided schema."
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
                return _execute_checks(plan, df)

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
