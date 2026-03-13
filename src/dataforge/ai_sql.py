from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .ai_base import run_with_retry
from .plan_cache import PlanCache, cache_key
from . import run_state


BLOCKED_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "TRUNCATE", "GRANT", "REVOKE", "EXEC",
}


@dataclass
class SQLResult:
    sql: str
    params: Dict[str, Any]
    rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TableSchema:
    table_name: str
    columns: List[Dict[str, str]]   # [{"name": "id", "type": "integer"}, ...]


def _introspect_tables(db_connector: Callable, tables: List[str]) -> List[TableSchema]:
    conn = db_connector()
    try:
        cur = conn.cursor()
        schemas: List[TableSchema] = []
        for table in tables:
            cur.execute(
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_name = %s "
                "ORDER BY ordinal_position",
                (table,),
            )
            cols = [{"name": row[0], "type": row[1]} for row in cur.fetchall()]
            if not cols:
                raise ValueError(f"Table '{table}' not found or has no columns.")
            schemas.append(TableSchema(table_name=table, columns=cols))
        cur.close()
        return schemas
    finally:
        conn.close()


def _validate_sql_plan(plan: Dict[str, Any], tables: List[str]) -> None:
    sql = plan.get("sql", "").strip()
    if not sql:
        raise ValueError("Plan must include a non-empty 'sql' field.")

    first_word = sql.split()[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise ValueError(f"Only SELECT/WITH queries are allowed. Got: {first_word}")

    sql_upper = sql.upper()
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_upper):
            raise ValueError(f"Blocked SQL keyword detected: {keyword}")

    table_names_lower = {t.lower() for t in tables}
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE):
        ref = match.group(1).lower()
        if ref not in table_names_lower and ref != "information_schema":
            raise ValueError(f"SQL references unknown table: {ref}. Allowed: {tables}")

    if "params" in plan and not isinstance(plan["params"], dict):
        raise ValueError("params must be a dict.")


def _execute_sql(db_connector: Callable, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    conn = db_connector()
    try:
        cur = conn.cursor()
        exec_sql = re.sub(r":(\w+)", r"%(\1)s", sql)
        cur.execute(exec_sql, params)
        col_names = [desc[0] for desc in cur.description]
        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
        cur.close()
        return rows
    finally:
        conn.close()


def expect_columns(*cols: str) -> Callable[[SQLResult], None]:
    def _check(result: SQLResult) -> None:
        if not result.rows:
            return
        actual_cols = set(result.rows[0].keys())
        missing = [c for c in cols if c not in actual_cols]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {sorted(actual_cols)}")
    return _check


def expect_non_empty(result: SQLResult) -> None:
    if not result.rows:
        raise ValueError("Query returned no rows.")


def ai_sql(
    *,
    db_connector: Callable,
    tables: List[str],
    llm_caller: Callable[[str, str], str],
    post_conditions: Optional[List[Callable[[SQLResult], None]]] = None,
    max_attempts: int = 3,
    compile_only: bool = False,
    plan_cache: Optional[PlanCache] = None,
):
    """
    Decorator that turns a docstring into an LLM-compiled SQL query,
    validates it against the allowlist, and executes it.

    ``db_connector`` must be a callable ``() -> connection`` returning a
    DB-API 2.0 compatible connection. The module never imports a specific
    driver — pass any connector (psycopg2, snowflake.connector, etc.).
    Returns a SQLResult, or the validated plan dict if compile_only=True.
    """
    post_conditions = post_conditions or []

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nl_spec = (func.__doc__ or "").strip()
            if not nl_spec:
                raise ValueError("AI SQL function must have a docstring with instructions.")

            table_schemas = _introspect_tables(db_connector, tables)
            schema_info = [
                {
                    "table": ts.table_name,
                    "columns": ts.columns,
                }
                for ts in table_schemas
            ]

            system_prompt = (
                "You are a strict SQL compiler that converts natural language into a JSON object "
                "containing a parameterized SELECT query.\n"
                "Return ONLY valid JSON. No prose.\n\n"
                "Schema:\n"
                '{"sql": "SELECT ... WHERE col >= :param_name", "params": {"param_name": "value"}}\n\n'
                "Rules:\n"
                "- Only SELECT or WITH (CTE) queries. No INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.\n"
                "- Use :param_name placeholders for literal values (dates, numbers, strings).\n"
                "- Only reference tables provided in the schema.\n"
                "- Use standard SQL syntax compatible with the provided database.\n"
                "- If runtime_params are provided, include them in the plan params dict."
            )

            base_user: Dict[str, Any] = {
                "request": nl_spec,
                "tables": schema_info,
            }
            if kwargs:
                base_user["runtime_params"] = kwargs

            def build_payload(last_error: Optional[str]) -> str:
                payload = dict(base_user)
                if last_error:
                    payload["previous_error"] = last_error
                    payload["instruction"] = "Fix the SQL plan to satisfy the validation rules."
                return json.dumps(payload, indent=2)

            def process_result(plan: Dict[str, Any]) -> Any:
                _validate_sql_plan(plan, tables)
                if compile_only:
                    return plan
                sql = plan["sql"].strip()
                params = plan.get("params", {})
                rows = _execute_sql(db_connector, sql, params)
                result = SQLResult(sql=sql, params=params, rows=rows)
                for check in post_conditions:
                    check(result)
                return result

            run_state.last_run_from_cache = False
            if plan_cache and not compile_only:
                key = cache_key(nl_spec, {
                    "tables": tables,
                    "schema": schema_info,
                    "runtime_params": sorted(kwargs.items()),
                })
                cached = plan_cache.get(key)
                if cached is not None:
                    run_state.last_run_from_cache = True
                    return process_result(cached)
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
