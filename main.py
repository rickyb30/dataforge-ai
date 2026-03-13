"""
DataForge — demo runner.

Demonstrates every core module (dataframe, features, profiling, quality, sql)
in a single file.  Run:  python main.py <demo> [--plan-cache true|false]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_t_start = time.perf_counter()


# ============================================================
# 1. LLM + plan cache setup
# ============================================================

from dataforge.llm import make_caller

llm = make_caller(
    provider="bedrock",
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region="us-east-1",
)
from dataforge.plan_cache import plan_cache


# ============================================================
# 2a. Demos — AI DataFrame
# ============================================================

from dataforge.ai_dataframe import ai_dataframe, expect_columns, expect_non_empty


@ai_dataframe(
    csv_path="data/people.csv",
    llm_caller=llm,
    post_conditions=[expect_columns("id", "first_name"), expect_non_empty],
    plan_cache=plan_cache,
)
def create_df() -> pd.DataFrame:
    """
    Read the dataset.
    Create a dataframe capturing id and first_name.
    Ensure id is numeric if possible.
    """


@ai_dataframe(
    csv_path="data/loan_applications.csv",
    llm_caller=llm,
    post_conditions=[expect_non_empty],
    plan_cache=plan_cache,
)
def loan_summary() -> pd.DataFrame:
    """
    Fill null or zero annual_income values with the median annual_income.
    Group by loan_purpose and compute the average credit_score and total loan_amount.
    Sort by total loan_amount descending.
    """


@ai_dataframe(
    csv_path="data/payments.csv",
    llm_caller=llm,
    post_conditions=[expect_columns("payment_date", "amount", "amount_7d_avg"), expect_non_empty],
    plan_cache=plan_cache,
)
def payments_rolling_avg() -> pd.DataFrame:
    """
    Sort by payment_date ascending.
    Compute a 7-row rolling mean of amount into amount_7d_avg.
    Select payment_date, amount, and amount_7d_avg.
    """


@ai_dataframe(
    csv_path="data/payments.csv",
    llm_caller=llm,
    post_conditions=[expect_non_empty],
    plan_cache=plan_cache,
)
def payments_by_status() -> pd.DataFrame:
    """
    Filter payments where status equals the value of runtime_params.status.
    Select id, customer_id, amount, and payment_date.
    Sort by amount descending.
    """


@ai_dataframe(
    csv_path="data/people.csv",
    llm_caller=llm,
    compile_only=True,
)
def inspect_plan() -> dict:
    """
    Select id, first_name, last_name.
    Sort by last_name ascending.
    """


# ============================================================
# 2b. Demos — AI Features
# ============================================================

from dataforge.ai_features import (
    ai_features,
    expect_columns as feat_expect_columns,
    expect_feature_types,
)


@ai_features(
    csv_path="data/loan_applications.csv",
    llm_caller=llm,
    post_conditions=[
        feat_expect_columns("debt_to_income", "age_bucket"),
        expect_feature_types({"debt_to_income": "float"}),
    ],
    plan_cache=plan_cache,
)
def build_features() -> pd.DataFrame:
    """
    Create debt_to_income = total_debt / annual_income.
    Bucket age into: <25, 25-34, 35-44, 45-54, 55+.
    Drop records where annual_income is null or zero.
    """


@ai_features(
    csv_path="data/loan_applications.csv",
    llm_caller=llm,
    post_conditions=[
        feat_expect_columns("credit_score_norm", "annual_income_norm"),
        expect_feature_types({"credit_score_norm": "float", "annual_income_norm": "float"}),
    ],
    plan_cache=plan_cache,
)
def normalize_features() -> pd.DataFrame:
    """
    Normalize credit_score using min-max scaling into credit_score_norm.
    Normalize annual_income using z-score scaling into annual_income_norm.
    Drop applicant_name.
    """


@ai_features(
    csv_path="data/loan_applications.csv",
    llm_caller=llm,
    post_conditions=[expect_feature_types({"annual_income": "float"})],
    plan_cache=plan_cache,
)
def encode_loan_purpose() -> pd.DataFrame:
    """
    Drop records where annual_income is null.
    One-hot encode loan_purpose, dropping the original column.
    Keep all other columns.
    """


@ai_features(
    csv_path="data/loan_applications.csv",
    llm_caller=llm,
    compile_only=True,
)
def inspect_features_plan() -> dict:
    """
    Normalize credit_score using min-max scaling into credit_score_norm.
    Bucket age into: <30, 30-50, 50+.
    Drop applicant_name.
    """


# ============================================================
# 2c. Demos — AI Profiling
# ============================================================

from dataforge.ai_profiling import ai_profile


@ai_profile(
    csv_path="data/people.csv",
    llm_caller=llm,
    plan_cache=plan_cache,
)
def profile_people():
    """
    Summarize row count, null counts, cardinality, and basic stats for numeric fields.
    Highlight potential outliers in age.
    """


# ============================================================
# 2d. Demos — AI Quality
# ============================================================

from dataforge.ai_quality import ai_quality


@ai_quality(
    csv_path="data/payments.csv",
    llm_caller=llm,
    plan_cache=plan_cache,
)
def payments_quality():
    """
    id must be unique.
    amount must be non-negative.
    currency must be one of: USD, CAD, EUR.
    customer_id null rate must be less than 0.1%.
    """


# ============================================================
# 2e. Demos — AI SQL
# ============================================================

from dataforge.ai_sql import (
    ai_sql,
    expect_columns as sql_expect_columns,
    expect_non_empty as sql_expect_non_empty,
    SQLResult,
)


def db_connector():
    import psycopg2
    return psycopg2.connect(
        host="localhost", port=5432, dbname="testdb",
        user="postgres", password="dataforge",
    )


@ai_sql(
    db_connector=db_connector,
    tables=["orders", "customers"],
    llm_caller=llm,
    post_conditions=[sql_expect_columns("customer_id", "name", "total_spend"), sql_expect_non_empty],
    plan_cache=plan_cache,
)
def top_customers() -> SQLResult:
    """
    Return total spend by customer for the last 90 days.
    Join orders to customers.
    Only include customers with total spend over 500.
    Order by total spend descending.
    """


@ai_sql(
    db_connector=db_connector,
    tables=["orders"],
    llm_caller=llm,
    post_conditions=[sql_expect_non_empty],
    plan_cache=plan_cache,
)
def orders_in_range() -> SQLResult:
    """
    Return order_id, customer_id, amount, and order_date.
    Filter orders where order_date is between runtime_params.start_date and runtime_params.end_date.
    Order by order_date ascending.
    """


# ============================================================
# 3. CLI runner
# ============================================================

DEMOS = {
    "dataframe":          ("AI DataFrame — basic",           lambda: create_df()),
    "dataframe-adv":      ("AI DataFrame — group_by / fill_null", lambda: loan_summary()),
    "dataframe-rolling":  ("AI DataFrame — rolling window",  lambda: payments_rolling_avg()),
    "dataframe-params":   ("AI DataFrame — runtime params",  lambda: payments_by_status(status="completed")),
    "dataframe-inspect":  ("AI DataFrame — compile_only",    lambda: inspect_plan()),
    "sql":                ("AI SQL — top customers",         lambda: top_customers()),
    "sql-params":         ("AI SQL — runtime params",        lambda: orders_in_range(start_date="2025-01-01", end_date="2025-03-01")),
    "features":           ("AI Features — derive + bucketize", lambda: build_features()),
    "features-normalize": ("AI Features — normalize",        lambda: normalize_features()),
    "features-encode":    ("AI Features — one_hot_encode",   lambda: encode_loan_purpose()),
    "features-inspect":   ("AI Features — compile_only",     lambda: inspect_features_plan()),
    "quality":            ("AI Quality",                     lambda: payments_quality()),
    "profiling":          ("AI Profiling",                   lambda: profile_people()),
}


def run_demo(name: str, plan_cache_enabled: bool = True) -> None:
    plan_cache.enabled = plan_cache_enabled
    label, func = DEMOS[name]
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0

    if type(result).__name__ == "DataFrame":
        print(result.to_string())
        print("\nDtypes:\n", result.dtypes)
    elif type(result).__name__ == "SQLResult":
        print("SQL:", result.sql)
        print("Params:", result.params)
        print("\nRows:")
        for row in result.rows:
            print(row)
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)

    print(f"\nResponse time: {elapsed:.2f}s (total wall: {time.perf_counter() - _t_start:.2f}s)")
    from dataforge import run_state
    if run_state.last_run_from_cache:
        print("Plan: from cache (LLM skipped)")
    else:
        print("Plan: from LLM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataForge demo runner")
    parser.add_argument(
        "module",
        choices=list(DEMOS),
        help="Which demo to run: " + ", ".join(DEMOS),
    )
    parser.add_argument(
        "--plan-cache",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        metavar="true|false",
        help="Use plan cache to reuse compiled plans for same intent+context (default: true)",
    )
    args = parser.parse_args()
    run_demo(args.module, plan_cache_enabled=args.plan_cache)
