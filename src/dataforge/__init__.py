"""DataForge — governed data operations through natural language."""

from .ai_dataframe import ai_dataframe, expect_columns, expect_non_empty
from .ai_features import (
    ai_features,
    expect_columns as feat_expect_columns,
    expect_feature_types,
)
from .ai_profiling import ai_profile
from .ai_quality import ai_quality
from .ai_sql import ai_sql, SQLResult
from .plan_cache import plan_cache, PlanCache, cache_key

__all__ = [
    "ai_dataframe",
    "ai_features",
    "ai_profile",
    "ai_quality",
    "ai_sql",
    "expect_columns",
    "expect_non_empty",
    "feat_expect_columns",
    "expect_feature_types",
    "SQLResult",
    "plan_cache",
    "PlanCache",
    "cache_key",
]
