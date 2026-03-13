# DataForge
### Governed data operations through natural language

Thought to share something I've been building — a lightweight framework that lets you describe data operations in plain English inside a function, and returns a validated, production-ready result without any boilerplate. Unlike typical AI coding tools, it avoids code generation entirely — the LLM compiles your intent into a governed execution plan that runs inside the platform, keeping auditability and control where they belong.

---

## Core concept

Treat the LLM as a **compiler**, not an executor. You describe what you want in a function docstring, the LLM produces a structured JSON plan, and a deterministic engine executes it. The LLM never runs code, never touches your data, and cannot produce operations outside a governed allowlist.

```
Natural language spec
      ↓
Schema + sample context (grounding)
      ↓
LLM compiles plan (JSON / IR)
      ↓
Validation (allowlist + schema checks)
      ↓
Safe executor (pandas / SQL / etc.)
      ↓
Post-conditions (output contract)
      ↓
Return DataFrame / Table / Report
```

---

## Project structure

```
dataforge-ai/
  src/dataforge/       
    __init__.py
    ai_base.py
    ai_dataframe.py
    ai_features.py
    ai_profiling.py
    ai_quality.py
    ai_sql.py
    plan_cache.py
    run_state.py
    llm.py             ← unified LLM factory
    llm_bedrock.py     ← AWS Bedrock provider
    llm_openai.py      ← OpenAI API provider
    llm_anthropic.py   ← Anthropic API provider
  main.py              ← demo runner
  data/
    people.csv
    loan_applications.csv
    payments.csv
  pyproject.toml
  README.md
```
---

## Installation

```bash
pip install dataforge-ai
```

With optional extras:

```bash
pip install dataforge-ai[bedrock]     # AWS Bedrock support
pip install dataforge-ai[openai]      # OpenAI API support
pip install dataforge-ai[anthropic]   # Anthropic API support
pip install dataforge-ai[sql]         # PostgreSQL support
pip install dataforge-ai[snowflake]   # Snowflake support
pip install dataforge-ai[all]         # everything
```

Or install from source:

```bash
git clone https://github.com/meetrickyb/dataforge-ai.git
cd dataforge-ai
pip install -e .
```

---

## Quick start

```python
from dataforge import ai_dataframe, expect_columns, expect_non_empty
from dataforge.llm import make_caller

# AWS Bedrock
llm = make_caller(
    provider="bedrock",
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region="us-east-1",
)

# Or OpenAI API
# llm = make_caller(
#     provider="openai",
#     model_id="gpt-4.1-mini",
#     api_key="sk-...",
# )

@ai_dataframe(
    csv_path="data/people.csv",
    llm_caller=llm,
    post_conditions=[expect_columns("id", "first_name"), expect_non_empty],
)
def create_df():
    """
    Read the dataset.
    Create a dataframe capturing id and first_name.
    Ensure id is numeric if possible.
    """

df = create_df()   # returns a real, validated pandas DataFrame
```

### Custom LLM provider

The framework accepts any callable with the signature `(system: str, user: str) -> str`. To use a different LLM backend, write a simple wrapper:

```python
def my_llm(system: str, user: str) -> str:
    # call your model here
    return response_text
```

No changes needed in the decorators or core — the LLM provider is just a callable.

### Running demos

`main.py` includes demos for every module:

```bash
python main.py dataframe            # basic DataFrame creation
python main.py dataframe-adv        # group_by + fill_null
python main.py dataframe-rolling    # rolling window
python main.py dataframe-params     # runtime parameters
python main.py dataframe-inspect    # compile_only (returns plan, no execution)
python main.py features             # derive + bucketize
python main.py features-normalize   # min-max / z-score normalization
python main.py features-encode      # one-hot encoding
python main.py features-inspect     # compile_only
python main.py quality              # data quality checks
python main.py profiling            # dataset profiling
python main.py sql                  # SQL compilation + execution
python main.py sql-params           # SQL with runtime parameters
```

Disable the plan cache to force an LLM call:

```bash
python main.py dataframe --plan-cache false
```

### SQL demos — local PostgreSQL setup

The `sql` and `sql-params` demos require a PostgreSQL instance. The easiest way is Docker:

```bash
docker run -d --name dataforge-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=dataforge \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  postgres:16
```

Then seed the test tables:

```bash
PGPASSWORD=dataforge psql -h localhost -U postgres -d testdb -f data/seed_postgres.sql
```

This creates `customers` and `orders` tables with sample data. After seeding, the SQL demos are ready:

```bash
python main.py sql           # top customers by spend
python main.py sql-params    # orders filtered by date range
```

### Using Snowflake

The `ai_sql` module is database-agnostic — pass any DB-API 2.0 connector. For Snowflake:

```bash
pip install dataforge-ai[snowflake]
```

```python
import snowflake.connector
from dataforge import ai_sql, SQLResult
from dataforge.llm import make_caller

llm = make_caller(provider="bedrock", model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region="us-east-1")

def sf_connector():
    return snowflake.connector.connect(
        account="your_account",
        user="your_user",
        password="your_password",
        database="your_db",
        schema="your_schema",
        warehouse="your_wh",
    )

@ai_sql(
    db_connector=sf_connector,
    tables=["orders", "customers"],
    llm_caller=llm,
)
def top_customers() -> SQLResult:
    """Return total spend by customer for the last 90 days."""

result = top_customers()
```

---

## LLM providers

All providers are accessed through a single unified factory:

```python
from dataforge.llm import make_caller
```

| Provider | Backend | Key params |
|---|---|---|
| `"bedrock"` | AWS Bedrock Converse | `model_id`, `region` |
| `"openai"` | OpenAI API | `model_id`, `api_key`, `base_url` |
| `"anthropic"` | Anthropic API | `model_id`, `api_key`, `max_tokens` |

`make_caller(provider=..., **kwargs)` returns a `(system, user) -> str` callable.
The framework also accepts any custom callable with that signature.

---

## Core modules

| Module | What it does |
|---|---|
| `dataforge.ai_dataframe` | NL → validated DataFrame transformations (select, filter, rename, limit, sort, group_by, deduplicate, fill_null, type_cast, join). Supports CSV and Parquet. |
| `dataforge.ai_sql` | NL → governed SQL compilation with schema grounding + parameterized execution. Database-agnostic (Postgres, Snowflake, etc.). |
| `dataforge.ai_features` | NL → reproducible feature pipelines (filter, derive, bucketize, one_hot_encode, normalize, etc.) with type validation. |
| `dataforge.ai_quality` | NL → rule sets and checks that produce structured quality reports (unique, not_null, non_negative, in_set, null_rate). |
| `dataforge.ai_profiling` | NL → deterministic profiling summaries (row_count, null_counts, cardinality, numeric_stats, outliers). |

All modules share a common base (`ai_base.py`) for JSON parsing, retry logic, and context types. Each domain module brings its own validator, executor, and post-condition helpers.

---

## Why not just generate code?

Code generation is a **human-in-the-loop** workflow — someone writes a prompt, reviews the output, and decides to run it. That works for development, but it doesn't scale to automated, repeatable data operations in a governed environment.

When you generate code, you are trusting the model to produce something correct, safe, and compliant every single time — with no structural guarantee it will. DataForge flips the model: the LLM compiles intent into a constrained plan, and a deterministic engine executes it. The output contract is defined upfront, not hoped for.

**The honest tradeoff:** code generation is more flexible. If an operation is complex enough to fall outside the allowlist, you would still write it by hand. DataForge is the right tool for the governed, repeatable subset — not a replacement for all data engineering.

---

## How does the LLM stay in bounds?

The framework does not rely on the LLM behaving well. There are four hard layers:

1. **LLM output is JSON, not code** — even a misbehaving model cannot execute anything; it can only produce text that gets parsed into a plan.
2. **`_validate_plan` rejects anything outside the allowlist** — if the model produces an unsupported operation, it fails before execution.
3. **Column grounding** — the model can only reference columns that exist in the provided schema. Hallucinated fields fail validation.
4. **Post-conditions** — even a plan that passes structural validation must satisfy the output contract, or it retries and eventually raises.

The LLM is treated as **untrusted by design** — like user input, everything it produces goes through validation before anything executes.

**The honest caveat:** the LLM could produce a plan that is structurally valid but semantically wrong (e.g., filtering on the wrong value). Post-conditions catch shape and contract issues, but domain correctness still depends on well-written docstrings and meaningful post-conditions. That is the human's responsibility.

---

## Deterministic execution — a key design feature

A common concern with LLM-based tools is non-determinism: run the same thing twice, get different results. DataForge addresses this by separating **compilation** from **execution**.

The execution engine is fully deterministic. Given the same compiled plan and the same data, it produces the exact same result every time. There is no randomness, no model inference, and no ambiguity in the executor — it is a straightforward plan interpreter.

The compilation step (LLM → plan) is inherently non-deterministic — the same docstring might produce slightly different but functionally equivalent plans across runs. However, this non-determinism is **bounded and removable**:

1. **Bounded** — the plan must pass structural validation and post-conditions. Different plans that satisfy the same contract are functionally equivalent.
2. **Removable** — once a plan is compiled and validated, it can be cached and replayed without the LLM. In production, you pin a known-good plan and run it deterministically with zero LLM involvement.

This is fundamentally different from working with LLMs or agents directly, where non-determinism affects every step: the reasoning, the tool selection, the execution order, and the output format. With DataForge, non-determinism exists at exactly one step, and that step can be eliminated entirely.

```
Development:   docstring → LLM compiles plan → validate → execute → result
                              (non-deterministic)    (deterministic)

Production:    cached plan → execute → result
                    (fully deterministic, no LLM)
```

---

## How is this different from AI agents?

An AI agent is autonomous — it decides what steps to take, what tools to call, and in what order. You give it a goal and trust it to figure out the path. DataForge gives the LLM exactly one job: compile a natural language instruction into a structured plan. After that, the LLM is done. A deterministic engine takes over.

| | AI Agent | DataForge |
|---|---|---|
| LLM role | Decision-maker, executor | Compiler only |
| Autonomy | High — multi-step, open-ended | None beyond plan generation |
| Trust model | Trust the agent to act correctly | LLM output is untrusted, always validated |
| Execution | Agent runs tools, code, APIs | Controlled engine with allowlist ops only |
| Determinism | Non-deterministic by nature | Same plan = same result, always |
| Auditability | Hard — reasoning is opaque | Full — intent, plan, validation all inspectable |
| Failure modes | Open-ended, hard to predict | Bounded — fails at validation or post-condition |

DataForge is architecturally the **opposite** of an agent — instead of giving the LLM more capability and trusting it to stay in bounds, it takes capability away and enforces bounds structurally. The LLM cannot act outside the allowlist because the executor simply does not implement anything outside it. With agents, the constraint is a guideline; here it is a hard wall.

**The honest tradeoff:** agents handle open-ended, complex, multi-step tasks that DataForge cannot. DataForge is the right model when you know the output type upfront, need reproducibility, and operate in a governed environment — which describes most production data work.

---

## Governance model

- **Allowlist ops:** extensions require explicit addition to the allowlist and validation logic — no silent capability creep.
- **Context policy:** only approved data sources and schema metadata are injected into the LLM prompt.
- **Logging policy:** store intent, compiled plan, and validation outcomes for every run.
- **Security policy:** no network calls, filesystem writes, or arbitrary code execution from within the executor.
- **Provider policy:** LLM provider and model are set by the caller, not the framework — teams use org-approved models.

---

## Non-negotiables for governed rollout

- Schema grounding: model sees real columns, dtypes, and sample rows — no hallucinated fields.
- Constrained operations: only transformations that can be audited and supported are on the allowlist.
- Deterministic execution: the IR is executed by a stable engine (pandas / SQL), not re-interpreted.
- Post-conditions: assertions define the output contract and prevent silent drift.
- Observability: intent, plan, inputs metadata, and validation outcomes are logged together.
