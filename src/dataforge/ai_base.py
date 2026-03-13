from __future__ import annotations

import csv as _csv_mod
import json
import re
from typing import Any, Callable, Dict, List, Optional


def csv_columns(path: str) -> List[str]:
    """Read just the header row of a CSV — no pandas needed."""
    with open(path, newline="") as f:
        return next(_csv_mod.reader(f))


def extract_json(text: str) -> Dict[str, Any]:
    """
    Model may return JSON wrapped in fences / prose. Extract first JSON object.
    """
    text = re.sub(r"```(json)?", "", text).replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output did not contain a valid JSON object.")
        return json.loads(text[start : end + 1])


def run_with_retry(
    *,
    llm_caller: Callable[[str, str], str],
    system_prompt: str,
    build_payload: Callable[[Optional[str]], str],
    process_result: Callable[[Dict[str, Any]], Any],
    max_attempts: int = 3,
    out_plan: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Generic retry loop used by all domain modules.

    Parameters
    ----------
    llm_caller      : ``(system, user) -> str`` callable from an LLM provider.
    system_prompt   : System prompt sent to the LLM.
    build_payload   : ``(last_error | None) -> str`` — builds the user payload.
                      Receives the previous error message (or None on first attempt)
                      so it can include repair instructions.
    process_result  : ``(parsed_json) -> result`` — validates the parsed plan and
                      produces the final result. Should raise on validation failure.
    max_attempts    : Number of retries before giving up.
    out_plan       : If provided, a dict to receive the successful plan (e.g. for caching).
                      Set as out_plan["plan"] = plan after process_result(plan) succeeds.
    """
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        user_payload = build_payload(last_error)
        raw = llm_caller(system_prompt, user_payload)

        try:
            plan = extract_json(raw)
            result = process_result(plan)
            if out_plan is not None:
                out_plan["plan"] = plan
            return result
        except Exception as e:
            last_error = f"Attempt {attempt} failed: {type(e).__name__}: {e}"

    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_error}")
