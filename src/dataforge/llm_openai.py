from __future__ import annotations

from typing import Callable


def make_caller(*, model_id: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> Callable[[str, str], str]:
    """
    Return a callable ``(system, user) -> str`` backed by the OpenAI API.

    Parameters
    ----------
    model_id : Model name, e.g. ``"gpt-4.1-mini"``
    api_key  : OpenAI API key
    base_url : API base URL, defaults to ``"https://api.openai.com/v1"``
    """
    client = None

    def _get_client():
        nonlocal client
        if client is None:
            import openai
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
        return client

    def call(system: str, user: str) -> str:
        resp = _get_client().responses.create(
            model=model_id,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return resp.output_text

    return call
