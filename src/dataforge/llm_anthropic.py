from __future__ import annotations

from typing import Callable


def make_caller(*, model_id: str, api_key: str, max_tokens: int = 1200) -> Callable[[str, str], str]:
    """
    Return a callable ``(system, user) -> str`` backed by the Anthropic API.

    Parameters
    ----------
    model_id   : Model name, e.g. ``"claude-sonnet-4-20250514"``
    api_key    : Anthropic API key
    max_tokens : Maximum tokens in the response, defaults to 1200
    """
    client = None

    def _get_client():
        nonlocal client
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        return client

    def call(system: str, user: str) -> str:
        resp = _get_client().messages.create(
            model=model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.content[0].text

    return call
