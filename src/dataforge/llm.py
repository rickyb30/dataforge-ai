from __future__ import annotations

from typing import Callable

PROVIDERS = ("bedrock", "openai", "azure_openai", "anthropic")


def make_caller(*, provider: str, **kwargs) -> Callable[[str, str], str]:
    """
    Unified factory — returns a ``(system, user) -> str`` callable for the
    chosen LLM provider.

    Parameters
    ----------
    provider : ``"bedrock"``, ``"openai"``, ``"azure_openai"``, or ``"anthropic"``
    **kwargs : Forwarded to the provider's ``make_caller``:
               - bedrock:      ``model_id``, ``region``
               - openai:       ``model_id``, ``api_key``, ``base_url``
               - anthropic:    ``model_id``, ``api_key``, ``max_tokens``
    """
    if provider == "bedrock":
        from dataforge.llm_bedrock import make_caller as _make
    elif provider == "openai":
        from dataforge.llm_openai import make_caller as _make
    elif provider == "anthropic":
        from dataforge.llm_anthropic import make_caller as _make
    else:
        raise ValueError(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    return _make(**kwargs)
