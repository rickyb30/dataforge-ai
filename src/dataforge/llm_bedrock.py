from __future__ import annotations

from typing import Callable

import boto3


def make_caller(*, model_id: str, region: str = "us-east-1") -> Callable[[str, str], str]:
    """
    Return a callable ``(system, user) -> str`` backed by AWS Bedrock Converse.

    Parameters
    ----------
    model_id : Bedrock model ID, e.g. ``"us.anthropic.claude-sonnet-4-5-20250929-v1:0"``
    region   : AWS region, defaults to ``"us-east-1"``
    """

    def call(system: str, user: str) -> str:
        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 1200},
        )
        blocks = resp["output"]["message"]["content"]
        return "".join(block.get("text", "") for block in blocks if isinstance(block, dict))

    return call
