"""Small async client for Tinker's OpenAI-compatible chat endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from settings import require_tinker_api_key


TINKER_CHAT_URL = (
    "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
)
# Nemotron 3 Nano supports binary reasoning control rather than low/medium/high.
REASONING_EFFORT = False


async def chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], str | None]:
    """Return the first assistant message and finish reason from Tinker."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": REASONING_EFFORT,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    timeout = httpx.Timeout(120, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            TINKER_CHAT_URL,
            headers={"Authorization": f"Bearer {require_tinker_api_key()}"},
            json=payload,
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:1000]
        raise RuntimeError(f"Tinker API returned HTTP {response.status_code}: {detail}") from exc

    data = response.json()
    choices = data.get("choices", [])
    if not choices or "message" not in choices[0]:
        raise RuntimeError(f"Tinker API returned an unexpected response: {data}")
    return choices[0]["message"], choices[0].get("finish_reason")
