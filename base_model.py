#!/usr/bin/env python3
"""Prompt a Tinker base model without giving it any tools (ablation)."""

from __future__ import annotations

import argparse
import asyncio

from settings import DEFAULT_MODEL
from tinker_api import chat_completion

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's requested output format exactly."
)


async def ask_base_model(
    user_prompt: str,
    *,
    model_name: str,
) -> str:
    """Sample a model directly; there is intentionally no tool declaration or loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response, finish_reason = await chat_completion(model=model_name, messages=messages)
    answer = response.get("content", "") or ""
    if finish_reason == "length" and not answer:
        raise RuntimeError("model exhausted its output-token budget before answering")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is (17 * 23) / 4?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(asyncio.run(ask_base_model(args.prompt, model_name=args.model)))


if __name__ == "__main__":
    main()
