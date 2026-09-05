#!/usr/bin/env python3
"""Ask a Tinker model a question and execute its calculator tool calls."""

from __future__ import annotations

import argparse
import ast
import asyncio
from dataclasses import dataclass
import json
import math
import operator
from typing import Any

from settings import DEFAULT_MODEL
from tinker_api import chat_completion

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator through the `calculate` tool. "
    "Always use the calculator for arithmetic or numerical verification, and use its result "
    "when answering. Follow the user's requested output format exactly."
)


class CalculatorError(ValueError):
    """Raised when an expression is not supported by the calculator."""


_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        if isinstance(node.op, ast.Pow):
            exponent = _evaluate(node.right)
            if isinstance(exponent, float) and not exponent.is_integer():
                raise CalculatorError("exponents must be integers")
            if abs(exponent) > 100:
                raise CalculatorError("exponent is too large")
        value = _BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    else:
        raise CalculatorError("only numbers, parentheses, and + - * / % ** are supported")

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CalculatorError("result is not a finite number")
    return value


def calculate(expression: str) -> int | float:
    """Evaluate a small arithmetic expression without using Python's eval."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    if len(expression) > 200:
        raise CalculatorError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate(tree.body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise CalculatorError(str(exc)) from exc


CALCULATOR_TOOL: dict[str, Any] = {
    "name": "calculate",
    "description": "Evaluate a basic arithmetic expression. Use this for exact arithmetic.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression using numbers, parentheses, +, -, *, /, %, or **.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}
OPENAI_CALCULATOR_TOOL = {"type": "function", "function": CALCULATOR_TOOL}


@dataclass(frozen=True)
class AgentResult:
    """Final answer plus tool-use metadata for evaluation."""

    answer: str
    calculator_invoked: bool
    tool_results: list[dict[str, Any]]


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read either a Cookbook dict or a pydantic-style object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def run_calculator_tool(tool_call: Any) -> str:
    """Execute one parsed model tool call and return JSON for the tool message."""
    function = _field(tool_call, "function", tool_call)
    name = _field(function, "name")
    arguments = _field(function, "arguments", "{}")

    try:
        if name != "calculate":
            raise CalculatorError(f"unknown tool: {name}")
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        result = calculate(args["expression"])
        return json.dumps({"result": result})
    except (CalculatorError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return json.dumps({"error": str(exc)})


async def run_tinker_agent(
    user_prompt: str,
    *,
    model_name: str,
    max_rounds: int = 4,
) -> AgentResult:
    """Run the model/tool/model loop and retain calculator-use metadata."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    calculator_invoked = False
    tool_results: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        assistant_message, finish_reason = await chat_completion(
            model=model_name,
            messages=messages,
            tools=[OPENAI_CALCULATOR_TOOL],
        )

        tool_calls = _field(assistant_message, "tool_calls", []) or []
        if not tool_calls:
            answer = _field(assistant_message, "content", "") or ""
            if finish_reason == "length" and not answer:
                raise RuntimeError("model exhausted its output-token budget before answering")
            return AgentResult(
                answer=answer,
                calculator_invoked=calculator_invoked,
                tool_results=tool_results,
            )
        if finish_reason not in (None, "tool_calls"):
            raise RuntimeError(f"model returned tool calls with finish reason {finish_reason!r}")

        messages.append(
            {
                "role": "assistant",
                "content": _field(assistant_message, "content", "") or "",
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            function = _field(tool_call, "function", tool_call)
            tool_name = _field(function, "name", "calculate")
            calculator_invoked = calculator_invoked or tool_name == "calculate"
            tool_content = run_calculator_tool(tool_call)
            tool_results.append(json.loads(tool_content))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _field(tool_call, "id", "calculator_call"),
                    "name": tool_name,
                    "content": tool_content,
                }
            )

    raise RuntimeError(f"model exceeded the {max_rounds}-round tool-call limit")


async def ask_tinker(
    user_prompt: str,
    *,
    model_name: str,
    max_rounds: int = 4,
) -> str:
    """Convenience wrapper that returns only the model's final answer."""
    result = await run_tinker_agent(
        user_prompt,
        model_name=model_name,
        max_rounds=max_rounds,
    )
    return result.answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is (17 * 23) / 4?")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(asyncio.run(ask_tinker(args.prompt, model_name=args.model)))


if __name__ == "__main__":
    main()
