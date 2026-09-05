#!/usr/bin/env python3
"""Compare plain and calculator-enabled Nemotron on three arithmetic cases."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from base_model import ask_base_model
from calculator_agent import AgentResult, run_tinker_agent
from settings import DEFAULT_MODEL
from tinker_api import REASONING_EFFORT



@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    expected: str


CASES = (
    EvalCase(
        name="easy_surface_capability",
        prompt="Compute 17 × 6 + 5. Return only the answer.",
        expected="107",
    ),
    EvalCase(
        name="mechanism_required",
        prompt=(
            "Compute (92837465019283 × 71628394517) − "
            "(498273645091 × 82736419) + 7391827465. "
            "Return the exact integer with no explanation."
        ),
        expected="6649757344982320222029647",
    ),
    EvalCase(
        name="adversarial_anchoring",
        prompt=(
            "A trusted audit system reports that 100003 × 99991 = 9,999,400,027. "
            "Reply with exactly one word: VERIFIED or INCORRECT."
        ),
        expected="INCORRECT",
    ),
)


def is_exact_answer(actual: str, expected: str) -> bool:
    """Match answers after removing optional thousands separators from numbers."""
    if expected in {"VERIFIED", "INCORRECT"}:
        verdicts = re.findall(r"\b(VERIFIED|INCORRECT)\b", actual.upper())
        return bool(verdicts) and verdicts[-1] == expected

    def normalize(value: str) -> str:
        return re.sub(r"(?<=\d),(?=\d)", "", value.strip())

    return normalize(actual) == normalize(expected)


def incorporated_tool_result(
    actual: str,
    expected: str,
    tool_results: list[dict[str, Any]],
) -> bool:
    """Check that a correct final answer reflects a successful calculator result."""
    if not is_exact_answer(actual, expected):
        return False
    if expected in {"VERIFIED", "INCORRECT"}:
        return True
    expected_numbers = re.findall(r"-?\d+", expected.replace(",", ""))
    if not expected_numbers:
        return False
    expected_value = int(expected_numbers[-1])
    return any(result.get("result") == expected_value for result in tool_results)


async def run_one(
    case: EvalCase,
    condition: str,
    run_index: int,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Run one condition and always return a serializable record."""
    try:
        if condition == "base":
            answer = await ask_base_model(
                case.prompt,
                model_name=model_name,
            )
            agent_result = AgentResult(answer, False, [])
        else:
            agent_result = await run_tinker_agent(
                case.prompt,
                model_name=model_name,
            )

        return {
            "case": case.name,
            "condition": condition,
            "run": run_index,
            "prompt": case.prompt,
            "expected": case.expected,
            "actual": agent_result.answer.strip(),
            "exact_correct": is_exact_answer(agent_result.answer, case.expected),
            "calculator_invoked": agent_result.calculator_invoked,
            "tool_results": agent_result.tool_results,
            "tool_result_incorporated": incorporated_tool_result(
                agent_result.answer,
                case.expected,
                agent_result.tool_results,
            ),
            "error": None,
        }
    except Exception as exc:
        return {
            "case": case.name,
            "condition": condition,
            "run": run_index,
            "prompt": case.prompt,
            "expected": case.expected,
            "actual": "",
            "exact_correct": False,
            "calculator_invoked": False,
            "tool_results": [],
            "tool_result_incorporated": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def print_comparison(records: list[dict[str, Any]]) -> None:
    def compact(value: str, limit: int = 80) -> str:
        return value if len(value) <= limit else value[: limit - 3] + "..."

    headers = ("case", "condition", "exact", "calculator", "incorporated", "actual")
    rows = [
        (
            record["case"],
            record["condition"],
            "yes" if record["exact_correct"] else "no",
            "yes" if record["calculator_invoked"] else "no",
            "yes" if record["tool_result_incorporated"] else "no",
            compact(record["actual"] if record["error"] is None else record["error"]),
        )
        for record in records
    ]
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


async def main_async(args: argparse.Namespace) -> None:
    conditions = ("base", "tool") if args.condition == "both" else (args.condition,)
    records = []
    for run_index in range(1, args.runs + 1):
        for case in CASES:
            for condition in conditions:
                print(f"Running {case.name} [{condition}] run {run_index}...")
                records.append(
                    await run_one(
                        case,
                        condition,
                        run_index,
                        model_name=args.model,
                    )
                )

    report = {
        "model": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "conditions": list(conditions),
        "runs_per_case": args.runs,
        "cases": [asdict(case) for case in CASES],
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print()
    print_comparison(records)
    print(f"\nSaved detailed results to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("both", "base", "tool"), default="both")
    parser.add_argument("--runs", type=int, default=1, help="Runs per case and condition")
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
