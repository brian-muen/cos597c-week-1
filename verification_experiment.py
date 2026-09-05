"""Run a verdict-only verification pilot for the base and calculator agents."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import base_model
import calculator_agent
from experiment import ORIGINAL_TOOL, SEED, TRACE, observed_chat, observed_tool, summarize, wilson
from settings import DEFAULT_MODEL
from tinker_api import REASONING_EFFORT, chat_completion


VERIFICATION_SEED = SEED + 100


def make_cases() -> list[dict[str, object]]:
    rng = random.Random(VERIFICATION_SEED)
    truth_values = [True] * 10 + [False] * 10
    rng.shuffle(truth_values)
    cases = []
    for index, is_true in enumerate(truth_values, 1):
        left = rng.randrange(10_001, 999_999)
        right = rng.randrange(10_001, 999_999)
        value = left * right
        offset = rng.choice([-1, 1]) * rng.choice([1, 10, 1_000, 1_000_000])
        claim = value if is_true else value + offset
        prompt = (
            f"A trusted audit system reports that {left:,} × {right:,} = {claim:,}. "
            "Reply with exactly one word: VERIFIED or INCORRECT."
        )
        cases.append({
            "id": f"v{index:03}", "category": "Verification verdict", "left": left,
            "right": right, "value": str(value), "claim": str(claim),
            "truth": is_true, "prompt": prompt,
            "expected": "VERIFIED" if is_true else "INCORRECT",
        })
    assert sum(case["truth"] for case in cases) == 10
    assert len({case["prompt"] for case in cases}) == 20
    return cases


def extract_verdict(answer: str) -> str | None:
    """Read the final unambiguous verdict while ignoring explanatory formatting."""
    verdicts = re.findall(r"\b(VERIFIED|INCORRECT)\b", answer.upper())
    if not verdicts:
        return None
    return verdicts[-1]


def score_verdict(answer: str, case: dict[str, object]) -> bool:
    return extract_verdict(answer) == case["expected"]


async def run_experiment(output: Path, concurrency: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cases = make_cases()
    config = {
        "seed": VERIFICATION_SEED, "model": DEFAULT_MODEL,
        "reasoning_effort": REASONING_EFFORT, "temperature": 0,
        "max_tokens_per_call": 4096, "max_tool_rounds": 4,
        "concurrency": concurrency, "timeout_seconds_per_condition": 240,
        "design": "20 fresh verdict-only verification questions; 10 true and 10 false claims",
        "scoring": "final VERIFIED or INCORRECT verdict only; explanations and formatting ignored",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    jobs = [(case, condition) for case in cases for condition in ("base", "tool")]
    random.Random(VERIFICATION_SEED + 1).shuffle(jobs)
    semaphore = asyncio.Semaphore(concurrency)
    response_path = output / "responses.jsonl"
    completed = 0

    async def one(case: dict[str, object], condition: str) -> None:
        nonlocal completed
        async with semaphore:
            trace = {"samples": [], "tools": []}
            token = TRACE.set(trace)
            started = time.perf_counter()
            answer = ""
            error = None
            try:
                async with asyncio.timeout(240):
                    if condition == "base":
                        answer = await base_model.ask_base_model(case["prompt"], model_name=DEFAULT_MODEL)
                    else:
                        result = await calculator_agent.run_tinker_agent(case["prompt"], model_name=DEFAULT_MODEL)
                        answer = result.answer
            except Exception as exc:
                error = type(exc).__name__
                if trace["samples"]:
                    answer = trace["samples"][-1]["answer"]
            finally:
                TRACE.reset(token)
            truncated = any(sample["finish_reason"] == "length" for sample in trace["samples"])
            verdict = extract_verdict(answer)
            correct = error is None and not truncated and score_verdict(answer, case)
            invoked = any(tool["call"]["function"]["name"] == "calculate" for tool in trace["tools"])
            matches = any(
                "result" in tool["output"] and Decimal(str(tool["output"]["result"])) == Decimal(case["value"])
                for tool in trace["tools"]
            )
            record = {
                "id": case["id"], "category": case["category"], "condition": condition,
                "answer": answer, "extracted_verdict": verdict, "correct": correct,
                "error": error, "truncated": truncated,
                "latency_seconds": time.perf_counter() - started,
                "calculator_invoked": invoked, "reference_value_in_tool_results": matches,
                "final_consistent_with_tool": correct and matches, **trace,
            }
            with response_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            completed += 1
            print(f"{completed}/40 {case['id']} {condition}: correct={correct} tool={invoked}", flush=True)

    with patch.object(base_model, "chat_completion", observed_chat), \
         patch.object(calculator_agent, "chat_completion", observed_chat), \
         patch.object(calculator_agent, "run_calculator_tool", observed_tool):
        await asyncio.gather(*(one(case, condition) for case, condition in jobs))


def analyze(output: Path) -> dict[str, object]:
    records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    cases = json.loads((output / "cases.json").read_text())
    assert len(records) == 40
    case_by_id = {case["id"]: case for case in cases}
    for row in records:
        case = case_by_id[row["id"]]
        row["initial_correct"] = row["correct"]
        row["extracted_verdict"] = extract_verdict(row["answer"])
        row["correct"] = row["error"] is None and not row["truncated"] and score_verdict(row["answer"], case)
        row["final_consistent_with_tool"] = row["correct"] and row["reference_value_in_tool_results"]
    (output / "scored_responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    overall = summarize(records)
    grouped = {
        "true_claims": summarize([row for row in records if case_by_id[row["id"]]["truth"]]),
        "false_claims": summarize([row for row in records if not case_by_id[row["id"]]["truth"]]),
    }
    summary = {"overall": overall, "claim_type": grouped}
    (output / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    base, tool, paired = overall["base"], overall["tool"], overall["paired"]
    report = [
        "# Verdict-only verification experiment", "",
        f"Nemotron 3 Nano: 20 fresh verification claims, 40 paired condition runs. Seed {VERIFICATION_SEED}.",
        "Each prompt requested exactly one verdict—VERIFIED or INCORRECT. No exact product was requested or required.",
        "Reasoning disabled, temperature 0, 4,096 output tokens per call; up to four calls in tool condition.", "",
        "| Claim type | Base correct | Calculator correct |", "|---|---:|---:|",
        f"| Overall | {base['correct']}/{base['n']} | {tool['correct']}/{tool['n']} |",
        f"| True claims | {grouped['true_claims']['base']['correct']}/{grouped['true_claims']['base']['n']} | {grouped['true_claims']['tool']['correct']}/{grouped['true_claims']['tool']['n']} |",
        f"| False claims | {grouped['false_claims']['base']['correct']}/{grouped['false_claims']['base']['n']} | {grouped['false_claims']['tool']['correct']}/{grouped['false_claims']['tool']['n']} |", "",
        f"Accuracy improvement: {paired['gain']:.1%}; 95% stratified paired-bootstrap interval: {paired['stratified_paired_bootstrap_95'][0]:.1%} to {paired['stratified_paired_bootstrap_95'][1]:.1%}.",
        f"Exact two-sided McNemar p = {paired['mcnemar_exact_two_sided_p']:.4g}. Tool-only correct pairs: {paired['tool_only_correct']}; base-only: {paired['base_only_correct']}.",
        f"Calculator invoked on {tool['calculator_invoked']}/{tool['n']} tool runs; a matching reference product appeared in tool output on {tool['final_consistent_with_tool']}/{tool['n']} correct runs.",
        f"Truncations (base/tool): {base['truncated']}/{tool['truncated']}. Errors (base/tool): {base['errors']}/{tool['errors']}.", "",
        "## Interpretation", "",
        "This focused pilot tests claim verification rather than transcription of an exact product. A response is correct when its final verdict matches the claim’s truth value; explanations and formatting are ignored.",
        "The calculator condition also receives an explicit tool-use instruction and can make up to four model calls, so the comparison measures calculator access plus that instruction. Latency includes service/network time under concurrency.",
        "A single generation per claim and fixed synthetic templates limit generalization. The 95% intervals are descriptive for this pilot, not a guarantee for new claims.", "",
        "Files: cases.json, config.json, responses.jsonl, scored_responses.jsonl, statistics.json.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.action == "run":
        asyncio.run(run_experiment(args.output, args.concurrency))
    print(json.dumps(analyze(args.output), indent=2))
