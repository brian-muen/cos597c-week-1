#!/usr/bin/env python3
"""Serve the local test bench and proxy model requests without exposing the API key."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from base_model import ask_base_model
from calculator_agent import run_tinker_agent
from evaluate import CASES
from experiment import semantic_score
from verification_experiment import extract_verdict
from settings import DEFAULT_MODEL


SITE_DIR = Path(__file__).with_name("site").resolve()
RESULTS_DIR = SITE_DIR.parent / "experiments" / "nemotron-calculator-60"
VERIFICATION_RESULTS_DIR = SITE_DIR.parent / "experiments" / "nemotron-verification-verdict-20"


async def compare_models(prompt: str) -> dict[str, object]:
    """Run both conditions concurrently and return independent result cards."""
    preset = next((case for case in CASES if case.prompt == prompt.strip()), None)

    async def run(mode: str) -> dict[str, object]:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(240):
                if mode == "base":
                    answer = await ask_base_model(prompt, model_name=DEFAULT_MODEL)
                    result = {"answer": answer, "calculator_invoked": False, "tool_results": []}
                else:
                    agent = await run_tinker_agent(prompt, model_name=DEFAULT_MODEL)
                    result = {
                        "answer": agent.answer,
                        "calculator_invoked": agent.calculator_invoked,
                        "tool_results": agent.tool_results,
                    }
            if preset:
                is_verdict_only = preset.expected in {"VERIFIED", "INCORRECT"}
                if is_verdict_only:
                    result["correct"] = extract_verdict(result["answer"]) == preset.expected
                else:
                    is_verification = preset.expected.startswith("INCORRECT:")
                    expected_case = {
                        "expected": preset.expected,
                        "claim": "preset" if is_verification else None,
                        "value": preset.expected.split(":", 1)[-1].strip(),
                    }
                    result["correct"] = semantic_score(result["answer"], expected_case)
            else:
                result["correct"] = None
        except TimeoutError:
            result = {"error": "This model exceeded the four-minute time limit. Try again."}
        except Exception:
            result = {"error": "The model request failed. Please try again or check the server configuration."}
        result["latency_seconds"] = time.perf_counter() - started
        return result

    base, tool = await asyncio.gather(run("base"), run("tool"))
    expected = preset.expected if preset else None
    return {"base": base, "tool": tool, "expected": expected}


def experiment_results() -> dict[str, object]:
    """Expose aggregate statistics and timing points used by the results dashboard."""
    stats = json.loads((RESULTS_DIR / "statistics.json").read_text())
    timings = []
    for line in (RESULTS_DIR / "scored_responses.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            timings.append({"condition": row["condition"], "seconds": row["latency_seconds"]})
    verification = json.loads((VERIFICATION_RESULTS_DIR / "statistics.json").read_text())
    return {"statistics": stats, "timings": timings, "verification": verification}


class TestBenchHandler(BaseHTTPRequestHandler):
    """Serve static assets and a small JSON endpoint for the test bench."""

    server_version = "TinkerTestBench/0.1"

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/results":
            try:
                self._send_json(experiment_results())
            except (OSError, ValueError, KeyError):
                self._send_json({"error": "Saved experiment results are unavailable."}, status=503)
            return
        if path == "/experiment-report.md":
            try:
                body = (RESULTS_DIR / "report.md").read_bytes()
            except OSError:
                self.send_error(404, "Report not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/verification-report.md":
            try:
                body = (VERIFICATION_RESULTS_DIR / "report.md").read_bytes()
            except OSError:
                self.send_error(404, "Verification report not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/":
            path = "/index.html"

        candidate = (SITE_DIR / path.lstrip("/")).resolve()
        if SITE_DIR not in candidate.parents or not candidate.is_file():
            self.send_error(404, "Not found")
            return

        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/run", "/api/compare"}:
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 20_000:
                raise ValueError("prompt payload is too large")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            prompt = request.get("prompt", "")
            mode = request.get("mode", "tool")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            if len(prompt) > 10_000:
                raise ValueError("prompt is too long")
            if path == "/api/compare":
                self._send_json(asyncio.run(compare_models(prompt)))
                return
            if mode not in {"base", "tool"}:
                raise ValueError("mode must be 'base' or 'tool'")

            if mode == "base":
                answer = asyncio.run(ask_base_model(prompt, model_name=DEFAULT_MODEL))
                result = {
                    "answer": answer,
                    "calculator_invoked": False,
                    "tool_results": [],
                }
            else:
                agent_result = asyncio.run(run_tinker_agent(prompt, model_name=DEFAULT_MODEL))
                result = {
                    "answer": agent_result.answer,
                    "calculator_invoked": agent_result.calculator_invoked,
                    "tool_results": agent_result.tool_results,
                }
            self._send_json(result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # Keep server errors readable without exposing a traceback.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=502)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TestBenchHandler)
    print(f"Test bench running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping test bench.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
