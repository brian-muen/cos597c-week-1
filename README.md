# Tinker + calculator scaffold

This is a minimal tool-calling loop for a Tinker sampling client:

1. Declare a `calculate` tool in the model's prompt.
2. Sample the model.
3. Parse its tool call.
4. Execute the calculator locally.
5. Append the tool result and sample again for the final answer.

The calculator uses Python's AST parser and an allowlist of arithmetic nodes; it does not call `eval`.

## Setup

Requires Python 3.11+ and a Tinker API key.

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Open `.env` and paste the key after `TINKER_API_KEY=`. The file is already gitignored. Shell
environment variables take precedence if you prefer `export TINKER_API_KEY="..."`.

The default is Tinker's `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` model. The scripts use
Tinker's hosted OpenAI-compatible chat endpoint, which applies the model's chat template server-side.
Nemotron 3 Nano only supports binary reasoning control, so both evaluation conditions disable
extended reasoning (the closest supported low-reasoning setting) and use the same 4,096-token limit.

Run the calculator-enabled condition:

```bash
python calculator_agent.py "What is 12.5 * (8 - 3)?"
```

## Base-model ablation

`base_model.py` uses the same model, renderer, sampling parameters, and prompt format, but provides no calculator tool:

```bash
python base_model.py "What is 12.5 * (8 - 3)?"
```

This makes it a direct comparison against `calculator_agent.py`.

## Three-case comparison

Run the easy, mechanism-required, and adversarial-anchoring cases under both conditions:

```bash
python evaluate.py
```

The command prints a comparison table and writes `eval_results.json`. Each record includes
exact-answer correctness (allowing optional numeric thousands separators), whether the calculator
was invoked, the returned tool values, and whether the final answer correctly incorporated a tool
result. Use `--runs 3` for repeated trials or
`--condition base` / `--condition tool` to run one side only.

## Local test bench

Start the interactive website; the API key stays in the local Python server:

```bash
python web_server.py
```

Then open <http://127.0.0.1:8000>. It includes the three evaluation prompts, base/tool toggles,
custom prompts, and a result trace showing whether `calculate` was invoked. Stop the server with
`Ctrl-C`.

For another Tinker model, pass its model ID or set `TINKER_MODEL`:

```bash
python calculator_agent.py \
  --model "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" \
  "What is 2 ** 10?"
```

Run the local, API-free tests:

```bash
pytest
```

Tinker renders and parses the structured tool call; the application owns calculator execution and
the follow-up model/tool loop.

## 60-question experiment

The completed paired pilot and graphs are in `experiments/nemotron-calculator-60/report.md`.
It uses 20 straightforward questions, 20 large-integer questions, and 20 verification claims
(10 true, 10 false). Each question is tested once per condition. The model uses the same settings
as the interactive demo. Results include final-answer accuracy, tool use, latency, truncation,
95% intervals and an exact paired McNemar test.

To generate a fresh run of the same seeded suite, choose a new output directory:

```bash
python -m pip install -e '.[experiment,dev]'
python experiment.py run --output experiments/my-run
```

To rebuild statistics and PNG/SVG graphs from saved responses without model calls:

```bash
python experiment.py analyze --output experiments/nemotron-calculator-60
```

The experiment's final-answer extractor accepts explanations and math presentation around the
final value. It never uses a reference number merely appearing somewhere in the response as
evidence of correctness. Original responses and initial scores are preserved separately from
the corrected scores; the report documents the scoring correction and limits of the pilot.
