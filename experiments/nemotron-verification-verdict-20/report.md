# Verdict-only verification experiment

Nemotron 3 Nano: 20 fresh verification claims, 40 paired condition runs. Seed 5972126.
Each prompt requested exactly one verdict—VERIFIED or INCORRECT. No exact product was requested or required.
Reasoning disabled, temperature 0, 4,096 output tokens per call; up to four calls in tool condition.

| Claim type | Base correct | Calculator correct |
|---|---:|---:|
| Overall | 10/20 | 20/20 |
| True claims | 0/10 | 10/10 |
| False claims | 10/10 | 10/10 |

Accuracy improvement: 50.0%; 95% stratified paired-bootstrap interval: 30.0% to 70.0%.
Exact two-sided McNemar p = 0.001953. Tool-only correct pairs: 10; base-only: 0.
Calculator invoked on 20/20 tool runs; a matching reference product appeared in tool output on 20/20 correct runs.
Truncations (base/tool): 0/0. Errors (base/tool): 0/0.

## Interpretation

This focused pilot tests claim verification rather than transcription of an exact product. A response is correct when its final verdict matches the claim’s truth value; explanations and formatting are ignored.
The calculator condition also receives an explicit tool-use instruction and can make up to four model calls, so the comparison measures calculator access plus that instruction. Latency includes service/network time under concurrency.
A single generation per claim and fixed synthetic templates limit generalization. The 95% intervals are descriptive for this pilot, not a guarantee for new claims.

Files: cases.json, config.json, responses.jsonl, scored_responses.jsonl, statistics.json.
