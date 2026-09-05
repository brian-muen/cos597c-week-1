# Calculator access experiment

Nemotron 3 Nano: 60 fresh questions, 120 paired condition runs. Seed 5972026.
Reasoning disabled, temperature 0, 4,096 output tokens per model call; up to four calls in tool condition.

| Question group | Base correct | Calculator correct |
|---|---:|---:|
| Overall | 21/60 | 60/60 |
| Straightforward | 18/20 | 20/20 |
| Large integers | 0/20 | 20/20 |
| Verification | 3/20 | 20/20 |

Paired accuracy improvement: 65.0%; 95% stratified paired bootstrap interval: 58.3% to 71.7%.
Exact two-sided McNemar p = 3.638e-12. Tool-only correct pairs: 39; base-only: 0.
Calculator invoked on 60/60 tool runs; final answer correct and a tool value matching the reference: 60/60.
Truncations (base/tool): 4/0. Errors (base/tool): 0/0.
Median end-to-end seconds (base/tool): 0.77/1.52.

## Design and interpretation

This is an exploratory pilot on a fixed synthetic distribution, not a general math benchmark. Twenty questions per group are enough to reveal large differences but leave wide category intervals. Ten verification claims are true and ten false. All questions and exact integer references were saved before any generation; Python integer references were checked independently with 100-digit Decimal arithmetic. No model failures were removed or replaced, and no retries or prompt tuning were performed after inspecting results.

The initial string scorer was found to reject correct answers accompanied by explanations or LaTeX boxes. After inspecting outputs, we added reference-blind final-answer extraction and applied it uniformly to both conditions. Original responses and initial scores remain in responses.jsonl; corrected scores and extracted answers are in scored_responses.jsonl. This scoring correction was made after collection began and is not preregistered.

The conditions use the existing separate system prompts: the tool prompt explicitly instructs calculator use. Thus the estimated effect is calculator access plus that instruction. Tool runs may use more model calls. Latency includes service/network time under concurrency and is descriptive, not an isolated speed benchmark.

Accuracy accepts thousands separators, whitespace, simple Markdown, answer labels and exact decimal/scientific representations. It requires the right verification verdict and exact value, with no numerical tolerance. Truncated responses and execution/API errors count as unsuccessful end-to-end attempts; their counts are reported separately. Tool-result consistency is an observable agreement check, not proof that the model causally used that value. A correct multi-step tool solution without a single reference-valued tool output can be undercounted by this metric.

Accuracy error bars are 95% Wilson intervals. Difference intervals resample paired questions within categories (10,000 bootstrap draws). McNemar tests paired binary outcomes; category comparisons are descriptive and not used for separate significance claims (category p-values in the JSON are unadjusted). Bootstrap intervals can collapse in categories with identical observed outcomes; they do not imply population certainty. Template dependence and a single generation per question limit generalization. Reasoning was disabled in the API request, but some responses still contained visible working.

Files: cases.json, config.json, responses.jsonl, scored_responses.jsonl, statistics.json, results.png and results.svg.
