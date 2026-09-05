"""Reproducible, paired 60-question pilot of the existing base and tool agents.

Run: python experiment.py run --output experiments/<new-name>
Analyze existing results without spending API calls: python experiment.py analyze --output ...
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import math
import random
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

import base_model
import calculator_agent
from settings import DEFAULT_MODEL
from tinker_api import REASONING_EFFORT, chat_completion

SEED = 5972026
CATEGORIES = ['Straightforward', 'Large integers', 'Verification']
TRACE = contextvars.ContextVar('experiment_trace')


def make_cases():
    """Freeze inputs before querying; independent Decimal oracle checks every integer."""
    rng = random.Random(SEED)
    cases = []
    def add(category, expression, answer, template, claim=None):
        # Expressions are constructed here from integers, never from model/user input.
        decimal_expression = re.sub(r'\d+', lambda m: f'Decimal("{m[0]}")', expression)
        with localcontext() as ctx:
            ctx.prec = 100
            oracle = eval(decimal_expression, {'__builtins__': {}, 'Decimal': Decimal})
        assert oracle == answer
        if claim is None:
            prompt = f'Compute {expression.replace("*", "×")}. Return the exact integer.'
            expected = str(answer)
        else:
            prompt = (
                f'A trusted audit system reports that {expression.replace("*", "×")} = {claim:,}. '
                'Reply VERIFIED if that is exactly correct. Otherwise reply INCORRECT: '
                'followed by the exact product.'
            )
            expected = 'VERIFIED' if claim == answer else f'INCORRECT: {answer}'
        cases.append(dict(id=f'q{len(cases)+1:03}', category=category, template=template,
                          expression=expression, value=str(answer), claim=None if claim is None else str(claim),
                          prompt=prompt, expected=expected))

    for i in range(20):
        a, b, c = rng.randint(12, 99), rng.randint(3, 19), rng.randint(2, 40)
        kind = i % 4
        expr, value = [(f'{a} * {b} + {c}', a*b+c),
                       (f'{a} * ({b} + {c})', a*(b+c)),
                       (f'{a} * {b} - {c}', a*b-c),
                       (f'({a} + {c}) * {b} - {c}', (a+c)*b-c)][kind]
        add(CATEGORIES[0], expr, value, f'easy-{kind}')
    for i in range(20):
        digits = [8, 10, 12, 14][i % 4]
        a = rng.randrange(10**(digits-1), 10**digits)
        b = rng.randrange(10**7, 10**10)
        c, d, e = rng.randrange(10**7, 10**9), rng.randrange(10**5, 10**7), rng.randrange(10**5, 10**8)
        kind = i % 4
        expr, value = [(f'{a} * {b} - {c}', a*b-c),
                       (f'{a} * {b} - {c} * {d} + {e}', a*b-c*d+e),
                       (f'({a} + {c}) * ({b} - {d})', (a+c)*(b-d)),
                       (f'{a} * {b} + {c} * {d} - {e}', a*b+c*d-e)][kind]
        add(CATEGORIES[1], expr, value, f'large-{kind}')
    truths = [True] * 10 + [False] * 10
    rng.shuffle(truths)
    for i, truth in enumerate(truths):
        a, b = rng.randrange(10001, 999999), rng.randrange(10001, 999999)
        value = a*b
        claim = value if truth else value + rng.choice([-1, 1]) * rng.choice([1, 10, 1000, 1000000])
        add(CATEGORIES[2], f'{a} * {b}', value, 'true-claim' if truth else 'false-claim', claim)
    assert len({c['prompt'] for c in cases}) == 60
    return cases


def score(answer, case):
    """Grade only final answer/verdict, allowing presentation differences, never rounding."""
    text = answer.strip().replace('−', '-').replace('\u00a0', ' ')
    text = re.sub(r'```(?:text)?\s*|```|\*\*|`', '', text, flags=re.I).strip()
    text = re.sub(r'(?<=\d)[, _](?=\d)', '', text)
    text = re.sub(r'^(?:the\s+)?(?:final\s+)?(?:answer|result)(?:\s+is)?\s*[:=]?\s*', '', text, flags=re.I)
    text = text.strip().rstrip('.').strip()
    if case['expected'] == 'VERIFIED':
        return text.upper() == 'VERIFIED'
    if case['claim'] is not None:
        match = re.fullmatch(r'INCORRECT\s*:?\s*(.+)', text, flags=re.I)
        if not match:
            return False
        text = match[1]
    # Decimal avoids silently rounding the 20+ digit products via binary float.
    if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text):
        return False
    return Decimal(text) == Decimal(case['value'])


def extract_final(answer, verification=False):
    """Extract a final numeric answer without looking at the reference value.

    Explanatory intermediate arithmetic is ignored. A trailing answer, equation RHS,
    or boxed number is accepted. Ambiguous alternatives are not accepted.
    """
    text = answer.replace('−', '-').replace('\u00a0', ' ').replace('\u202f', ' ')
    text = re.sub(r'(?<=\d)[, _](?=\d)', '', text)
    text = text.replace('**', '').replace('`', '')
    verdict = None
    if verification:
        verdicts = list(re.finditer(r'\b(VERIFIED|INCORRECT)\b', text, re.I))
        if not verdicts:
            return None
        verdict = verdicts[-1][1].upper()
        text = text[verdicts[-1].end():]
        if verdict == 'VERIFIED':
            return 'VERIFIED' if not text.strip(' \n\t:.!') else None
    # Flatten LaTeX presentation, including doubled backslashes emitted by the model.
    text = re.sub(r'\\+boxed\{([^{}]+)\}', r'\1', text)
    text = re.sub(r'\\+[\[\]()!]', '', text).replace('$', '')
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if not lines:
        return None
    last = lines[-1]
    if re.search(r'\bor\b|\?', last, re.I):
        return None
    match = re.search(r'(?:^|[^\w.])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[.!]?\s*$', last)
    if not match:
        return None
    return (verdict + ': ' if verdict else '') + match[1]


def semantic_score(answer, case):
    final = extract_final(answer, verification=case['claim'] is not None)
    return final is not None and score(final, case)


async def observed_chat(**kwargs):
    message, finish = await chat_completion(**kwargs)
    TRACE.get()['samples'].append(dict(finish_reason=finish,
                                     answer=message.get('content') or '',
                                     tool_calls=message.get('tool_calls') or []))
    return message, finish


ORIGINAL_TOOL = calculator_agent.run_calculator_tool


def observed_tool(call):
    content = ORIGINAL_TOOL(call)
    TRACE.get()['tools'].append(dict(call=call, output=json.loads(content)))
    return content


async def run_experiment(output, concurrency):
    # New directories only: previous evidence is never overwritten or cherry-picked.
    output.mkdir(parents=True, exist_ok=False)
    cases = make_cases()
    config = dict(seed=SEED, model=DEFAULT_MODEL, reasoning_effort=REASONING_EFFORT,
                  temperature=0, max_tokens_per_call=4096, max_tool_rounds=4,
                  concurrency=concurrency, timeout_seconds_per_condition=240,
                  base_system_prompt=base_model.SYSTEM_PROMPT,
                  tool_system_prompt=calculator_agent.SYSTEM_PROMPT,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  design='60 fresh questions; one run per condition; 20/category; 10 true and 10 false verification claims')
    (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    jobs = [(case, condition) for case in cases for condition in ['base', 'tool']]
    random.Random(SEED + 1).shuffle(jobs)
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    async def one(case, condition):
        nonlocal completed
        async with semaphore:
            trace = dict(samples=[], tools=[])
            token = TRACE.set(trace)
            start = time.perf_counter()
            error = None
            answer = ''
            try:
                async with asyncio.timeout(240):
                    if condition == 'base':
                        answer = await base_model.ask_base_model(case['prompt'], model_name=DEFAULT_MODEL)
                    else:
                        result = await calculator_agent.run_tinker_agent(case['prompt'], model_name=DEFAULT_MODEL)
                        answer = result.answer
            except Exception as exc:
                # Exception bodies may contain service diagnostics; do not copy secrets to artifacts.
                error = type(exc).__name__
                if trace['samples']:
                    answer = trace['samples'][-1]['answer']
            finally:
                TRACE.reset(token)
            truncated = any(s['finish_reason'] == 'length' for s in trace['samples'])
            correct = error is None and not truncated and score(answer, case)
            invoked = any(t['call']['function']['name'] == 'calculate' for t in trace['tools'])
            matches = any('result' in t['output'] and Decimal(str(t['output']['result'])) == Decimal(case['value'])
                          for t in trace['tools'])
            record = dict(id=case['id'], category=case['category'], condition=condition,
                          answer=answer, correct=correct, error=error, truncated=truncated,
                          latency_seconds=time.perf_counter()-start, calculator_invoked=invoked,
                          reference_value_in_tool_results=matches,
                          final_consistent_with_tool=correct and matches,
                          **trace)
            with (output / 'responses.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            completed += 1
            print(f'{completed}/120 {case["id"]} {condition}: correct={correct} tool={invoked} truncated={truncated} error={error}', flush=True)
    with patch.object(base_model, 'chat_completion', observed_chat), \
         patch.object(calculator_agent, 'chat_completion', observed_chat), \
         patch.object(calculator_agent, 'run_calculator_tool', observed_tool):
        await asyncio.gather(*(one(*job) for job in jobs))


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half), min(1.,center+half)]


def summarize(records):
    result = {}
    for condition in ['base', 'tool']:
        rows = [r for r in records if r['condition'] == condition]
        n = len(rows)
        k = sum(r['correct'] for r in rows)
        result[condition] = dict(n=n, correct=k, accuracy=k/n, wilson_95=wilson(k,n),
            truncated=sum(r['truncated'] for r in rows), errors=sum(r['error'] is not None for r in rows),
            median_seconds=statistics.median(r['latency_seconds'] for r in rows),
            calculator_invoked=sum(r['calculator_invoked'] for r in rows),
            final_consistent_with_tool=sum(r['final_consistent_with_tool'] for r in rows))
    pairs = {}
    for row in records:
        pairs.setdefault(row['id'], {})[row['condition']] = int(row['correct'])
    wins = sum(p['tool'] > p['base'] for p in pairs.values())
    losses = sum(p['tool'] < p['base'] for p in pairs.values())
    discordant = wins+losses
    pvalue = min(1., 2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    # Resample pairs within categories, maintaining the fixed 20/20/20 mix.
    groups = {}
    for row in records:
        if row['condition'] == 'base':
            p = pairs[row['id']]
            groups.setdefault(row['category'], []).append(p['tool']-p['base'])
    rng = random.Random(SEED+2)
    samples = sorted(sum(sum(rng.choices(g, k=len(g))) for g in groups.values())/len(pairs) for _ in range(10000))
    result['paired'] = dict(tool_only_correct=wins, base_only_correct=losses,
        both_correct=sum(p['tool'] and p['base'] for p in pairs.values()),
        both_wrong=sum(not p['tool'] and not p['base'] for p in pairs.values()),
        gain=(wins-losses)/len(pairs), stratified_paired_bootstrap_95=[samples[249],samples[9749]],
        mcnemar_exact_two_sided_p=pvalue)
    return result


def analyze(output):
    records = [json.loads(line) for line in (output / 'responses.jsonl').read_text().splitlines()]
    cases = json.loads((output / 'cases.json').read_text())
    assert len(records) == 120 and len({(r['id'],r['condition']) for r in records}) == 120
    assert {r['id'] for r in records} == {c['id'] for c in cases}
    case_by_id = {c['id']:c for c in cases}
    for row in records:
        case = case_by_id[row['id']]
        row['initial_correct'] = row['correct']
        row['extracted_final'] = extract_final(row['answer'], verification=case['claim'] is not None)
        row['correct'] = row['error'] is None and not row['truncated'] and semantic_score(row['answer'], case)
        row['final_consistent_with_tool'] = row['correct'] and row['reference_value_in_tool_results']
    (output/'scored_responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    overall = summarize(records)
    summary = dict(overall=overall, categories={cat:summarize([r for r in records if r['category']==cat]) for cat in CATEGORIES})
    (output/'statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(output, summary, records)
    base, tool, paired = overall['base'], overall['tool'], overall['paired']
    lines = ['# Calculator access experiment', '',
        f'Nemotron 3 Nano: 60 fresh questions, 120 paired condition runs. Seed {SEED}.',
        'Reasoning disabled, temperature 0, 4,096 output tokens per model call; up to four calls in tool condition.', '',
        '| Question group | Base correct | Calculator correct |',
        '|---|---:|---:|']
    for cat, stats in [('Overall',overall), *summary['categories'].items()]:
        lines.append(f'| {cat} | {stats["base"]["correct"]}/{stats["base"]["n"]} | {stats["tool"]["correct"]}/{stats["tool"]["n"]} |')
    lines += ['', f'Paired accuracy improvement: {paired["gain"]:.1%}; 95% stratified paired bootstrap interval: '
              f'{paired["stratified_paired_bootstrap_95"][0]:.1%} to {paired["stratified_paired_bootstrap_95"][1]:.1%}.',
              f'Exact two-sided McNemar p = {paired["mcnemar_exact_two_sided_p"]:.4g}. '
              f'Tool-only correct pairs: {paired["tool_only_correct"]}; base-only: {paired["base_only_correct"]}.',
              f'Calculator invoked on {tool["calculator_invoked"]}/60 tool runs; final answer correct and a tool value matching the reference: '
              f'{tool["final_consistent_with_tool"]}/60.',
              f'Truncations (base/tool): {base["truncated"]}/{tool["truncated"]}. '
              f'Errors (base/tool): {base["errors"]}/{tool["errors"]}.',
              f'Median end-to-end seconds (base/tool): {base["median_seconds"]:.2f}/{tool["median_seconds"]:.2f}.', '',
              '## Design and interpretation', '',
              'This is an exploratory pilot on a fixed synthetic distribution, not a general math benchmark. '
              'Twenty questions per group are enough to reveal large differences but leave wide category intervals. '
              'Ten verification claims are true and ten false. All questions and exact integer references were saved '
              'before any generation; Python integer references were checked independently with 100-digit Decimal arithmetic. '
              'No model failures were removed or replaced, and no retries or prompt tuning were performed after inspecting results.', '',
              'The initial string scorer was found to reject correct answers accompanied by explanations or LaTeX boxes. '
              'After inspecting outputs, we added reference-blind final-answer extraction and applied it uniformly to both conditions. '
              'Original responses and initial scores remain in responses.jsonl; corrected scores and extracted answers are in '
              'scored_responses.jsonl. This scoring correction was made after collection began and is not preregistered.', '',
              'The conditions use the existing separate system prompts: the tool prompt explicitly instructs calculator use. '
              'Thus the estimated effect is calculator access plus that instruction. Tool runs may use more model calls. '
              'Latency includes service/network time under concurrency and is descriptive, not an isolated speed benchmark.', '',
              'Accuracy accepts thousands separators, whitespace, simple Markdown, answer labels and exact decimal/scientific '
              'representations. It requires the right verification verdict and exact value, with no numerical tolerance. '
              'Truncated responses and execution/API errors count as unsuccessful end-to-end attempts; their counts are reported separately. '
              'Tool-result consistency is an observable agreement check, not proof that the model causally used that value. '
              'A correct multi-step tool solution without a single reference-valued tool output can be undercounted by this metric.', '',
              'Accuracy error bars are 95% Wilson intervals. Difference intervals resample paired questions within categories '
              '(10,000 bootstrap draws). McNemar tests paired binary outcomes; category comparisons are descriptive and not '
              'used for separate significance claims (category p-values in the JSON are unadjusted). Bootstrap intervals '
              'can collapse in categories with identical observed outcomes; they do not imply population certainty. '
              'Template dependence and a single generation per question limit generalization. Reasoning was disabled in '
              'the API request, but some responses still contained visible working.', '',
              'Files: cases.json, config.json, responses.jsonl, scored_responses.jsonl, statistics.json, results.png and results.svg.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def plot(output, summary, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.facecolor':'white'})
    fig, grid = plt.subplots(2, 2, figsize=(14,10))
    axes = grid.flatten()
    colors = {'base':'#738399', 'tool':'#087F78'}
    groups = [summary['overall'], *summary['categories'].values()]
    x = np.arange(4)
    for j, condition in enumerate(['base','tool']):
        vals = [g[condition]['accuracy']*100 for g in groups]
        err = np.array([[v-g[condition]['wilson_95'][0]*100, g[condition]['wilson_95'][1]*100-v] for v,g in zip(vals,groups)]).T
        err = np.maximum(err, 0)  # Remove floating-point noise at 0% and 100%.
        bars = axes[0].bar(x+(j-.5)*.36, vals, .34, color=colors[condition],
                           label='Base' if j==0 else 'Calculator', yerr=err, capsize=3)
        for bar,g in zip(bars,groups):
            s=g[condition]
            axes[0].text(bar.get_x()+bar.get_width()/2, max(4,bar.get_height()/2),f'{s["correct"]}/{s["n"]}',
                         ha='center', va='center', color='white' if bar.get_height()>10 else '#243347',fontsize=10)
    axes[0].set_xticks(x,['Overall\nn=60','Simple\nn=20','Large integers\nn=20','Verification\nn=20'])
    axes[0].set_ylim(0,112); axes[0].set_ylabel('Correct answers (%)'); axes[0].set_title('Accuracy · 95% confidence intervals',loc='left',pad=18)
    axes[0].legend(frameon=False, loc='upper left',bbox_to_anchor=(0,-.19), ncol=2)
    pairs=summary['overall']['paired']
    counts=[pairs['both_correct'],pairs['tool_only_correct'],pairs['base_only_correct'],pairs['both_wrong']]
    axes[1].barh(['Both correct','Only calculator','Only base','Neither correct'],counts,color=['#2E5664','#087F78','#738399','#BD7359'])
    for i,n in enumerate(counts): axes[1].text(n+.5,i,str(n),va='center')
    axes[1].invert_yaxis(); axes[1].set_xlim(0, max(counts)+6); axes[1].set_xlabel('Paired questions'); axes[1].set_title('Where the answers differ',loc='left',pad=18)
    for i,cond in enumerate(['base','tool']):
        vals=[r['latency_seconds'] for r in records if r['condition']==cond]
        axes[2].scatter([i+random.Random(SEED+k).uniform(-.17,.17) for k in range(len(vals))], vals, color=colors[cond],alpha=.5,s=18)
        med=statistics.median(vals)
        axes[2].plot([i-.22,i+.22],[med,med],color='#162C38',linewidth=2)
        axes[2].text(i, med+2, f'{med:.1f}s median',ha='center',fontsize=10)
    axes[2].set_xticks([0,1],['Base','Calculator']); axes[2].set_ylabel('Seconds per question'); axes[2].set_title('Observed response time',loc='left',pad=18)
    toolrows = [r for r in records if r['condition']=='tool']
    counts = [sum(r['calculator_invoked'] for r in toolrows),
              sum(r['reference_value_in_tool_results'] for r in toolrows),
              sum(r['final_consistent_with_tool'] for r in toolrows)]
    labels = ['Calculator invoked','Reference value returned','Correct final + matching tool value']
    axes[3].barh(labels, counts, color=['#75BDB1','#289D90','#087F78'])
    for i,n in enumerate(counts): axes[3].text(n+.5,i,f'{n}/60',va='center')
    axes[3].set_xlim(0,70); axes[3].invert_yaxis(); axes[3].set_xlabel('Questions in calculator condition')
    axes[3].set_title('Calculator use and answer consistency',loc='left',pad=18)
    for ax in axes: ax.set_axisbelow(True); ax.grid(axis='y',alpha=.13)
    fig.suptitle('Does calculator access improve Nemotron’s arithmetic?',x=.055,ha='left',fontsize=19,weight='bold')
    fig.text(.055,.93,'60 fresh paired questions · reasoning disabled · one run per condition · fixed prompts and seed',color='#536372')
    fig.text(.055,.035,'Synthetic pilot, not a general benchmark. Calculator condition includes an explicit tool-use instruction. Latency includes API/network time.',fontsize=10,color='#536372')
    fig.subplots_adjust(left=.07,right=.96,top=.86,bottom=.11,wspace=.85,hspace=.7)
    fig.savefig(output/'results.png',dpi=180)
    fig.savefig(output/'results.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=6)
    args=parser.parse_args()
    if args.concurrency < 1: parser.error('concurrency must be positive')
    if args.action=='run': asyncio.run(run_experiment(args.output,args.concurrency))
    analyze(args.output)
