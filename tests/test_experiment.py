from decimal import Decimal

from calculator_agent import calculate
from experiment import make_cases, score, summarize, semantic_score, extract_final


def test_frozen_suite_has_balanced_claims_and_exact_references():
    cases = make_cases()
    assert len(cases) == 60
    assert sum(c['expected'] == 'VERIFIED' for c in cases) == 10
    assert sum(c['expected'].startswith('INCORRECT') for c in cases) == 10
    for case in cases:
        assert calculate(case['expression']) == int(case['value'])
        assert score(case['expected'], case)


def test_scoring_ignores_presentation_but_rejects_wrong_values_and_verdicts():
    case = make_cases()[20]
    value = int(case['value'])
    assert score(f'**Answer: {value:,}.**', case)
    assert score(f'{value}.000', case)
    assert score(f'{Decimal(value):E}', case)
    assert not score(str(value+1), case)
    assert not score(f'{value} or {value+1}', case)
    claim = next(c for c in make_cases() if c['expected'].startswith('INCORRECT'))
    assert not score('VERIFIED', claim)
    assert not score(claim['value'], claim)
    assert score(f'incorrect: {int(claim["value"]):,}', claim)


def test_paired_statistic_uses_discordant_pairs():
    records = []
    for i in range(10):
        for condition in ['base', 'tool']:
            records.append(dict(id=str(i),category='test',condition=condition,
                                correct=condition=='tool',truncated=False,error=None,
                                latency_seconds=1,calculator_invoked=condition=='tool',
                                final_consistent_with_tool=condition=='tool'))
    stats = summarize(records)
    assert stats['paired']['gain'] == 1
    assert stats['paired']['mcnemar_exact_two_sided_p'] == 2/1024
    assert stats['paired']['tool_only_correct'] == 10


def test_final_answer_extraction_does_not_credit_intermediate_numbers():
    case = make_cases()[0]
    n = case['value']
    assert semantic_score(f'Here is an explanation.\nThe exact result is **{n}**.', case)
    assert semantic_score(r'\\boxed{' + n + '}', case)
    assert not semantic_score(f'The intermediate value is {n}.\nFinal answer: 123', case)
    assert not semantic_score(f'{n} or 123', case)
    assert extract_final('incorrect: 9,999,399,973', verification=True) == 'INCORRECT: 9999399973'
