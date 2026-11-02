"""Summarize paired frozen-suite rows, retaining all planned cases in denominators.

Consumes scores from the pinned task scorer; does not score token equality or
claim an official full benchmark. Exact marginal binomial intervals plus a
Bonferroni union bound give a conservative paired difference interval.
"""
import argparse
import hashlib
import json
import math
import pathlib

CONTRACT_FIELDS = ('checkpoint_identity', 'tokenizer_identity', 'template_identity',
                   'precision', 'temperature', 'top_p', 'sampling_seed',
                   'output_budget', 'context_capacity', 'prompt_policy')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def binomial_cdf(k, n, probability):
    if k < 0:
        return 0.0
    if k >= n or probability == 0:
        return 1.0
    if probability == 1:
        return 0.0
    terms = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) + i * math.log(probability)
                      + (n - i) * math.log1p(-probability)) for i in range(k + 1)]
    return min(1.0, math.fsum(terms))


def invert_cdf(k, n, target):
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clopper_pearson(k, n, alpha=.05):
    assert type(k) is int and type(n) is int and n > 0 and 0 <= k <= n
    return [0.0 if k == 0 else invert_cdf(k - 1, n, 1 - alpha / 2),
            1.0 if k == n else invert_cdf(k, n, alpha / 2)]


def paired_interval(gained, lost, n):
    # Each marginal interval has 97.5% coverage. They need not be independent:
    # union bound >=95% for p_gain - p_loss, clipped to its possible range.
    gain = clopper_pearson(gained, n, .025)
    loss = clopper_pearson(lost, n, .025)
    return [max(-1.0, gain[0] - loss[1]), min(1.0, gain[1] - loss[0])]


def normalize_rows(suite, expected):
    rows = {}
    for row in suite.get('rows', []):
        index = row['source_index']
        assert type(index) is int and index in expected, ('unplanned case', index)
        assert index not in rows, ('duplicate case', index)
        assert row['question_SHA256'] == expected[index], ('question differs', index)
        for key in ('correct', 'natural_EOS', 'parse_failure', 'cap_hit'):
            assert type(row[key]) is bool, (index, key)
        assert not (row['natural_EOS'] and row['cap_hit']), ('EOS/cap conflict', index)
        assert not row['correct'] or (row['natural_EOS'] and not row['parse_failure']
                                      and not row['cap_hit'] and not row.get('error'))
        rows[index] = row
    return rows


def arm_summary(rows, expected):
    n = len(expected)
    correct = sum(row['correct'] for row in rows.values())
    return dict(planned=n, observed=len(rows), missing_or_unfinished=n-len(rows),
                correct=correct, accuracy_over_planned=correct/n,
                exact_binomial95=clopper_pearson(correct, n),
                coverage=len(rows)/n,
                natural_EOS=sum(row['natural_EOS'] for row in rows.values()),
                caps=sum(row['cap_hit'] for row in rows.values()),
                parse_failures=sum(row['parse_failure'] for row in rows.values()),
                explicit_errors=sum(bool(row.get('error')) for row in rows.values()))


def summarize(fixture, baseline, candidate, name, fixture_sha, margin=None):
    cases = fixture['cases']
    expected = {}
    for case in cases:
        index = case['source_index']
        assert type(index) is int and index not in expected
        expected[index] = hashlib.sha256(case['question'].encode()).hexdigest()
    assert expected, 'empty suite'
    old_suite, new_suite = baseline['suites'][name], candidate['suites'][name]
    assert old_suite['fixture_SHA256'] == new_suite['fixture_SHA256'] == fixture_sha
    old, new = normalize_rows(old_suite, expected), normalize_rows(new_suite, expected)
    gained, lost, both, neither = 0, 0, 0, 0
    for index in expected:
        left = bool(old.get(index, {}).get('correct', False))
        right = bool(new.get(index, {}).get('correct', False))
        gained += not left and right
        lost += left and not right
        both += left and right
        neither += not left and not right
    n = len(expected)
    interval = paired_interval(gained, lost, n)
    contract_old = baseline.get('comparison_contract', {})
    contract_new = candidate.get('comparison_contract', {})
    absent = [field for field in CONTRACT_FIELDS
              if field not in contract_old or field not in contract_new]
    mismatched = [field for field in CONTRACT_FIELDS
                  if field in contract_old and field in contract_new
                  and contract_old[field] != contract_new[field]]
    complete = len(old) == len(new) == n
    comparable = not absent and not mismatched
    # Do not turn all-zero scores or missing results into a qualification.
    informative = sum(row['correct'] for row in old.values()) > 0
    if margin is not None:
        assert 0 < margin < 1
    result = dict(suite=name, fixture_SHA256=fixture_sha, planned_cases=n,
                  baseline=arm_summary(old, expected), candidate=arm_summary(new, expected),
                  paired=dict(gained=gained, lost=lost, both_correct=both,
                              neither_correct=neither, candidate_minus_baseline=(gained-lost)/n,
                              conservative_exact95=interval),
                  interval_method='97.5% two-sided Clopper-Pearson for each paired gain/loss probability; Bonferroni >=95% difference interval',
                  comparison_contract_missing=absent, comparison_contract_mismatched=mismatched,
                  both_arms_complete=complete, contract_comparable=comparable,
                  baseline_informative=informative, declared_margin=margin,
                  within_declared_margin_on_this_suite=(complete and comparable and informative
                        and interval[0] > -margin) if margin is not None else None,
                  full_benchmark_proven=False, global_no_regression_proven=False,
                  limitations=['Pinned scorer correctness consumed, not independently reparsed from raw answers here',
                               'Custom subset and historical exposure must be disclosed by run manifest',
                               'Checkpoint identity must disclose header/index versus full payload hashing scope',
                               'No client timing/SLO or performance claim from correctness rows',
                               'Missing cases count incorrect in planned denominator; incompleteness blocks margin qualification',
                               'No margin assumed when none was declared before the run'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ('fixture', 'baseline', 'candidate', 'output'):
        parser.add_argument('--'+field, type=pathlib.Path, required=True)
    parser.add_argument('--suite', required=True, choices=['GSM8K', 'MMLU'])
    parser.add_argument('--declared-margin', type=float)
    args = parser.parse_args()
    documents = [json.loads(path.read_bytes()) for path in [args.fixture, args.baseline, args.candidate]]
    result = summarize(*documents, args.suite, sha(args.fixture), args.declared_margin)
    result['source_SHA256'] = {field: sha(getattr(args, field)) for field in ('fixture', 'baseline', 'candidate')}
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['suite', 'planned_cases', 'paired', 'both_arms_complete']}))
