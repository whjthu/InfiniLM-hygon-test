"""Check scoring-denominator and exact paired-interval boundaries, CPU only."""
import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time

G = pathlib.Path(__file__).resolve().parents[3]
module_path = G / 'tools/summarize_paired_quality.py'
spec = importlib.util.spec_from_file_location('paired_quality', module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
started = time.perf_counter()
checks = []

# Independent closed forms for the two extreme Clopper-Pearson cases.
for n in [1, 8, 64, 256]:
    assert math.isclose(m.clopper_pearson(0, n)[1], 1 - .025**(1/n), abs_tol=2e-12)
    assert math.isclose(m.clopper_pearson(n, n)[0], .025**(1/n), abs_tol=2e-12)
checks.append('exact_binomial_extreme_closed_forms')
for n in range(1, 9):
    for k in range(n + 1):
        expected = sum(math.comb(n, i) * .3**i * .7**(n-i) for i in range(k+1))
        assert math.isclose(m.binomial_cdf(k, n, .3), expected, abs_tol=2e-13)
checks.append('binomial_CDF_against_independent_small_polynomial')

# Exhaust all paired multinomial outcomes: check actual interval coverage,
# rather than only checking that the implementation reproduces itself.
n, p_gain, p_loss = 8, .15, .35
coverage = 0.0
for gain in range(n + 1):
    for loss in range(n - gain + 1):
        neutral = n - gain - loss
        probability = (math.factorial(n) / math.factorial(gain) / math.factorial(loss)
                       / math.factorial(neutral) * p_gain**gain * p_loss**loss
                       * (1-p_gain-p_loss)**neutral)
        low, high = m.paired_interval(gain, loss, n)
        if low <= p_gain-p_loss <= high:
            coverage += probability
assert .95 <= coverage <= 1.000000000001
checks.append('exact_multinomial_enumerated_coverage_at_least95')

cases = [dict(source_index=i, question='Frozen problem '+str(i)) for i in range(64)]
fixture = dict(cases=cases)
contract = {field: 'same-pinned-value' for field in m.CONTRACT_FIELDS}
rows = [dict(source_index=c['source_index'], question_SHA256=hashlib.sha256(c['question'].encode()).hexdigest(),
             correct=True, natural_EOS=True, parse_failure=False, cap_hit=False) for c in cases]
arm = dict(comparison_contract=contract, suites=dict(GSM8K=dict(fixture_SHA256='frozenSHA', rows=rows)))
result = m.summarize(fixture, arm, arm, 'GSM8K', 'frozenSHA')
assert result['baseline']['correct'] == 64 and result['paired']['both_correct'] == 64
assert result['within_declared_margin_on_this_suite'] is None
assert not result['global_no_regression_proven']
assert result['paired']['conservative_exact95'][0] < 0
checks.append('identical_scores_still_finite_sample_uncertainty_no_implicit_margin')

partial = copy.deepcopy(arm)
partial['suites']['GSM8K']['rows'].pop()
result = m.summarize(fixture, arm, partial, 'GSM8K', 'frozenSHA', .1)
assert result['candidate']['missing_or_unfinished'] == 1
assert result['candidate']['accuracy_over_planned'] == 63/64
assert result['paired']['lost'] == 1 and not result['within_declared_margin_on_this_suite']
checks.append('missing_case_denominator_retained_blocks_margin')

bad = copy.deepcopy(arm)
bad['suites']['GSM8K']['rows'][0].update(correct=False, natural_EOS=False, cap_hit=True)
bad['suites']['GSM8K']['rows'][1].update(correct=False, parse_failure=True)
result = m.summarize(fixture, arm, bad, 'GSM8K', 'frozenSHA')
assert result['candidate']['caps'] == result['candidate']['parse_failures'] == 1
assert result['paired']['lost'] == 2 and result['candidate']['correct'] == 62
checks.append('caps_and_parse_failures_preserved_in_paired_score')

for name in ['duplicate', 'wrong_question', 'unplanned', 'correct_cap', 'fixture_mismatch']:
    bad = copy.deepcopy(arm)
    rr = bad['suites']['GSM8K']['rows']
    if name == 'duplicate': rr.append(copy.deepcopy(rr[0]))
    elif name == 'wrong_question': rr[0]['question_SHA256'] = 'other'
    elif name == 'unplanned': rr[0]['source_index'] = 999
    elif name == 'correct_cap': rr[0].update(natural_EOS=False, cap_hit=True)
    else: bad['suites']['GSM8K']['fixture_SHA256'] = 'other'
    try:
        m.summarize(fixture, arm, bad, 'GSM8K', 'frozenSHA')
    except AssertionError:
        checks.append('reject_'+name)
    else:
        raise AssertionError(name+' accepted')
for name in ['missing_contract', 'changed_budget']:
    bad = copy.deepcopy(arm)
    if name == 'missing_contract': bad.pop('comparison_contract')
    else: bad['comparison_contract']['output_budget'] = 'different'
    result = m.summarize(fixture, arm, bad, 'GSM8K', 'frozenSHA', .1)
    assert not result['contract_comparable'] and not result['within_declared_margin_on_this_suite']
    checks.append('unqualified_'+name)
zero = copy.deepcopy(arm)
for row in zero['suites']['GSM8K']['rows']: row['correct'] = False
result = m.summarize(fixture, zero, zero, 'GSM8K', 'frozenSHA', .1)
assert not result['baseline_informative'] and not result['within_declared_margin_on_this_suite']
checks.append('all_zero_baseline_not_qualification')
historical_format = []
if len(sys.argv) == 2:
    # Same historical file supplied twice exercises the real serialized schema;
    # this is explicitly not a new OFF/ON comparison or Model experiment.
    archive_path = pathlib.Path(sys.argv[1])
    archive = json.loads(archive_path.read_bytes())
    existing = archive['Deep']['phases']['quality']['result']
    for name, filename, known_correct in [('GSM8K', 'gsm8k-fixed32.json', 32),
                                           ('MMLU', 'mmlu-fixed32.json', 27)]:
        fixture_path = G / 'code/bw1000/public-native-bridge/Deep-semantic68-owner-r320/benchmark' / filename
        fixture = json.loads(fixture_path.read_bytes())
        receipt = m.summarize(fixture, existing, existing, name, m.sha(fixture_path))
        assert receipt['baseline']['correct'] == known_correct
        assert receipt['baseline']['natural_EOS'] == 32
        assert not receipt['contract_comparable']
        assert receipt['within_declared_margin_on_this_suite'] is None
        historical_format.append(dict(suite=name, observed=32, existing_correct=known_correct,
                                      self_comparison_only=True, new_Model_result=False))
    checks.append('actual_Deep1221_two_pilot_row_formats_and_question_hashes')
print(json.dumps(dict(completed=True, CPU_only=True, GPU_initialized=False,
                      Model_loaded=False, actual_local_checks=checks,
                      exact_enumerated_coverage=coverage, seconds=time.perf_counter()-started,
                      source_SHA256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                      historical_format_only=historical_format,
                      scope='Synthetic statistical and consumer boundary checks, no actual benchmark or quality qualification'), indent=2))
