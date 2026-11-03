"""Small synthetic reader-boundary checks; no claimed real r445 observations."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pd445_original_reader', D / 'analyze_original.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rejected(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Invalid evidence accepted')


def main():
    assert m.answer('#### A', 'S')[0]
    assert m.answer('#### B\n#### A', 'S')[0]
    assert not m.answer('#### A\n#### B', 'S')[0]
    assert not m.answer('Choice A without required marker', 'S')[0]
    gold = json.dumps(m.GOLD)
    assert m.answer(gold, 'L')[0] and m.answer('```json\n' + gold + '\n```', 'L')[0]
    wrong = copy.deepcopy(m.GOLD)
    wrong['total'] = str(wrong['total'])
    assert not m.answer(json.dumps(wrong), 'L')[0]
    assert not m.answer('prose\n```json\n' + gold + '\n```', 'L')[0]
    assert not m.answer('{"total":1,"total":2}', 'L')[0]
    req = dict(output_budget=1024, fixture='S', mode='PD')
    source = dict(generated_token_ids=[99, 100, 1], finish_reason='eos', natural_EOS=True,
                  quality=dict(content='#### A', task_correct=True))
    row = dict(actual_admission_monotonic=10., scheduled_arrival_monotonic=9.,
               commands=[dict(operation='prefill', client_send_monotonic=11.)],
               client_token_ID_timeline=[dict(seq=i, token_id=token, received_monotonic=t, delivered_monotonic=t)
                                         for i, (token, t) in enumerate([(99, 12.), (100, 16.), (1, 17.)])],
               client=dict(completed=True, error=None, generated_received=3, visible_tokens=2, admitted_monotonic=10.,
                           client_token_ID_TTFT_s=2., client_token_ID_decode_tok_s=.25,
                           client_token_ID_ITL_p50_s=4., client_token_ID_ITL_p95_s=4.,
                           client_token_ID_ITL_p99_s=4., client_token_ID_ITL_max_s=4., token_ID_completion_wall_s=8.))
    result, ids = m.token_metrics(row, source, req)
    assert result['client_admission_to_dispatch_queue_s'] == 1. and result['client_token_ID_decode_tok_s'] == .25
    assert result['natural_EOS'] and result['task_correct'] and result['visible'] == 2 and ids == [99, 100, 1]
    bad = copy.deepcopy(source)
    bad['generated_token_ids'] = [99, 1, 100]
    rejected(lambda: m.token_metrics(row, bad, req))
    bad = copy.deepcopy(source)
    bad['quality']['task_correct'] = False
    rejected(lambda: m.token_metrics(row, bad, req))
    bad = copy.deepcopy(row)
    bad['client_token_ID_timeline'][1]['delivered_monotonic'] = 11.
    rejected(lambda: m.token_metrics(bad, source, req))
    bad = copy.deepcopy(row)
    bad['client']['client_token_ID_decode_tok_s'] = 1.
    rejected(lambda: m.token_metrics(bad, source, req))
    assert m.elapsed_seconds('1-02:03:04') == 93784
    assert not any(n in sys.modules for n in ('torch', 'numpy', 'infinilm'))
    print(json.dumps(dict(passed=True, scope='Synthetic offline reader/parser/clock rejection only; no actual r445 data',
                          actual_request_or_device_qualified=False, GPU=0, jobs=0, Native_dlopen=False,
                          independent_boundary_cases=15)))


if __name__ == '__main__':
    main()
