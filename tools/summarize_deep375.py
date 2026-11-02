"""Read-only paired timings from the root's sealed 264-wire/gold audit.

The original audit reparses each original wire and Native timeline on the
server. This consumer verifies its sealed bytes and summarizes those rows; it
does not rerun the model or reinterpret backend ticks as client streaming.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

AUDIT_SHA = '09671c90e2782e808741be5f7a42d12d29bbf34a6855b4554b4e93d2d3dd010e'
HANDLES = ['910374.1304', '910374.1305', '910374.1301', '910371.701']


def summarize(report):
    raw = json.dumps(report, indent=2).encode()
    assert hashlib.sha256(raw).hexdigest() == AUDIT_SHA, 'Sealed original audit bytes differ'
    assert report['completed'] and report['passed'] and report['all32_normal_lifecycle']
    assert [x['actual_GPU'] for x in report['replicas']] == HANDLES
    assert all(x['rows'] == 66 and x['normal_lifecycle'] for x in report['replicas'])
    rows = report['request_metadata']
    assert len(rows) == 264 and sum(not x['warmup'] for x in rows) == 256
    assert all(x['natural_EOS'] and not x['cap_hit'] and not x['parse_failure'] for x in rows)
    pairs = {}
    for row in rows:
        if row['warmup']:
            continue
        pair = pairs.setdefault(row['case_id'], {})
        assert row['arm'] not in pair
        pair[row['arm']] = row
    assert len(pairs) == 128 and all(set(pair) == {0, 1} for pair in pairs.values())
    for pair in pairs.values():
        assert all(pair[0][key] == pair[1][key] for key in
                   ['case_id', 'question_SHA256', 'shard', 'suite', 'output_budget', 'prompt_tokens'])
        assert all(math.isfinite(pair[arm]['backend_emitter_decode_tok_s'])
                   and pair[arm]['backend_emitter_decode_tok_s'] > 0 for arm in [0, 1])

    def costs(selected):
        ratios = [p[1]['backend_emitter_decode_tok_s'] / p[0]['backend_emitter_decode_tok_s'] for p in selected]
        arms = {}
        for arm in [0, 1]:
            values = [p[arm] for p in selected]
            rates = [r['backend_emitter_decode_tok_s'] for r in values]
            arms[str(arm)] = dict(cases=len(values),
                backend_emitter_decode_median_tok_s=statistics.median(rates),
                strict_above15=sum(v > 15 for v in rates),
                generated_tokens_including_EOS=sum(r['generated_tokens'] for r in values),
                recorded_client_HTTP_wall_sum_s=sum(r['client_HTTP_wall_s'] for r in values),
                client_incremental_qualified=False)
        return dict(arms=arms, paired_rate_ratio_median=statistics.median(ratios),
                    candidate_faster_pairs=sum(v > 1 for v in ratios),
                    candidate_slower_pairs=sum(v < 1 for v in ratios),
                    identical_generated_ID_hash_pairs=sum(p[0]['generated_IDs_SHA256'] == p[1]['generated_IDs_SHA256'] for p in selected),
                    output_texts_may_differ=True, endpoint_drift_estimate=None)

    return dict(original_audit_SHA256=AUDIT_SHA, scope=report['scope'],
                source_reader_SHA256='ee15b280b7ded754f11c3061e0105b0872192bca7a3482d95559b81f185721a8',
                actual_GPU=HANDLES, actual_CPU_keepers=[x['actual_CPU_keeper'] for x in report['replicas']],
                completed_requests=264, evaluated_pairs=128, warm_requests=8,
                all32_normal_lifecycle=True, all_original_post_clean=True,
                quality=report['quality'],
                costs={suite: costs([p for p in pairs.values() if suite == 'ALL' or p[0]['suite'] == suite])
                       for suite in ['GSM8K', 'MMLU', 'ALL']},
                per_replica={str(i): costs([p for p in pairs.values() if p[0]['shard'] == i]) for i in range(4)},
                global_no_regression_proven=False, full_benchmark_proven=False,
                client_streaming_SLO_qualified=False,
                limits=['Historical fixed custom0shot64GSM+64MMLU, not heldout or official fullsuite.',
                        'Native976f HC epilogue OFF/ON, same-source paired within replica; a314 fullpre not run.',
                        'Different generated texts/lengths allowed and reported; HTTP sums are not fixed-token latency.',
                        'Rates include original emitter/counter recording overhead in both arms, not instrumentation-free service.',
                        'Balanced heterogeneous adjacent pairs do not yield homogeneous endpoint drift.',
                        'Backend emitter rates are not incremental client TTFT/ITL; no strict15 qualification.',
                        'No predeclared noninferiority margin; intervals do not prove global no-regression.',
                        'Per-request sums across four replicas are not concurrent makespan or GPU utilization.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sealed_report', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.sealed_report.read_bytes())), ensure_ascii=False, indent=2))
