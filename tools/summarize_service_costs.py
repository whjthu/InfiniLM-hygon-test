"""Summarize existing GLM/Deep owner receipts without running model code.

Reads a report, checks its identity and completed request timelines, and emits
numbers only. Partial live requests are retained as pending, never as failures
or successful samples. Backend emitter metrics are not client streaming SLOs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

MODELS = {
    'GLM': {'native': '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',
            'topology': (8, 4, 32), 'eos': {154820, 154827, 154829},
            'vocab': 154880, 'rows_key': 'service_requests'},
    'Deep': {'native': 'afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2',
             'topology': (8, 1, 8), 'eos': {1}, 'vocab': 129280,
             'rows_key': 'requests'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or nonnumeric timing')
    return value


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * fraction
    lower = int(x)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (x - lower)


def request_cost(row, spec):
    ids = row['generated_token_ids']
    ticks = row['backend_token_monotonic']
    require(ids and len(ids) == len(ticks) == row['actual_generated_tokens'], 'Token/timeline count differs')
    require(all(type(i) is int and 0 <= i < spec['vocab'] for i in ids), 'Invalid generated token')
    ticks = [number(t) for t in ticks]
    require(all(a <= b for a, b in zip(ticks, ticks[1:])), 'Backward token clock')
    accepted = number(row['admission']['accepted_monotonic'])
    started, ended = number(row['started_monotonic']), number(row['ended_monotonic'])
    require(accepted <= started <= ticks[0] <= ticks[-1] <= ended, 'Request timeline outside owner interval')
    require(row['normal_reset_before'] is True and row['normal_reset_after'] is True, 'Reset incomplete')
    budget = row['max_new_tokens']
    prompt = row['prompt_tokens']
    require(type(prompt) is int and type(budget) is int and 0 < prompt and 0 < budget <= 8192
            and prompt + budget <= 32768 and len(ids) <= budget, 'Context or output budget invalid')
    reason = row['native_finish_reason']
    require(reason in ('eos_token', 'length'), 'Unrecognized finish reason')
    require((ids[-1] in spec['eos']) if reason == 'eos_token' else
            (len(ids) == budget and ids[-1] not in spec['eos']), 'EOS/cap evidence differs')
    visible = [t for t, token in zip(ticks, ids) if token not in spec['eos']]
    gaps = [b - a for a, b in zip(visible, visible[1:])]
    host = row['Native_forward_host_summary']
    pf, dc = number(host['prefill_s']), number(host['decode_s'])
    wall = ended - started
    require(wall > 0 and pf >= 0 and dc >= 0 and pf + dc <= wall + max(.01, wall * 1e-5),
            'Native enclosure exceeds same-owner wall')
    return {'admitted_index': row['admitted_index'], 'phase': row.get('phase', 'unspecified'),
            'prompt_tokens': prompt, 'output_budget': budget, 'generated_tokens': len(ids),
            'visible_tokens': len(visible), 'natural_EOS': reason == 'eos_token',
            'controlled_or_unexpected_cap': reason == 'length', 'owner_wall_s': wall,
            'queue_s': started - accepted, 'backend_emitter_first_wait_s': visible[0] - accepted if visible else None,
            'backend_emitter_decode_tok_s': (len(visible) - 1) / (visible[-1] - visible[0])
            if len(visible) > 1 and visible[-1] > visible[0] else None,
            'backend_emitter_ITL_p95_s': percentile(gaps, .95),
            'backend_emitter_ITL_p99_s': percentile(gaps, .99),
            'Native_prefill_host_enclosure_s': pf, 'Native_decode_host_enclosure_s': dc,
            'Native_prefill_fraction_of_owner_wall': pf / wall,
            'Native_decode_fraction_of_owner_wall': dc / wall,
            'outside_Native_host_enclosure_s': max(0., wall - pf - dc)}


def summarize(report, model, expected_step, expected_manifest):
    spec = MODELS[model]
    require(report['actual_step'] == expected_step, 'Actual step differs')
    require(report['source_manifest_SHA256'] == expected_manifest, 'Source manifest differs')
    require(report['Native_SHA256'] == spec['native'], 'Native differs')
    topology = (report['TP'], report['PP'], report.get('cards', report.get('all_serving_cards')))
    require(topology == spec['topology'], 'Topology differs')
    rows, pending = [], []
    for index, row in enumerate(report[spec['rows_key']]):
        require(row['admitted_index'] == index, 'Nonsequential admission')
        if 'ended_monotonic' not in row or not row.get('normal_reset_after') or 'final_metadata' not in row:
            pending.append({'admitted_index': index, 'prompt_tokens': row.get('prompt_tokens'),
                            'reported_error': bool(row.get('error') or row.get('reset_error'))})
            continue
        rows.append(request_cost(row, spec))
    rates = [r['backend_emitter_decode_tok_s'] for r in rows if r['backend_emitter_decode_tok_s'] is not None]
    return {'model': model, 'actual_step': expected_step, 'source_manifest_SHA256': expected_manifest,
            'report_completed_flag': report.get('completed') is True,
            'completed_requests_with_checked_timelines': len(rows), 'pending_requests': pending,
            'requests': rows, 'backend_emitter_decode_median_tok_s': statistics.median(rates) if rates else None,
            'client_incremental_metrics_qualified': False, 'new_speedup_or_Model_qualification': False,
            'limits': ['Live snapshot may be incomplete; request samples are heterogeneous.',
                       'Emitter timestamps exclude EOS/control IDs and use (n-1)/(last-first).',
                       'Buffered HTTP does not provide incremental client TTFT or ITL.',
                       'Native host enclosure includes waits/host work; it is not GPU busy or removable time.',
                       'Do not sum pipeline-stage enclosures or infer physical maximum from these observations.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--expected-step', required=True)
    p.add_argument('--manifest-sha', required=True)
    args = p.parse_args()
    raw = args.report.read_bytes()
    result = summarize(json.loads(raw), args.model, args.expected_step, args.manifest_sha)
    result['receipt_SHA256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
