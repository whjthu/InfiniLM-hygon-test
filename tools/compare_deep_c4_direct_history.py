"""Compare two complete guarded DeepSeek C4 diagnostic outputs without runtime calls."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct


REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / 'code/bw1000/Deep-C4-causal-consumer-r483/causal-input-v1/INPUT_BINDING.json'
INPUT_SHA = '33b0fe310e0cbb6c55e9a994e539fe17a9a912de65d107338ef52616fec41aa7'
METADATA = REPO / 'code/bw1000/Deep-C4-causal-consumer-r483/checkpoint-input-v2/CHECKPOINT_METADATA.json'
METADATA_SHA = 'f56b6d38bdf6097488e707b200ea2cc8b4466ce27750de1c038a6be987a4cf00'
BASELINE_NATIVE_SHA = '90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1'
FLOOR = 16 * 1024**3
WEIGHT_BYTES = 13144320
CACHE_BYTES = 128 * 8448
COUNT_KEYS = ('prepares', 'commits', 'aborts', 'resets')
GUARD_KEYS = ('input_audits', 'input_buffers_checked', 'input_bytes_checked')
OUTPUT_LABELS = ('warm_abort', 'primary', 'full_boundary')
SAMPLED = {('history', 0), ('history', 512), ('warm_abort', 8496), ('primary', 8496),
           ('full_boundary', 32767), ('rewrite', 0)}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def read_pinned(path, wanted):
    data = path.read_bytes()
    require(digest(wanted) and sha256(data).hexdigest() == wanted, f'bad byte identity for {path.name}')
    return data


def output_values(root, label, plan, records):
    width = plan['compressed_after']
    size = min(512, width)
    require(records.keys() == {'scores', 'indices'}, f'missing score or TopK sample for {label}')
    specs = {'scores': ([1, width], 10, 4 * width, 'f', width),
             'indices': ([1, size], 3, 8 * size, 'q', size)}
    payloads, decoded = {}, {}
    for key, (shape, dtype, count, fmt, length) in specs.items():
        ref = records[key]
        meta = ref['metadata']
        require((meta['shape'], meta['dtype'], meta['bytes']) == (shape, dtype, count) and meta['contiguous'] is True,
                f'wrong geometry for {label} {key}')
        path = root / f'{label}-{key}.bin'
        require(path.stat().st_size == count, f'wrong byte count for {label} {key}')
        data = read_pinned(path, ref['SHA256'])
        require(len(data) == count, f'wrong byte count for {label} {key}')
        payloads[key] = data
        decoded[key] = struct.unpack(f'<{length}{fmt}', data)
    scores, topk = decoded['scores'], decoded['indices']
    require(all(map(math.isfinite, scores)) and len(set(topk)) == len(topk) and all(0 <= i < width for i in topk)
            and [scores[i] for i in topk] == sorted(scores, reverse=True)[:size], f'invalid scores or TopK for {label}')
    return payloads


def read_arm(root, ref, binding, weight_keys, range_sha):
    outcome = json.loads(read_pinned(root / 'weighted-outcome.json', ref['outcome_SHA256']))
    progress = [json.loads(line) for line in read_pinned(root / 'progress.jsonl', ref['progress_SHA256']).splitlines()]
    require(progress and progress[0].get('event') == 'admitted' and progress[0].get('planned_states') == 71
            and progress[0].get('planned_outputs') == 6 and progress[0].get('original_receipt_SHA256') == range_sha
            and progress[-1] == {'event': 'weighted_complete', 'complete': True},
            'original weighted input or completion receipt differs')
    close = outcome['closed']
    require(len(close) == 4 and all(type(value) is int for value in close) and min(close[:2]) >= FLOOR
            and close[2:] == [1, 1], 'guarded owner close or 16 GiB floor differs')
    checkpoint = outcome['checkpoint']
    require(checkpoint['bytes'] == WEIGHT_BYTES and checkpoint['loaded'] == sorted(weight_keys),
            'original checkpoint range load differs')
    diagnostic = outcome['diagnostic']
    require(diagnostic['performance_eligible'] is False and diagnostic['counts'] == {key: binding[key] for key in COUNT_KEYS}
            and diagnostic['final_past'] == binding['final_past'], 'frozen guarded diagnostic scope differs')
    records, payloads = diagnostic['records'], {}
    planned = binding['steps']
    require(len(records) == len(planned) == 71, 'full C4 step denominator was not preserved')
    require([row['record'] for row in progress if row.get('event') == 'step_complete'] == records and
            any(row.get('event') == 'closed' and row.get('driver_free_before_after_close') == close[:2] for row in progress),
            'guarded outcome differs from the durable step or close records')
    counts = [7, 7, WEIGHT_BYTES + 7 * 512]
    audit = diagnostic['input_audit']
    require([audit['postload'][key] for key in GUARD_KEYS] == counts and
            audit['postload']['input_guard_bytes_each_side'] == 256, 'checkpoint input guards differ')
    for index, (row, plan) in enumerate(zip(records, planned)):
        require(row['step'] == index and row['label'] == plan['label'], 'unexpected C4 step order')
        if plan['action'] == 'reset':
            require(row['after']['state']['all_zero'] is True and
                    row['before']['raw_cache']['SHA256'] == row['after']['raw_cache']['SHA256'],
                    'reset state or raw cache changed')
            continue
        require((row['past_before'], row['tokens'], row['input_pins']) == (plan['past'], plan['tokens'], plan['inputs']),
                'C4 causal inputs differ from the sealed plan')
        counts[0] += 1
        counts[1] += 3
        counts[2] += sum(pin['bytes'] for pin in plan['inputs'].values()) + 3 * 512
        observed = row['prepare_input_audit']
        require([observed[key] for key in GUARD_KEYS] == counts and
                observed['input_guard_bytes_each_side'] == 256, 'C4 prepare input guards differ')
        if row['label'] in OUTPUT_LABELS:
            payloads[row['label']] = output_values(root, row['label'], plan, row['outputs'])
    require([audit['final'][key] for key in GUARD_KEYS] == counts and
            audit['final']['input_guard_bytes_each_side'] == 256, 'final C4 input guards differ')
    fields = ('label', 'past', 'tokens', 'variant')
    observed_samples = [tuple(row[key] for key in fields) for row in diagnostic['numeric_audit']]
    planned_samples = [tuple(row[key] for key in fields) for row in planned if (row['label'], row.get('past')) in SAMPLED]
    require(observed_samples == planned_samples, 'projection reference sampling changed')
    cache = {}
    for row in diagnostic['cache_audit']:
        step = row['step']
        require(type(step) is int and 0 <= step < len(planned) and step not in cache, 'bad cache audit index')
        plan, report = planned[step], row['report']
        require((plan['label'], plan['past']) in SAMPLED and report['checked_cache_bytes'] == CACHE_BYTES
                and digest(report['after_SHA256']), 'raw cache audit differs')
        new_slots = plan['compressed_after'] - plan['past'] // 4
        require(report['pending_slots'] == new_slots and report['committed_slots'] == (new_slots if plan['commit'] else 0),
                'cache transaction slot accounting differs')
        cache[step] = report['after_SHA256']
    require(len(cache) == len(SAMPLED) and set(payloads) == set(OUTPUT_LABELS), 'missing C4 output or cache sample')
    return payloads, cache


def compare(manifest_path, manifest_sha):
    manifest = json.loads(read_pinned(manifest_path, manifest_sha))
    range_sha = manifest['original_range_receipt_SHA256']
    arms = manifest['arms']
    require(digest(range_sha) and arms.keys() == {'baseline', 'candidate'}, 'pair identity or original range hash missing')
    base, cand = (arms[key] for key in ('baseline', 'candidate'))
    require(base['Native_SHA256'] == BASELINE_NATIVE_SHA and digest(cand['Native_SHA256']) and
            cand['Native_SHA256'] != BASELINE_NATIVE_SHA, 'missing the declared baseline and candidate Native identities')
    roots = [(manifest_path.parent / row['directory']).resolve(strict=True) for row in (base, cand)]
    require(roots[0] != roots[1] and base['outcome_SHA256'] != cand['outcome_SHA256'], 'duplicated C4 arm evidence')
    binding = json.loads(read_pinned(INPUT, INPUT_SHA))
    weight_keys = json.loads(read_pinned(METADATA, METADATA_SHA))['weights'].keys()
    a, ca = read_arm(roots[0], base, binding, weight_keys, range_sha)
    b, cb = read_arm(roots[1], cand, binding, weight_keys, range_sha)
    sample_results = {}
    for label in OUTPUT_LABELS:
        score_match = a[label]['scores'] == b[label]['scores']
        topk_match = a[label]['indices'] == b[label]['indices']
        sample_results[label] = dict(scores_byte_exact=score_match, topk_bytes_equal=topk_match,
                                     same_scores_topk_ties=score_match and not topk_match)
    return dict(pair_manifest_SHA256=manifest_sha, original_range_receipt_SHA256=range_sha,
                causal_binding_SHA256=INPUT_SHA, full_steps_per_arm=71, raw_outputs_per_arm=6,
                declared_Native_SHA256={arm: ref['Native_SHA256'] for arm, ref in arms.items()},
                sample_results=sample_results, raw_cache_digests_equal_by_step={step: ca[step] == cb[step] for step in ca},
                exact_scores_all_samples=all(row['scores_byte_exact'] for row in sample_results.values()),
                operator_branch_execution_verified=False, performance_eligible=False, Model_GO=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.manifest_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
