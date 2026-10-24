"""One terminal audit; negative configuration outcomes are retained, never rebaselined."""
import hashlib
import json
import math
import pathlib
import subprocess
import sys

r = pathlib.Path.home() / 'hygon-inference-refactor-20261014'
tag = sys.argv[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = [v.split('|') for v in subprocess.check_output(['sacct', '-S', '2026-10-24', '-j', tag, '-n', '-P', '-o', 'JobID,State,ExitCode'], text=True).splitlines()]
assert len(rows) == 1 and rows[0][1:] == ['COMPLETED', '0:0']
p = r / 'results' / f'deep-B64-M64-r8644d-{tag}.json'
out = p.with_name(p.stem + '-audit.json')
assert not out.exists()
q = json.loads(p.read_text())
assert q['completed'] and q['engine_worker_shutdown_completed'] and not q['performance_adopted']
manifest = json.loads((r / 'incoming/Deep-B64-M64-config-r8644d-manifest.json').read_text())
for n, h in manifest['sources_SHA256'].items():
    assert sha(r / 'incoming' / n) == h
assert q['library_sha256'] == manifest['DSO_SHA256']
ref = json.loads((r / 'incoming/Deep-B32-samegeom-reference-r7269.json').read_text())
gold = [ref['generated_ids'][i % 32] for i in range(64)]
trials = q['config_trials_r8644']
assert [c['micro'] for c in trials] == [64, 32, 32, 64][:len(trials)]
for c in trials:
    if not c['complete']:
        assert q['outcome'] == 'NEGATIVE_QUALITY_STOP' and 'failure' in c
        assert c is trials[-1]
        continue
    assert c['generated_ids'] == gold and c['full_IDs_EOS_exact'] and c['gate26_passed']
    assert c['CPUargmax_every_window']
    ws = c['CAPI_windows']
    assert [(v['step'], v['chunk'], v['rows']) for v in ws] == [(0, a, 4) for a in range(0, 64, 4)] + [(s, a, c['micro']) for s in range(1, 128) for a in range(0, 64, c['micro'])]
    assert all(math.isfinite(v['seconds']) and v['seconds'] > 0 for v in ws)
    pre = sum(v['seconds'] for v in ws if v['step'] == 0)
    dec = sum(v['seconds'] for v in ws if v['step'] > 0)
    assert math.isclose(pre, c['native_prefill_seconds'], rel_tol=1e-12)
    assert math.isclose(dec, c['native_decode_seconds'], rel_tol=1e-12)
    assert math.isclose(8128 / dec, c['native_decode_tok_s'], rel_tol=1e-12)
    assert math.isclose(c['native_decode_tok_s'] / 8, c['native_decode_tok_s_per_card'], rel_tol=1e-12)
    assert math.isclose(127 / dec, c['amortized_native_decode_tok_s_per_logical_row'], rel_tol=1e-12)
    assert math.isclose(8192 / (pre + dec), c['native_E2E_output_tok_s'], rel_tol=1e-12)
    assert c['projection_counter_delta'] == [43 * 127 * (64 // c['micro'])] * 8
    stamps = c['request_completion_offsets_seconds']
    assert len(stamps) == len(c['request_metrics']) == 64
    for row, (ts, metrics) in enumerate(zip(stamps,c['request_metrics'])):
        assert len(ts) == 128 and all(math.isfinite(t) and t > 0 for t in ts)
        gaps = [b-a for a,b in zip(ts,ts[1:])]
        assert min(gaps) > 0 and metrics['row'] == row and metrics['non_EOS_output_tokens'] == 128
        assert math.isclose(metrics['first_to_last_seconds'],ts[-1]-ts[0],rel_tol=1e-12)
        assert math.isclose(metrics['decode_tok_s'],127/(ts[-1]-ts[0]),rel_tol=1e-12)
        assert metrics['TTFT_seconds'] == ts[0] and metrics['max_gap_seconds'] == max(gaps)
        ordered = sorted(gaps);position = .95*(len(ordered)-1);lo = int(position)
        p95 = ordered[lo]+(position-lo)*(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])
        assert math.isclose(metrics['ITL_p95_seconds'],p95,rel_tol=1e-12)
for key in ['owner_affinity_before_weight_load', 'owner_affinity_after_weight_load', 'owner_affinity']:
    a = q[key]
    assert a['verified'] and a['width'] == 4
    owners = a['actual_owner_report']['owners']
    assert sorted(o['device'] for o in owners) == list(range(8))
    masks = [int(v) for o in owners for v in o['verified_mask'].split(',')]
    assert len(masks) == len(set(masks)) == 32
assert q['finite_before_passed']
if q.get('quality_passed'):
    assert q['finite_after_restore_passed']
qualified = bool(len(trials) == 4 and q.get('quality_passed') and all(c['complete'] for c in trials))
assert q['performance_qualified'] == qualified
result = {'completed': True, 'step': tag, 'normal8owner_shutdown': True,
          'source_report_SHA256': sha(p), 'outcome': q['outcome'], 'performance_qualified': qualified,
          'performance_adopted': False, 'negative_preserved_no_resubmit': True,
          'scope': q['scope'], 'trials': [{k: v for k, v in c.items() if k not in ['generated_ids', 'CAPI_windows', 'per_step_completion_seconds']} for c in trials]}
with out.open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
