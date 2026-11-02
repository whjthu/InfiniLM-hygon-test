"""Read sealed scalar reports only; never loads Runtime or submits work."""
import argparse
import hashlib
import json
import math
from pathlib import Path


PINS = {
    'watch.json': '0404082c841fc608af7b2afb56e43ca89416601dcb17da5f80216d91fba17f13',
    'driver.json': '6edf0f20b537a6c8fec2a9238cebd297ae78a8a977529ce5f1c6d55199eb7bbf',
}


def audit(directory):
    directory = Path(directory)
    reports = {}
    for name, expected in PINS.items():
        raw = (directory / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        reports[name] = json.loads(raw)
    w, d = reports['watch.json'], reports['driver.json']
    assert w['completed'] and w['passed'] and d['completed'] and d['passed']
    assert w['actual_step'] == d['actual_GPU'] == '910374.1332'
    assert w['Slurm_COMPLETED0'] and w['srun_exit0'] and w['normal1']
    assert w['last_observation']['srun_exit'] == 0
    assert w['normal_runtime_close'] and d['normal_runtime_close']
    assert d['GPU_count'] == 1 and not d['Model_loaded'] and not d['Model_gain']
    assert w['before'][0]['actual_step'] == '910374.1331'
    assert w['post'][0]['actual_step'] == '910374.1333'
    for post in w['post']:
        assert post['passed'] and not post['KFD_pids']
        assert not post['matching_GPU_driver_PIDs'] and not post['expected_PIDs_still_exist']
    s = d['session']
    assert s['normal_runtime_close'] and d['Torch_after_Runtime_close']
    assert s['foreign_thread_refused_without_owner_poison']
    assert s['actual_capture_CAPI_requires_2operators_1device_segment_0host_segments_empty_fallback']
    assert s['counters'] == dict(prepares=1, captures=1, eager_calls=99,
                                 replay_calls=100, workspace_bytes=8389120, phase=2)
    assert [p['pattern'] for p in s['patterns']] == [0, 1, 2]
    summaries = []
    numeric_count = 0
    maxima = {kind: 0.0 for kind in ('F32', 'BF16')}
    for p in s['patterns']:
        assert [x['mode'] for x in p['warm']] == [0, 1]
        assert [x['mode'] for x in p['formal']] == [0, 1, 1, 0]
        for key, loops in [('warm', 1), ('formal', 16)]:
            for x in p[key]:
                assert x['loops'] == loops and x['numeric']['passed']
                for field in ('whole_completed_wall_ms_per_iteration',
                              'internal_loop_and_terminal_drain_ms_per_iteration'):
                    assert math.isfinite(x[field]) and x[field] > 0
                for kind in maxima:
                    n = x['numeric'][kind]
                    assert n['finite'] and n['passed']
                    assert math.isfinite(n['relative_RMS']) and n['relative_RMS'] <= 1 / 64
                    assert math.isfinite(n['maxabs_over_reference_max'])
                    assert n['maxabs_over_reference_max'] <= 1 / 8
                    maxima[kind] = max(maxima[kind], n['relative_RMS'])
                numeric_count += 1
        a, b, c, e = [x['whole_completed_wall_ms_per_iteration'] for x in p['formal']]
        off, on = (a + e) / 2, (b + c) / 2
        summaries.append(dict(pattern=p['pattern'], eager_ms=off, graph_ms=on,
                              graph_over_eager=on / off, latency_reduction_fraction=1 - on / off,
                              rate_gain_fraction=off / on - 1,
                              eager_endpoint_drift_fraction=e / a - 1,
                              adjacent_pair_graph_over_eager=[b / a, c / e],
                              both_adjacent_pairs_faster=b < a and c < e,
                              formal_full_CAPI_wall_ms=[a, b, c, e]))
    return dict(completed=True, passed=True, actual_GPU=w['actual_step'],
                original_SHA256=PINS, counters=s['counters'], patterns=summaries,
                numeric_epoch_checks=numeric_count, max_relative_RMS=maxima,
                setup_ms={k: d[k] for k in ('Torch_device_setup_ms', 'DSO_load_ms', 'Context_init_ms')},
                prepare_ms=s['prepare_ms'], capture_ms=s['capture_ms'],
                normal_Runtime_close=True, independent_PID_KFD_post_clean=True,
                qualification='Actual fixed-buffer private Runtime/provider graph function PASS',
                performance_gate_predeclared=False, Model_gain=False, next_action='PARK_MODEL_BINDING_STOP_GRID',
                limitations=[
                    'Synthetic fixed 16x2048x512 F32 GEMM and BF16 Cast; no Model, TP or collective graph.',
                    'Full completed CAPI loop wall includes terminal drain; input copy, checks and CPU oracle excluded in both arms.',
                    'Private eager sets stream and workspace each call; production Generic is not this baseline.',
                    'One ABBA per pattern; retain all endpoint drift, including zero-pattern 9.57%.',
                    'Guards/readonly checks are asserted by successful CAPI outside timing, not independently dumped tensors.',
                    'Fixed-pointer graph to dynamic Model binding/amortization UNKNOWN; no additional loading justified.',
                    'Card residency is not GPU busy; no device busy or energy measurement.',
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sealed_directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.sealed_directory), indent=2))
