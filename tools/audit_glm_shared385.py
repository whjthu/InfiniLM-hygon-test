"""Read original eight-rank scalar records; recalculate the frozen layer gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def audit(path):
    raw = Path(path).read_bytes(); snapshot = json.loads(raw)
    owners, references = {}, []
    for ref in snapshot['owners']:
        r = ref['scalar_report']
        assert hashlib.sha256(json.dumps(r, indent=2).encode()).hexdigest() == ref['original_SHA256']
        assert r['actual_step'] == '910371.716' and r['normal'] and r['completed']
        assert r['Runtime_weak_expired'] and r['normal_reset_destroy'] and r['Torch_after_destroy']
        assert r['measurement_calls'] == len(r['trials']) == 96 and r['warm_calls'] == len(r['warm']) == 4
        assert r['profile'] == 0 and r['TP'] == 8 and r['layer'] == 38
        for t in r['trials'] + r['warm']:
            mode = t['mode']
            assert t['finite'] and t['canary'] and t['input_readonly']
            assert t['relative_RMS'] == t['ownrepeat_relative_RMS'] == 0
            assert t['shared_M1_counts_delta'] == [1, mode, mode, mode, 0]
            assert t['vector_counts_delta'] == [0, 1, 0, 1, 0, 0, 0, 0]
            assert t['route_U'] == 8 and t['route_maxrows'] == 1
        owners[r['rank']] = r
        references.append(dict(rank=r['rank'], path=ref['path'], SHA256=ref['original_SHA256'], PCI=r['physical_PCI']))
    assert set(owners) == set(range(8)) and len({r['physical_PCI'] for r in owners.values()}) == 8
    assert all(r['input_fixtures'] == owners[0]['input_fixtures'] for r in owners.values())
    key = lambda t: (t['fixture'], t['pair'], t['arm'], t['rep'])
    lookup = {rank: {key(t): t for t in r['trials']} for rank, r in owners.items()}
    aligned = []
    for k in sorted(lookup[0]):
        mode = lookup[0][k]['mode']
        assert all(lookup[rank][k]['mode'] == mode for rank in range(8))
        aligned.append(dict(fixture=k[0], pair=k[1], arm=k[2], rep=k[3], mode=mode,
                            max8_complete_driver_wall_s=max(lookup[rank][k]['complete_driver_wall_s'] for rank in range(8))))
    gate = owners[0]['performance_gate']; summary = []
    assert all(r['performance_gate'] == gate for r in owners.values())
    for fixture in range(2):
        pairs = []
        for pair in range(4):
            means = {}
            for mode in [0, 1]:
                times = [t['max8_complete_driver_wall_s'] for t in aligned if (t['fixture'], t['pair'], t['mode']) == (fixture, pair, mode)]
                assert len(times) == 6
                means[mode] = statistics.mean(times)
            pairs.append(dict(pair=pair, OFF_s=means[0], ON_s=means[1], paired_rate_gain=means[0] / means[1] - 1))
        gain = statistics.median(p['paired_rate_gain'] for p in pairs)
        positive = sum(p['paired_rate_gain'] > 0 for p in pairs)
        drift = abs(pairs[-1]['OFF_s'] / pairs[0]['OFF_s'] - 1)
        passed = gain >= gate['minimum_median_rate_gain_per_input'] and positive >= gate['minimum_positive_pairs_per_input'] and drift <= gate['maximum_OFF_endpoint_drift']
        summary.append(dict(fixture=fixture, pairs=pairs, median_paired_rate_gain=gain, positive_pairs=positive, OFF_endpoint_drift=drift, gate_passed=passed))
    assert not all(r['gate_passed'] for r in summary)
    return dict(state='ROOT_RECALCULATED_COMPONENT_GATE_FAIL_STOP', actual_GPU='910371.716',
                original_scalar_snapshot_SHA256=hashlib.sha256(raw).hexdigest(), original_owners=references,
                reader_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                raw_owner_JSON_hashes_verified=True, aligned_trials=96, numeric_max_relative_RMS=0,
                all8_normal=True, summary=summary, original_gate=gate, retained=False,
                clock='Aligned trial max of8 local elapsed enclosures; not one global makespan or GPUbusy time.',
                scope='Whole checkpointlayer38 TP8 with two synthetic hidden/prefix fixtures; sharedM1 only; no Model/benchmark/publicSLO qualification.',
                cleanup_reference='results/bw1000/2026-11-02/GLM-shared-M1-716-terminal-component-gate-FAIL-clean.json',
                no_threshold_change=True, no_grid_expansion=True, Model_gain=False,
                prior_CPU774_failure_retained=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('snapshot', type=Path)
    print(json.dumps(audit(parser.parse_args().snapshot), ensure_ascii=False, indent=2))
