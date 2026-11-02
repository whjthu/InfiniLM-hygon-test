"""Read-only root audit of aligned eight-owner HC component measurements."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics


def paired(values):
    assert len(values) == 20
    groups = []
    for start in range(0, 20, 4):
        left, a, b, right = values[start:start+4]
        baseline, candidate = (left+right)/2, (a+b)/2
        drift = right/left-1
        groups.append(dict(group=start//4, ON_over_OFF=candidate/baseline,
                           OFF_endpoint_drift=drift,
                           gain_exceeds_absolute_drift=1-candidate/baseline > abs(drift)))
    baseline = statistics.mean(v for i, v in enumerate(values) if i % 4 in (0, 3))
    candidate = statistics.mean(v for i, v in enumerate(values) if i % 4 in (1, 2))
    return dict(OFF_us=baseline*1e6, ON_us=candidate*1e6, ON_over_OFF=candidate/baseline,
                groups=groups, all5_same_direction=all(v['ON_over_OFF'] < 1 for v in groups),
                all5_exceed_drift=all(v['gain_exceeds_absolute_drift'] for v in groups))


def audit(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'd8f09440fa4e863263507f1007f131646372dd88a69997283e659be1ca5c93ee'
    outer = json.loads(raw)
    a = outer['archive']
    owners, report, watch = a['owners'], a['report'], a['watch']
    assert report['completed'] and report['passed'] and not report['Model_loaded']
    assert report['actual_step'] == '910374.1298'
    assert report['processes'] == 1 and report['owner_threads'] == 8
    assert len(owners) == 8 and {o['rank'] for o in owners} == set(range(8))
    assert {o['hardware']['Runtime_device'] for o in owners} == set(range(8))
    assert all(o['hardware']['visible_device_list'] == '0,1,2,3,4,5,6,7' for o in owners)
    owner_metrics, max_error = [], 0.0
    for owner in owners:
        assert owner['completed'] and owner['passed'] and owner['normal_close']
        assert owner['Runtime_expired'] == 1 and owner['destroy_code'] == 0
        assert owner['weights_state_initial'] == owner['weights_state_final']
        assert len(owner['calls']) == 20 and len(owner['warmup']) == 2
        for i, row in enumerate(owner['calls']):
            assert row['trial'] == i and row['mode'] == [0, 1, 1, 0][i % 4]
            assert row['repeats'] == 32 and row['GPU_hits'] == row['mode']*32
            for total, repeats in [('CAPI_wall_s', 'repeat_CAPI_wall_s'),
                                   ('native_component_s', 'repeat_native_component_s')]:
                assert len(row[repeats]) == 32
                assert all(math.isfinite(v) and v > 0 for v in row[repeats])
                assert math.isclose(row[total], math.fsum(row[repeats]), rel_tol=1e-10)
            assert math.isfinite(row['max_NRMSE'])
            max_error = max(max_error, row['max_NRMSE'])
        owner_metrics.append(dict(rank=owner['rank'],
            fixture={k: owner['fixture'][k] for k in ['layer', 'kind', 'seed', 'input_kind']},
            CAPI=paired([r['CAPI_wall_s']/32 for r in owner['calls']]),
            Cpp=paired([r['native_component_s']/32 for r in owner['calls']])))
    aligned = {}
    for field in ['CAPI_wall_s', 'native_component_s', 'local_loop_wall_s']:
        aligned[field] = paired([max(o['calls'][i][field] for o in owners)/32 for i in range(20)])
    measurements = [r for r in report['protocol']['records'] if r['phase']['kind'] == 'measurement']
    assert len(measurements) == 20
    aligned['whole_checked_phase_s'] = paired([row['whole_phase_wall_s']/32 for row in measurements])
    assert watch['terminal'][2:4] == ['COMPLETED', '0:0']
    assert watch['normal_close_all8'] and watch['threads_joined'] == [True]*8
    assert watch['post']['kfd_pids'] == [] and watch['post']['matching_component_processes'] == []
    for parent, handle in [('910374', '910374.1298'), ('910407', '910407.746')]:
        match = [line.split('|') for line in a['sacct'][parent] if line.startswith(handle+'|')]
        assert len(match) == 1 and match[0][2:4] == ['COMPLETED', '0:0']
    all_groups = [group for owner in owner_metrics for group in owner['CAPI']['groups']]
    return dict(root_aligned_audit_passed=True, actual_GPU='910374.1298', CPU_keeper='910407.746',
                local_original_archive_SHA256=digest,
                remote_original_archive_path=outer['remote_archive_path'],
                remote_original_archive_SHA256=outer['remote_archive_SHA256'],
                all8_normal_and_state_hit_checks_passed=True, max_observed_NRMSE=max_error,
                measurement_calls_per_owner=640, measurement_calls_all8=5120,
                aligned_same_trial_max8=aligned, owner_fixture_metrics=owner_metrics,
                CAPI_owner_groups_faster=sum(v['ON_over_OFF'] < 1 for v in all_groups),
                CAPI_owner_groups_exceed_drift=sum(v['gain_exceeds_absolute_drift'] for v in all_groups),
                maximum_absolute_owner_OFF_drift=max(abs(v['OFF_endpoint_drift']) for v in all_groups),
                post=watch['post'],
                decision='Retain component opt-in; prepare default-OFF Native CPU integration for next independently necessary Model residency, no dedicated reload',
                limitations=['Eight owners have distinct layer0/20 attn/ffn two-seed fixtures, not actual coherent TP8 model activations',
                             'Runtime logical indices0..7 and Slurm8card assignment recorded; per-owner PCI identifiers absent',
                             'Same-trial max8 CAPI and Cpp intervals improve; outer checked phase has no stable improvement',
                             'No kernel attribution or GPU utilization from host/process wall',
                             'No per-layer multiplication or model speed projection used as evidence',
                             'Root recomputed scalar timing and checked sealed validation; no raw activation re-evaluation',
                             'No new GPU, Model, or broad quality qualification'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result['aligned_same_trial_max8'][k]['ON_over_OFF']
                      for k in result['aligned_same_trial_max8']}))
