"""Recompute complete-phase paired costs from original Kimi386 scalar reports.

This reader launches no work and imports no model/device libraries. Numeric
receipts remain device-reported checks, not a second reconstruction of tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as stats

GPU_MF = '90110b97a62e2fea74d4f4dfc95d7c9665ade3b4494e7b2c7165853eab549436'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original(snapshot, key):
    ref = snapshot[key]
    value = ref['data']
    assert any(sha((json.dumps(value, ensure_ascii=a, indent=2) + '\n').encode()) == ref['SHA256'] for a in [True, False])
    return value


def audit(path):
    raw = Path(path).read_bytes()
    snapshot = json.loads(raw)
    watch, driver = original(snapshot, '-watch.json'), original(snapshot, 'driver')
    assert watch['manifest_SHA256'] == driver['source_manifest_SHA256'] == GPU_MF
    assert watch['actual_GPU'] == driver['actual_step'] == '910371.713'
    assert watch['CPU_keeper'] == '910407.773'
    assert watch['completed'] and watch['passed'] and watch['srun_exit_code'] == 0
    assert watch['terminal'][2:4] == ['COMPLETED', '0:0']
    accounting = {v.split('|')[0]: v.split('|') for v in snapshot['accounting']}
    assert accounting['910407.773'][2:4] == ['COMPLETED', '0:0']
    assert accounting['910371.714'][2:4] == ['COMPLETED', '0:0']
    post = watch['post']
    assert post['KFD_available'] and not post['kfd_pids'] and not post['matching_component_processes']
    assert driver['completed'] and driver['passed'] and driver['normal_Runtime_close'] and driver['normal_session_close']
    assert driver['all20_contexts_closed'] and driver['original_model_parameter_identity_unchanged']
    assert driver['GPU_cards'] == 1 and driver['target_cards'] == 0 and driver['profile'] is False
    assert driver['actual_device']['visible_count'] == 1 and 'gfx936' in driver['actual_device']['gcnArchName']
    assert driver['fixture']['raw_dim'] == 35840 and driver['fixture']['period_rows'] == 512
    records, fences = driver['phase_records'], driver['memory_fences']
    assert len(records) == 20 and len(fences) == 1270
    assert all(v['floor_passed'] and v['free_after_bytes'] >= 4 << 30 for v in fences)
    rows = []
    modes = [False, True, False, True, True, False, False, True, True, False]
    for i, r in enumerate(records):
        L = [8635, 32638][i // 10]
        n = (L + 511) // 512
        assert r['ordinal'] == i and r['L'] == L and r['discard_raw'] is modes[i % 10]
        assert r['phase'] == ('warm' if i % 10 < 2 else 'formal')
        assert r['passed'] and r['completed'] and r['normal_session_close'] and r['profile'] is False
        expected = ['begin'] + ['append_prompt'] * n + ['seal', 'query'] + ['append_committed', 'query'] * 7 + ['retire', 'close']
        assert [v['cmd'] for v in r['commands']] == expected
        assert len(r['queries']) == 8 and r['fence_end'] - r['fence_start'] == n + 23
        assert [v['past'] for v in r['queries']] == [L + 8 * j for j in range(8)]
        assert all(len(v['proposals']) == 7 and all(type(t) is int and 0 <= t < 163840 for t in v['proposals']) for v in r['queries'])
        for kind in ['hidden', 'logits']:
            check = r['last_query_' + kind]
            assert check['finite'] and check['NRMSE'] == check['maxabs'] == 0
        assert r['all8_proposal_position_matches'] == r['proposal_positions'] == 56
        payload = r['payload_before_retire']
        assert payload['raw'] == (0 if r['discard_raw'] else (L + 56) * 35840 * 2)
        assert payload['projected'] == (L + 56) * 7168 * 2
        trims = sum(bool(v['trimmed']) for v in fences[r['fence_start']:r['fence_end']])
        if r['phase'] == 'formal':
            assert trims == 0
        keys = ['ordinal', 'L', 'phase', 'arm', 'complete_phase_wall_s', 'prefix_begin_append_seal_wall_s',
                'eight_query_and_seven_append_wall_s', 'query_dispatch_sum_s', 'committed_append_dispatch_sum_s',
                'Torch_peak_allocated_bytes', 'Torch_peak_reserved_bytes', 'payload_before_retire']
        rows.append(dict({k: r[k] for k in keys}, all8_query_proposals_match=True,
                         final_query_hidden_logits_reported_maxabs=0, allocator_trim_count=trims))
    summaries = []
    metrics = ['complete_phase_wall_s', 'prefix_begin_append_seal_wall_s', 'eight_query_and_seven_append_wall_s',
               'query_dispatch_sum_s', 'committed_append_dispatch_sum_s', 'Torch_peak_allocated_bytes', 'Torch_peak_reserved_bytes']
    for L in [8635, 32638]:
        selected = [r for r in rows if r['L'] == L and r['phase'] == 'formal']
        off, on = [[r for r in selected if r['arm'] == arm] for arm in ['OFF', 'ON']]
        paired = []
        for start in [0, 4]:
            block = selected[start:start + 4]
            assert [r['arm'] for r in block] == ['OFF', 'ON', 'ON', 'OFF']
            a, b = [r['complete_phase_wall_s'] for r in [block[0], block[3]]]
            c, d = [r['complete_phase_wall_s'] for r in [block[1], block[2]]]
            paired.append(dict(ON_over_OFF=(c + d) / (a + b),
                               OFF_bracket_spread_fraction=abs(b - a) / ((a + b) / 2)))
        values = {}
        for metric in metrics:
            a, b = [stats.mean(r[metric] for r in arm) for arm in [off, on]]
            values[metric] = dict(OFF_mean=a, ON_mean=b, ON_over_OFF=b / a, OFF_minus_ON=a - b)
        assert all(v['ON_over_OFF'] < 1 and 1 - v['ON_over_OFF'] > v['OFF_bracket_spread_fraction'] for v in paired)
        summaries.append(dict(L=L, formal_OFF=4, formal_ON=4, pairs=paired, metrics=values))
    return dict(state='ROOT_AUDITED_SCOPED_DRAFT_PHASE_GAIN_NORMAL_CLEAN', observed_UTC=snapshot['observed_UTC'],
                actual_GPU='910371.713', CPU_keeper='910407.773', post='910371.714',
                source_claim_commit=watch['publication_commit'], GPU_manifest_SHA256=GPU_MF,
                original_driver_path=snapshot['driver']['path'], original_driver_SHA256=snapshot['driver']['SHA256'],
                original_watch_path=snapshot['-watch.json']['path'], original_watch_SHA256=snapshot['-watch.json']['SHA256'],
                original_snapshot_SHA256=sha(raw), original_JSON_hashes_verified=True,
                reader=str(Path(__file__).resolve()), reader_SHA256=sha(Path(__file__).read_bytes()),
                summaries=summaries, phase_records=rows, planned_actual_phases=20, warm=4, formal=16,
                query_calls=160, committed_appends=140, command_count=1190, original_memory_fences=1270,
                allocator_trim_count=driver['allocator_trim_count'], trim_scope='one near32k OFF warm phase; zero formal trims',
                strict_load_and_fixture_setup_s=driver['strict_load_and_fixture_setup_s'], whole_driver_wall_s=driver['whole_driver_wall_s'],
                accounting=list(accounting.values()), normal_Runtime_close=True, normal_session_close=True, post_clean=True,
                scope='One actual pinned 5layer draft and borrowedhead; periodic synthetic BF16 features/full35840 projection; complete emptycontext prefix+8query/7committed8+retire/close; no target residency.',
                limits=['GPU scalar numeric receipts verified; root does not reconstruct hidden/logit tensors from raw activations.',
                        'Finalquery hidden/logits and all8proposal diagnostics are not target acceptance/quality evidence.',
                        'Postphase numeric/save cost excluded from phase but included in whole driver.',
                        'Explicit original4GiB memory guard and allocation stats remain in both arms; profile OFF.',
                        'Torch peak allocated differs from reserved; unchanged reserved is not a physical capacity gain.',
                        'Query timing and target decode are not accelerated by this result; no Model/endtoend/SSE/15tok qualification.',
                        'GPU/keeper elapsed is residence/cleanup cost, not GPUbusy utilization.'],
                Model_gain_qualified=False, acceptance_qualified=False, target_Model_loads=0, automatic_reload=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('snapshot', type=Path)
    print(json.dumps(audit(p.parse_args().snapshot), ensure_ascii=False, indent=2))
