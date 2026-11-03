"""Standard-library audit of immutable r426 original bytes; never contacts cluster."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

MF = '04e61f257f3bf68df64f614a323cac93d7daf471de913964e451be94c3de95ee'
LIB = '90a970c759908c866fa9802208fd729929bcfdd14dfbcbbd5a15cb227b44beb9'
MODES = [0, 1, 0, 1, 1, 0]
sha = lambda b: hashlib.sha256(b).hexdigest()


def read(seal, repo):
    raw = Path(seal).read_bytes()
    envelope = json.loads(raw)
    if 'rawtext' in envelope:
        b = envelope['rawtext'].encode()
        assert len(b) == envelope['bytes'] and sha(b) == envelope['SHA256']
        archive_path, archive_sha = envelope['path'], envelope['SHA256']
        data = json.loads(b)
        archive_bytes = len(b)
    else:
        data, archive_path, archive_sha = envelope, str(seal), sha(raw)
        archive_bytes = len(raw)
    assert data['read_only'] and data['Slurm_submitted'] == 0
    files = {}
    for f in data['files']:
        b = f['rawtext'].encode()
        assert len(b) == f['bytes'] and sha(b) == f['SHA256'], f['path']
        assert f['path'] not in files
        files[f['path']] = f
    get = lambda suffix: next(f for p, f in files.items() if p.endswith(suffix))
    watch = json.loads(get('/GLM426-watch-v1.json')['rawtext'])
    assert watch['completed'] and watch['passed'] and watch['normal8']
    assert watch['GPU_source_manifest_SHA256'] == MF
    assert watch['publication_commit'] == '1af49df863f0b9e338506febf218fc90ea6a5062'
    assert watch['actual_step'] == '910371.751' and watch['srun_exit0'] and watch['Slurm_COMPLETED0']
    log = get('/GLM-r426-selected-plan-layer38-v1-910371.751.log')
    assert log['SHA256'] == watch['log_SHA256'] and 'GLM426_ALL8_NORMAL' in log['rawtext']
    for probe, step in [('before', '910371.750'), ('post', '910371.752')]:
        d = watch[probe]
        assert d['passed'] and d['actual_step'] == step and d['CPU_only']
        assert d['KFD_pids'] == d['matching_GPU_driver_PIDs'] == d['expected_PIDs_still_exist'] == []
        assert len(set(d['physical_PCI'])) == 8
    accounting = {}
    for line in data['accounting']:
        fields = line.split('|')
        accounting[fields[0]] = fields
    elapsed_s = {}
    for step in ['910371.750', '910371.751', '910371.752', '910407.856']:
        row = accounting[step]
        assert row[2:4] == ['COMPLETED', '0:0'], row
        hh, mm, ss = map(int, row[4].split(':'))
        elapsed_s[step] = hh * 3600 + mm * 60 + ss
    gpu = repo / 'code/bw1000/GLM-NoV-selected-FP8-plan-r426/gpu-v1'
    mraw = (gpu / 'GPU_SOURCE_MANIFEST.json').read_bytes()
    assert sha(mraw) == MF
    manifest = json.loads(mraw)
    for name, digest in manifest['files_SHA256'].items():
        assert sha((gpu / name).read_bytes()) == digest, name
    owners = []
    references = {}
    numeric = []
    free = {k: [] for k in ['free_entry', 'free_after_selected', 'free_after_output']}
    final = dict(created=3, destroyed=3, hits=21, misses=3, evictions=0,
                 entries=0, bytes=0, highwater=20447232, runs=24, invalid=0)
    for expected in watch['owners']:
        f = files[expected['path']]
        assert f['SHA256'] == expected['SHA256']
        d = json.loads(f['rawtext'])
        rank = d['rank']
        assert rank == expected['rank'] and d['PID'] == expected['PID']
        assert d['physical_PCI'] == expected['physical_PCI'] and d['gcnArchName'].split(':')[0] == 'gfx936'
        assert d['GPU_source_manifest_SHA256'] == MF and d['library_SHA256'] == LIB
        for name, provider in d['actual_symbol_owners'].items():
            expected_provider = ('/refactor/candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so'
                                 if name == 'glm_fp8_create_r776' else
                                 '/refactor/candidates/GLM-NoV-selected-FP8-plan-r426-component-link-v1/libGLM_NoV_selected_plan_attention_r426.so')
            assert provider == expected_provider, (rank, name, provider)
        for k in ['completed', 'normal', 'normal_destroy', 'Runtime_weak_expired', 'Torch_after_destroy',
                  'policy_closed', 'NoV_fixed', 'plan_reuse_only_variable', 'weights_loaded_once',
                  'input_immutable', 'destroy_attempted_once', 'Runtime_expired']:
            assert d[k] is True, (rank, k)
        assert d['destroy_RC'] == 0 and d['partial_error'] is None
        assert d['query_calls'] == 48 and d['seed_calls'] == 432
        assert d['warm_calls'] == 2 and d['measurement_calls'] == 4
        assert d['load']['keys'] == 12 and d['load']['checkpoint_load_passes'] == 1
        assert not d['Model'] and not d['wholeModel_loaded'] and not d['indexer_included']
        assert d['profile'] == 0 and d['plan_cache_final_stats'] == final
        assert len(d['CPU_affinity']) == 4 and d['input_shape'] == [32768, 6144]
        assert len(d['records']) == 6
        for i, rec in enumerate(d['records']):
            assert (rec['index'], rec['mode'], rec['warm']) == (i, MODES[i], i < 2)
            assert rec['phase_complete'] and rec['NoV_fixed']
            assert rec['prefix_rows'] == 32760 and rec['seed_calls'] == 64
            assert rec['tail_seed_calls'] == 8 and rec['total_projected_rows'] == 32768
            assert len(rec['queries']) == 8
            assert rec['final_clear']['entries'] == rec['final_clear']['bytes'] == rec['final_clear']['invalid'] == 0
            for cycle, query in enumerate(rec['queries']):
                assert query['cycle'] == cycle and query['past'] == 32760 + cycle
                result, mode = query['result'], MODES[i]
                assert result['canary'] and result['identity_valid'] and not result['numeric_alert']
                n = result['native']
                assert (n['count'], n['union_U'], n['past'], n['seeded']) == (1, 2048, 32760 + cycle, 32761 + cycle)
                assert n['mode'] and n['persistent_V_bytes'] == 0 and n['profile'] == 0
                assert n['selectedV_HOST_completed_expand_calls'] == 1 and n['selectedV_max_block_rows'] == 8192
                delta = result['plan_stats_delta']
                expected_delta = dict(created=int(mode and cycle == 0), hits=int(mode and cycle > 0), runs=mode, invalid=0)
                for k, v in expected_delta.items():
                    assert delta[k] == v, (rank, i, cycle, k, delta)
                for met in result['numerics']:
                    assert met['finite'] and all(math.isfinite(met[k]) for k in ['NRMSE', 'maxabs', 'reference_RMS'])
                    assert met['NRMSE'] <= .02
                    key = rank, cycle, met['name']
                    references.setdefault(key, met['SHA256'])
                    assert references[key] == met['SHA256']  # Observed, not a general quality requirement.
                    numeric.append(met)
                for k in free:
                    free[k].append(n[k])
        owners.append(d)
    owners.sort(key=lambda x: x['rank'])
    assert [d['rank'] for d in owners] == list(range(8))
    assert len({d['physical_PCI'] for d in owners}) == 8
    masks = [set(d['CPU_affinity']) for d in owners]
    assert sum(map(len, masks)) == len(set().union(*masks)) == 32
    assert len({d['synthetic_hidden_SHA256'] for d in owners}) == 1

    def measures(rec):
        qs = rec['queries']
        return dict(whole_phase_checked_s=rec['whole_phase_checked_wall_s'],
                    eight_query_checked_sequence_s=rec['eight_query_sequence_including_tail_seed_and_clear_s'],
                    complete_attention_CAPI_eight_sum_ms=sum(q['result']['native']['complete_attention_CAPI_ms'] for q in qs),
                    dispatch_eight_sum_ms=1000 * sum(q['result']['dispatch_wall_s'] for q in qs),
                    checked_query_eight_sum_ms=1000 * sum(q['checked_query_wall_s'] for q in qs),
                    first_query_CAPI_ms=qs[0]['result']['native']['complete_attention_CAPI_ms'],
                    remaining_seven_CAPI_sum_ms=sum(q['result']['native']['complete_attention_CAPI_ms'] for q in qs[1:]),
                    prefix_seed_s=rec['seed_wall_s'])
    per_rank = [[measures(r) for r in d['records']] for d in owners]
    # Maximum of rank-local durations; never subtract absolute clocks across ranks.
    sequences = [dict(index=i, mode=MODES[i], warm=i < 2,
                     max_rank_local={k: max(per_rank[r][i][k] for r in range(8)) for k in per_rank[0][i]}) for i in range(6)]
    def contrast(values):
        off, on = statistics.mean([values[2], values[5]]), statistics.mean([values[3], values[4]])
        return dict(OFF_mean=off, ON_mean=on, ON_over_OFF=on / off,
                    reduction_percent=100 * (1 - on / off),
                    OFF_endpoint_drift_percent=100 * (values[5] / values[2] - 1),
                    adjacent_ON_over_OFF=[values[3] / values[2], values[4] / values[5]])
    contrasts = {k: contrast([s['max_rank_local'][k] for s in sequences]) for k in per_rank[0][0]}
    same_rank = {k: [contrast([r[i][k] for i in range(6)]) for r in per_rank] for k in per_rank[0][0]}
    return dict(passed=True, actual_GPU='910371.751', actual_keeper='910407.856', actual_post='910371.752',
                original_archive=dict(path=archive_path, SHA256=archive_sha, bytes=archive_bytes),
                source_manifest_SHA256=MF, library_SHA256=LIB, original_files_SHA256={p: f['SHA256'] for p, f in files.items()},
                original_accounting=data['accounting'], elapsed_s=elapsed_s,
                allocated_service_card_seconds=8 * elapsed_s['910371.751'],
                logical_queries_per_owner=48, measured_queries_per_owner=32, warm_queries_per_owner=16,
                total_owner_queries=384, seed_calls_per_owner=432,
                normal_owners=8, source_members_verified=len(manifest['files_SHA256']),
                final_plan_stats_each_owner=final,
                finite_numeric_summaries=len(numeric), max_NRMSE=max(n['NRMSE'] for n in numeric),
                max_abs_error=max(n['maxabs'] for n in numeric), observed_matching_summary_hashes=True,
                sequences=sequences, formal_contrasts=contrasts, same_rank_formal_contrasts=same_rank,
                free_snapshots_bytes={k: dict(min=min(v), max=max(v)) for k, v in free.items()},
                load_wall_s=dict(min=min(d['load_wall_s'] for d in owners), max=max(d['load_wall_s'] for d in owners)),
                qualification='Actual layer38 TP8 fullNoV attention synthetic geometry only; no indexer/Model/routing/quality/online qualification.',
                timing='All six sequences retained, warm0/1 excluded from ABBA2/3/4/5. Max rank-local duration per sequence; no cross-clock subtraction. CAPI and dispatch nested, do not add. Checked wholephase includes prefix/tail seed, metrics/JSON/validation observer, clear; not observer-free production wall.',
                memory='Snapshots not allocator peaks; retained plan highwater19.5MiB/owner, no Model-fit claim.',
                decision='Retain component Native improvement for root review; checked wholephase has no robust reduction. No new GPU/Model authorized by this result.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seal', type=Path)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    q = read(args.seal, args.repo)
    args.output.write_text(json.dumps(q, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=q['passed'], output=str(args.output), SHA256=sha(args.output.read_bytes()), formal=q['formal_contrasts'])))
