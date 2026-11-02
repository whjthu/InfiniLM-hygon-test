"""Audit original PD382 JSON snapshots without submitting work or loading models.

Recompute private token-ID client metrics and compare P/D state digests and
continuation traces. Live rows remain pending. Final qualification additionally
requires the original keeper's terminal, normal owner exits and independent post.
No cross-node monotonic subtraction or GPU busy inference is made.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

G = Path(__file__).resolve().parents[1]
PARSER = G / 'code/bw1000/Deep-HC-T1-Model-r366/owner-v2'
sys.path.insert(0, str(PARSER))
from content_json import parse_final_object

RUN = 'pd-20261102-a'
MF = '8c65930f1b436456b54745fad0ea5ba65c2bc4bff82fb98ce6eb8e9f62734370'
NATIVE = 'bc1421f05553444c3d1f4e96515d6a0fd12ee39dc37583b60dc94da52b10cf92'
GOLD = {'left.json': {'nonce': '74c6b6493c9da03cf59935f73adb08c0', 'sum': 202076},
        'right.json': {'nonce': 'c69771de85bab17b01a2871597d3fcd1', 'sum': 275061}, 'total': 477137}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values); point = (len(values) - 1) * fraction; lo = int(point)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (point - lo)


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8), (a, b)


def original(snapshot, name):
    ref = snapshot[name]; data = ref['data']
    assert digest((json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()) == ref['SHA256'], name
    return data


def checked_stage(report, request, operation):
    assert report['actual_Native'] and report['completed']
    assert report['request'] == request and report['operation'] == operation
    assert report['normal_reset_before'] and report['normal_reset_after']
    assert math.isfinite(report['started_local_monotonic'])
    assert report['started_local_monotonic'] <= report['ended_local_monotonic']


def checked_answer(report):
    ids = report['generated_token_ids']
    assert ids and len(ids) <= 128 and 1 not in ids[:-1]
    natural = report['finish_reason'] == 'eos'
    assert report['finish_reason'] in ['eos', 'length'] and report['natural_EOS'] == natural
    assert ids[-1] == 1 if natural else (len(ids) == 128 and 1 not in ids)
    try:
        parsed, presentation = parse_final_object(report['quality']['content'])
        correct = canonical(parsed) == canonical(GOLD)
    except (ValueError, TypeError):
        correct, presentation = False, None
    assert report['quality']['task_correct'] == correct
    return ids, presentation, correct, natural


def audit(path):
    raw = Path(path).read_bytes(); snapshot = json.loads(raw)
    assert snapshot['run'] == RUN
    sealed = 'files' in snapshot
    if sealed:
        archive = snapshot
        snapshot = dict(observed_UTC=archive['observed_UTC'], run=RUN, accounting=archive['accounting'])
        assert archive['all_frozen_source_members_unchanged']
        directory = Path(path).parent / 'Deep382-PD836-sealed'
        for ref in archive['files']:
            name = Path(ref['path']).name
            local = directory / name
            data = local.read_bytes()
            assert len(data) == ref['bytes'] and digest(data) == ref['SHA256'], name
            if name.endswith('.json'):
                snapshot[name] = dict(SHA256=ref['SHA256'], data=json.loads(data))
    plan_raw = (G / 'code/bw1000/Deep-PD-disaggregation-r382/BENCHMARK_PLAN.json').read_bytes()
    assert digest(plan_raw) == 'ff0902853e1d1eb78b1cbb732f3d9e1b70f36581c4f970485dbd9de279366680'
    plan = json.loads(plan_raw)
    expected = [(g, n) for g in plan['planned_groups'] for n in range(g['requests'])]
    watch = original(snapshot, 'Deep-PD-r382-' + RUN + '-watch.json')
    client_document = original(snapshot, 'Deep382-' + RUN + '-client.json')
    client = dict(client_document.get('partial', client_document))
    for key in ['source_manifest_SHA256', 'role_ready_receipts']:
        client[key] = client_document[key]
    roles = {role: original(snapshot, 'Deep382-' + RUN + '-' + role + '.json') for role in ['P', 'D']}
    assert watch['actual_GPU'] == '910376.836' and watch['source_manifest_SHA256'] == MF
    assert client['source_manifest_SHA256'] == MF and client['planned_external'] == len(client['requests']) == 15
    for role, node in [('P', 'a02r4n12'), ('D', 'a02r4n13')]:
        assert roles[role]['node'] == node and roles[role]['actual_step'] == '910376.836'
        ready = client['role_ready_receipts'][role]
        assert ready['actual_Native'] and ready['Native_SHA256'] == NATIVE
        assert (ready['TP'], ready['PP'], ready['rows'], ready['capacity']) == (8, 1, 1, 32768)
        assert ready['fixed_HC_epilogue_mode'] == 0 and len(ready['actual_devices']) == 8
        assert all(v['SHA256'] == NATIVE for v in ready['actual_PD_symbol_providers'].values())
    rows, pending = [], []
    for index, row in enumerate(client['requests']):
        request = row['request']; assert request['index'] == index and request['epoch'] == index + 1
        group, ordinal = expected[index]
        assert (request['group'], request['phase'], request['mode']) == (group['id'], group['phase'], group['mode'])
        assert request['co_role'] == (group['co_nodes'][ordinal] if group['mode'] == 'co' else 'P')
        assert request['arrival_offset_s'] == group.get('arrival_offsets_s', [0.] * group['requests'])[ordinal]
        assert request['prompt_IDs_SHA256'] == plan['fixture']['actual_prompt_token_ids_SHA256']
        assert request['past'] == 8489 and request['output_budget'] == 128
        if not row.get('client', {}).get('completed') or not row.get('P_report' if request['mode'] == 'PD' else request['co_role'] + '_report'):
            pending.append(dict(index=index, admitted=row.get('admitted', False), error=row.get('client', {}).get('error')))
            continue
        cc = row['client']; timeline = row['client_token_ID_timeline']
        assert cc['error'] is None and cc['generated_received'] == len(timeline)
        assert [r['seq'] for r in timeline] == list(range(len(timeline)))
        assert all(type(r['token_id']) is int and 0 <= r['token_id'] < 129280 for r in timeline)
        assert all(math.isfinite(r[k]) for r in timeline for k in ['received_monotonic', 'delivered_monotonic'])
        assert all(r['received_monotonic'] <= r['delivered_monotonic'] for r in timeline)
        assert all(a['delivered_monotonic'] <= b['delivered_monotonic'] for a, b in zip(timeline, timeline[1:]))
        early = bool(row.get('D_skipped_early_P_EOS'))
        source = row[request['co_role'] + '_report'] if request['mode'] == 'co' else row['P_report'] if early else row['D_report']
        checked_stage(source, request, 'co' if request['mode'] == 'co' else 'prefill' if early else 'import_decode')
        ids, presentation, correct, natural = checked_answer(source)
        assert ids == [r['token_id'] for r in timeline]
        assert len(ids) == cc['generated_received'] and len(ids) - int(natural) == cc['visible_tokens']
        ticks = [r['delivered_monotonic'] for r in timeline if r['token_id'] != 1]
        assert not ticks or ticks[0] >= cc['admitted_monotonic']
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        ttft = ticks[0] - cc['admitted_monotonic'] if ticks else None
        rate = (len(ticks) - 1) / (ticks[-1] - ticks[0]) if len(ticks) > 1 else None
        if ttft is not None: close(ttft, cc['client_token_ID_TTFT_s'])
        else: assert cc['client_token_ID_TTFT_s'] is None
        if rate is not None: close(rate, cc['client_token_ID_decode_tok_s'])
        for name, value in [('p50', quantile(gaps, .5)), ('p95', quantile(gaps, .95)),
                            ('p99', quantile(gaps, .99)), ('max', max(gaps) if gaps else None)]:
            if value is not None: close(value, cc['client_token_ID_ITL_' + name + '_s'])
        item = dict(index=index, group=request['group'], phase=request['phase'], mode=request['mode'],
                    generated=len(ids), visible=len(ticks), task_correct=correct, natural_EOS=natural,
                    capped=not natural,
                    generated_IDs_SHA256=digest(json.dumps(ids, separators=(',', ':')).encode()),
                    client_token_ID_TTFT_s=ttft, client_token_ID_decode_tok_s=rate,
                    client_token_ID_ITL_max_s=max(gaps) if gaps else None,
                    client_token_ID_ITL_p95_s=quantile(gaps, .95), client_token_ID_ITL_p99_s=quantile(gaps, .99),
                    recorded_client_completion_s=cc['token_ID_completion_wall_s'],
                    queue_included_from_client_admission=True, HTTP_SSE_text_SLO_qualified=False)
        close(cc['admitted_monotonic'], row['actual_admission_monotonic'])
        command = row['commands'][0]
        assert command['operation'] == ('co' if request['mode'] == 'co' else 'prefill')
        queue_s = command['client_send_monotonic'] - cc['admitted_monotonic']
        assert queue_s >= 0
        item.update(client_admission_to_dispatch_queue_s=queue_s,
                    scheduled_arrival_to_actual_admission_s=row['actual_admission_monotonic'] - row['scheduled_arrival_monotonic'],
                    Native_decode_local_s=source['Native_forward_summary']['decode_wall_s'])
        if early:
            assert request['mode'] == 'PD' and ids == [1] and 'D_report' not in row
            assert [r['role'] for r in timeline] == ['P']
            item.update(early_P_EOS_without_D=True, actual_complete_state_handoff=False)
        elif request['mode'] == 'PD':
            P, D = row['P_report'], row['D_report']; checked_stage(P, request, 'prefill')
            assert P['Native_forward_summary']['prefill_input_tokens'] == 8489
            assert P['Native_forward_summary']['prefill_calls'] == 9 and P['Native_forward_summary']['decode_calls'] == 0
            assert D['Native_forward_summary']['prefill_input_tokens'] == D['Native_forward_summary']['prefill_calls'] == 0
            assert D['Native_forward_summary']['decode_calls'] == len(ids) - 1
            assert D['decode_trace'] == [dict(kind='decode_summary', calls=len(ids) - 1, first_past=8489, last_total=8489 + len(ids) - 1)]
            assert P['generated_token_ids_before_handoff'] == [ids[0]]
            assert [r['role'] for r in timeline] == ['P'] + ['D'] * (len(ids) - 1)
            assert D['actual_Native_import_published'] and D['all8_import_publish_contract_completed']
            evidence = D['Native_import_evidence']
            assert evidence['all8_validate_then_commit'] and not evidence['poisoned']
            assert (evidence['shards'], evidence['blocks'], evidence['payload_bytes']) == (8, 2184, 372424784)
            a, b = P['transfer'], D['transfer']
            assert a['destination_published'] and b['published']
            assert a['payload_bytes'] == b['payload_bytes'] == evidence['payload_bytes']
            assert len(a['block_SHA256']) == len(b['block_SHA256']) == 2184 and a['block_SHA256'] == b['block_SHA256']
            assert P['source_snapshot_released_after_whole_ticket_ACK']
            item.update(actual_complete_state_handoff=True, D_prompt_tokens_recomputed=0,
                        Native_prefill_local_s=P['Native_forward_summary']['prefill_wall_s'],
                        state_bytes=a['payload_bytes'], all2184_block_digests_match=True,
                        source_snapshot_local_s=P['export_snapshot_local_wall_s'],
                        source_connect_export_ACK_local_s=P['source_connect_export_ACK_local_wall_s'],
                        destination_receive_verify_commit_ACK_local_s=D['receive_validate_commit_ACK_local_wall_s'],
                        transfer_clocks_overlap_do_not_sum=True)
        else:
            assert source['Native_forward_summary']['prefill_input_tokens'] == 8489
            assert source['Native_forward_summary']['prefill_calls'] == 9
            assert source['Native_forward_summary']['decode_calls'] == len(ids) - 1
            assert all(r['role'] == request['co_role'] for r in timeline)
            item['Native_prefill_local_s'] = source['Native_forward_summary']['prefill_wall_s']
        rows.append(item)
    groups = []
    for group in client['groups']:
        assert group['all_planned_requests_terminal']
        selected = [r for r in rows if r['group'] == group['id']]
        assert len(selected) == group['requests']
        wall = group['client_group_end_monotonic'] - group['client_group_start_monotonic']
        assert math.isfinite(wall) and wall > 0
        visible = sum(r['visible'] for r in selected)
        correct_tokens = sum(r['visible'] for r in selected if r['task_correct'] and r['natural_EOS'])
        fast_tokens = sum(r['visible'] for r in selected if r['task_correct'] and r['natural_EOS'] and (r['client_token_ID_decode_tok_s'] or 0) > 15)
        groups.append(dict(id=group['id'], phase=group['phase'], mode=group['mode'], requests=len(selected),
                           client_group_wall_s=wall, visible_output_tokens=visible, serving_cards=16,
                           output_tokens_s=visible / wall, output_tokens_s_per_all_service_card=visible / wall / 16,
                           task_correct_natural_EOS_output_tokens=correct_tokens,
                           task_correct_natural_EOS_tokens_s_per_all_service_card=correct_tokens / wall / 16,
                           strict15_task_correct_tokens_s=fast_tokens / wall,
                           resident_card_seconds=16 * wall, includes_idle_P_or_D=True))
    comparisons = {}
    by_group = {g['id']: g for g in groups}
    for phase, pairs in [('latency', [(2, 3), (5, 4)]), ('capacity', [(6, 7), (9, 8)])]:
        paired = []
        for co_id, pd_id in pairs:
            if co_id not in by_group or pd_id not in by_group:
                continue
            co, pd = by_group[co_id], by_group[pd_id]
            assert co['phase'] == pd['phase'] == phase and co['mode'] == 'co' and pd['mode'] == 'PD'
            paired.append(dict(co_group=co_id, PD_group=pd_id,
                               PD_over_co_group_wall_ratio=pd['client_group_wall_s'] / co['client_group_wall_s'],
                               PD_over_co_output_per_card_ratio=pd['output_tokens_s_per_all_service_card'] / co['output_tokens_s_per_all_service_card']))
        comparisons[phase] = dict(complete_two_pair_comparison=len(paired) == 2, pairs=paired)
        if len(paired) == 2:
            co = [by_group[i] for i, _ in pairs]; pd = [by_group[j] for _, j in pairs]
            comparisons[phase].update(co_mean_group_wall_s=statistics.mean(g['client_group_wall_s'] for g in co),
                                      PD_mean_group_wall_s=statistics.mean(g['client_group_wall_s'] for g in pd),
                                      PD_over_co_group_wall_ratio=sum(g['client_group_wall_s'] for g in pd) / sum(g['client_group_wall_s'] for g in co),
                                      PD_over_co_aggregate_output_per_card_ratio=(sum(g['visible_output_tokens'] for g in pd) / sum(g['client_group_wall_s'] for g in pd)) / (sum(g['visible_output_tokens'] for g in co) / sum(g['client_group_wall_s'] for g in co)))
    pipeline = []
    for gid in [7, 8]:
        cohort = [r for r in client['requests'] if r['request']['group'] == gid]
        if gid not in by_group:
            continue
        first, second = cohort
        dispatch = second['commands'][0]['client_send_monotonic']
        d_import = next(c['client_send_monotonic'] for c in first['commands'] if c['operation'] == 'import')
        d_done = next(e['client_received_monotonic'] for e in first['events'] if e['role'] == 'D' and e['kind'] == 'stage_done')
        assert d_import < dispatch < d_done and second['commands'][0]['other_role_busy_at_dispatch']
        pipeline.append(dict(group=gid, P2_dispatched_while_D1_busy=True,
                             client_observed_D1_remaining_enclosure_s=d_done - dispatch,
                             actual_GPU_busy_overlap_fraction='UNKNOWN; command/receipt envelopes only'))
    accounting = {r.split('|')[0]: r.split('|') for r in snapshot['accounting']}
    gpu = accounting['910376.836']; keeper = accounting['910407.771']
    elapsed = gpu[4].split('-')
    h, minute, second = map(int, elapsed[-1].split(':'))
    gpu_elapsed_s = (int(elapsed[0]) * 86400 if len(elapsed) == 2 else 0) + h * 3600 + minute * 60 + second
    final = bool(watch.get('completed'))
    post = watch.get('post', watch.get('post_after_failure', []))
    post_clean = bool(len(post) == 2 and all(r['KFD_available'] and not r['kfd_pids'] and not r['owned_process_PIDs'] and all(r['ports_bindable'].values()) for r in post))
    owners_destroyed = False
    if final:
        owners_destroyed = all(r.get('normal_actual8owner_destroy') and r.get('normal_stage_loop_close') and r.get('normal_control_socket_close') for r in roles.values())
        for role in roles.values():
            owners = role['owner_exit_observation']['owners']
            assert len(owners) == 8 and len({r['tid'] for r in owners}) == 8
            assert not any(r['task_exists_after_destroy'] for r in owners)
    clean = False
    if final and watch.get('terminal', [None] * 4)[2:4] == ['COMPLETED', '0:0'] and watch.get('srun_exit_code') == 0:
        assert len(rows) == 15 and not pending and len(groups) == 10
        assert watch['srun_exit_code'] == 0 and watch['terminal'][2:4] == ['COMPLETED', '0:0']
        for role in roles.values():
            assert role['completed'] and role['passed'] and role['normal_actual8owner_destroy']
            assert role['normal_control_socket_close'] and role['normal_stage_loop_close']
            owners = role['owner_exit_observation']['owners']
            assert len(owners) == 8 and len({r['tid'] for r in owners}) == 8
            assert not any(r['task_exists_after_destroy'] for r in owners)
        assert len(watch['post']) == 2
        for post in watch['post']:
            assert post['KFD_available'] and not post['kfd_pids'] and not post['owned_process_PIDs']
            assert all(post['ports_bindable'].values())
        clean = keeper[2:4] == ['COMPLETED', '0:0'] and gpu[2:4] == ['COMPLETED', '0:0']
    formal = [r for r in rows if r['phase'] != 'warm']
    return dict(observed_UTC=snapshot['observed_UTC'], run=RUN, actual_GPU='910376.836', actual_CPU_keeper='910407.771',
                original_snapshot_SHA256=digest(raw), original_snapshot=str(Path(path).resolve()),
                raw_JSON_hashes_verified=True, per_request_timeline_gold_independently_reparsed=True,
                original_client_completed=bool(client_document.get('completed')),
                original_client_error=client_document.get('error'),
                keeper_error=watch.get('error'), role_errors={r: d.get('error') for r, d in roles.items()},
                forced_scancel=watch.get('forced_scancel'), original_srun_exit_code=watch.get('srun_exit_code'),
                all16_actual_owners_destroyed=owners_destroyed, independent_post_clean=post_clean,
                post_observations=post,
                planned_external=15, completed_checked=len(rows), pending=pending, requests=rows, groups=groups,
                formal_comparisons=comparisons, pipeline_enclosures=pipeline,
                role_parameter_load_local_s={role: ready['load_seconds'] for role, ready in client['role_ready_receipts'].items()},
                formal_completed=len(formal), strict15_private_client=sum((r['client_token_ID_decode_tok_s'] or 0) > 15 for r in formal),
                original_keeper_terminal=final, final_normal16owner_and_post_clean=clean,
                accounting_GPU=gpu, accounting_CPU_keeper=keeper,
                all16card_step_elapsed_card_seconds=16 * gpu_elapsed_s,
                elapsed_card_seconds_are_residency_not_GPU_busy=True,
                final_scope_qualified=bool(clean and len(formal) == 12 and all(r['task_correct'] and r['natural_EOS'] for r in rows)),
                full15_measurement_and_quality_pass=bool(len(rows) == 15 and len(groups) == 10 and len(formal) == 12 and all(r['task_correct'] and r['natural_EOS'] for r in rows)),
                all_service_cards=16, public_HTTP_SSE_SLO_qualified=False,
                scope='Historical fixed complete8489tool prompt/output128/ctx32768;samePDDSO HC0, C1 and two-request cohorts;noonlinearrivalSLO or broadqualityproof',
                limits=['Warm results are not formal speedup samples.',
                        'Private token-ID client metrics are not public HTTP/SSE text metrics.',
                        'D2H/snapshot and source RPC envelope are labeled; source/destination RPC enclosures overlap and must not be added.',
                        'Ordinary-host staged path; no explicit pinned allocation or deviceDirect/RDMA qualification.',
                        'Group card-seconds include idle resident roles; load/warm/final destroy require separate accounting.',
                        'No cross-node clock subtraction, GPU utilization, kernel attribution, or physical peak inference.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('snapshot', type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.snapshot), ensure_ascii=False, indent=2))
