"""Read sealed r445 originals only. No service imports, GPU or subprocess calls.

Adapted from tools/audit_pd382.py. A successful audit validates the evidence;
quality, lifecycle and measured performance are separate returned conclusions.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

NATIVE = 'bc1421f05553444c3d1f4e96515d6a0fd12ee39dc37583b60dc94da52b10cf92'
FIXTURE_SHA = {'L': '43377e6a70efdf8960989559c957241f9e725f8a2c2a55b9bf438c331275cb6d',
               'S': '33d2f8c70c967e10177f03eb1b44ff81e2f5629e5aec44650450e98d466ef37e'}
GOLD = {'left.json': {'nonce': '74c6b6493c9da03cf59935f73adb08c0', 'sum': 202076},
        'right.json': {'nonce': 'c69771de85bab17b01a2871597d3fcd1', 'sum': 275061}, 'total': 477137}
NODES = {'P': 'a02r4n12', 'D': 'a02r4n13'}
H = lambda b: hashlib.sha256(b).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def answer(content, fixture):
    assert isinstance(content, str)
    if fixture == 'S':
        matches = re.findall(r'####\s*([ABCD])\b', content)
        return bool(matches and matches[-1] == 'A'), 'Original last #### letter'
    text = content.strip()
    presentation = 'bare_JSON'
    if text.startswith('```'):
        match = re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```', text)
        if not match:
            return False, 'Rejected fence or surrounding prose'
        text, presentation = match.group(1), 'whole_json_fence'
    try:
        value = decode(text)
    except (ValueError, TypeError):
        return False, 'Parse failure'
    return typed_equal(value, GOLD), presentation


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * p
    lo = int(point)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (point - lo)


def finite(value):
    assert type(value) in (int, float) and math.isfinite(value)
    return value


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(finite(a), finite(b), rel_tol=1e-11, abs_tol=1e-8), (a, b)


def load_seal(path):
    raw = Path(path).read_bytes()
    seal = decode(raw)
    files, hashes = {}, {}
    for ref in seal['files']:
        name = PurePosixPath(ref['path']).name
        assert name and name not in files
        # Existing PD382 sealing uses gzip_base64. No remote reads or implicit
        # fallback to mutable running files are permitted by this reader.
        data = gzip.decompress(base64.b64decode(ref['gzip_base64'], validate=True))
        assert len(data) == ref['bytes'] and H(data) == ref['SHA256'], name
        files[name], hashes[name] = data, H(data)
    return seal, files, hashes, H(raw)


def stage(report, request, operation):
    assert report['actual_Native'] is True and report['completed'] is True
    assert report['request'] == request and report['operation'] == operation
    assert report['normal_reset_before'] is True and report['normal_reset_after'] is True
    assert finite(report['ended_local_monotonic']) >= finite(report['started_local_monotonic'])


def token_metrics(row, source, request):
    ids = source['generated_token_ids']
    budget = request['output_budget']
    assert ids and len(ids) <= budget and all(type(i) is int and 0 <= i < 129280 for i in ids)
    natural = source['finish_reason'] == 'eos'
    assert source['finish_reason'] in ('eos', 'length') and source['natural_EOS'] is natural
    assert 1 not in ids[:-1]
    assert ids[-1] == 1 if natural else len(ids) == budget and 1 not in ids
    correct, presentation = answer(source['quality']['content'], request['fixture'])
    assert source['quality']['task_correct'] is correct
    timeline, client = row['client_token_ID_timeline'], row['client']
    assert client['error'] is None and client['completed'] is True
    assert [v['seq'] for v in timeline] == list(range(len(ids)))
    assert [v['token_id'] for v in timeline] == ids
    assert client['generated_received'] == len(ids)
    assert all(finite(v['received_monotonic']) <= finite(v['delivered_monotonic']) for v in timeline)
    assert all(a['delivered_monotonic'] <= b['delivered_monotonic'] for a, b in zip(timeline, timeline[1:]))
    admitted = finite(row['actual_admission_monotonic'])
    close(admitted, client['admitted_monotonic'])
    assert row['scheduled_arrival_monotonic'] <= admitted <= timeline[0]['delivered_monotonic']
    ticks = [v['delivered_monotonic'] for v in timeline if v['token_id'] != 1]
    assert client['visible_tokens'] == len(ticks)
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    # Delivery can share a clock tick; no finite token rate is invented then.
    rate = (len(ticks) - 1) / (ticks[-1] - ticks[0]) if len(ticks) > 1 and ticks[-1] > ticks[0] else None
    ttft = ticks[0] - admitted if ticks else None
    close(ttft, client['client_token_ID_TTFT_s'])
    close(rate, client['client_token_ID_decode_tok_s'])
    for key, value in [('p50', quantile(gaps, .5)), ('p95', quantile(gaps, .95)),
                       ('p99', quantile(gaps, .99)), ('max', max(gaps) if gaps else None)]:
        close(value, client['client_token_ID_ITL_' + key + '_s'])
    first_command = row['commands'][0]
    assert first_command['operation'] == ('co' if request['mode'] == 'co' else 'prefill')
    queue = finite(first_command['client_send_monotonic']) - admitted
    assert queue >= 0
    end = admitted + finite(client['token_ID_completion_wall_s'])
    assert end >= timeline[-1]['delivered_monotonic']
    return dict(generated=len(ids), visible=len(ticks), task_correct=correct, presentation=presentation,
                natural_EOS=natural, capped=not natural, generated_IDs_SHA256=H(json.dumps(ids, separators=(',', ':')).encode()),
                client_token_ID_TTFT_s=ttft, client_token_ID_decode_tok_s=rate,
                client_token_ID_ITL_p50_s=quantile(gaps, .5), client_token_ID_ITL_p95_s=quantile(gaps, .95),
                client_token_ID_ITL_p99_s=quantile(gaps, .99), client_token_ID_ITL_max_s=max(gaps) if gaps else None,
                client_admission_to_dispatch_queue_s=queue,
                scheduled_arrival_to_actual_admission_s=admitted - row['scheduled_arrival_monotonic'],
                client_completion_wall_s=client['token_ID_completion_wall_s'], client_completion_monotonic=end), ids


def elapsed_seconds(value):
    parts = value.split('-')
    h, m, s = map(int, parts[-1].split(':'))
    return (int(parts[0]) * 86400 if len(parts) == 2 else 0) + h * 3600 + m * 60 + s


def audit(path, *, run, source_sha, gpu, keeper, plan_path, plan_sha):
    assert re.fullmatch('910376[.][0-9]+', gpu) and re.fullmatch('910407[.][0-9]+', keeper)
    assert re.fullmatch('[0-9a-f]{64}', source_sha) and re.fullmatch('[0-9a-f]{64}', plan_sha)
    seal, files, hashes, archive_sha = load_seal(path)
    assert seal['run'] == run and seal['all_frozen_source_members_unchanged'] is True
    raw_plan = Path(plan_path).read_bytes()
    assert H(raw_plan) == plan_sha
    plan = decode(raw_plan)
    assert plan['context'] == 32768 and plan['prefill_chunk'] == 1024 and plan['planned_external_requests'] == 22
    assert plan['planned_role_request_lifecycles']['total_stage_lifecycles'] == 32
    assert set(plan['fixtures']) == {'L', 'S'}
    for key, spec in plan['fixtures'].items():
        assert spec['SHA256'] == FIXTURE_SHA[key]
        assert spec['actual_prompt_tokens'] == (8489 if key == 'L' else 150)
        assert spec['output_budget'] == (128 if key == 'L' else 1024)
    expected = [(g, i) for g in plan['planned_groups'] for i in range(g['requests'])]
    assert len(expected) == 22
    assert [g['id'] for g in plan['planned_groups']] == list(range(8))
    assert [g['mode'] for g in plan['planned_groups']] == ['co', 'co', 'PD', 'PD', 'co', 'PD', 'PD', 'co']
    assert sum(g['requests'] for g in plan['planned_groups'] if g['phase'] == 'warm') == 6
    assert all(g['phase'] == 'capacity' and g['requests'] == 4 for g in plan['planned_groups'][4:])
    assert plan['planned_groups'][4]['fixtures'] == plan['planned_groups'][5]['fixtures'] == ['L', 'S', 'S', 'L']
    assert plan['planned_groups'][6]['fixtures'] == plan['planned_groups'][7]['fixtures'] == ['S', 'L', 'L', 'S']
    assert all(plan['planned_groups'][i]['co_nodes'] == ['P', 'D', 'P', 'D'] for i in (4, 7))
    def original(name):
        return decode(files[name])
    watch = original('Deep-PD-r445-' + run + '-watch.json')
    document = original('Deep382-' + run + '-client.json')
    client = document.get('partial', document)
    roles = {role: original('Deep382-' + run + '-' + role + '.json') for role in ('P', 'D')}
    assert watch['actual_GPU'] == gpu and watch['source_manifest_SHA256'] == source_sha
    assert document['run'] == run and document['source_manifest_SHA256'] == source_sha
    assert document['all_service_cards'] == 16 and client['planned_external'] == len(client['requests']) == 22
    ready = document['role_ready_receipts']
    for role in ('P', 'D'):
        r = ready[role]
        assert roles[role]['node'] == NODES[role] and roles[role]['actual_step'] == gpu
        assert r['actual_Native'] is True and r['Native_SHA256'] == NATIVE
        assert (r['TP'], r['PP'], r['rows'], r['capacity']) == (8, 1, 1, 32768)
        assert r['fixed_HC_epilogue_mode'] == 0 and len(r['actual_devices']) == 8
        assert r['actual_PD_symbol_providers'] and all(v['SHA256'] == NATIVE for v in r['actual_PD_symbol_providers'].values())
        for key, spec in plan['fixtures'].items():
            inp = r['inputs'][key]
            assert inp['fixture_SHA256'] == spec['SHA256'] and inp['actual_prompt_tokens'] == spec['actual_prompt_tokens']
            assert inp['single_runtime_encode'] is True and inp['no_service_request_enqueued'] is True
            assert re.fullmatch('[0-9a-f]{64}', inp['prompt_IDs_SHA256'])
            assert inp['prompt_IDs_SHA256'] == ready['P']['inputs'][key]['prompt_IDs_SHA256']
            if spec['actual_prompt_token_ids_SHA256'] is not None:
                assert inp['prompt_IDs_SHA256'] == spec['actual_prompt_token_ids_SHA256']
    rows, pending = [], []
    for index, row in enumerate(client['requests']):
        req = row['request']
        g, ordinal = expected[index]
        fixture = g['fixtures'][ordinal]
        spec = plan['fixtures'][fixture]
        assert (req['index'], req['epoch'], req['ordinal_in_group']) == (index, index + 1, ordinal)
        assert (req['group'], req['mode'], req['phase'], req['fixture']) == (g['id'], g['mode'], g['phase'], fixture)
        assert req['co_role'] == (g['co_nodes'][ordinal] if g['mode'] == 'co' else 'P')
        assert req['arrival_offset_s'] == g['arrival_offsets_s'][ordinal]
        assert req['past'] == spec['actual_prompt_tokens'] and req['output_budget'] == spec['output_budget']
        assert req['prompt_IDs_SHA256'] == ready['P']['inputs'][fixture]['prompt_IDs_SHA256']
        early = bool(row.get('D_skipped_early_P_EOS'))
        source_key = req['co_role'] + '_report' if req['mode'] == 'co' else 'P_report' if early else 'D_report'
        if not row.get('client', {}).get('completed') or source_key not in row:
            pending.append(dict(index=index, fixture=fixture, mode=req['mode'], admitted=row.get('admitted', False), error=row.get('client', {}).get('error')))
            continue
        source = row[source_key]
        stage(source, req, 'co' if req['mode'] == 'co' else 'prefill' if early else 'import_decode')
        item, ids = token_metrics(row, source, req)
        item.update(index=index, group=req['group'], ordinal=ordinal, phase=req['phase'], mode=req['mode'], fixture=fixture,
                    co_role=req['co_role'], output_budget=req['output_budget'], prompt_tokens=req['past'])
        prefill = row['P_report'] if req['mode'] == 'PD' else source
        if req['mode'] == 'PD':
            stage(prefill, req, 'prefill')
        f = prefill['Native_forward_summary']
        assert f['prefill_input_tokens'] == req['past'] and f['prefill_calls'] == math.ceil(req['past'] / 1024)
        trace = prefill['prefill_trace']
        cursor = 0
        for part in trace:
            count = min(1024, req['past'] - cursor)
            assert part['kind'] == 'prefill' and (part['past'], part['tokens'], part['total']) == (cursor, count, cursor + count)
            cursor += count
        assert cursor == req['past'] and len(trace) == f['prefill_calls']
        item.update(Native_prefill_local_s=finite(f['prefill_wall_s']),
                    Native_decode_local_s=finite(source['Native_forward_summary'].get('decode_wall_s', 0.)),
                    prefill_stage_local_enclosure_s=prefill['ended_local_monotonic'] - prefill['started_local_monotonic'])
        if early:
            assert req['mode'] == 'PD' and ids == [1] and 'D_report' not in row
            assert [e['role'] for e in row['client_token_ID_timeline']] == ['P']
            item['early_P_EOS_without_migration'] = True
        elif req['mode'] == 'PD':
            d = source
            assert f['decode_calls'] == 0 and prefill['generated_token_ids_before_handoff'] == [ids[0]]
            df = d['Native_forward_summary']
            assert df['prefill_calls'] == df['prefill_input_tokens'] == 0 and df['decode_calls'] == len(ids) - 1
            assert d['decode_trace'] == [dict(kind='decode_summary', calls=len(ids) - 1, first_past=req['past'], last_total=req['past'] + len(ids) - 1)]
            assert [e['role'] for e in row['client_token_ID_timeline']] == ['P'] + ['D'] * (len(ids) - 1)
            e, a, b = d['Native_import_evidence'], prefill['transfer'], d['transfer']
            assert d['actual_Native_import_published'] is True and d['all8_import_publish_contract_completed'] is True
            assert e['all8_validate_then_commit'] is True and not e['poisoned'] and e['shards'] == 8 and e['blocks'] == 2184
            assert a['destination_published'] is True and b['published'] is True and prefill['source_snapshot_released_after_whole_ticket_ACK'] is True
            assert a['payload_bytes'] == b['payload_bytes'] == e['payload_bytes'] > 0
            assert len(a['block_SHA256']) == len(b['block_SHA256']) == 2184 and a['block_SHA256'] == b['block_SHA256']
            assert all(re.fullmatch('[0-9a-f]{64}', v) for v in a['block_SHA256'])
            ticks = row['client_token_ID_timeline']
            item.update(actual_complete_state_handoff=True, D_prompt_tokens_recomputed=0, state_bytes=e['payload_bytes'],
                        first_P_to_first_D_client_gap_s=ticks[1]['delivered_monotonic'] - ticks[0]['delivered_monotonic'],
                        source_snapshot_local_s=finite(prefill['export_snapshot_local_wall_s']),
                        source_connect_export_ACK_local_s=finite(prefill['source_connect_export_ACK_local_wall_s']),
                        destination_prepare_import_local_s=finite(d['prepare_unpublished_import_local_wall_s']),
                        destination_receive_verify_commit_ACK_local_s=finite(d['receive_validate_commit_ACK_local_wall_s']),
                        source_destination_transfer_clocks_overlap_do_not_sum=True)
        else:
            assert source['Native_forward_summary']['decode_calls'] == len(ids) - 1
            assert all(e['role'] == req['co_role'] for e in row['client_token_ID_timeline'])
        rows.append(item)
    groups = []
    for group in client['groups']:
        gid = group['id']
        spec = plan['planned_groups'][gid]
        assert spec['id'] == gid and group['all_planned_requests_terminal'] is True
        selected = [r for r in rows if r['group'] == gid]
        assert len(selected) == group['requests'] == spec['requests']
        start, end = finite(group['client_group_start_monotonic']), finite(group['client_group_end_monotonic'])
        wall = end - start
        assert wall > 0 and max(r['client_completion_monotonic'] for r in selected) <= end
        for rawrow in client['requests']:
            if rawrow['request']['group'] == gid:
                close(rawrow['scheduled_arrival_monotonic'], start + rawrow['request']['arrival_offset_s'])
                assert all(start <= e['client_received_monotonic'] <= end for e in rawrow['events'])
        visible = sum(r['visible'] for r in selected)
        good = sum(r['visible'] for r in selected if r['task_correct'] and r['natural_EOS'])
        groups.append(dict(id=gid, phase=spec['phase'], mode=spec['mode'], fixture_order=spec['fixtures'], requests=len(selected),
                           client_group_wall_s=wall, visible_output_tokens=visible, useful_gold_EOS_tokens=good,
                           serving_cards=16, visible_tokens_s_per_card=visible / wall / 16,
                           useful_gold_EOS_tokens_s_per_card=good / wall / 16, resident_card_seconds=16 * wall,
                           queue_TTFT_p95_s=quantile([r['client_token_ID_TTFT_s'] for r in selected if r['client_token_ID_TTFT_s'] is not None], .95)))
    assert len({g['id'] for g in groups}) == len(groups)
    by = {g['id']: g for g in groups}
    pairs = []
    for coid, pdid in [(4, 5), (7, 6)]:
        if coid in by and pdid in by:
            co, pd = by[coid], by[pdid]
            assert co['fixture_order'] == pd['fixture_order'] and co['mode'] == 'co' and pd['mode'] == 'PD'
            pairs.append(dict(co_group=coid, PD_group=pdid, fixture_order=co['fixture_order'],
                              PD_over_co_wall=pd['client_group_wall_s'] / co['client_group_wall_s'],
                              PD_over_co_visible_per_card=pd['visible_tokens_s_per_card'] / co['visible_tokens_s_per_card'],
                              PD_over_co_useful_per_card=pd['useful_gold_EOS_tokens_s_per_card'] / co['useful_gold_EOS_tokens_s_per_card'] if co['useful_gold_EOS_tokens_s_per_card'] else None))
    drift = {}
    for mode, ends in [('co', (4, 7)), ('PD', (5, 6))]:
        if all(g in by for g in ends):
            drift[mode] = dict(group_ids=ends, later_over_earlier_wall=by[ends[1]]['client_group_wall_s'] / by[ends[0]]['client_group_wall_s'],
                               includes_fixture_order_change=True, pure_same_order_repeat_drift_qualified=False)
    aggregates = {}
    for mode in ('co', 'PD'):
        gs = [g for g in groups if g['phase'] == 'capacity' and g['mode'] == mode]
        if gs:
            wall = sum(g['client_group_wall_s'] for g in gs)
            aggregates[mode] = dict(complete_both_cohorts=len(gs) == 2, requests=sum(g['requests'] for g in gs),
                                    group_wall_s=wall, all16card_seconds=16 * wall,
                                    visible_tokens_s_per_card=sum(g['visible_output_tokens'] for g in gs) / wall / 16,
                                    useful_gold_EOS_tokens_s_per_card=sum(g['useful_gold_EOS_tokens'] for g in gs) / wall / 16)
    account = {}
    for r in seal['accounting']:
        r = r.split('|') if isinstance(r, str) else r
        assert r[0] not in account
        account[r[0]] = r
    assert gpu in account and keeper in account
    owners = {}
    for role, r in roles.items():
        obs = r.get('owner_exit_observation', {}).get('owners', [])
        owners[role] = bool(len(obs) == 8 and len({v['tid'] for v in obs}) == 8 and not any(v['task_exists_after_destroy'] for v in obs)
                            and r.get('normal_actual8owner_destroy') and r.get('normal_stage_loop_close') and r.get('normal_control_socket_close'))
    post = watch.get('post', watch.get('post_after_failure', []))
    post_clean = bool(len(post) == 2 and {r['node'] for r in post} == set(NODES.values()) and all(r['KFD_available'] and not r['kfd_pids'] and not r['owned_process_PIDs'] and all(r['ports_bindable'].values()) for r in post))
    # Keeper passed also includes answer quality. Normal shutdown must remain
    # independently reportable if a fully drained run has a wrong answer.
    lifecycle = bool(watch.get('completed') and watch.get('srun_exit_code') == 0 and document.get('execution_shutdown_completed')
                     and all(owners.values()) and post_clean and account[gpu][2:4] == ['COMPLETED', '0:0'] and account[keeper][2:4] == ['COMPLETED', '0:0'])
    if lifecycle:
        assert not pending and len(rows) == 22 and len(groups) == 8
        assert all(r.get('completed') and r.get('passed') for r in roles.values())
        assert sum(len(r['stage_reports']) for r in roles.values()) == 32 - sum(bool(r.get('early_P_EOS_without_migration')) for r in rows)
        for role in ('P', 'D'):
            stopped = document['role_stopped_receipts'][role]
            assert stopped['stage_reports'] == roles[role]['stage_reports']
            assert stopped['normal_actual8owner_destroy'] is True and stopped['normal_stage_loop_close'] is True
            ref = watch['role_reports'][role]
            assert ref['actual_step'] == gpu and PurePosixPath(ref['path']).name == 'Deep382-' + run + '-' + role + '.json'
            assert ref['SHA256'] == hashes[PurePosixPath(ref['path']).name]
        assert len({r['actual_step'] for r in post}) == 1
        for r in post:
            assert account[r['actual_step']][2:4] == ['COMPLETED', '0:0']
    lifecycle_checks = dict(watch_completed=bool(watch.get('completed')), watch_passed=bool(watch.get('passed')),
                            srun_exit_code=watch.get('srun_exit_code'), client_execution_shutdown_completed=bool(document.get('execution_shutdown_completed')),
                            original_client_error=document.get('error'), watch_error=watch.get('error'),
                            role_errors={r: roles[r].get('error') for r in roles}, forced_scancel=watch.get('forced_scancel'))
    return dict(audit_passed=True, scope='Two frozen fixtures, queued external C4; Native active C1 per role, no batch4 or publicSSE claim',
                run=run, actual_GPU=gpu, CPU_keeper=keeper, Native_SHA256=NATIVE, source_manifest_SHA256=source_sha,
                original_seal_SHA256=archive_sha, original_file_SHA256=hashes, plan_SHA256=plan_sha,
                planned_external=22, completed_checked=len(rows), pending=pending, requests=rows, groups=groups,
                same_order_pairs=pairs, endpoint_changes_not_pure_drift=drift, formal_16card_aggregates=aggregates,
                formal_complete=len(pairs) == 2 and not pending, original_client_completed=bool(document.get('completed')),
                all22_gold_natural_EOS=bool(len(rows) == 22 and all(r['task_correct'] and r['natural_EOS'] for r in rows)),
                owner_destroyed=owners, independent_post_clean=post_clean, normal_lifecycle_qualified=lifecycle, lifecycle_checks=lifecycle_checks,
                role_load_local_s={r: ready[r]['load_seconds'] for r in ready}, accounting_GPU=account[gpu], accounting_keeper=account[keeper],
                all16card_residency_seconds=16 * elapsed_seconds(account[gpu][4]),
                warm_group_card_seconds=sum(g['resident_card_seconds'] for g in groups if g['phase'] == 'warm'),
                Model_new_general_quality_qualified=False, public_HTTP_SSE_SLO_qualified=False, GPU_busy_qualified=False,
                limits=['All planned failed/unstarted requests remain in denominator; no speed conclusion from partial pairs.',
                        'Useful throughput counts only task-correct natural-EOS visible tokens; all-role residency includes idle cards.',
                        'P and D local durations may overlap; never subtract remote monotonic clocks or sum both transfer enclosures.',
                        'Endpoint changes include the two frozen request orders; there is no independent same-order drift repeat.',
                        'Private token-ID clocks are not per-user public text delivery or a15tok/s SLO.',
                        'Native local forward durations and role spans are not exclusive kernels, GPU busy or allocator attribution.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('seal', type=Path)
    for name in ('run', 'source-sha', 'gpu', 'keeper', 'plan', 'plan-sha'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.seal, run=a.run, source_sha=a.source_sha, gpu=a.gpu, keeper=a.keeper, plan_path=a.plan, plan_sha=a.plan_sha)
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if a.output:
        a.output.write_text(raw, encoding='utf-8')
    print(raw, end='')


if __name__ == '__main__':
    main()
