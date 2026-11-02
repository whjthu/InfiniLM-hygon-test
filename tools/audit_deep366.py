"""Root read-only audit of the sealed six-request Model archive (no GPU/imports)."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics

ARCHIVE_SHA = '1278a801bf459af812b574f6568f798dee04a1e0075bb87c3fc9d8b9ed10f721'
NATIVE_SHA = '976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, ('duplicate key', key)
        result[key] = value
    return result


def parse(text):
    def invalid(value):
        raise ValueError(value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid)


def audit(path, metadata_path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA
    outer = parse(raw)
    a = outer['archive']
    report, client, fixture = a['report'], a['client'], a['fixture']
    # Recompute the tool answer from actual opaque IDs, irrespective of reply order.
    tool_calls = {}
    for message in fixture['body']['messages']:
        for call in message.get('tool_calls', []):
            assert call['id'] not in tool_calls
            tool_calls[call['id']] = parse(call['function']['arguments'])['path']
    gold = {}
    tool_order = []
    for message in fixture['body']['messages']:
        if message['role'] != 'tool':
            continue
        tool_order.append(message['tool_call_id'])
        payload = parse(message['content'])
        records = {record['id']: record['value'] for record in payload['records']}
        assert len(records) == len(payload['records'])
        values = [records[key] for key in payload['query_ids']]
        assert all(type(value) is int for value in values)
        gold[tool_calls[message['tool_call_id']]] = {
            'nonce': payload['nonce'], 'sum': sum(values)}
    assert tool_order == list(reversed(tool_calls))
    gold['total'] = sum(value['sum'] for value in gold.values())
    assert gold == fixture['expected_answer']
    assert report['Native_SHA256'] == NATIVE_SHA
    assert report['completed'] and report['passed']
    assert report['actual_HTTP'] == report['actual_Native'] == 6
    assert client['actual_HTTP_attempts'] == client['actual_HTTP_completed'] == 6
    assert client['actual_Native_completed'] == 6 and client['no_retries']
    assert len(report['requests']) == len(client['requests']) == len(a['wires']) == 6
    rows, total_forwards, previous = [], 0, None
    for index, (native, measured, wire) in enumerate(zip(
            report['requests'], client['requests'], a['wires'])):
        mode = [0, 1, 0, 1, 1, 0][index]
        assert native['HC_mode'] == measured['hc_epilogue_mode'] == mode
        assert native['admitted_index'] == measured['index'] == index
        assert native['normal_reset_before'] and native['normal_reset_after']
        assert native['prompt_tokens'] == measured['prompt_tokens'] == 8489
        assert native['prefill_binding']['prefill_chunk'] == measured['chunk'] == 1024
        assert native['admission']['history_render_audit']['Native_prompt_token_ids_SHA256'] == fixture['actual_prompt_token_ids_SHA256']
        assert wire['request']['messages'] == fixture['body']['messages']
        assert wire['request']['max_tokens'] == 128
        ids = native['generated_token_ids']
        assert len(ids) == native['actual_generated_tokens'] == 98 and ids[-1] == 1
        assert ids.count(1) == 1 and native['native_finish_reason'] == 'eos_token'
        choice = wire['response']['choices'][0]
        assert wire['response']['id'] == native['request_id']
        assert choice['finish_reason'] == 'stop'
        content = choice['message']['content'].strip()
        fenced = content.startswith('```json\n') and content.endswith('\n```')
        result = parse(content[8:-4] if fenced else content)
        assert result == gold
        assert type(result['total']) is int
        assert all(type(result[key]['sum']) is int for key in tool_calls.values())
        ticks = [tick for tick, token in zip(native['backend_token_monotonic'], ids) if token != 1]
        assert len(native['backend_token_monotonic']) == 98 and len(ticks) == 97
        assert all(math.isfinite(tick) for tick in ticks)
        assert all(right > left for left, right in zip(ticks, ticks[1:]))
        rate = 96 / (ticks[-1] - ticks[0])
        assert math.isclose(rate, measured['backend_emitter_decode_tok_s'], rel_tol=1e-12)
        prefills = [entry for entry in native['Native_continuation'] if entry['kind'] == 'prefill']
        past, tokens = 0, []
        for entry in prefills:
            count = min(1024, 8489 - past)
            assert entry['past'] == past and entry['tokens'] == count
            past += count
            assert entry['total'] == past
            tokens.append(count)
        assert past == 8489 and len(prefills) == 9
        tokens += [1] * 97
        coverage = native['HC_forward_coverage']
        assert len(coverage) == len(tokens) == 106
        aggregate = [0] * 7
        for entry, count in zip(coverage, tokens):
            expected = [0, 0, 86, 86 if count == 1 else 0,
                        86 if count == 1 and mode == 1 else 0, 86 if count != 1 else 0, 0]
            assert entry['mode'] == mode and entry['input_tokens'] == count
            assert entry['per_owner_delta'] == [expected] * 8
            assert entry['candidate_completed_branch'] == (count == 1 and mode == 1)
            aggregate = [x + y for x, y in zip(aggregate, expected)]
        before, after = native['HC_counters_before_request'], native['HC_counters_after_request']
        assert len(before) == len(after) == 8
        assert previous is None or before == previous
        assert all([y - x for x, y in zip(left, right)] == aggregate for left, right in zip(before, after))
        previous = after
        forward = native['Native_forward_host_summary']
        assert forward['prefill_calls'] == 9 and forward['decode_calls'] == 97
        total_forwards += len(coverage)
        rows.append(dict(index=index, warmup=index < 2, mode=mode, natural_EOS=True,
                         visible_tokens=97, task_correct=True, single_complete_JSON_fence=fenced,
                         Native_prefill_s=forward['prefill_s'], Native_decode_s=forward['decode_s'],
                         HTTP_wall_s=measured['client_HTTP_wall_s'], backend_visible_decode_tok_s=rate,
                         backend_TTFT_s=ticks[0] - native['started_monotonic']))
    owners = report['owner_exit_observation']['owners']
    assert len(owners) == len({owner['tid'] for owner in owners}) == 8
    assert {owner['device'] for owner in owners} == set(range(8))
    assert not any(owner['task_exists_after_destroy'] for owner in owners)
    assert report['normal_server_shutdown'] and report['normal_Native_shutdown']
    assert not report['cleanup']['forced_cleanup'] and not report['cleanup']['errors']
    assert not report['cleanup']['HTTP_thread_alive_after_cleanup']
    assert a['legacy']['engine_worker_shutdown_completed'] and total_forwards == 636
    watch = a['watch']
    assert watch['terminal'][2:4] == ['COMPLETED', '0:0']
    assert watch['post_empty'] and not watch['post']['kfd_pids'] and watch['post']['HTTP43882_bindable']
    metadata_bytes = metadata_path.read_bytes()
    metadata = parse(metadata_bytes)
    for parent, handle in [('910374', '910374.1295'), ('910407', '910407.735')]:
        matching = [row.split('|') for row in metadata['Slurm'][parent] if row.startswith(handle + '|')]
        assert len(matching) == 1 and matching[0][2:4] == ['COMPLETED', '0:0']
    assert metadata['Native_SHA256'] == NATIVE_SHA
    providers = metadata['actual_provider']['actual_new_Native_providers']
    assert len(providers) == 9 and all(value['SHA256'] == NATIVE_SHA for value in providers.values())
    paired = {}
    for field in ['Native_prefill_s', 'Native_decode_s', 'HTTP_wall_s', 'backend_visible_decode_tok_s', 'backend_TTFT_s']:
        off = statistics.mean(rows[i][field] for i in [2, 5])
        on = statistics.mean(rows[i][field] for i in [3, 4])
        paired[field] = dict(OFF=off, ON=on, ON_over_OFF=on/off,
                             change_percent=100*(on/off-1),
                             OFF_endpoint_drift_percent=100*(rows[5][field]/rows[2][field]-1),
                             ON_endpoint_drift_percent=100*(rows[4][field]/rows[3][field]-1))
    return dict(root_independent_audit_passed=True, actual_GPU='910374.1295', CPU_keeper='910407.735',
                scope='One8489token realtool request; TP8PP1; 2warm+1ABBA; output128; Native976f same coverage OFF/ON',
                Native_SHA256=NATIVE_SHA, local_original_archive_SHA256=ARCHIVE_SHA,
                remote_original_archive_path=outer['remote_archive_path'],
                remote_original_archive_SHA256=outer['remote_archive_SHA256'],
                provider_lifecycle_metadata_SHA256=hashlib.sha256(metadata_bytes).hexdigest(),
                rows=rows, paired=paired, real_tool_gold_recomputed=gold,
                all6_full_input_correct_natural_EOS=True, all8_normal_owner_threads_exited=True,
                actual_forward_calls=total_forwards, all8_each86_HC_coverage=True,
                counter_scope='CPU branch completed after drain, not hardware counters',
                post=watch['post'], strict15_cases=sum(row['backend_visible_decode_tok_s'] > 15 for row in rows),
                adopted='Scoped Model decode and HTTP improvement; opt-in retained, default OFF',
                limitations=['One request and one ABBA, no broad quality noninferiority',
                             'Same instrumented976f OFF, not uninstru afc6 timing',
                             'Buffered HTTP; no client incremental TTFT/online SLO qualification',
                             'Prefill difference not attributed to T1-only hook',
                             'Memory snapshots are not peak; no GPU utilization claim',
                             'Single complete JSON fence semantic PASS; not strict bare JSON format PASS'],
                raw_payloads_published=False, no_new_GPU_probe=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--metadata', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.metadata)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['paired']))
