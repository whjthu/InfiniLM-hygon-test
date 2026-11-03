"""Offline audit of sealed original public Kimi requests; stdlib, no device imports."""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(text):
    def unique(pairs):
        value = {}
        for k, v in pairs:
            assert k not in value, 'Duplicate JSON key'
            value[k] = v
        return value
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def parse_sse(raw):
    calls, identifiers = {}, set()
    finish = usage = None
    done = False
    for line in raw.decode().splitlines():
        if not line or line.startswith(':'):
            continue
        assert line.startswith('data: ')
        if line[6:] == '[DONE]':
            done = True
            break
        q = strict_json(line[6:])
        assert 'error' not in q
        if q.get('id'):
            identifiers.add(q['id'])
        if q.get('usage') is not None:
            usage = q['usage']
        for choice in q.get('choices', []):
            assert choice.get('index', 0) == 0
            finish = choice.get('finish_reason') or finish
            for part in choice.get('delta', {}).get('tool_calls', []):
                c = calls.setdefault(part.get('index', 0), dict(id='', function=dict(name='', arguments='')))
                c['id'] += part.get('id', '')
                for key in ('name', 'arguments'):
                    c['function'][key] += part.get('function', {}).get(key, '')
    assert done and len(identifiers) == 1 and list(calls) == [0]
    return dict(id=identifiers.pop(), choices=[dict(message=dict(tool_calls=[calls[0]]), finish_reason=finish)], usage=usage)


def audit(path):
    raw = path.read_bytes()
    assert sha(raw) == 'b236d665a613ffe57c072e8a33d53c2ecba6e2b3b30ca9c99767352450126f26'
    seal = json.loads(gzip.decompress(raw))
    files = {}
    for name, row in seal['files'].items():
        value = base64.b64decode(row['base64'], validate=True)
        assert len(value) == row['bytes'] and sha(value) == row['SHA256'], name
        files[name] = value
    read = lambda name: strict_json(files[name])
    watch, config, card = [read(n) for n in ('watch.json', 'deployment/config.json', 'deployment/FIXTURE_CARD.json')]
    assert watch['completed'] and watch['passed'] and not watch['forced_cleanup']
    assert watch['actual_GPU'] == '910380.572' and watch['actual_client'] == '910380.573' and watch['CPU_keeper'] == '910407.865'
    assert watch['publication_commit'] == '346fe342e0b95b60dda2b1769e658b32c4a76684'
    for name, expected in [('provider', 'fc7e8bc3e4b846929c7616477283071ab136c8595a5374584641a394b5ce6be6'),
                           ('deployment', 'ca86eb7b3ad9f1cf6167ad93796e3aef8f29688be11a449281850c9d14d22102'),
                           ('public-parent', '122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724')]:
        assert sha(files[name + '/SOURCE_MANIFEST.json']) == expected
        for relative, digest in read(name + '/SOURCE_MANIFEST.json')['files_SHA256'].items():
            assert sha(files[name + '/' + relative]) == digest
    model = read('provider/qualified-model.template.json')
    native = '8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
    assert config['Native_SHA256'] == model['native_DSO_SHA256'] == native
    assert sha(files['provider/qualified-model.template.json']) == config['model_manifest_SHA256']
    client = read('client/result.json')
    assert client['completed'] and client['passed'] and client['attempted_HTTP'] == client['completed_HTTP'] == len(client['rows']) == 2
    signatures, loading, masks, timings = [], [], [], []
    owners = []
    for node in range(4):
        report = read('run/Kimi-public-r428-public-kimi-20261103-a-node%d.json' % node)
        claim = read('run/claim-public-kimi-20261103-a-node%d.json' % node)
        assert report['completed'] and report['protocol_before_destroy_passed'] and report['all8_owned_workers_shutdown_completed']
        assert not report['cleanup_errors'] and not report['service_cleanup_errors']
        assert report['library_sha256'] == native and report['public_source_manifest_SHA256'] == watch['provider_source_SHA256']
        assert claim['actual_step'] == watch['actual_GPU'] and claim['node'] == report['node'] == node
        assert claim['source_manifest_SHA256'] == report['public_source_manifest_SHA256']
        assert report['TP'] == 16 and report['PP'] == 2 and report['PP_ranges'] == [[0, 46], [46, 93]]
        assert report['loaded_keys'] == report['actual_engine_weight_keys']['key_count']
        assert report['actual_allstage_weight_keys']['all93_layer_ids_exactly_once_across_PPstages']
        assert report['preload_memory_gate']['minimum_free_bytes'] > 4 * 1024**3
        for snapshot in ('actual_loaded_libraries_before_create', 'actual_loaded_libraries_after_server_import', 'actual_loaded_libraries'):
            for library, digest in model['loaded_library_SHA256'].items():
                assert report[snapshot][library]['SHA256'] == digest
        affinity = report['owner_affinity']
        assert affinity['verified'] and affinity['width'] == 4
        node_masks = []
        for owner in affinity['actual_owner_report']['owners']:
            assert owner['changes'] == 1
            subset = list(map(int, owner['verified_mask'].split(',')))
            assert len(subset) == 4
            node_masks.extend(subset)
        assert len(node_masks) == len(set(node_masks)) == 32
        masks.append(dict(node=node, distinct_logical_CPUs=len(set(node_masks))))
        rows = report['agent_requests']
        assert len(rows) == 2 and report['serving_protocol_complete']
        sig = []
        calls = 0
        for i, row in enumerate(rows):
            ids = row['generated_ids']
            assert ids and all(type(v) is int and v >= 0 for v in ids)
            assert ids[-1] in (163585, 163586) and not any(t in (163585, 163586) for t in ids[:-1])
            assert row['normal_reset'] and row['finish'] == 'eos' and not row.get('error') and not row.get('cleanup_errors')
            assert row['output_budget'] == [4096, 8192][i] and row['prefill_mode'] == 'W512'
            h = row['history_render_audit']
            assert h['qualified'] and h['RC_order_preserved'] and h['RC_one_contribution_per_source_field'] and not h['markers_sent_to_Native']
            assert h['Native_prompt_token_ids_SHA256'] == card['plan'][i]['current_prompt_IDs_SHA256']
            observations = row['long_prefill_observations']
            assert len(observations) == 1 and observations[0]['full_prompt_consumed']
            assert observations[0]['input_tokens_committed'] == row['input_tokens'] == [8635, 9613][i]
            assert row['input_tokens'] + row['output_budget'] <= 32768
            calls += math.ceil(row['input_tokens'] / 512) + len(ids) - 1
            sig.append([row['request_id'], h['Native_prompt_token_ids_SHA256'], ids])
            if node == 0:
                ticks = row['backend_token_monotonic']
                assert len(ticks) == len(ids) and all(math.isfinite(v) for v in ticks)
                assert all(a <= b for a, b in zip(ticks, ticks[1:]))
                assert row['started_monotonic'] <= ticks[0] <= ticks[-1] <= row['finished_monotonic']
                visible = [t for token, t in zip(ids, ticks) if token not in (163585, 163586)]
                timings.append(dict(index=i, input_tokens=row['input_tokens'], generated_including_EOS=len(ids), visible=len(visible),
                    backend_dequeue_to_first_emit_s=ticks[0] - row['started_monotonic'],
                    backend_visible_decode_tps=(len(visible) - 1) / (visible[-1] - visible[0]),
                    client_wall_s=client['rows'][i]['client_clock']['wall_s'],
                    client_first_semantic_delta_s=client['rows'][i]['client_clock']['first_payload_delta_s']))
        assert report['agent_native_forward_calls'] == calls == 215
        signatures.append(sig)
        loading.append(dict(node=node, keys=report['loaded_keys'], create_s=report['create_seconds'], load_s=report['loading_seconds'],
                            host_loader_phase_profile=report['host_loader_phase_profile']))
        owners.append(report)
    assert all(s == signatures[0] for s in signatures)
    seen = set()
    for i in range(2):
        body, encoded, original = read('client/request-%d.json' % i), read('encoding/case-%d.json' % i), read('fixtures/source-%d.json' % i)
        original = original['body'] if i == 0 else original
        if i:
            original = copy.deepcopy(original)
            original.update(stream=True, stream_options=dict(include_usage=True))
        assert body == original == encoded['body']
        assert sha(files['client/request-%d.json' % i]) == card['plan'][i]['actual_request_SHA256']
        prompt_ids = encoded['prompt_token_ids']
        assert len(prompt_ids) == [8635, 9613][i]
        assert sha(json.dumps(prompt_ids, separators=(',', ':')).encode()) == card['plan'][i]['current_prompt_IDs_SHA256']
        tools = [strict_json(m['content']) for m in body['messages'] if m['role'] == 'tool' and 'records' in m['content']]
        assert len(tools) == 1
        records = tools[0]
        values = {v['id']: v['value'] for v in records['records']}
        assert len(values) == len(records['records'])
        selected = [values[k] for k in records['query_ids']]
        assert len(selected) == 4 and all(type(v) is int for v in selected)
        expected = dict(a=selected[0], b=selected[1]) if i == 0 else dict(path='final.json', content=dict(nonce=records['nonce'], total=sum(selected)))
        wire = files['client/response-%d.wire' % i]
        response = parse_sse(wire) if i else strict_json(wire)
        choice = response['choices'][0]
        assert choice['finish_reason'] == 'tool_calls' and len(choice['message']['tool_calls']) == 1
        call = choice['message']['tool_calls'][0]
        assert call['function']['name'] == ['calculate', 'write_file'][i]
        arguments = strict_json(call['function']['arguments'])
        if i:
            arguments['content'] = strict_json(arguments['content'])
            assert type(arguments['content']['total']) is int
        else:
            assert all(type(arguments[k]) is int for k in ('a', 'b'))
        assert arguments == expected
        history = {c['id'] for m in body['messages'] for c in m.get('tool_calls', [])}
        assert call['id'] and call['id'] not in history
        assert all(m['tool_call_id'] in history for m in body['messages'] if m['role'] == 'tool')
        c = client['rows'][i]
        assert response['id'] == signatures[0][i][0] == c['request_id'] and response['id'] not in seen
        seen.add(response['id'])
        assert sha(wire) == c['wire_SHA256']
        usage = response['usage']
        assert all(type(usage[k]) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        assert usage['prompt_tokens'] == len(prompt_ids) and usage['completion_tokens'] == len(signatures[0][i][2])
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    assert len(watch['post']) == 4
    for post in watch['post']:
        assert not post['kfd_pids'] and not post['matching_component_processes'] and not post['expected_PIDs_still_exist']
        assert post['ports_bindable'] == {'19743': True, '26743': True, '43844': True}
    for step in ('910380.572', '910380.573', '910380.574', '910407.865'):
        account = [r for r in seal['accounting'] if r[0] == step]
        assert len(account) == 1 and account[0][2:4] == ['COMPLETED', '0:0']
    return dict(passed=True, archive_SHA256=sha(raw), raw_files_verified=len(files), actual_GPU='910380.572', actual_client='910380.573',
        keeper='910407.865', post='910380.574', original_epoch=watch['original_GPU_start_unix'], external_requests=2,
        normal_four_leaders_32_workers=True, all_four_request_signatures_equal=True, natural_EOS_and_integer_tool_gold=True,
        Native_CAPI_per_leader=215, timings=timings, loading=loading, affinity=masks, post_clean=True,
        new_causal_speedup=False, incremental_client_token_SLO=False, full_benchmark=False,
        scope='Qualified8df TP16PP2/32cards/32768 ordinary public HTTP+thinking SSE two complete tool histories only. P0 buffered text; backend node0 dequeue/reset-to-emitter clock is not client token delivery or GPU-event time.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = audit(args.archive)
    with args.output.open('x', encoding='utf8') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: value[k] for k in ('passed', 'raw_files_verified', 'external_requests', 'Native_CAPI_per_leader', 'timings')}))
