"""Offline global512 audit from immutable original files, never shard averages.

Bundle JSON maps logical file names to {path,bytes,SHA256}. Paths are relative
to the bundle, confined below it; weights, code and network are never loaded.
Only the three explicitly pinned, reviewed stdlib consumers are imported.
Missing/in-flight shards retain their original planned denominators.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

PROVIDER = '822f91e71088cf6c63828f442b956ab6d59744becdbf23433b8ddeef2769f4bb'
NATIVE = 'a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
PINS = {
    'scoring': '5965c7282e005f60c12cc0f2f719bafa9a87f084e748eb2d148e7ad27d53428d',
    'coverage': '38e4a90ccda81b2fdeb10bbde700a62ea057668b832642e1d61e36935adc0deb',
    'wire_client_r290': '045ea460cd6a95277ae60fe787860960160b9eae4537dcafe02ee2a34dd3af14',
}
INPUTS = {
    'inputs/cases.json': 'c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38',
    'inputs/requests-private.json': '5b90259f33666c99e5c74f41733899aaaeaf545481faf57e259c1dd233bb420d',
    'inputs/encoding.json': '04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a',
}
H = lambda b: hashlib.sha256(b).hexdigest()
C = lambda q: json.dumps(q, ensure_ascii=False, separators=(',', ':')).encode()


def consumers():
    out = {}
    for name, digest in PINS.items():
        path = Path(__file__).resolve().parent.parent / (name + '.py')
        assert H(path.read_bytes()) == digest, ('Reviewed consumer changed', name)
        spec = importlib.util.spec_from_file_location('offline450_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out[name] = module
    return out


class Bundle:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.base = self.manifest.parent
        self.raw = self.manifest.read_bytes()
        q = json.loads(self.raw)
        assert q['schema_version'] == 1 and q['provider_source_SHA256'] == PROVIDER
        self.recording = q.get('diagnostic_recording_only', False)
        assert type(self.recording) is bool
        self.files = q['files']
        self.used = {}

    def has(self, name):
        return name in self.files

    def read(self, name):
        pin = self.files[name]
        path = (self.base / pin['path']).resolve()
        assert path.is_relative_to(self.base), ('Outside original bundle', name)
        raw = path.read_bytes()
        assert len(raw) == pin['bytes'] and H(raw) == pin['SHA256'], name
        self.used[name] = dict(bytes=len(raw), SHA256=H(raw))
        return raw

    def json(self, name):
        return json.loads(self.read(name))


def response_row(entry, case, expected_body, recorded, owner, body, wire, modules):
    assert body == C(expected_body) and H(body) == entry['request_body_SHA256']
    assert recorded['logical_index'] == entry['logical_index']
    assert recorded['case_id'] == entry['case_id'] and recorded['mode'] == entry['mode']
    assert recorded['request_body_SHA256'] == H(body) and recorded['wire_SHA256'] == H(wire)
    assert recorded['prompt_IDs_SHA256'] == entry['prompt_IDs_SHA256']
    proof = modules['coverage'].validate(owner, entry)
    assert proof == recorded['actual_owner_coverage']
    binding = owner['r450_binding_proof']
    assert binding['actual_HC_groups_per_owner'] == 86 and binding['HC0_fixed']
    assert set(binding['providers'].values()) == {'/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'}
    assert recorded['prompt_tokens'] == entry['prompt_tokens']
    ids = owner['generated_token_ids']
    assert all(type(v) is int and 0 <= v < 129280 for v in ids)
    assert recorded['actual_generated_including_EOS'] == len(ids)
    assert recorded['generated_IDs_SHA256'] == H(C(ids))
    natural = owner['native_finish_reason'] == 'eos_token'
    if natural:
        assert ids[-1] == 1 and ids.count(1) == 1
    else:
        assert len(ids) == 4096 and 1 not in ids and owner['native_finish_reason'] == 'length'
    ticks = owner['backend_token_monotonic']
    assert len(ticks) == len(ids) and all(math.isfinite(t) for t in ticks)
    assert all(b > a for a, b in zip(ticks, ticks[1:]))
    history = owner['admission']['history_render_audit']
    assert history['qualified'] and history['prequeue_bound']
    assert history['Native_prompt_token_ids_SHA256'] == entry['prompt_IDs_SHA256']
    assert not history['pending_thinking_prefix_open']
    assert owner['admission']['text_delivery'] == 'P1'
    assert owner['final_metadata']['buffered_P0_text'] is False
    content, tools, parse_failed = '', False, False
    if recorded['HTTP_status'] != 200:
        parse_failed = True
    else:
        try:
            message, finish, usage = modules['wire_client_r290'].sse_message(wire)
        except (AssertionError, ValueError, KeyError, TypeError, RuntimeError):
            parse_failed = True
        else:
            assert usage == dict(prompt_tokens=entry['prompt_tokens'], completion_tokens=len(ids),
                                 total_tokens=entry['prompt_tokens'] + len(ids))
            identifiers = {part['id'] for line in wire.splitlines()
                           if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]'
                           for part in [json.loads(line[6:])] if part.get('id')}
            assert identifiers == {owner['request_id']}
            assert finish == recorded['HTTP_finish']
            assert finish in [('stop' if natural else 'length'), 'tool_calls']
            content, tools = message.get('content') or '', bool(message.get('tool_calls'))
    score = modules['scoring'].grade(case, content, natural_EOS=natural,
                                      cap_hit=not natural, unexpected_tool=tools)
    if parse_failed:
        score.update(correct=False, parse_failure=True)
    for key, value in score.items():
        assert recorded[key] == value, ('Original wire regrade differs', entry['logical_index'], key)
    assert bool(recorded['public_parse_error']) == parse_failed
    return dict(logical_index=entry['logical_index'], case_id=case['id'], suite=case['suite'],
                mode=entry['mode'], **score)


def analyze(manifest, deployment_sha):
    bundle = Bundle(manifest)
    modules = consumers()
    for name, digest in INPUTS.items():
        assert H(bundle.read(name)) == digest, name
    cases = bundle.json('inputs/cases.json')['cases']
    assets = bundle.json('inputs/requests-private.json')
    encoded = bundle.json('inputs/encoding.json')
    assert encoded['passed'] and encoded['actual_Deep_codec'] and encoded['actual_encodings'] == 256
    frozen = assets['logical_plan']
    assert len(cases) == 256 and len(frozen) == 512
    assert H(C(frozen)) == '6b2f8f7dd85c61408dbfb830009fa796c7c302d3a761a5271e27d100fb900d34'
    by_case = {r['id']: r for r in cases}
    by_request = {r['case_id']: r for r in assets['requests']}
    lookup = {r['logical_index']: r for r in frozen}
    assert len(by_case) == len(by_request) == 256 and len(lookup) == 512
    deployment = bundle.json('deployment/SOURCE_MANIFEST.json')
    assert H(bundle.read('deployment/SOURCE_MANIFEST.json')) == deployment_sha
    assert deployment['provider_source_SHA256'] == PROVIDER
    shards = bundle.json('deployment/SHARDS.json')
    assert len(shards) == 8 and [s['index'] for s in shards] == list(range(8))
    assert deployment['files_SHA256']['SHARDS.json'] == H(bundle.read('deployment/SHARDS.json'))
    assigned, verified, observations, normal = [], [], [], []
    for shard in shards:
        j = shard['index']
        key = 'j%d/' % j
        plan_raw = bundle.read('deployment/' + shard['plan'])
        config_raw = bundle.read('deployment/' + shard['config'])
        assert H(plan_raw) == deployment['plans_SHA256'][str(j)]
        assert H(config_raw) == deployment['configs_SHA256'][str(j)]
        plan, config = json.loads(plan_raw), json.loads(config_raw)
        rows = plan['assigned']
        assert len(rows) == 64 and plan['warmups'] == 0
        assert config['r450_plan'] == rows and config['r450_frozen512'] == plan['frozen512'] == frozen
        expected_cases = [c for suite in ['gsm8k', 'mmlu']
                          for c in [c for c in cases if c['suite'] == suite][j::8]]
        expected = [next(r for r in frozen if r['case_id'] == c['id'] and r['mode'] == mode)
                    for i, c in enumerate(expected_cases) for mode in ([0, 1] if i % 2 == 0 else [1, 0])]
        assert rows == expected and config['node'] == shard['node']
        assert config['Native_SHA256'] == NATIVE and config['HC_mode'] == config['r450_r366_fixed'] == 0
        assert all(r == lookup[r['logical_index']] for r in rows)
        assigned.extend(r['logical_index'] for r in rows)
        required = [key + s for s in ['watch.json', 'client.json', 'owner.json', 'ready.json']]
        if not all(bundle.has(s) for s in required):
            observations.append(dict(shard=j, state='UNVERIFIED_OR_NOT_STARTED', verified_requests=0))
            continue
        watch, client, owner, ready = [bundle.json(s) for s in required]
        assert deployment['GPU_GO'] and deployment['Model_GO'] and config['execute_authorized']
        identity = ready['identity']
        assert watch['shard'] == j and watch['provider_source_SHA256'] == PROVIDER
        assert watch['deployment_source_SHA256'] == deployment_sha
        assert identity['actual_step'] == watch['actual_GPU'] and identity['run'] == shard['run']
        assert identity['source_manifest_SHA256'] == PROVIDER
        assert identity['provider_config_SHA256'] == H(config_raw) and owner['identity'] == identity
        assert ready['Model_READY'] and ready['Native_internal_owner_count'] == 8
        assert owner['Native_SHA256'] == NATIVE and owner['HC_mode'] == 0
        assert owner['TP'] == 8 and owner['PP'] == owner['cache_rows'] == 1 and owner['cache_capacity'] == 32768
        assert owner['loaded_key_count'] == 67569
        assert client['global_planned_HTTP'] == 512 and client['assigned_HTTP'] == 64
        assert len(client['rows']) == client['completed_HTTP'] <= client['attempted_HTTP'] <= 64
        assert [r['logical_index'] for r in client['rows']] == [r['logical_index'] for r in rows[:len(client['rows'])]]
        for entry, recorded in zip(rows, client['rows']):
            found = [r for r in owner['requests'] if r.get('r450_logical_index') == entry['logical_index']]
            assert len(found) == 1
            stem = key + '%04d' % entry['logical_index']
            verified.append(response_row(entry, by_case[entry['case_id']], by_request[entry['case_id']]['body'],
                           recorded, found[0], bundle.read(stem + '.request.json'), bundle.read(stem + '.wire'), modules))
        complete = client['completed'] and client['execution_passed'] and client['completed_HTTP'] == 64
        clean = False
        if watch['completed'] and watch['passed'] and complete and bundle.has(key + 'closed.json'):
            closed = bundle.json(key + 'closed.json')
            front, stopped = bundle.json(key + 'frontend.json'), bundle.json(key + 'STOP.json')
            assert front['identity'] == stopped['identity'] == identity
            assert closed['identity'] == identity and closed['normal'] and closed['engine_destroy_returned_zero']
            assert closed['observed_all8_threads_exited'] and owner['completed'] and owner['normal_Native_shutdown']
            assert owner['engine_worker_shutdown_completed'] and owner['observed_all8_threads_exited']
            affinity = owner['owner_affinity_after_weight_load']['actual_owner_report']['owners']
            assert len(affinity) == 8 and {r['device'] for r in affinity} == set(range(8))
            masks = [int(cpu) for r in affinity for cpu in r['verified_mask'].split(',')]
            assert len(masks) == len(set(masks)) == 32 and all(r['changes'] == 1 for r in affinity)
            threads = owner['owner_thread_observation']
            assert threads['status'] == 'OBSERVED' and len(threads['owners']) == 8
            assert len({r['TID'] for r in threads['owners']}) == 8
            assert watch['normal8internalowner_destroy'] and watch['lifecycle_scope_pass'] and not watch['forced_cleanup']
            post = watch['post']
            assert len(post) == 1 and post[0]['node'] == shard['node']
            assert not post[0]['kfd_pids'] and not post[0]['matching_component_processes'] and not post[0]['expected_PIDs_still_exist']
            assert post[0]['ports_bindable'] == {'43882': True}
            clean = True
        normal.append(clean)
        observations.append(dict(shard=j, verified_requests=len(client['rows']), complete=complete,
                                 normal_GPU_and_post=clean, actual_GPU=watch['actual_GPU'],
                                 CPU_keeper=watch['CPU_keeper'], actual_client=watch['actual_client']))
    assert len(assigned) == len(set(assigned)) == 512 and set(assigned) == set(lookup)
    assert len({r['logical_index'] for r in verified}) == len(verified)
    summary = modules['scoring'].summarize(cases, verified)
    # CPU keeper accounting is independently required for full normal lifecycle.
    keepers_normal = False
    if bundle.has('accounting.json'):
        accounting = bundle.json('accounting.json')
        rows_by_id = {r[0]: r for r in accounting}
        assert len(rows_by_id) == len(accounting)
        keepers_normal = len(observations) == 8 and all(
            all(h in rows_by_id and rows_by_id[h][2:4] == ['COMPLETED', '0:0']
                for h in [o.get('actual_GPU'), o.get('CPU_keeper'), o.get('actual_client')]) for o in observations)
    return dict(original_bundle_SHA256=H(bundle.raw), deployment_manifest_SHA256=deployment_sha,
                verified_original_files=bundle.used, verified_requests=len(verified), planned_requests=512,
                missing_or_unverified_requests=512-len(verified), per_shard=observations, summary=summary,
                complete_all_planned_quality=len(verified)==512,
                diagnostic_recording_only=bundle.recording,
                actual_Model_quality_data_complete=not bundle.recording and len(verified)==512,
                normal_lifecycle_qualified=not bundle.recording and len(normal)==8 and all(normal) and keepers_normal,
                default_promotion=False, noninferiority_proven=False, hardware_maximum_or_online_SLO_proven=False,
                full_benchmark=False, actual_terminal_lifecycle_branch_must_be_checked_against_original_run=True,
                no_new_HTTP=True, no_new_GPU=True, aggregate_method='Original row union and wire regrade; never average per-shard summaries')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--deployment-sha', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = analyze(args.bundle, args.deployment_sha)
    Path(args.output).write_bytes((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k not in ['verified_original_files', 'summary']}, ensure_ascii=False))
