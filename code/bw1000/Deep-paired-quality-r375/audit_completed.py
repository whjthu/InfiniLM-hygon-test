"""Read-only independent wire/gold/Native lifecycle audit; reuse root r374 CI.

Run after the existing four watchers reach terminal. No model imports or post
probes. Raw questions/answers/IDs remain on the server; only metadata is output.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import sys
import traceback
from fractions import Fraction

R = pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
S = R / 'incoming/Deep-paired-quality-r375/owner'
LABEL = 'Deep-paired-quality-r375'
OWNER_SHA = '1b93c3db22ab59c6768e401dc46f71b630a768934fe316135b53d780d1b3fafb'
CONTROLLER_SHA = 'e868b3124fae4ff3facbeca2ba1c9e4a295d7b8fd5beeb3e3196dae75b151e57'
ENCODING_SHA = 'f17342d2ce78e73b9f8853720c6b477c947ccf8d71a7503dcd7e04e391e16447'
NATIVE_SHA = '976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def numeric(text):
    found = re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)', text)
    try:
        return Fraction(found[-1].replace(',', '')) if found else None
    except (ValueError, ZeroDivisionError):
        return None


def letter(text):
    found = re.findall(r'####\s*([ABCD])\b', text)
    return ord(found[-1]) - 65 if found else None


def audit(scorer_path, scorer_sha):
    assert sha(S / 'SOURCE_MANIFEST.json') == OWNER_SHA
    source_manifest = json.loads((S / 'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(S / name) == expected for name, expected in source_manifest['files_SHA256'].items())
    assert sha(scorer_path) == scorer_sha
    spec = importlib.util.spec_from_file_location('pinned_root_r374', scorer_path)
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    sys.path.insert(0, str(S))
    import quality_plan
    quality_plan.R = R  # Host alias of the same already-pinned source data.
    refs, cases = quality_plan.load_data()
    from client_owner_r281 import backend_metrics
    ep = R / 'candidates/Deep-paired-quality-r375/CPU-910407.749/encoded/encoding.json'
    assert sha(ep) == ENCODING_SHA
    encoding = json.loads(ep.read_bytes())
    contract = encoding['comparison_contract']
    arms = {str(a): {'comparison_contract': contract, 'suites': {name: {'rows': []} for name in ['GSM8K', 'MMLU']}} for a in [0, 1]}
    q = dict(completed=False, passed=False, read_only=True, GPU=0, Model_loads=0, HTTP_submitted=0,
             no_new_post_probe=True, raw_payloads_published=False, replicas=[], request_metadata=[],
             root_CI_consumer_SHA256=scorer_sha, independent_wire_gold_reparse=True,
             scope='Custom0shot64GSM+64MMLU historical fixed cases, Deep976f OFF/ON four independent8card replicas; not official benchmark/onlineSLO/global no-regression')
    all_normal = True
    for replica in range(4):
        wp = R / 'results' / (LABEL + '-r%d-watch.json' % replica)
        watch = json.loads(wp.read_bytes())
        assert watch['completed'], 'Original run is still active; do not resubmit or claim a terminal audit'
        assert watch['owner_manifest_SHA256'] == OWNER_SHA and watch['controller_manifest_SHA256'] == CONTROLLER_SHA
        actual = watch.get('actual_GPU')
        assert re.fullmatch(r'910(?:374|371)[.]\d+', actual)
        normal = bool(watch.get('Model_qualified') and watch.get('post_empty') and watch.get('srun_exit_code') == 0)
        all_normal &= normal
        post = watch.get('post', {})
        if normal:
            assert not post['kfd_pids'] and post['HTTP43882_bindable'] and post['owner_tids_all_gone']
            assert len(post['owner_tids']) == 8
        audit_replica = dict(replica=replica, actual_GPU=actual, actual_CPU_keeper=watch['CPU_keeper_actual'],
                             watcher_SHA256=sha(wp), normal_lifecycle=normal, existing_post=post,
                             terminal=watch.get('terminal'), srun_exit_code=watch.get('srun_exit_code'), rows=0)
        q['replicas'].append(audit_replica)
        owner_path = R / 'results' / (LABEL + '-' + actual + '.json')
        client_root = R / 'results' / (LABEL + '-' + actual) / 'prefill-pair'
        client_path = client_root / 'result.json'
        if not owner_path.exists() or not client_path.exists():
            audit_replica['missing_owner_or_client'] = True
            all_normal = False
            continue
        owner = json.loads(owner_path.read_bytes()); client = json.loads(client_path.read_bytes())
        assert owner['actual_step'] == actual and owner['replica'] == replica
        assert owner['Native_SHA256'] == NATIVE_SHA and owner['source_manifest_SHA256'] == OWNER_SHA
        assert owner['comparison_contract'] == client['comparison_contract'] == contract
        assert owner['prepared_entry_identity'] == encoding['prepared_entry_identity']
        frozen = quality_plan.plan(replica)
        assert len(frozen) == 66
        if normal:
            assert owner['completed'] and owner['passed'] and owner['normal_Native_shutdown'] and owner['normal_server_shutdown']
            assert owner['actual_HTTP'] == owner['actual_Native'] == owner['attempted_Native'] == owner['completed_Native'] == 66
            assert client['actual_HTTP_attempts'] == client['actual_HTTP_completed'] == client['actual_Native_completed'] == 66
            assert owner['owner_exit_observation']['all8_unique_threads_exited']
            assert len(client['requests']) == len(owner['requests']) == 66
        audit_replica.update(owner_SHA256=sha(owner_path), client_SHA256=sha(client_path), rows=len(client['requests']))
        for index, recorded in enumerate(client['requests']):
            entry = frozen[index]; case = cases[entry['case_id']]; r = owner['requests'][index]
            assert all(recorded[key] == value for key, value in entry.items())
            folder = client_root / ('request-%03d' % index)
            request_path, wire_path = folder / 'request-001.request.json', folder / 'request-001.wire'
            assert sha(request_path) == recorded['request_SHA256'] and sha(wire_path) == recorded['wire_SHA256']
            request = json.loads(request_path.read_bytes()); wire = json.loads(wire_path.read_bytes())
            assert request['messages'] == quality_plan.messages(case) and request['max_tokens'] == 4096 and request['temperature'] == 0
            assert request['stream'] is False and request['tool_choice'] == 'none' and request['chat_template_kwargs']['enable_thinking'] is False
            assert r['request_id'] == wire['id'] and r['admitted_index'] == index
            assert r['admission']['diagnostic_user'] == request['user'] == 'Deep375-r%d-i%d-mode%d' % (replica, index, entry['arm'])
            assert r['max_new_tokens'] == 4096 and r['normal_reset_before'] and r['normal_reset_after']
            encoded = encoding['cases'][entry['case_id']]
            assert r['prompt_tokens'] == encoded['prompt_tokens'] == wire['usage']['prompt_tokens']
            assert r['admission']['history_render_audit']['Native_prompt_token_ids_SHA256'] == encoded['prompt_IDs_SHA256']
            ids = r['generated_token_ids']; ticks = r['backend_token_monotonic']
            assert len(ids) == len(ticks) == r['actual_generated_tokens'] == wire['usage']['completion_tokens']
            assert ids and all(type(x) is int and 0 <= x < 129280 for x in ids)
            assert all(math.isfinite(x) for x in ticks) and all(a <= b for a, b in zip(ticks, ticks[1:]))
            assert r['admission']['accepted_monotonic'] <= r['started_monotonic'] <= ticks[0] <= ticks[-1] <= r['ended_monotonic']
            natural = r['native_finish_reason'] == 'eos_token'
            assert (wire['choices'][0]['finish_reason'] == 'stop') == natural
            assert (ids[-1] == 1 and 1 not in ids[:-1]) if natural else (len(ids) == 4096 and 1 not in ids)
            text = wire['choices'][0]['message'].get('content') or ''
            value = numeric(text) if case['suite'] == 'GSM8K' else letter(text)
            gold = Fraction(case['gold_numeric'].replace(',', '')) if case['suite'] == 'GSM8K' else case['answer']
            correct = bool(value is not None and value == gold and natural)
            assert recorded['correct'] == correct and recorded['parse_failure'] == (value is None)
            assert recorded['natural_EOS'] == natural and recorded['cap_hit'] == (not natural)
            assert r['prefill_binding']['planned_case'] == entry and r['HC_mode'] == entry['arm']
            sizes = [x['tokens'] for x in r['Native_continuation'] if x['kind'] == 'prefill']
            assert sum(sizes) == r['prompt_tokens'] and all(0 < x <= 1024 for x in sizes)
            sizes += [1] * r['Native_forward_host_summary']['decode_calls']
            assert len(sizes) == len(r['HC_forward_coverage'])
            for size, coverage in zip(sizes, r['HC_forward_coverage']):
                delta = [0, 0, 86, 86 if size == 1 else 0, 86 if size == 1 and entry['arm'] else 0, 86 if size != 1 else 0, 0]
                assert coverage['input_tokens'] == size and coverage['mode'] == entry['arm']
                assert coverage['per_owner_delta'] == [delta] * 8
            metrics = backend_metrics(r)
            assert all(recorded[key] == val for key, val in metrics.items())
            independent = dict(recorded, correct=correct, parse_failure=value is None, natural_EOS=natural, cap_hit=not natural)
            independent.update(**metrics, wire_gold_independently_reparsed=True,
                               generated_IDs_SHA256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                               HTTP_wall_scope='Actual client recorded wall; raw wire has no independent HTTP clock',
                               full_Native_enclosure_s=r['ended_monotonic'] - r['started_monotonic'])
            q['request_metadata'].append(independent)
            if not entry['warmup']:
                arms[str(entry['arm'])]['suites'][case['suite']]['rows'].append(independent)
    q['quality'] = {}
    for suite in ['GSM8K', 'MMLU']:
        fixture = {'suite': suite, 'cases': [case for key, case in cases.items() if key != 'warm' and case['suite'] == suite]}
        fixture_sha = hashlib.sha256(quality_plan.canonical(fixture)).hexdigest()
        assert len(fixture['cases']) == 64
        for arm in arms.values(): arm['suites'][suite]['fixture_SHA256'] = fixture_sha
        q['quality'][suite] = scorer.summarize(fixture, arms['0'], arms['1'], suite, fixture_sha)
    q.update(completed=True, passed=bool(all_normal and all(x['both_arms_complete'] and x['contract_comparable'] for x in q['quality'].values())),
             all32_normal_lifecycle=bool(all_normal), full_benchmark_proven=False, global_no_regression_proven=False,
             root_CI_scope='Consumes independently reparsed rows from this audit; no new CI implementation or declared noninferiority margin',
             timing_scope='Per-request complete HTTP/Native-emitter/Native-forward enclosures; no cross-node clock subtraction, GPUbusy or onlineSLO claim')
    return q


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scorer', type=pathlib.Path, required=True); ap.add_argument('--scorer-sha', required=True); ap.add_argument('--output', type=pathlib.Path, required=True)
    args = ap.parse_args()
    try: result = audit(args.scorer, args.scorer_sha)
    except BaseException as error: result = dict(completed=True, passed=False, read_only=True, GPU=0, Model_loads=0, error=repr(error), traceback=traceback.format_exc())
    with args.output.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: result.get(k) for k in ['completed', 'passed', 'error', 'all32_normal_lifecycle']}))
