"""Offline r444 originals only. No collection, HTTP, jobs, tokenizer or Native.

Reads private CPU918 and per-slot byte seals; never imports a running provider.
Missing slots/outcomes retain the full 256-question/512-request denominator.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath

R = '/public/home/aclsylqidf/hygon-inference-refactor-20261014'
CPU918 = '71e514ed04967b1b67c19f6e19decc92a3c0d9d09450eff659f610efa3005133'
ENC = '1e083b8241422ed87511d83fa8585bfc19eb356e539d42acfa80b4c10cc8351c'
DATA = 'c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38'
PLAN = '0224a12e37732ed9823f662300d49b419777c914d2cbee56819eb6e066c0b694'
NATIVE = '9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
DEPLOY = '56133dff89d8579b070f64d31f1adad2aa0582cea0cd9168ed164065162df6d9'
PROVIDER = 'eaf5ae8c5333e0601ce2bec7fdd27e41e0a110b534660e27f55c362915cb96a0'
EOS = {154820, 154827, 154829}
sha = lambda raw: hashlib.sha256(raw).hexdigest()
compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def key(path):
    path = str(path)
    if path.startswith('/refactor/'):
        path = R + path[len('/refactor'):]
    return path


class Originals:
    def __init__(self):
        self.files = {}
        self.archives = []

    def add(self, path, expected=None):
        blob = Path(path).read_bytes()
        if expected is not None:
            assert sha(blob) == expected, 'Archive pin mismatch'
        seal = json.loads(gzip.decompress(blob) if blob[:2] == b'\x1f\x8b' else blob)
        items = seal['files']
        entries = items.items() if isinstance(items, dict) else ((q['path'], q) for q in items)
        count = 0
        for name, row in entries:
            if 'base64' in row:
                raw = base64.b64decode(row['base64'], validate=True)
            else:
                raw = gzip.decompress(base64.b64decode(row['gzip_base64'], validate=True))
            assert sha(raw) == row['SHA256'] and len(raw) == row['bytes'], name
            name = key(name)
            assert name not in self.files or self.files[name] == raw, 'Conflicting original bytes'
            self.files[name] = raw
            count += 1
        self.archives.append(dict(path=str(path), SHA256=sha(blob), files=count))

    def raw(self, path):
        return self.files[key(path)]

    def data(self, path):
        return json.loads(self.raw(path))


def leaf(name, path, expected):
    assert sha(path.read_bytes()) == expected, 'Readonly scorer/source pin mismatch'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_inputs(store):
    base = R + '/incoming/GLM-W4-fewshot-quality-r444'
    data = store.raw(base + '/fixtures-r1/cases.json')
    plan = store.raw(base + '/fixtures-r1/requests-and-pairs.json')
    enc = store.raw(R + '/candidates/GLM-W4-fewshot-quality-r444-CPU-v1/encoding.json')
    assert (sha(data), sha(plan), sha(enc)) == (DATA, PLAN, ENC)
    cases, plan, encoding = json.loads(data)['cases'], json.loads(plan), json.loads(enc)
    assert encoding['passed'] and encoding['completed'] and encoding['actual_checkpoint_CPU_processor']
    assert not encoding['Torch_CUDA_initialized'] and not encoding['Native_constructed']
    assert len(cases) == len(encoding['cases']) == 256 and len(plan['comparison_schedule']) == 512
    assert [c['id'] for c in cases] == [r['case_id'] for r in plan['requests']] == [r['case_id'] for r in encoding['cases']]
    encoded = {r['case_id']: r for r in encoding['cases']}
    exposed = [c['id'] for c in cases if encoded[c['id']]['planned_pair_control_exposed']]
    assert len(exposed) == 160 and len(cases) - len(exposed) == 96
    for c, request in zip(cases, plan['requests']):
        e = encoded[c['id']]
        assert sha(compact(request['body'])) == request['compact_body_SHA256'] == e['request_body_SHA256']
        private = store.raw(e['private_path'])
        assert sha(private) == e['private_SHA256']
        p = json.loads(private)
        assert sha(compact(p['prompt_token_ids'])) == e['prompt_IDs_SHA256']
        assert len(p['prompt_token_ids']) == e['prompt_tokens'] and e['prompt_tokens'] + 4096 <= 32768
        assert e['planned_pair_control_exposed'] == (len(e['prefill_chunk_tokens']) > 2)
    return cases, plan, encoded, exposed


def tokens(row, encoded, wire, window, verify):
    ids, ticks = row['generated_token_ids'], row['backend_token_monotonic']
    assert ids and all(type(t) is int and 0 <= t < 154880 for t in ids)
    assert len(ids) == row['actual_generated_tokens'] == row['generated_tokens']
    assert sha(compact(ids)) == row['generated_token_ids_SHA256']
    assert len(ticks) == len(ids) and all(math.isfinite(t) for t in ticks)
    assert all(a <= b for a, b in zip(ticks, ticks[1:]))
    assert row['admission']['accepted_monotonic'] <= row['started_monotonic'] <= ticks[0] <= ticks[-1] <= row['ended_monotonic']
    natural = row['native_finish_reason'] == 'eos_token'
    assert row['native_finish_reason'] in ('eos_token', 'length')
    assert (ids[-1] in EOS and not any(t in EOS for t in ids[:-1])) if natural else (len(ids) == 4096 and not any(t in EOS for t in ids))
    assert row['generated_natural_EOS'] is natural
    assert row['normal_reset_before'] and row['normal_reset_after'] and row['original_reset_all16_completed']
    assert row['actual_prompt_token_ids_SHA256'] == encoded['prompt_IDs_SHA256']
    assert row['prompt_tokens'] == encoded['prompt_tokens'] and row['max_new_tokens'] == row['requested_output_budget'] == 4096
    continuation = row['Native_continuation']
    assert all(c['kind'] in ('prefill', 'decode') for c in continuation)
    assert [c['tokens'] for c in continuation if c['kind'] == 'prefill'] == encoded['prefill_chunk_tokens']
    assert len([c for c in continuation if c['kind'] == 'decode']) == len(ids) - 1
    assert all(c['tokens'] == 1 for c in continuation if c['kind'] == 'decode')
    verify.verify_trace(row, window, encoded)
    verify.verify_native_calls(row, finite_expected=True)
    assert len(row['native_calls']) == len(encoded['prefill_chunk_tokens']) + len(ids) - 1
    assert [(c['prefill'], c['T']) for c in row['native_calls']] == [(c['kind']=='prefill', c['tokens']) for c in continuation]
    usage = wire['usage']
    assert all(type(usage[k]) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    assert usage['prompt_tokens'] == encoded['prompt_tokens'] and usage['completion_tokens'] == len(ids)
    assert usage['total_tokens'] == usage['prompt_tokens'] + len(ids)
    assert wire['choices'][0]['finish_reason'] in ('stop', 'length', 'tool_calls')
    assert (wire['choices'][0]['finish_reason'] != 'length') if natural else (wire['choices'][0]['finish_reason'] == 'length')
    return natural


def lifecycle(store, folder, watch, owners, failures, verify, client_rows, encoded):
    count, capi_count = 0, 0
    for stage in range(2):
        masks = []
        for rank in range(8):
            try:
                owner = owners[(stage, rank)]
                life = store.data(folder + '/lifecycle-%d-%d.json' % (stage, rank))
                assert owner['completed'] and owner['normal_shutdown'] and owner['worker_shutdown_completed'] and owner['post_close_Torch_liveness']
                assert life['normal'] and life['completed'] and life['destroy_attempts'] == life['shutdown_attempts'] == 1
                assert life['destroy_RC'] == life['shutdown_RC'] == 0 and not life['cleanup_errors'] and life['primary_error'] is None
                assert owner['all16_preweight_8GiB_passed'] and owner['all16_loaded_and_postload_4GiB_passed'] and owner['constructor_noV_verified']
                assert owner['constructor_noV_census']['cache_category_bytes'].get('projected_value', 0) == 0
                a, b = owner['owner_affinity_after_load'], owner['owner_affinity_final']
                assert a['width'] == b['width'] == 4
                fields = lambda v: sorted((r['device'], r['changes'], r['verified_mask']) for r in v['actual_owner_report']['owners'])
                assert fields(a) == fields(b)
                mine = next(r for r in b['actual_owner_report']['owners'] if r['device'] == rank)
                mask = set(map(int, mine['verified_mask'].split(',')))
                assert mine['changes'] == 1 and len(mask) == 4 and all(not mask & old for old in masks)
                masks.append(mask)
                rows = owner['service_requests']
                assert owner['attempted_Native'] == owner['completed_Native'] == len(rows) == 64
                assert len(client_rows) == 64
                for i, c in enumerate(client_rows):
                    row = rows[i]
                    assert row['request_index'] == row['admitted_index'] == i == c['local_owner_index']
                    assert row['generated_token_ids_SHA256'] == c['generated_IDs_SHA256']
                    assert row['generated_tokens'] == c['generated_tokens']
                    assert row['generated_natural_EOS'] == c['natural_EOS'] and row['original_reset_all16_completed']
                    assert row['prefill_control_window'] == c['window'] and row['native_profile'] == 'glm-nov16-w' + str(c['window'])
                    assert row['actual_prompt_token_ids_SHA256'] == encoded[c['case_id']]['prompt_IDs_SHA256']
                    assert row['prompt_tokens'] == encoded[c['case_id']]['prompt_tokens'] and row['requested_output_budget'] == 4096
                    verify.verify_trace(row, c['window'], encoded[c['case_id']])
                    verify.verify_native_calls(row, finite_expected=stage == 1)
                    assert len(row['native_calls']) == len(encoded[c['case_id']]['prefill_chunk_tokens']) + c['generated_tokens'] - 1
                    capi_count += len(row['native_calls'])
                count += 1
            except (AssertionError, KeyError, ValueError, TypeError) as error:
                failures.append(dict(scope='owner-lifecycle', stage=stage, rank=rank, error=repr(error)))
    try:
        post = watch.get('post', watch.get('post_after_failure', []))
        assert len(post) == 2 and len({p['node'] for p in post}) == 2
        assert len({(p['node'],pci)for p in post for pci in p['physical_PCI']})==16
        for p in post:
            assert p['CPU_only'] and not p['GPU_initialized'] and p['KFD_available']
            assert not p['kfd_pids'] and not p['matching_component_processes'] and not p['expected_PIDs_still_exist']
            assert p['ports_bindable'] == {'43843': True, '43844': True} and p['TCP_connections_made'] == 0
    except (AssertionError, KeyError, TypeError) as error:
        failures.append(dict(scope='independent-post', error=repr(error)))
    return count, capi_count


def observed_clocks(row, client):
    ids,ticks=row['generated_token_ids'],row['backend_token_monotonic']
    visible=[t for token,t in zip(ids,ticks)if token not in EOS]
    gaps=sorted(b-a for a,b in zip(visible,visible[1:]))
    def percentile(p):
        if not gaps:return None
        k=(len(gaps)-1)*p;i=int(k)
        return gaps[i]+(gaps[min(i+1,len(gaps)-1)]-gaps[i])*(k-i)
    ttft=visible[0]-row['admission']['accepted_monotonic']if visible else None
    speed=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and visible[-1]>visible[0]else None
    def equal(a,b):return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
    assert client['visible_tokens']==len(visible)
    assert equal(ttft,client['backend_TTFT_from_admission_s'])and equal(speed,client['backend_visible_decode_tok_s'])
    assert equal(percentile(.95),client['backend_ITL_P95_s'])and equal(percentile(.99),client['backend_ITL_P99_s'])
    assert type(client['HTTP_wall_s'])in(int,float)and math.isfinite(client['HTTP_wall_s'])and client['HTTP_wall_s']>=0
    calls=row['native_calls'];summary=row['Native_forward_host_summary'];assert summary==client['Native_forward_host_summary']
    return dict(HTTP_wall_reported_client_s=client['HTTP_wall_s'],visible_tokens=len(visible),including_EOS=len(ids),backend_TTFT_recomputed_s=ttft,backend_visible_decode_recomputed_tps=speed,ITL_P95_recomputed_s=percentile(.95),ITL_P99_recomputed_s=percentile(.99),Native_CAPI_sum_s=math.fsum(c['Native_CAPI_s']for c in calls),Native_prefill_CAPI_sum_s=math.fsum(c['Native_CAPI_s']for c in calls if c['prefill']),Native_decode_CAPI_sum_s=math.fsum(c['Native_CAPI_s']for c in calls if not c['prefill']),Native_forward_host_summary=summary)


def timing_summary(rows):
    def summary(values):
        values=sorted(v for v in values if v is not None)
        if not values:return None
        return dict(n=len(values),min=values[0],max=values[-1],mean=math.fsum(values)/len(values),median=(values[(len(values)-1)//2]+values[len(values)//2])/2)
    metrics=['HTTP_wall_reported_client_s','backend_TTFT_recomputed_s','backend_visible_decode_recomputed_tps','ITL_P95_recomputed_s','ITL_P99_recomputed_s','Native_CAPI_sum_s','Native_prefill_CAPI_sum_s','Native_decode_CAPI_sum_s']
    per_window={str(w):{k:summary(r['timing'][k]for r in rows if r['window']==w)for k in metrics}for w in [2,4]}
    paired={}
    for r in rows:paired.setdefault(r['case_id'],{})[r['window']]=r['timing']
    ratios={k:summary(v[4][k]/v[2][k]for v in paired.values()if set(v)=={2,4}and v[2][k]is not None and v[4][k]is not None and v[2][k]>0)for k in metrics}
    return dict(per_window=per_window,complete_same_case_W4_over_W2_ratios=ratios,scope='Verified observed requests only. Backend ticks/CAPI sums independently recomputed; HTTPwall is recorded scalar from frozenclient because raw has no HTTPstart/end pair. Chunk/control callbacks are not hardware counters. No SLO, decoder15 claim or global performance inference.')


def analyze(index_path):
    index = json.loads(Path(index_path).read_bytes())
    store = Originals()
    store.add(index['CPU918_archive'], CPU918)
    for archive in index['execution_archives']:
        store.add(archive['path'], archive['SHA256'])
    cases, plan, encoded, exposed = pinned_inputs(store)
    parent = Path(__file__).resolve().parents[2] / 'GLM-W4-fewshot-quality-r444'
    score = leaf('frozen_r444_score', parent/'scoring.py', 'c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c')
    verify = leaf('frozen_r444_owner_checks', parent/'client-v1/owner_validation.py', '78dae335f87e5841de2a1904db73d79ca4f57420a0704f7a6d5532cab76e571a')
    ordinal = {c['id']: i for i, c in enumerate(cases)}
    expected_cases = {s: [c['id'] for c in cases if (ordinal[c['id']]//2)%8 == s] for s in range(8)}
    slots = {x['slot']: x for x in index['slots']}
    assert len(slots) == len(index['slots']) and set(slots)=={0}, 'This binding is original886 singleton only'
    all_rows, status = [], []
    for slot in range(8):
        failures, rows = [], []
        result = dict(slot=slot, planned_questions=32, planned_HTTP=64, supplied=slot in slots, failures=failures)
        status.append(result)
        if slot not in slots:
            result.update(state='UNSUPPLIED_NOT_ASSUMED_FAILED_OR_RUNNING', verified_HTTP=0, normal_owners=0)
            continue
        spec = slots[slot]
        try:
            watch = store.data(spec['watch_path'])
            assert watch['slot'] == slot and watch['planned_global_HTTP'] == 512 and watch['HTTP_planned'] == 64
            assert watch['deployment_source_SHA256'] == DEPLOY
            assert watch['provider_source_SHA256'] == PROVIDER
            assert (watch['actual_GPU'],watch['actual_client'],watch['CPU_keeper'])==('910376.886','910376.887','910407.950')
            assert watch['startup_accepted']is True, 'Read original accepted startup; never reapply startup2700 to ongoing service'
            epoch=watch['original_GPU_start_unix']
            for field,seconds in [('service_deadline_unix',28200),('Native_deadline_unix',28800),('GPU_deadline_unix',29400),('keeper_deadline_unix',32400)]:assert math.isclose(watch[field]-epoch,seconds,abs_tol=1e-6)
            folder, client_path = spec['run_directory'], spec['client_path']
            client = store.data(client_path)
            assert client['shard_index'] == slot and client['shards'] == 8 and client['planned_global_HTTP'] == 512
            assert client['encoding_SHA256'] == ENC and client['Native'] == NATIVE
            assert client['actual_step'] == watch['actual_GPU'] and client['provider_source_SHA256'] == watch['provider_source_SHA256']
            if watch.get('client_result_SHA256') is not None:
                assert sha(store.raw(client_path)) == watch['client_result_SHA256']
            expected = [x for x in plan['comparison_schedule'] if x['case_id'] in expected_cases[slot]]
            assert len(expected) == 64 and client['planned_case_ids'] == expected_cases[slot]
            owners = {}
            for stage in range(2):
                for rank in range(8):
                    try:
                        owner = store.data(folder + '/owner-%d-%d.json' % (stage, rank))
                        # Original production reports identify rank/stage through the
                        # actual partition and the synchronous Native layer registry.
                        # There are no top-level owner.stage/owner.rank fields.
                        partition = owner['actual_partition']
                        assert (partition['stage'], partition['device'], partition['PP']) == (stage, rank, 2)
                        assert (partition['begin'], partition['end']) == ((0, 38) if stage == 0 else (38, 78))
                        registry = owner['registry_owner_before_weightIO']
                        assert registry['all16passed'] and registry['synchronous_direct_CAPI_owner_thread']
                        identities = registry['identities']['identities']
                        assert len(identities) == partition['end'] - partition['begin']
                        assert {(r['stage'], r['rank'], r['layer']) for r in identities} == {(stage, rank, layer) for layer in range(partition['begin'], partition['end'])}
                        life_identity = store.data(folder + '/lifecycle-%d-%d.json' % (stage, rank))
                        assert (life_identity['stage'], life_identity['rank'], life_identity['actual_step'], life_identity['run'], life_identity['source_manifest_SHA256']) == (stage, rank, watch['actual_GPU'], client['run'], PROVIDER)
                        assert owner['run'] == client['run'] and owner['actual_step'] == watch['actual_GPU']
                        assert owner['Native_SHA256'] == NATIVE and owner['source_manifest_SHA256'] == owner['service_profile_manifest_SHA256'] == watch['provider_source_SHA256']
                        owners[(stage, rank)] = owner
                    except (KeyError, AssertionError, ValueError, TypeError) as error:
                        failures.append(dict(scope='owner-identity', stage=stage, rank=rank, error=repr(error)))
            seen = set()
            for i, c in enumerate(client['rows']):
                try:
                    planned = expected[i]
                    assert (c['case_id'], c['window'], c['global_schedule_index']) == (planned['case_id'], planned['window'], planned['index'])
                    assert c['local_owner_index'] == i and (c['case_id'], c['window']) not in seen
                    seen.add((c['case_id'], c['window']))
                    where = str(PurePosixPath(client_path).parent / ('%03d-%s-W%d' % (i, c['case_id'], c['window'])))
                    body, wire_raw = store.raw(where+'/request.body.json'), store.raw(where+'/response.wire')
                    headers = store.data(where+'/headers.json')
                    assert headers == {'Content-Type': 'application/json', 'X-InfiniLM-Native-Profile': 'glm-nov16-w'+str(c['window'])}
                    assert sha(body) == c['request_body_SHA256'] == planned['compact_body_SHA256'] == encoded[c['case_id']]['request_body_SHA256']
                    assert sha(wire_raw) == c['response_wire_SHA256'] and c['HTTP_status'] == 200 and c['output_budget'] == 4096
                    wire = json.loads(wire_raw)
                    assert 'error' not in wire and len(wire['choices']) == 1 and wire['id'] == c['response_id']
                    message = wire['choices'][0]['message']
                    assert message['role'] == 'assistant'
                    content = message.get('content') or ''
                    assert isinstance(content, str)
                    emitter = owners[(1, 0)]['service_requests'][i]
                    assert emitter['request_id'] == wire['id']
                    natural = tokens(emitter, encoded[c['case_id']], wire, c['window'], verify)
                    assert c['generated_IDs_SHA256'] == emitter['generated_token_ids_SHA256'] and c['natural_EOS'] is natural
                    graded = score.grade(cases[ordinal[c['case_id']]], content, natural_EOS=natural, cap_hit=not natural, unexpected_tool=bool(message.get('tool_calls')))
                    assert all(type(c[k]) is bool for k in ('correct', 'answer_matches_gold', 'parse_failure', 'natural_EOS', 'cap_hit', 'unexpected_tool_intent'))
                    for k in ('correct', 'answer_matches_gold', 'parse_failure', 'natural_EOS', 'cap_hit', 'unexpected_tool_intent', 'parsed_answer'):
                        assert c[k] == graded[k], 'Reported score differs from original wire/gold'
                    rows.append(dict(case_id=c['case_id'], window=c['window'], slot=slot, timing=observed_clocks(emitter,c), **graded))
                except (AssertionError, KeyError, IndexError, ValueError, TypeError) as error:
                    failures.append(dict(scope='request-original', local_index=i, error=repr(error)))
            normal, capi = lifecycle(store, folder, watch, owners, failures, verify, client['rows'], encoded)
            result.update(verified_HTTP=len(rows), normal_owners=normal, client_completed=client['completed'],
                          actual_CAPI_verified=capi,
                          client_execution_passed=client['execution_passed'], attempted_HTTP=client['actual_HTTP'],
                          pending_attempt=client['pending_attempt'], client_errors=client['errors'],
                          original_watch_completed=watch['completed'], original_watch_passed=watch['passed'],
                          original_watch_error=watch.get('error'), actual_GPU=watch.get('actual_GPU'),
                          actual_client=watch.get('actual_client'), actual_keeper=watch.get('CPU_keeper'))
            result['original_controller_normal'] = bool(watch['completed'] and watch['passed'] and not watch.get('forced_cleanup', False))
            # Accounting is an original text artifact, not a claim inferred from JSON.
            accounting = store.raw(spec['accounting_path']).decode()
            records = [line.split('|') for line in accounting.splitlines() if '|' in line]
            for step in (watch['actual_GPU'], watch['actual_client'], watch['CPU_keeper']):
                matching = [r for r in records if r[0] == step]
                assert len(matching) == 1 and matching[0][2:4] == ['COMPLETED', '0:0'], 'Actual terminal accounting unavailable/failed'
                assert matching[0][4].isdigit(), 'Sealed accounting column5 must be ElapsedRaw'
                limits={'910376.886':29400,'910376.887':28200,'910407.950':32400};assert 0<=int(matching[0][4])<=limits[step]
            result['terminal_accounting_normal'] = True
        except (AssertionError, KeyError, IndexError, ValueError, TypeError) as error:
            failures.append(dict(scope='slot-handoff-or-accounting', error=repr(error)))
        all_rows.extend(rows)
        result['verified_HTTP'] = len(rows)
        result['state'] = 'FULL_ORIGINALS_VERIFIED' if len(rows) == 64 and not failures and result.get('original_controller_normal') and result.get('client_execution_passed') and not result.get('client_errors') else 'PARTIAL_OR_FAILED_EVIDENCE_RETAINED'
    summary = score.summarize(cases, all_rows, exposure_case_ids=exposed)
    equivalent = [c for c in cases if c['id'] not in set(exposed)]
    summary['control_equivalent_96'] = score.summarize(equivalent, [r for r in all_rows if r['case_id'] in {c['id'] for c in equivalent}])
    selected=[c for c in cases if c['id']in expected_cases[0]]
    selected_exposed=[i for i in exposed if i in expected_cases[0]]
    selected_equivalent=[c for c in selected if c['id']not in selected_exposed]
    conditional=score.summarize(selected,all_rows,exposure_case_ids=selected_exposed)
    conditional['control_equivalent']=score.summarize(selected_equivalent,[r for r in all_rows if r['case_id']in {c['id']for c in selected_equivalent}])
    full = all(s['state'] == 'FULL_ORIGINALS_VERIFIED' for s in status)
    return dict(schema=1, actual_original_audit=True, full512_evidence_verified=full,
                planned_questions=256, planned_HTTP=512, verified_HTTP=len(all_rows),
                unverified_or_missing_HTTP=512-len(all_rows), slots=status, quality=summary,
                admitted_slot0_quality=conditional,admitted_slot0_questions=32,admitted_slot0_HTTP=64,admitted_slot0_exposed=len(selected_exposed),admitted_slot0_equivalent=len(selected_equivalent),timing=timing_summary(all_rows),
                input_and_execution_archives=store.archives, input_pins=dict(cases=DATA, plan=PLAN, encoding=ENC),
                control_exposed_questions=160, control_equivalent_questions=96,
                noninferiority_proven=False, broad_quality_qualified=False, GPU_or_Model_executed_by_reader=False,
                deadline_scope='Original watch arithmetic and actual Slurm ElapsedRaw verified. Native28800 is source-enforced; no absent absolute Native completion timestamp is invented. Accepted startup never re-gated by current total elapsed.',
                limits='Actual public response content scored; token IDs independently hashed/EOS checked, not independently tokenizer-decoded. Observed-pair intervals remain conditional with partial coverage. Wrong/cap/parse outcomes retained. No all-correct gate, no SLO or speed comparison.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('index', help='Private absolute archive and slot artifact paths; see README')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    q = analyze(a.index)
    Path(a.output).write_bytes((json.dumps(q, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps({k:q[k] for k in ('full512_evidence_verified', 'planned_HTTP', 'verified_HTTP', 'unverified_or_missing_HTTP')}))


if __name__ == '__main__':
    main()
