"""Audit retained AR stages of one failed long-pair run, without loading Native."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def audit(collection, fixtures):
    records = collection['reports']
    nodes = [x['record'] for x in records[:4]]
    watch = records[-1]['record']
    assert watch['completed'] and watch['passed'] is False and watch['post_empty']
    assert watch['state'] == 'FAILED_NO_RETRY_RELOAD'
    assert len(watch['post']) == 5
    assert all(p['passed'] and p['KFD_empty'] and not p['kfd_pids']
               and all(p['ports_bindable'].values()) for p in watch['post'])
    assert {n['node'] for n in nodes} == set(range(4))
    expected = []
    for f in fixtures['fixtures']:
        data = json.loads(next(m['content'] for m in f['body']['messages'] if m['role'] == 'tool'))
        values = {r['id']: r['value'] for r in data['records']}
        gold = dict(a=values[data['query_ids'][0]], b=values[data['query_ids'][1]])
        assert gold == f['expected_calculate_arguments']
        assert digest(f['prompt_tokens']) == f['prompt_ID_SHA256']
        expected.append((f, gold))
    signatures = []
    summaries = []
    for node in nodes:
        assert node['actual_step'] == '910380.536' and node['completed'] is False
        pair = node['pair_report']
        assert pair['completed'] is False and len(pair['requests']) == 2
        rows = []
        for r, (f, gold) in zip(pair['requests'], expected):
            assert r['name'] == f['name'] and r['mode'] == 'AR'
            assert r['input_tokens'] == f['input_tokens']
            assert r['prompt_SHA256'] == f['prompt_ID_SHA256']
            assert r['completed'] and r['normal_reset'] and r['finish'] == 'eos'
            ids = r['generated_ids']
            assert len(ids) == r['tokens'] == 59 and ids[-1] in f['eos']
            assert digest(ids) == r['ids_SHA256']
            grade = r['quality']
            assert grade['passed'] and grade['details']['passed']
            assert grade['details']['arguments'] == gold
            assert all(type(v) is int for v in grade['details']['arguments'].values())
            events = r['emitter_events']
            assert [v for event in events for v in event['tokens']] == ids
            times = [event['at_s'] for event in events]
            assert all(math.isfinite(t) for t in times) and all(b > a for a, b in zip(times, times[1:]))
            row = {k: r[k] for k in ['name', 'mode', 'input_tokens', 'prompt_SHA256', 'tokens',
                    'ids_SHA256', 'finish', 'normal_reset', 'TTFT_s', 'generation_local_wall_s',
                    'request_through_reset_local_wall_s']}
            row.update(visible_tokens=len(ids)-1, independently_derived_arguments=gold,
                       runtime_pinned_parser_grade=grade['details'],
                       visible_backend_decode_tok_s=(len(ids)-2)/(times[-2]-times[0]),
                       selected_including_EOS_decode_tok_s=(len(ids)-1)/(times[-1]-times[0]),
                       Native_calls=len(r['Native_calls']),
                       emitter_events_SHA256=digest(events), Native_calls_SHA256=digest(r['Native_calls']))
            rows.append(row)
        signatures.append([(r['prompt_SHA256'], r['ids_SHA256'], r['tokens'], r['finish']) for r in rows])
        summaries.append(dict(node=node['node'], loading_seconds=node['loading_seconds'], requests=rows,
                              first_capture_validation_recorded='first_capture_validation' in node,
                              normal_all8_destroy_recorded=node.get('all8_owned_workers_shutdown_completed', False)))
    assert all(s == signatures[0] for s in signatures)
    return dict(schema=1, run=watch['run'], outcome='FAIL_AFTER_TWO_AR_BEFORE_DRAFT_AUTHENTICATION',
                root_audit_passed=True, whole_run_passed=False, performance_gain_qualified=False,
                HTTP=False, client_SLO_qualified=False,
                actual=dict(target='910380.536', draft='910371.641', CPU='910407.666'),
                accounting=collection['accounting'],
                source_manifests=dict(controller=watch['source_manifest_SHA256'],
                    target=nodes[0]['source_manifest_SHA256'], Native=nodes[0]['Native_SHA256']),
                raw_reports=[{k: r[k] for k in ['path', 'SHA256', 'bytes']} for r in records],
                four_leader_two_AR_signatures_equal=True,
                nodes=summaries, draft=records[4]['record'], controller_error=dict(
                    type=watch['error_type'], message=watch['error'],
                    passive_listener_recorded='passive_listener' in watch),
                forced_cleanup=watch['owned_forced_cleanup'], post=watch['post'],
                normal_lifecycle_qualified=False,
                limitations=['Two AR stages retain scoped quality and emitter evidence only.',
                    'No completed spec arm, acceptance, capture or speedup qualification.',
                    'Reported tool arguments are independently checked against frozen records; output decoding used the actual pinned runtime parser.',
                    'Raw activations and large report/Native-call traces remain remote and local; SHA references are archived.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('collection', type=Path)
    p.add_argument('--fixtures', type=Path, default=Path('code/bw1000/Kimi-long-pair-driver-r337/frozen_inputs.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(json.loads(a.collection.read_bytes()), json.loads(a.fixtures.read_bytes()))
    result['collection_SHA256'] = hashlib.sha256(a.collection.read_bytes()).hexdigest()
    result['fixtures_SHA256'] = hashlib.sha256(a.fixtures.read_bytes()).hexdigest()
    with a.output.open('x', encoding='utf8', newline='\n') as out:
        json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ['outcome', 'root_audit_passed', 'whole_run_passed']}))
