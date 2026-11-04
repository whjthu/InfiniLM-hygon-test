"""Offline audit of the original failed 889 attempt, without a new request.

Inputs remain private: immutable gzip containing explicit raw file/hash records.
Only reviewed stdlib response/coverage consumers are loaded from source-v1.
The candidate 2048 and all unexecuted requests remain in the denominator.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

H = lambda raw: hashlib.sha256(raw).hexdigest()
ARCHIVE = '8ddb9aa83b3f3e0ef4ac051e36e32f82fb183624cbc40fff8af74244efd4f3db'
PREFIX = 'Deep-r451-public-chunk-pair-chunk-pair-20261104-e'
RUN = 'chunk-pair-20261104-e'
NATIVE = 'a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
PINS = {
    'model_counts': 'b82e86ec8401ab8a8764e0d92f3f3cb89a5ec30cdd281e221549faf18ff9f337',
    'wire_client_r290': '045ea460cd6a95277ae60fe787860960160b9eae4537dcafe02ee2a34dd3af14',
    'content_json': '2f47434e84a7cab184feca163620b27883e629335f4c07ce5cc4df1f0dd179d5',
}


def consumer(name):
    path = Path(__file__).resolve().parents[1] / 'source-v1' / (name + '.py')
    assert H(path.read_bytes()) == PINS[name], name
    spec = importlib.util.spec_from_file_location('offline451_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(path):
    raw = Path(path).read_bytes()
    assert H(raw) == ARCHIVE
    bundle = json.loads(gzip.decompress(raw))
    files = {}
    for item in bundle['files']:
        value = base64.b64decode(item['base64'], validate=True)
        assert len(value) == item['bytes'] and H(value) == item['SHA256']
        assert item['path'] not in files
        files[item['path']] = value
    assert len(files) == 19
    get = lambda name: json.loads(files[name])
    base = 'results/Deep-public-prefill-2048-r451/' + RUN
    client = 'results/' + PREFIX + '-client'
    watch = get('results/' + PREFIX + '-watch.json')
    report = get(base + '/owner.json')
    result = get(client + '/result.json')
    assert watch['actual_GPU'] == '910376.889' and watch['CPU_keeper'] == '910407.960'
    assert watch['actual_client'] == '910376.890' and watch['completed'] and not watch['passed']
    assert watch['original_GPU_start_unix'] == 1793759189.4451034
    assert watch['forced_cleanup'] and report['Native_SHA256'] == NATIVE
    assert result['completed'] and not result['passed']
    assert result['planned_HTTP'] == 6 and result['attempted_HTTP'] == 1
    assert result['completed_HTTP'] == 0 and result['rows'] == []
    assert 'fixture.py' in result['error_traceback'] and 'actual==expected' in result['error_traceback']
    assert len(report['requests']) == 1
    row = report['requests'][0]
    proof = consumer('model_counts').validate(row, 0)
    request_raw = files[client + '/request-0.request.json']
    body = json.loads(request_raw)
    assert H(request_raw) == result['current_attempt']['request_SHA256']
    assert body['stream'] and body['max_tokens'] == 128 and body['text_delivery'] == 'P1'
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    calls, expected = {}, {}
    for message in body['messages']:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls', []):
                assert call['function']['name'] == 'read_file' and call['id'] not in calls
                calls[call['id']] = json.loads(call['function']['arguments'])['path']
        if message['role'] == 'tool':
            data = json.loads(message['content'])
            values = {r['id']: r['value'] for r in data['records']}
            selected = [values[key] for key in data['query_ids']]
            assert len(selected) == 4 and all(type(v) is int for v in selected)
            expected[calls[message['tool_call_id']]] = dict(nonce=data['nonce'], sum=sum(selected))
    assert set(expected) == {'left.json', 'right.json'}
    expected['total'] = expected['left.json']['sum'] + expected['right.json']['sum']
    wire = files[client + '/response-0.wire']
    assert H(wire) == result['current_attempt']['wire_SHA256']
    message, finish, usage = consumer('wire_client_r290').sse_message(wire)
    assert finish == 'stop' and not message.get('tool_calls')
    actual, presentation = consumer('content_json').parse_final_object(message['content'])
    assert actual.keys() == expected.keys() and actual != expected
    assert all(actual[k]['nonce'] == expected[k]['nonce'] for k in ['left.json', 'right.json'])
    assert all(type(actual[k]['sum']) is int for k in ['left.json', 'right.json'])
    assert type(actual['total']) is int
    history = row['admission']['history_render_audit']
    assert history['qualified'] and history['prequeue_bound']
    assert history['Native_prompt_token_ids_SHA256'] == '681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
    assert history['prompt_tokens'] == row['prompt_tokens'] == usage['prompt_tokens'] == 8489
    ids = row['generated_token_ids']
    assert len(ids) == usage['completion_tokens'] == 98 and ids[-1] == 1 and ids.count(1) == 1
    assert row['native_finish_reason'] == 'eos_token'
    assert row['actual_forward_completed'] == row['actual_forward_attempts'] == 106
    assert row['normal_reset_before'] and row['normal_reset_after']
    ticks = row['backend_token_monotonic']
    assert len(ticks) == len(ids) and all(b > a for a, b in zip(ticks, ticks[1:]))
    account = {r.split('|')[0]: r.split('|') for r in bundle['accounting']}
    assert account['910376.889'][2:4] == ['CANCELLED by 0', '0:15']
    assert account['910376.890'][2:4] == ['FAILED', '1:0']
    assert account['910407.960'][2:4] == ['FAILED', '1:0']
    assert account['910376.891'][2:4] == ['COMPLETED', '0:0']
    post = watch['post_after_failure']
    assert len(post) == 1 and post[0]['actual_step'] == '910376.891'
    assert post[0]['KFD_available'] and not post[0]['kfd_pids']
    assert not post[0]['matching_component_processes'] and not post[0]['expected_PIDs_still_exist']
    assert post[0]['ports_bindable'] == {'43882': True}
    assert all(v['closed'] and not v['PGID_still_live'] for v in watch['owned_cleanup'])
    return dict(audit_passed=True, archive_SHA256=H(raw), original_files=len(files),
                original_snapshot_UTC=bundle['UTC'], actual_GPU='910376.889', actual_client='910376.890',
                actual_keeper='910407.960', actual_post='910376.891', Native_SHA256=NATIVE,
                planned_HTTP=6, actual_HTTP=1, completed_model_responses=1, client_accepted=0,
                unexecuted_HTTP=5, candidate2048_executed=False, valid_ABBA_pairs=0,
                failure='First1024OFFwarm integeranswer mismatch, not parser/transport/cap failure',
                expected_sums={k: expected[k]['sum'] for k in ['left.json', 'right.json']}, expected_total=expected['total'],
                actual_sums={k: actual[k]['sum'] for k in ['left.json', 'right.json']}, actual_total=actual['total'],
                both_nonces_correct=True, presentation=presentation, natural_EOS=True,
                prompt_tokens=8489, generated_including_EOS=98, actual_forward_completed=106,
                prefill1024_calls=9, actual_T1_coverage=proof, all8_owner_counts_verified=True,
                Native_forward_host_summary=row['Native_forward_host_summary'], loading_s=report['loading_seconds'],
                normal_request_reset=True, normal_whole_lifecycle_qualified=False,
                forced_cleanup=True, independent_post_clean=True, accounting=list(account.values()),
                no_new_HTTP=True, no_new_GPU=True, no_performance_or_quality_promotion=True,
                interpretation='Semantic failure retained. Candidate benefit UNKNOWN. Same index83 or integer error elsewhere does not establish a common numeric root cause.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), ensure_ascii=False, indent=2, allow_nan=False))
