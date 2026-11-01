"""Independently audit the completed six-request GLM W2/W4 model pair."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def audit(q):
    w = q['watch']
    assert w['actual_GPU'] == '910376.823' and w['CPU_keeper_actual'] == '910407.645'
    assert w['terminal'][2:4] == ['COMPLETED', '0:0']
    assert '910407.645|GLM-prefill-W2-W4-Model-r308-keeper|COMPLETED|0:0|' in q['accounting']
    assert len(w['post']) == 4 and all(p['KFD_empty'] and not p['kfd_pids'] and p['port43843_bindable'] for p in w['post'])
    owners = q['owners']
    assert len(owners) == 32 and {(o['stage'], o['rank']) for o in owners} == {(s, r) for s in range(4) for r in range(8)}
    signatures = []
    for owner in owners:
        life = owner['lifecycle']
        assert life['actual_producer'] == '910376.823'
        assert life['library_sha256'] == '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
        for name in ['completed', 'worker_shutdown_completed', 'post_close_Torch_liveness',
                     'normal_reset_close', 'native_service_loop_completed', 'normal_shutdown']:
            assert life[name] is True, (owner['stage'], owner['rank'], name)
        rows = owner['requests']
        assert len(rows) == life['full_requests_completed'] == 6
        assert [r['prefill_control_window'] for r in rows] == [2, 4, 2, 4, 4, 2]
        assert [r['pair_index'] for r in rows] == list(range(6))
        assert [r['paired_warmup'] for r in rows] == [True, True, False, False, False, False]
        assert all(r['original_reset_all32_completed'] is True and r['generated_natural_EOS'] is True
                   and r['reason'] == 'eos_token' and r['prompt_tokens'] == 8900
                   and r['requested_output_budget'] == 4096 for r in rows)
        signatures.append([(r['actual_prompt_token_ids_SHA256'], r['generated_token_ids_SHA256'],
                            r['generated_tokens'], r['reason']) for r in rows])
    assert all(s == signatures[0] for s in signatures)
    emitter = next(o for o in owners if (o['stage'], o['rank']) == (3, 0))
    cleanup = emitter['lifecycle']['cleanup']
    assert emitter['lifecycle']['normal_HTTP_shutdown'] and not cleanup['forced_cleanup'] and not cleanup['errors']
    request = q['frozen_request']
    assert request['SHA256'] == '5ee8ea9cc99e12969d4b0505d4dd70f50e6436079206e8de5a6e908c5854c9dd'
    body = request['body']
    assert body['max_tokens'] == 4096 and body['stream'] is False
    data = json.loads(body['messages'][-1]['content'])
    values = {r['id']: r['value'] for r in data['records']}
    expected = dict(a=values[data['query_ids'][0]], b=values[data['query_ids'][1]])
    assert all(type(v) is int for v in expected.values())
    client = q['supervisor']
    assert client['completed'] and client['passed'] and client['actual_HTTP'] == client['actual_Native'] == 6
    cases = client['phases']['paired']['result']['cases']
    assert len(cases) == len(q['responses']) == 6
    measured = []
    for i, (r, c, response) in enumerate(zip(emitter['requests'], cases, q['responses'])):
        ids = r['generated_token_ids']; ticks = r['backend_token_monotonic']
        assert digest(ids) == r['generated_token_ids_SHA256']
        assert len(ids) == len(ticks) == r['generated_tokens'] == 732
        assert ids[-1] in [154820, 154827, 154829]
        assert not any(t in [154820, 154827, 154829] for t in ids[:-1])
        assert all(math.isfinite(t) for t in ticks) and all(a <= b for a, b in zip(ticks, ticks[1:]))
        assert r['admission']['accepted_monotonic'] <= r['started_monotonic'] <= ticks[0] <= ticks[-1] <= r['ended_monotonic']
        assert r['normal_reset_before'] and r['normal_reset_after']
        assert response['SHA256'] == c['wire_SHA256'] and response['index'] == i
        wire = response['wire']; choice = wire['choices'][0]
        assert wire['id'] == c['request_id'] == r['request_id']
        assert choice['finish_reason'] == 'tool_calls' and len(choice['message']['tool_calls']) == 1
        call = choice['message']['tool_calls'][0]
        args = json.loads(call['function']['arguments'])
        assert call['function']['name'] == 'calculate' and args == expected
        assert all(type(v) is int for v in args.values())
        assert wire['usage'] == dict(prompt_tokens=8900, completion_tokens=732, total_tokens=9632)
        raw = r['native_calls']
        prefill = [x for x in raw if x['prefill']]; decode = [x for x in raw if not x['prefill']]
        assert len(prefill) == 18 and sum(x['T'] for x in prefill) == 8900 and len(decode) == 731
        assert [x['past'] for x in prefill] == list(range(0, 8900, 512))
        assert all(x['T'] == 1 for x in decode)
        pre_s = sum(x['Native_CAPI_s'] for x in prefill)
        decode_s = sum(x['Native_CAPI_s'] for x in decode)
        assert abs(pre_s - r['Native_forward_host_summary']['prefill_s']) < 1e-9
        assert abs(decode_s - r['Native_forward_host_summary']['decode_s']) < 1e-9
        visible = ticks[:-1]
        item = dict(index=i, window=r['prefill_control_window'], warmup=i < 2,
                    prefill_CAPI_host_wall_s=pre_s, decode_CAPI_host_wall_s=decode_s,
                    HTTP_completion_wall_s=c['HTTP_wall_s'],
                    backend_first_wait_from_admission_s=visible[0]-r['admission']['accepted_monotonic'],
                    backend_emitter_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0]),
                    output_tokens_including_EOS=len(ids), visible_tokens=len(visible), natural_EOS=True,
                    correct_tool_arguments=True)
        if i >= 2: measured.append(item)
    comparisons = {}
    for metric in ['prefill_CAPI_host_wall_s', 'HTTP_completion_wall_s', 'backend_first_wait_from_admission_s', 'decode_CAPI_host_wall_s']:
        base = mean(r[metric] for r in measured if r['window'] == 2)
        candidate = mean(r[metric] for r in measured if r['window'] == 4)
        comparisons[metric] = dict(W2_mean=base, W4_mean=candidate,
                                  W2_over_W4=base/candidate, latency_reduction_percent=100*(1-candidate/base),
                                  adjacent_pair_W2_over_W4=[measured[0][metric]/measured[1][metric], measured[3][metric]/measured[2][metric]],
                                  W2_endpoint_drift_percent=100*(measured[3][metric]/measured[0][metric]-1),
                                  W4_endpoint_drift_percent=100*(measured[2][metric]/measured[1][metric]-1))
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='SCOPED_REAL_MODEL_W4_PREFILL_AND_HTTP_COMPLETION_GAIN_PASS_NORMAL_CLEAN',
        actual_GPU='910376.823 COMPLETED0/51m20s', CPU_keeper='910407.645 COMPLETED0/51m24s',
        workload=dict(model='GLM-5.3', cards=32, TP=8, PP=4, context=32768,
                      prompt_tokens=8900, output_budget=4096, max_requests=6, warmups=2, measured_ABBA_blocks=1),
        same_residency=True, identical_prompt_all_six=True, exact_token_hash_all32_verified=True,
        six_natural_EOS_and_exact_tool_arguments=True, all32_normal_exit=True, HTTP_normal_shutdown=True,
        four_node_KFD_empty_and_port43843_bindable=True, comparisons=comparisons, measured_requests=measured,
        scoped_Model_prefill_gain=True, scoped_Model_HTTP_completion_gain=True,
        decode_speedup_proven=False, fullbenchmark_no_regression_proven=False,
        client_incremental_SLO_qualified=False, stable_multiload_or_multiworkload_gain_proven=False,
        interpretation='One frozen real named-tool request and one post-warmup ABBA block. Prefill control scheduling only; decode is unchanged within variation. HTTP buffered until completion, so backend first-token timing is not client TTFT. No claim of GPU busy, kernel attribution, broad quality or >15tok/s.',
        source=dict(owner_manifest_SHA256=w['owner_manifest_SHA256'], controller_manifest_SHA256=w['controller_manifest_SHA256'],
                    Native_SHA256=emitter['lifecycle']['library_sha256'], publication_commit=w['publication_commit']),
        original_evidence=dict(watch_SHA256=q['watch_SHA256'], supervisor_SHA256=q['supervisor_SHA256'],
            owner_files=[{k:o[k] for k in ['stage','rank','path','SHA256']} for o in owners],
            raw_data_remains_remote=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input'); parser.add_argument('output')
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    result = audit(json.loads(raw))
    result['root_collected_input_SHA256'] = hashlib.sha256(raw).hexdigest()
    target = Path(args.output)
    with target.open('x', encoding='utf8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(result['comparisons'], indent=2))
