"""Independent root audit of the six-request scoped prefill pair; reads local raw evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def load(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, fixture):
    directory = Path(directory)
    report_path = directory/'Deep348r2-1257-service-report-raw.json'
    client_path = directory/'Deep348r2-1257-client-result-raw.json'
    wire_path = directory/'Deep348r2-1257-six-real-HTTP-wires.json'
    summary_path = directory/'Deep348r2-1257-qualified-prefill-pair-summary.json'
    r, c, w, summary = map(load, [report_path, client_path, wire_path, summary_path])
    assert sha(report_path) == summary['report_SHA256']
    assert sha(client_path) == summary['client_SHA256']
    f = load(fixture)
    calls = {v['id']: json.loads(v['function']['arguments'])['path']
             for m in f['body']['messages'] for v in m.get('tool_calls', [])}
    gold = {}
    for m in f['body']['messages']:
        if m['role'] != 'tool': continue
        data = json.loads(m['content']); values = {v['id']: v['value'] for v in data['records']}
        gold[calls[m['tool_call_id']]] = dict(nonce=data['nonce'], sum=sum(values[i] for i in data['query_ids']))
    gold['total'] = sum(v['sum'] for v in gold.values())
    assert gold == f['expected_answer']
    assert r['completed'] and r['passed'] and r['actual_HTTP'] == r['actual_Native'] == 6
    assert r['normal_server_shutdown'] and r['normal_Native_shutdown']
    cleanup = r['cleanup']
    assert not cleanup['forced_cleanup'] and not cleanup['errors'] and not cleanup['HTTP_thread_alive_after_cleanup']
    owners = r['owner_exit_observation']['owners']
    assert len(owners) == len({v['tid'] for v in owners}) == 8
    assert {v['device'] for v in owners} == set(range(8)) and not any(v['task_exists_after_destroy'] for v in owners)
    assert c['completed'] and c['passed'] and c['actual_HTTP_attempts'] == c['actual_HTTP_completed'] == 6
    assert len(r['requests']) == len(c['requests']) == len(w['rows']) == 6
    rows = []
    for i, (native, client, wire, chunk) in enumerate(zip(r['requests'], c['requests'], w['rows'], [1024,2048,1024,2048,2048,1024])):
        assert native['admitted_index'] == client['index'] == i and client['chunk'] == chunk
        assert native['prefill_binding']['prefill_chunk'] == chunk
        assert native['normal_reset_before'] and native['normal_reset_after']
        assert native['prompt_tokens'] == f['actual_prompt_tokens'] == 8489
        assert native['admission']['history_render_audit']['Native_prompt_token_ids_SHA256'] == f['actual_prompt_token_ids_SHA256']
        assert native['native_finish_reason'] == 'eos_token' and native['generated_token_ids'][-1] == 1
        assert len(native['generated_token_ids']) == native['actual_generated_tokens'] == 98
        response = wire['response']; choice = response['choices'][0]
        assert response['id'] == native['request_id'] == wire['request_id']
        assert wire['complete_frozen_messages_and_tools_equal'] and choice['finish_reason'] == 'stop'
        text = choice['message']['content'].strip()
        assert text.startswith('```json\n') and text.endswith('\n```')
        answer = json.loads(text[8:-4]); assert answer == gold
        for name in ['left.json', 'right.json']: assert type(answer[name]['sum']) is int
        assert type(answer['total']) is int
        assert wire['wire_SHA256'] == client['wire_SHA256']
        trace = [t for t in native['Native_continuation'] if t['kind'] == 'prefill']
        past = 0
        for t in trace:
            assert t['past'] == past and t['tokens'] == min(chunk, 8489-past)
            past += t['tokens']; assert t['total'] == past
        assert past == 8489 and len(trace) == math.ceil(8489/chunk)
        forward = native['Native_forward_host_summary']
        assert forward['prefill_calls'] == len(trace) and forward['decode_calls'] == 97
        ticks = [t for t, token in zip(native['backend_token_monotonic'], native['generated_token_ids']) if token != 1]
        assert len(ticks) == 97 and all(b > a for a,b in zip(ticks,ticks[1:]))
        visible_rate = (len(ticks)-1)/(ticks[-1]-ticks[0])
        assert math.isclose(visible_rate, client['backend_emitter_decode_tok_s'], rel_tol=1e-12)
        rows.append(dict(index=i, chunk=chunk, warmup=i<2, prefill_s=forward['prefill_s'],
                         HTTP_s=client['client_HTTP_wall_s'], decode_s=forward['decode_s'],
                         backend_TTFT_s=ticks[0]-native['started_monotonic'], visible_decode_tok_s=visible_rate))
    paired = {}
    for key in ['prefill_s','HTTP_s','decode_s','backend_TTFT_s','visible_decode_tok_s']:
        a = (rows[2][key]+rows[5][key])/2; b = (rows[3][key]+rows[4][key])/2
        paired[key] = dict(A1024=a, B2048=b, ratio_B_over_A=b/a,
                           A_endpoint_drift=rows[5][key]/rows[2][key]-1,
                           B_endpoint_drift=rows[4][key]/rows[3][key]-1)
    assert summary['post']['kfd_pids'] == [] and summary['post']['HTTP43882_bindable']
    assert '910374.1257|' in summary['sacct'] and '910407.669|' in summary['sacct']
    return dict(root_audit_passed=True, actual_GPU='910374.1257', CPU='910407.669', post='910374.1258',
                original_source_commit=summary['source_publication_commit'], Native_SHA256=r['Native_SHA256'],
                quality='6/6 independently recomputed exact file nonce/sums/total;98includingEOS/97visible;full8489input;normalreset',
                all8_unique_owner_threads_exited=True, normal_lifecycle=True, post_clean=True,
                rows=rows, paired=paired, request_body_published=False,
                evidence_SHA256={p.name:sha(p) for p in [report_path,client_path,wire_path,summary_path]},
                fixture_SHA256=sha(fixture), actual_profile_scope='Native wrapper host wall plus backend emitter and bufferedHTTP, not kernel timing',
                scope='One frozen actual8K C1 tool workload, one ABBA after two excludedwarmups. Keep opt-in2048 candidate for this scope; no globaldefault promotion.',
                decode_result='Observed about3% slower; no>15tok/s qualification',
                performance_round=True, full_goal_complete=False, limitations=['No broadbenchmark noninferiority','No clientincrementalTTFT/ITL SLO','Memory checkpoints are not measuredpeak','No GPUbusy inference','Callcount9to5 is observed; subkernel/communication contribution not isolated'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory',type=Path,default=Path('results/bw1000/2026-11-02'))
    p.add_argument('--fixture',type=Path,default=Path('code/bw1000/Deep-prefill-chunk-pair-r348/repair-r2/owner/frozen-toolprompt.json'))
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=audit(a.directory,a.fixture)
    with a.output.open('x',encoding='utf8',newline='\n') as out: json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result['paired']))
