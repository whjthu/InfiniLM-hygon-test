"""Emitter checks derived from r418; actual425 source/report and new HOST trace."""
import hashlib,json,math,time
from pathlib import Path
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
EOS={154820,154827,154829}
def read(p):return json.loads(Path(p).read_bytes())
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
def digest(v):return hashlib.sha256(v).hexdigest()

def verify_native_calls(row,finite_expected=True):
    calls=row['native_calls'];assert calls
    assert all(c['CAPI_returned'] and c['CAPI_RC']==0 and c['finite'] in (True,None) for c in calls)
    assert all(c['finite'] is (True if finite_expected else None) for c in calls), 'Stage-specific logits finite contract'
    assert all(math.isfinite(c['Native_CAPI_s']) and c['Native_CAPI_s']>=0 for c in calls)

def verify_trace(row,window,encoded):
    traces=row['prefill_window_audit'];assert len(traces)==1
    t=traces[0];chunks=encoded['prefill_chunk_tokens'];groups=encoded['expected_control_groups'][str(window)]
    assert t['prefill'] and t['requested_window']==window and t['planned_chunks']==len(chunks)
    assert t['planned_chunk_tokens']==chunks and t['completed'] and not t['canceled'] and t['error'] is None
    assert t['complete_prefill_window_counter_qualified'] and t['successful_forward_count']==len(chunks)
    assert t['sample_calls']==1 and t['execute_discard_calls']==1 and t['last_unvoted_successful_forwards']==0
    assert t['actual_committed_group_chunks']==groups and t['vote_callbacks']==len(groups)+1
    assert [c['tokens']for c in t['forward_calls']]==chunks and all(c['returned']for c in t['forward_calls'])
    assert [c['final']for c in t['forward_calls']]==[False]*(len(chunks)-1)+[True]
    cumulative=[0]
    for n in groups:cumulative.append(cumulative[-1]+n)
    assert [v['after_successful_forwards']for v in t['votes']]==cumulative
    assert all(v['cancel']==0 for v in t['votes'])
    assert t['planned_pair_control_can_differ']==(len(chunks)>2)
    assert t['W4_group_larger_than_W2_observed']==(window==4 and len(chunks)>2)
    return t

def owner_row(args, ready, response_id, local_index, window, encoded, usage, finish):
    end = min(time.monotonic()+8, time.monotonic()+max(0,args.deadline_unix-time.time()))
    while time.monotonic() < end:
        q = read(args.service_report)
        assert q['actual_step'] == args.expected_step and q['run'] == ready['identity']['run']
        assert q['Native_SHA256'] == NATIVE and q['source_manifest_SHA256'] == args.loader_manifest_sha
        assert q['service_profile_manifest_SHA256'] == args.profile_manifest_sha
        rows = [r for r in q['service_requests'] if r.get('request_id') == response_id]
        assert len(rows) <= 1
        if rows and rows[0].get('final_metadata') and rows[0].get('normal_reset_after'):
            row = rows[0]
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Completed original owner receipt unavailable; no retry')
    assert row['request_index'] == row['admitted_index'] == args.native_offset+local_index
    assert row['native_profile'] == 'glm-nov16-w'+str(window)
    assert row['prefill_control_window'] == window and row['service_profile_manifest_SHA256'] == args.profile_manifest_sha
    assert row['normal_reset_before'] and row['normal_reset_after'] and row['original_reset_all16_completed']
    a, h = row['admission'], row['admission']['history_render_audit']
    assert h['qualified'] and h['prequeue_bound'] and not h['markers_sent_to_Native']
    assert row['max_new_tokens'] == row['requested_output_budget'] == 4096
    assert row['prompt_tokens'] == encoded['prompt_tokens'] == usage['prompt_tokens'] == h['prompt_tokens']
    assert row['actual_prompt_token_ids_SHA256'] == encoded['prompt_IDs_SHA256'] == a['Native_prompt_token_ids_SHA256']
    assert sum(c['tokens'] for c in row['Native_continuation'] if c['kind']=='prefill') == row['prompt_tokens']
    assert all(c['kind'] in ('prefill','decode') for c in row['Native_continuation'])
    assert all(1 <= c['tokens'] <= 512 for c in row['Native_continuation'] if c['kind']=='prefill')
    assert row['prompt_tokens']+4096 <= 32768
    ids = row['generated_token_ids']
    assert ids and len(ids) == row['actual_generated_tokens'] == usage['completion_tokens']
    decode = [c for c in row['Native_continuation'] if c['kind']=='decode']
    assert len(decode)==len(ids)-1 and all(c['tokens']==1 for c in decode)
    assert all(type(t) is int and 0 <= t < 154880 for t in ids)
    assert digest(compact(ids)) == row['generated_token_ids_SHA256']
    natural = row['native_finish_reason'] == 'eos_token'
    assert row['native_finish_reason'] in ('eos_token','length')
    if natural:
        assert ids[-1] in EOS and not any(t in EOS for t in ids[:-1]) and finish != 'length'
    else:
        assert len(ids) == 4096 and not any(t in EOS for t in ids)
    ticks = row['backend_token_monotonic']
    assert len(ticks)==len(ids) and all(math.isfinite(t) for t in ticks)
    assert all(x<=y for x,y in zip(ticks,ticks[1:]))
    assert a['accepted_monotonic'] <= row['started_monotonic'] <= ticks[0] <= ticks[-1] <= row['ended_monotonic']
    visible = [t for t,i in zip(ticks,ids) if i not in EOS]
    gaps = sorted(y-x for x,y in zip(visible,visible[1:]))
    def percentile(p):
        if not gaps:return None
        k=(len(gaps)-1)*p;i=int(k);return gaps[i]+(gaps[min(i+1,len(gaps)-1)]-gaps[i])*(k-i)
    trace = verify_trace(row, window, encoded)
    assert row['generated_natural_EOS'] is natural
    verify_native_calls(row)
    return dict(prefill_window_audit=trace, natural_EOS=natural, cap_hit=not natural, generated_tokens=len(ids),
        visible_tokens=len(visible), generated_IDs_SHA256=row['generated_token_ids_SHA256'],
        backend_TTFT_from_admission_s=visible[0]-a['accepted_monotonic'] if visible else None,
        backend_visible_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0]) if len(visible)>1 and visible[-1]>visible[0] else None,
        backend_ITL_P95_s=percentile(.95),backend_ITL_P99_s=percentile(.99),
        Native_forward_host_summary=row['Native_forward_host_summary'],
        owner_row_compact_SHA256=digest(compact(row)),normal_reset=True)

