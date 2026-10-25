"""One bounded real-scheduler candidate: interleave existing M2 prefill with
one generic small-M decode wave, then finish active rows. No Engine threads,
padding, timestamp edits, or new network protocol. Geometry changes explicitly.
"""
import math

def run_interleaved(*, inputs, eos, forward, clock, state, label, limit=128, token_ready_clock=None):
    assert len(inputs)==16 and all(len(v)==512 for v in inputs)
    assert 1<limit<=128 and state=={}
    B=16;ids=[[]for _ in range(B)];pasts=[0]*B;times=[[]for _ in range(B)]
    state.update(label=label,completed=False,generated_ids=ids,token_ready_times_from_phase_start=times,
                 global_physical_rows=list(range(1,17)),input_lengths=[512]*16,forward_jobs=[],
                 timing_scope='Same frontend clock after actual tailfeedback/control return, emitter selected by caller; not HTTP/client. No delayed first-token emission.',
                 geometry='Prefill M2; one decode of all admitted active rows after each prefill part; N2/4/...16 then active EOS/completion shrink. Current I192 only N16, others original fallback.',
                 finaltoken_qualification='Independent frozen648/651 full128 IDs/EOS required; geometry change not mathematical equivalence claim.')
    start=clock();prefill_done=None
    def call(kind,rows,ins,oldpasts):
        assert rows==sorted(set(rows)) and all(0<=i<16 for i in rows)
        assert all(oldpasts[k]==pasts[i] for k,i in enumerate(rows))
        tick=clock();selected,seconds,_=forward(ins,oldpasts,[i+1 for i in rows],label+'-'+kind+str(len(state['forward_jobs'])),kind=='decode')
        returned=clock();assert len(selected)==len(rows) and all(type(v)is int and v>=0 for v in selected)
        state['forward_jobs'].append({'kind':kind,'rows':[i+1 for i in rows],'past_lengths':list(oldpasts),'input_lengths':[len(v)for v in ins],'N':sum(map(len,ins)),'callback_CAPI_seconds':seconds,'local_callback_complete_wall_seconds':returned-tick,'I192_expected_eligible':kind=='decode'and len(rows)==16})
        ready=returned if token_ready_clock is None else token_ready_clock()
        assert tick<=ready<=returned
        stamp=ready-start
        for k,i in enumerate(rows):
            ids[i].append(selected[k]);pasts[i]+=len(ins[k]);times[i].append(stamp)
            assert pasts[i]==512+len(ids[i])-1 and len(ids[i])<=limit
        return returned
    def active():return [i for i in range(B)if ids[i]and len(ids[i])<limit and ids[i][-1]not in eos]
    def decode():
        rows=active()
        if rows:call('decode',rows,[[ids[i][-1]]for i in rows],[pasts[i]for i in rows])
    for part in range(0,B,2):
        rows=[part,part+1];assert all(not ids[i]and pasts[i]==0 for i in rows)
        returned=call('prefill',rows,[inputs[i]for i in rows],[0,0])
        if part==B-2:prefill_done=returned-start
        decode()
    while active():decode()
    last=max(v[-1]for v in times);useful=[[(v,t)for v,t in zip(row,stamps)if v not in eos]for row,stamps in zip(ids,times)]
    assert all((len(row)==limit or row[-1]in eos)and not any(v in eos for v in row[:-1])for row in ids)
    post=[(v,t)for row,stamps in zip(ids,times)for j,(v,t)in enumerate(zip(row,stamps))if j>0 and v not in eos and t>=prefill_done];post_seconds=last-prefill_done
    state.update(completed=True,actual_final_pasts=pasts,TTFT_seconds=[v[0]for v in times],
                 service_wall_seconds=last,all_prefill_completed_seconds=prefill_done,
                 useful_outputs=sum(map(len,useful)),end_to_end_output_tok_s=sum(map(len,useful))/last,
                 post_all_prefill_decode_tok_s=len(post)/post_seconds if post_seconds>0 else None,
                 post_all_prefill_decode_scope='Only useful outputs at/after last prefill return divided by following wall; mixed-service early decode excluded, not interchangeable with accepted full decode benchmark.',
                 per_request_first_last_tok_s=[(len(row)-1)/(row[-1][1]-row[0][1])if len(row)>1 and row[-1][1]>row[0][1]else None for row in useful],
                 per_request_max_ITL_seconds=[max((b[1]-a[1]for a,b in zip(row,row[1:])),default=None)for row in useful],
                 cumulative_prefill_callback_wall_seconds=sum(v['local_callback_complete_wall_seconds']for v in state['forward_jobs']if v['kind']=='prefill'),
                 cumulative_decode_callback_wall_seconds=sum(v['local_callback_complete_wall_seconds']for v in state['forward_jobs']if v['kind']=='decode'),
                 callback_wall_limit='Local sequential host call+feedback windows can overlap GPU dependency; not GPU activity or removable wall budget.')
    return state
