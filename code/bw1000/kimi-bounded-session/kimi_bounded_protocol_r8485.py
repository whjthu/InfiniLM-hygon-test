"""One already-loaded real Engine per PP stage, TP8 each; finite single epoch.

Reuse r8235's verified loader/owner destroy path. No math compilation or HTTP.
Actual source/dependency and GPU qualification are still required before launch.
"""
import ctypes,hashlib,json,pathlib,threading,time
from online_kimi_provider_r8467 import KimiLoadedAdapter


def sequence_diff(actual,expected,eos):
    first=next((i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b),
               min(len(actual),len(expected)) if len(actual)!=len(expected) else None)
    return dict(exact=actual==expected,first_difference_ordinal0=first,
                actual_token=actual[first] if first is not None and first<len(actual) else None,
                expected_token=expected[first] if first is not None and first<len(expected) else None,
                actual_length=len(actual),expected_length=len(expected),
                actual_EOS_positions=[i for i,v in enumerate(actual) if v in eos],
                expected_EOS_positions=[i for i,v in enumerate(expected) if v in eos])


class BorrowedAdapter(KimiLoadedAdapter):
    """Engine ownership stays with old r8235 runner's checked finally destroy.

    close here closes this adapter only. No broker receipt proves native destroy;
    outer runner report+actual terminal/postKFD must prove all owner cleanup.
    """
    def close(self):self.closed=True


def run(g):
    lib,ctx,control,report=g['lib'],g['ctx'],g['control'],g['report']
    node,r,tag,check=g['node'],g['r'],g['tag'],g['check']
    assert (g['tp_nodes'],g['pp'])==(1,4)
    assert lib.modern_kimi_vocab_head_get_r8341()==0
    import os
    assert os.environ['INFINILM_KIMI_N4_TILE_R8167']=='0'
    frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json'
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625'
    frozen=json.loads(frozen_path.read_text())
    chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())
    known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
    eos=set(chat['eos_token_ids']);assert eos==set(frozen['eos_token_ids'])
    def gold_for(task):
        key='sort' if task.startswith('sort_probe') else task
        return [v['generated_ids'][0] for v in known['cases'] if v['task']==key and v['grouped']]
    early=[(task,seq) for task in chat['cases'] for seq in gold_for(task) if len(seq)<128 and seq[-1] in eos]
    assert early,'No independently frozen earlyEOS QA prompt available; do not fabricate EOS'
    early_task,early_gold=min(early,key=lambda x:len(x[1]))
    history=next(v for v in frozen['geometries'] if (v['B'],v['M'])==(16,4))['fixed_complete_IDs']
    inputs=[frozen['natural_input_ids'][i%2] for i in range(7)]+[chat['cases'][early_task]['input_ids']]
    report.update(online_protocol='r8485-bounded-prefill-wait-O-W-W-O',GPU_qualification=False,QA=[],early_EOS_task=early_task,
                  scope='Four actual live71 O/W/W/O epochs each samejobtrace CPUoracle; B16historical anchor separate; no clientdelivery' ,
                  online_inputs_SHA256=hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest())
    report['phase_timings']=[]
    report['static_results']=[]
    def memgate():
        assert all(free>512*1024**2 for free,total in g['memory']()),'512MiB forward floor'
    def adapter():
        return BorrowedAdapter(lib,ctx,stage=node,stages=4,rows=32,check=check,memory_gate=memgate,destroy=lambda p:None)
    def reset(label):
        check(lib.modern_kimi_engine_reset_r2475(ctx))
        control.collect('online-reset:'+label,{'all8owners_native_reset_completed':True})
        assert all(free>5*1024**3 for free,total in g['memory']()),'phase5GiBreserve'
    def static(prompts,label,expected,micro=4):
        phase_start=time.perf_counter();reset_start=phase_start
        report['phase']=label;g['save']()
        reset(label);reset_end=time.perf_counter();a=adapter();seq=0;outputs=[[] for _ in prompts];pasts=[0]*len(prompts)
        trace_path=r/'results'/f'Kimi-bounded-static-r8485-{tag}-node{node}-{label}.jsonl'
        trace=trace_path.open('x')
        def forward(indices,ins):
            nonlocal seq
            j=dict(seq=seq,kind='prefill' if all(pasts[i]==0 for i in indices) else 'decode',
                   keys=indices,rows=[i+1 for i in indices],pasts=[pasts[i] for i in indices],inputs=ins)
            call_start=time.perf_counter();ids=a.forward(j);local_return=time.perf_counter()
            packet=control.collect('static:'+label+':'+str(seq),{'ids':ids})
            ids=packet[3]['ids'];assert ids and len(ids)==len(indices)
            for i,w,v in zip(indices,map(len,ins),ids):outputs[i].append(v);pasts[i]+=w
            # Untimed static qualification: persist every returned token before
            # any final reference assertion. Do not use these fsync-heavy
            # controls to advertise throughput. Preserve actual source geometry.
            record=dict(label=label,job=seq,rows=j['rows'],pasts=j['pasts'],
                        widths=list(map(len,ins)),returned_emitter_IDs=ids,
                        local_CAPI_s=local_return-call_start,
                        coordinator_collect_s=time.perf_counter()-local_return,
                        phase_elapsed_s=time.perf_counter()-phase_start,
                        outputs_so_far=outputs)
            trace.write(json.dumps(record)+'\n');trace.flush();os.fsync(trace.fileno())
            report['static_active']=dict(label=label,job=seq,outputs=outputs,
                                         elapsed_s=time.perf_counter()-phase_start)
            g['save']()
            seq+=1
        for offset in range(0,len(prompts),2):
            ii=list(range(offset,min(offset+2,len(prompts))));forward(ii,[prompts[i] for i in ii])
        for step in range(127):
            active=[i for i,v in enumerate(outputs) if len(v)<128 and v[-1] not in eos]
            if not active:break
            for offset in range(0,len(active),micro):
                ii=active[offset:offset+micro];forward(ii,[[outputs[i][-1]] for i in ii])
        trace.close()
        phase_end=time.perf_counter()
        refs=[next((v for v in expected if v==outputs[0]),expected[0])] if len(prompts)==1 else expected
        diffs=[sequence_diff(v,ref,eos) for v,ref in zip(outputs,refs)]
        report['static_results'].append(dict(label=label,actual_outputs=outputs,expected_outputs=refs,
                                            per_request_diff=diffs,trace_path=str(trace_path),
                                            trace_SHA256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                                            reference_scope='Historical B16M4 full16 natural512 exactgeometry anchor; never substituted for N8 matchedtrace' if len(prompts)>1 else 'Independent frozen standalone QA'))
        report['phase_timings'].append(dict(label=label,reset_s=reset_end-reset_start,
                                           total_s=phase_end-phase_start,scope='Static qualification includes deliberate per-call persistence/control; not throughput'))
        g['save']() # full outputs + firstdiff/EOS/timing saved BEFORE assertion.
        assert all((len(v)==128 or v[-1] in eos) and not any(t in eos for t in v[:-1]) for v in outputs)
        if len(prompts)==1:assert outputs[0] in expected,(label,'complete_frozen_QA_ID_mismatch')
        else:assert outputs==expected,(label,'complete_frozen_static_ID_mismatch')
        a.close()
        return outputs
    for task,c in chat['cases'].items():
        static([c['input_ids']],'QA-before-'+task,gold_for(task),1)
        report['QA'].append(dict(epoch='before',task=task,complete_frozen_IDs_EOS=True));g['save']()
    report['before8QA_passed']=True
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1'
    control.collect('historical-anchor-policy1',{})
    anchor_inputs=[frozen['natural_input_ids'][i%2] for i in range(16)]
    static(anchor_inputs,'B16M4-historical-anchor',history,4)
    report['historical_B16M4_anchor_passed']=True;g['save']()
    flags={k:os.environ.get(k) for k in ['INFINILM_KIMI_N4_TILE_R8167','INFINILM_KIMI_VOCAB_HEAD_R8341','INFINILM_KIMI_PACKED_AUTO_R6136','INFINILM_KIMI_METADATA_PACK_R5858','INFINILM_KIMI_BOUNDED_BF16_R6054','INFINILM_KIMI_STATIC_DIV_R6020']}
    assert all(v=='0' for v in flags.values())
    from kimi_matched_trace_r8467 import digest
    identity=dict(DSO_SHA256=hashlib.sha256(g['library'].read_bytes()).hexdigest(),
                  SDK_SHA256=report['libinfiniops.so_sha256'],TP=8,PP=4,rows=32,pages_per_request=4,
                  config_SHA256=hashlib.sha256((g['model']/'config.json').read_bytes()).hexdigest(),
                  flags=flags,serving_policy=1,inputs_SHA256=digest(inputs))
    from kimi_bounded_batch_wait_r8485 import consume_plan,run_loaded_epochs
    plan,plan_SHA=consume_plan(r/'incoming/Kimi-fixed-arrival-plan-r8485.json',inputs)
    report['bounded_wait_plan_file_SHA256']=plan_SHA;g['save']()
    run_loaded_epochs(g,inputs,eos,reset,identity,plan)
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='0'
    control.collect('after-QA-policy0',{})
    for task,c in chat['cases'].items():
        static([c['input_ids']],'QA-after-'+task,gold_for(task),1)
        report['QA'].append(dict(epoch='after',task=task,complete_frozen_IDs_EOS=True));g['save']()
    report['serving_protocol_complete']=True
    report['GPU_qualification_pending_independent_terminal_audit']=True
    report['scope']='Historical exactB16 anchor plus four live O/W/W/O with sameTick control and fixed intended externaloffsets, each independent full-logit CPUoracle exacttrace; coordinator only. Old588 diagnostic, no client/puredecode/infinite-service claim.'
    g['save']()
