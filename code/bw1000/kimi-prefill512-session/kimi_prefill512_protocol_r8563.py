"""One load: independent baseline IDs, scoped candidate, then finite service.

Only the five KDA projections and uniform512 actual prefill are changed.
All outputs are persisted before classification. Complete token/EOS equality
is the Model criterion; intermediate bit equality is not a production gate.
"""
import os,json,time,ctypes,hashlib
from kimi_prefill512_scope_r8563 import ScopedProvider,MODE,ELIGIBLE
from kimi_bounded_protocol_r8485 import sequence_diff
from kimi_prefill512_live_r8563 import bounded_epoch
from kimi_matched_trace_r8467 import digest

def service_windows(events,cards=32):
    """One coordinator clock; explicit prefill-interleaved vs all-decode."""
    tokens=[e for e in events if e['kind']=='token']
    first=[e['t']for e in tokens if e['index']==0]
    assert len(first)==8
    end=max(e['t']for e in tokens);after=[e for e in tokens if e['index']>0 and not e['eos']]
    start_all=max(first);start_post=min(first)
    n_all=sum(e['t']>start_all for e in after)
    empty_post=not after or end<=start_post;empty_all=n_all==0 or end<=start_all
    post_rate=None if empty_post else len(after)/(end-start_post)
    all_rate=None if empty_all else n_all/(end-start_all)
    return dict(service_postfirst_window=[start_post,end],service_postfirst_tok_s=post_rate,
        service_postfirst_tok_s_per_card=None if post_rate is None else post_rate/cards,service_postfirst_empty=empty_post,
        service_postfirst_scope='Prefill can overlap/interleave; not pure decode',
        all_decode_window=[start_all,end],all_decode_tok_s=all_rate,
        all_decode_tok_s_per_card=None if all_rate is None else all_rate/cards,all_decode_tokens=n_all,all_decode_empty=empty_all,
        all_decode_scope='After every request firsttoken; no further prefill in this finite epoch; coordinator observed',
        client_delivery='NOT_MEASURED')

def run(g):
    lib,ctx,control,report=g['lib'],g['ctx'],g['control'],g['report']
    node,r,tag=g['node'],g['r'],g['tag']
    assert (g['tp_nodes'],g['pp'])==(1,4)
    assert os.environ['INFINILM_DENSE_PREFILL_F32_R2475']=='1'
    assert lib.modern_kimi_vocab_head_get_r8341()==0
    frozen=json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text())
    chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())
    known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
    eos=set(chat['eos_token_ids']);assert eos==set(frozen['eos_token_ids'])
    inputs=[frozen['natural_input_ids'][i%2]for i in range(8)]
    assert all(len(p)==512 for p in inputs)
    report.update(scope='Uniform512 fiveprojection Model candidate; unsupported/mixed1033 uses originalfallback. FinalIDs/EOS required; timing coordinator only.',
        candidate_revision=8563,QA=[],model_comparisons=[],phase_timings=[],candidate_qualified=False,
        default_OFF=True,actualprovider='unchanged dense_prefill_f32_r2475',native_math_flags=dict(reference=67,candidate=71),
        serving_flags64='same inherited Math11 scope; no SpecScope constructor',actual_inputs_SHA256=digest(inputs))
    report.update(distinct_prompt_count=len({digest(p)for p in inputs}),request_count=8,
        prompt_diversity_scope='Eight requests repeat two natural512 prompts fourtimes each. Routing is correlated; this finite cohort does not establish general diverse-user throughput limits.')
    counter=lib.modern_kimi_prefill512_counts_r8562
    counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];counter.restype=ctypes.c_int
    def counters():
        result=[]
        for d in range(8):
            v=(ctypes.c_uint64*3)();assert counter(d,v,3)==0;result.append(list(v))
        return result
    def reset(label):
        g['check'](lib.modern_kimi_engine_reset_r2475(ctx))
        control.collect('prefill512-reset:'+label,{'normal_reset_completed':True})
        assert all(free>5*1024**3 for free,total in g['memory']())
    def phase(label,mode,serving):
        # Previous phase has already drained and all-owner reset returns before
        # changing process-global flags. Every Engine call runs mainthread;
        # producer/socket threads only send/queue and never touch these flags.
        reset(label)
        assert os.environ.get(ELIGIBLE,'0')=='0'
        os.environ[MODE]=str(mode);os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=str(serving)
        control.collect('prefill512-mode:'+label,dict(mode=mode,serving=serving))
        report['phase']=label;g['save']()
    def adapter(full):
        def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
        return ScopedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full)
    def static(prompts,label,mode,prefill_micro,*,full=False,serving=1,expected=None):
        start=time.perf_counter();phase(label,mode,serving)
        a=adapter(full);seq=0;outputs=[[]for _ in prompts];pasts=[0]*len(prompts);records=[]
        path=r/'results'/f'Kimi-prefill512-{tag}-{label}-node{node}-outputs.json'
        def persist():
            value=dict(label=label,mode=mode,prefill_micro=prefill_micro,decode_micro=4,
                outputs=outputs,records=records,phase_s=time.perf_counter()-start,counters=counters())
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
        def call(ii,ins):
            nonlocal seq
            job=dict(seq=seq,kind='prefill'if all(pasts[i]==0 for i in ii)else'decode',keys=ii,rows=[i+1 for i in ii],pasts=[pasts[i]for i in ii],inputs=ins)
            tick=time.perf_counter();ids=a.forward(job)
            parts=control.collect('prefill512:'+label+':'+str(seq),dict(job_SHA256=digest(job),ids=ids,oracle=a.reference_observation if full and node==3 else None))
            assert all(v['job_SHA256']==digest(job)for v in parts)
            ids=parts[3]['ids'];assert ids and len(ids)==len(ii)
            oracle=parts[3]['oracle']
            for i,row,token in zip(ii,ins,ids):outputs[i].append(token);pasts[i]+=len(row)
            records.append(dict(job=job,ids=ids,oracle=oracle,call_collect_s=time.perf_counter()-tick));persist()
            if full:assert oracle['finite']and oracle['native_equals_CPU']
            seq+=1
        try:
            for offset in range(0,len(prompts),prefill_micro):
                ii=list(range(offset,min(offset+prefill_micro,len(prompts))));call(ii,[prompts[i]for i in ii])
            for _ in range(127):
                ii=[i for i,out in enumerate(outputs)if len(out)<128 and out[-1]not in eos]
                if not ii:break
                for offset in range(0,len(ii),4):
                    jj=ii[offset:offset+4];call(jj,[[outputs[i][-1]]for i in jj])
        finally:a.close()
        valid=all(out and (len(out)==128 or out[-1]in eos)and not any(v in eos for v in out[:-1])for out in outputs)
        diffs=[sequence_diff(v,ref,eos)for v,ref in zip(outputs,expected)]if expected is not None else None
        row=dict(label=label,mode=mode,actual_outputs=outputs,full_logit_reference=full,valid_EOS=valid,
            per_request_diff=diffs,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),counters=counters())
        report['model_comparisons'].append(row);report['phase_timings'].append(dict(label=label,wall_s=time.perf_counter()-start,scope='Untimed numerical controls include durable actuals/control/oracle; not TPS'))
        g['save']() # final complete IDs/EOS/firstdiff persisted before classification.
        assert valid
        return outputs
    def qa(mode,label):
        # Two independent frozen prompts, including naturally early EOS. No
        # cross-geometry B16 gold is applied to these standalone prompts.
        tasks=[]
        for task,c in chat['cases'].items():
            key='sort'if task.startswith('sort_probe')else task
            refs=[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']]
            tasks.append((min(map(len,refs)),task,c,refs))
        for _,task,c,refs in sorted(tasks)[:2]:
            actual=static([c['input_ids']],label+'-'+task,mode,1,full=False,serving=0)
            report['QA'].append(dict(label=label,task=task,exact=actual[0]in refs,actual_outputs=actual))
            g['save']()
    qa(0,'QA-before')
    baseline=static(inputs,'O-reference-M1',0,1,full=True)
    candidate=static(inputs,'N-candidate-M4',1,4,expected=baseline)
    # Baseline is frozen before candidate executes and never rewritten by N/W.
    report['baseline_IDs_SHA256']=digest(baseline);report['static_final_IDs_equal']=candidate==baseline;g['save']()
    offsets=json.loads((r/'incoming/Kimi-fixed-arrival-plan-r8485.json').read_text())['offsets_s']
    plan=dict(offsets_s=offsets,prompt_SHA256=[digest(p)for p in inputs])
    live=[]
    for i,(label,mode,wait)in enumerate([('O',0,0),('N',1,0),('waittrace',1,.01)]):
        phase('live-'+label,mode,1)
        info=bounded_epoch(g,inputs,eos,reset,full_reference=False,label='prefill512-'+label,port_delta=12000+i*50,wait_s=wait,arrival_plan=plan)
        ordered=[info['outputs'][record['request']]for record in info['producer_schedule']]
        diffs=[sequence_diff(v,ref,eos)for v,ref in zip(ordered,baseline)]
        events=[json.loads(v)for v in __import__('pathlib').Path(info['event_path']).read_text().splitlines()]
        windows=service_windows(events)
        row=dict(label=label,mode=mode,wait_s=wait,ordered_actual_outputs=ordered,per_request_diff=diffs,final_IDs_equal=ordered==baseline,
            producer_schedule=info['producer_schedule'],counters=counters(),metric=info['metric'],scope='Actual bounded coordinator-observed service, not replay/client, uniform512 only; raw aggregate_decode field uses service observationwindow and is not pure decode.')
        row['explicit_windows']=windows
        live.append(row);report['prefill512_live']=live;g['save']()
    qa(1,'QA-after')
    report['final_prefill512_counts']=counters()
    assert all(v[0]>0 and v[1]>0 and v[2]>0 for v in report['final_prefill512_counts'])
    report['candidate_qualified']=report['static_final_IDs_equal']and all(x['final_IDs_equal']for x in live)and all(x['exact']for x in report['QA'])
    report['generation_acceptance']='QUALIFIED_UNIFORM512_ONLY'if report['candidate_qualified']else'FAILED_OBSERVED'
    report['serving_protocol_complete']=True
    report['qualification_requires_actual_terminal_32normal_postKFD']=True;g['save']()
