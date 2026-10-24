"""One load: independent baseline IDs, scoped candidate, then finite service.

Only the five KDA projections and uniform512 actual prefill are changed.
All outputs are persisted before classification. Complete token/EOS equality
is the Model criterion; intermediate bit equality is not a production gate.
"""
import os,json,time,ctypes,hashlib
from kimi_prefill512_scope_r8563 import ScopedProvider,MODE,ELIGIBLE
from kimi_bounded_protocol_r8485 import sequence_diff
from kimi_static_cut_capture_r8572 import KEYS as TRACE_KEYS
from kimi_static_localize_r8572 import choose_detail
from kimi_matched_trace_r8467 import digest
from kimi_fixed512_prefill_r8624 import prefill_phase

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
    report.update(scope='OriginalMath fixedM512 PPprefill collect batching qualification only; decodeM4 unchanged; tailready/emitter andnode0visible separated. NotHTTP/client or fastestservicegain.',
        candidate_revision=8624,QA=[],model_comparisons=[],phase_timings=[],candidate_qualified=False,
        default_OFF=True,actualprovider='unchanged dense_prefill_f32_r2475',native_math_flags=dict(reference=67,candidate=71),
        serving_flags64='same inherited Math11 scope; no SpecScope constructor',actual_inputs_SHA256=digest(inputs))
    report.update(distinct_prompt_count=len({digest(p)for p in inputs}),request_count=8,
        prompt_diversity_scope='Eight requests repeat two natural512 prompts fourtimes each. Routing is correlated; this finite cohort does not establish general diverse-user throughput limits.')
    counter=lib.modern_kimi_prefill512_counts_r8562
    counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];counter.restype=ctypes.c_int
    ccl_counter=lib.modern_kimi_CCL512_counts_r8603
    ccl_counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];ccl_counter.restype=ctypes.c_int
    def ccl_counters():
        out=[]
        for d in range(8):
            v=(ctypes.c_uint64*3)();assert ccl_counter(d,v,3)==0;out.append(list(v))
        return out
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
        os.environ[MODE]=str(mode);os.environ['INFINILM_KIMI_CCL512_R8603']=str(mode);os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=str(serving)
        control.collect('prefill512-mode:'+label,dict(mode=mode,CCL512=mode,serving=serving))
        report['phase']=label;g['save']()
    def adapter(full):
        def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
        return ScopedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full)
    def static(prompts,label,mode,prefill_micro,*,full=False,serving=1,expected=None,timed=False,coarse=False,prefill_collect_each=None):
        start=time.perf_counter();phase(label,mode,serving)
        a=adapter(full);seq=0;outputs=[[]for _ in prompts];pasts=[0]*len(prompts);records=[]
        path=r/'results'/f'Kimi-fixed512-PP-{tag}-{label}-node{node}-outputs.json'
        first_call=None;decode_start=None;prefill_end=None;token_times=[[]for _ in prompts];trace_records=[]
        def persist():
            value=dict(label=label,mode=mode,prefill_micro=prefill_micro,decode_micro=4,
                outputs=outputs,records=records,phase_s=time.perf_counter()-start,counters=counters())
            value.update(CCL512_counters=ccl_counters(),token_times=token_times,clock='This PP owner mainthread monotonic after allowner token feedback; node0 authoritative; not client')
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
        def call(ii,ins):
            nonlocal seq,first_call,decode_start,prefill_end
            job=dict(seq=seq,kind='prefill'if all(pasts[i]==0 for i in ii)else'decode',keys=ii,rows=[i+1 for i in ii],pasts=[pasts[i]for i in ii],inputs=ins)
            trace=None;packed=None;old_trace={k:os.environ.get(k)for k in TRACE_KEYS}
            if coarse and 0 in ii and pasts[0]in(0,512):
                packed=ii.index(0);trace=r/'results'/('Kimi-fixed512-PP-trace-'+tag)/label/('stage'+str(node))/('job'+str(seq));trace.mkdir(parents=True,exist_ok=False)
                os.environ.update(INFINILM_KIMI_TRACE_DIR=str(trace),INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506=','.join(map(str,range(94))),INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506='stage_input,decoder,output_attn_res,final_norm',INFINILM_KIMI_TRACE_HASH_ONLY_R8506='1',INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508=str(packed),INFINILM_KIMI_KDA_TRACE_LAYER='999',INFINILM_KIMI_TRACE_LAYERS='1000',INFINILM_KIMI_TRACE_FULL_FIRST='1000',INFINILM_KIMI_TRACE_FULL_LAST='1000',INFINILM_KIMI_TRACE_KV_POOL_R8511='0')
            tick=time.perf_counter()
            if first_call is None:first_call=tick
            if job['kind']=='decode'and decode_start is None:decode_start=tick
            try:ids=a.forward(job)
            finally:
                for k,v in old_trace.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
            parts=control.collect('prefill512:'+label+':'+str(seq),dict(job_SHA256=digest(job),ids=ids,oracle=a.reference_observation if full and node==3 else None,trace_dir=str(trace)if trace else None))
            assert all(v['job_SHA256']==digest(job)for v in parts)
            ids=parts[3]['ids'];assert ids and len(ids)==len(ii)
            oracle=parts[3]['oracle']
            feedback=time.perf_counter()
            if job['kind']=='prefill':prefill_end=feedback
            for i,row,token in zip(ii,ins,ids):outputs[i].append(token);pasts[i]+=len(row);token_times[i].append(feedback)
            records.append(dict(job=job,ids=ids,oracle=oracle,call_collect_s=feedback-tick,feedback_t=feedback))
            trace_records.append(dict(job=job,packed_target=packed,perstage_trace_dirs=[v['trace_dir']for v in parts]))
            if not timed:persist()
            if full:assert oracle['finite']and oracle['native_equals_CPU']
            seq+=1
        try:
            if prefill_collect_each is None:
                for offset in range(0,len(prompts),prefill_micro):
                    ii=list(range(offset,min(offset+prefill_micro,len(prompts))));call(ii,[prompts[i]for i in ii])
            else:
                # Proposed ordinary-only fixedM512 arm. MODE0/CCL0, exact same
                # physicalrows, past0, input order. Existing native defer path.
                assert mode==0 and prefill_micro==1 and len(prompts)==8 and not full and not coarse
                jobs=[dict(seq=i,kind='prefill',keys=[i],rows=[i+1],pasts=[0],inputs=[prompts[i]])for i in range(8)]
                partial=[]
                def persist_prefill(record):
                    partial.append(record)
                    report.setdefault('fixed512_prefill_progress',{})[label]=dict(stage=node,actual_prefix=partial)
                    g['save']() # qualification only; ordinary callback is NOP.
                pref=prefill_phase(a,jobs,stage=node,control=control,label=label+':prefill',collect_each=prefill_collect_each,
                                   now_ns=lambda:int(time.perf_counter()*10**9),
                                   on_actual=(lambda record:None)if timed else persist_prefill)
                first_call=pref['clock']['phase_start_ns']*1e-9
                prefill_end=pref['clock']['phaseend_visible_ns']*1e-9
                own_first={v['seq']:v['visible_ns']*1e-9 for v in pref['feedback']}if node==0 else ({v['seq']:v['local_ready_ns']*1e-9 for v in pref['records']}if node==3 else {})
                for i,ids in enumerate(pref['ids']):
                    first_visible=own_first.get(i,prefill_end)
                    outputs[i].append(ids[0]);pasts[i]=512;token_times[i].append(first_visible)
                    records.append(dict(job=jobs[i],ids=ids,oracle=None,feedback_t=first_visible,
                        feedback_scope='Node0 actualcollect-visible; node3 own tailready; otherstages phaseend only. Not client/crosshost'))
                seq=8;report.setdefault('fixed512_prefill',[]).append(dict(label=label,**pref))
            for _ in range(127):
                ii=[i for i,out in enumerate(outputs)if len(out)<128 and out[-1]not in eos]
                if not ii:break
                for offset in range(0,len(ii),4):
                    jj=ii[offset:offset+4];call(jj,[[outputs[i][-1]]for i in jj])
        finally:
            finish=time.perf_counter();a.close();persist()
        valid=all(out and (len(out)==128 or out[-1]in eos)and not any(v in eos for v in out[:-1])for out in outputs)
        diffs=[sequence_diff(v,ref,eos)for v,ref in zip(outputs,expected)]if expected is not None else None
        row=dict(label=label,mode=mode,actual_outputs=outputs,full_logit_reference=full,valid_EOS=valid,
            per_request_diff=diffs,prefill_collect_each=prefill_collect_each,CCL512_counters=ccl_counters(),trace_records=trace_records,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),counters=counters())
        report['model_comparisons'].append(row);report['phase_timings'].append(dict(label=label,wall_s=time.perf_counter()-start,scope='Ordinary buffered timing'if timed else'Untimed numerical controls include durable actuals/control/oracle; not TPS'))
        if timed:
            decode_s=None if decode_start is None else finish-decode_start
            decode_actual=sum(max(0,len(v)-1)for v in outputs)
            decode_nonEOS=sum(sum(t not in eos for t in v[1:])for v in outputs)
            peruser=[]
            for out,ts in zip(outputs,token_times):
                dt=ts[-1]-ts[0]if len(ts)>1 else 0
                gaps=[b-a for a,b in zip(ts,ts[1:])]
                peruser.append(dict(actual_tokens=len(out),decode_tok_s=None if dt<=0 else (len(out)-1)/dt,decode_nonEOS_tok_s=None if dt<=0 else sum(t not in eos for t in out[1:])/dt,first_token_s=ts[0]-first_call,ITL_s=gaps))
            metric=dict(label=label,prefill_collect_each=prefill_collect_each,CCL512=mode,projection512=mode,requests=len(prompts),cards=32,prefill_s=prefill_end-first_call,prefill_input_tokens=sum(map(len,prompts)),prefill_input_tok_s=sum(map(len,prompts))/(prefill_end-first_call),decode_s=decode_s,decode_generated_tokens_including_EOS=decode_actual,decode_nonEOS_tokens=decode_nonEOS,decode_tok_s=None if not decode_s else decode_actual/decode_s,decode_nonEOS_tok_s=None if not decode_s else decode_nonEOS/decode_s,E2E_s=finish-first_call,E2E_output_tok_s=sum(map(len,outputs))/(finish-first_call),peruser=peruser,scope='Ordinary real synchronous Engine calls plus existing allowner control feedback. Numerical oracle, trace, output fsync, validation, reset excluded; first/last PP owner node0 monotonic feedback, client NOT_MEASURED. Two sequential micro4 decode groups; no HTTP/live claim.')
            metric['decode_tok_s_per_card']=None if metric['decode_tok_s']is None else metric['decode_tok_s']/32
            metric['prefill_input_tok_s_per_card']=metric['prefill_input_tok_s']/32
            metric['prefill_schedule']='diagnostic_original'if prefill_collect_each is None else('serial_helper'if prefill_collect_each else'pipeline_phaseend')
            report.setdefault('paired_timing',[]).append(metric)
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
    assert all(v['exact']for v in report['QA']),'OriginalQA failed; no candidate'
    historical=json.loads((r/'incoming/Kimi618-final-token-evidence-r8570.json').read_text())
    gold=next(v['actual_outputs']for v in historical['static']if v['label']=='O-reference-M1')
    assert historical['actual_handle']=='910375.618'and len(gold)==8 and all(len(v)==128 for v in gold)
    baseline=static(inputs,'O-reference-M1',0,1,full=True,expected=gold)
    report['historical618_exact_O8']=baseline==gold;report['baseline_IDs_SHA256']=digest(baseline);g['save']()
    assert baseline==gold,'OriginalO8 failed original618 independentgold; no candidate'
    candidate=static(inputs,'N-fixed512-PP-qualification',0,1,expected=baseline,prefill_collect_each=False)
    report['static_final_IDs_equal']=candidate==baseline;g['save']()
    if candidate==baseline:
        for repetition in range(2):
            for arm,serial in [('Oa',True),('Na',False),('Nb',False),('Ob',True)]:
                actual=static(inputs,'timing-'+str(repetition)+'-'+arm,0,1,expected=baseline,timed=True,prefill_collect_each=serial)
                if actual!=baseline:
                    report['timing_final_IDs_equal']=False;g['save']();break
            if report.get('timing_final_IDs_equal')is False:break
        else:report['timing_final_IDs_equal']=True
    else:report['paired_timing_skipped']='FAILED_OBSERVED full128/EOS/continuation; no timingpromotion'
    qa(0,'QA-after-restore')
    report['final_prefill512_counts']=counters();report['final_CCL512_counts']=ccl_counters()
    assert all(all(x==0 for x in v)for v in report['final_prefill512_counts']+report['final_CCL512_counts'])
    report['candidate_qualified']=report['static_final_IDs_equal']and report.get('timing_final_IDs_equal',False)and all(x['exact']for x in report['QA'])
    report['generation_acceptance']='QUALIFIED_FIXED512_PP_PREFILL_N8_DECODE_M4_ONLY'if report['candidate_qualified']else'FAILED_OBSERVED'
    report['qualification_scope']='Configuration/harness reuseofexisting594 deferredPP, notgainoverfastestexistingservice. Onlyprefill collectbatching changes, puredecodeM4 logic unchanged. NoHTTP/clientmeasurement.'
    report['AUX0_independent_arm']='NOT_RUN; no newmath/geometrymixedintoPPqualification'
    report['serving_protocol_complete']=True
    report['qualification_requires_actual_terminal_32normal_postKFD']=True;g['save']()
