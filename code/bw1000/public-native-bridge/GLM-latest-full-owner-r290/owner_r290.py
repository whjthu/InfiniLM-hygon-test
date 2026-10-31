"""Source-only paired W1/W2 same-context entry. No runtime/CCL/quality/speedup qualification."""
import ctypes,json,os,pathlib,queue,socket,subprocess,sys,threading,time,traceback,signal
from service_budget_r204 import NORMAL_CLEANUP_RESERVE

from glm_prefill_execute_r236 import execute_plan
from longcontext_wire_r204 import WIRE,CONTEXT,OUTPUT_MAX,WAIT,BEGIN,STOP,encode_wire,decode_wire,prefill_chunks

def finish_CPU_group(process,grace_seconds):
    if process is None:return False
    forced=False
    try:process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:forced=True
    if forced or process.returncode!=0:
        forced=True
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=10)
    return forced

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker,prepared):
    import torch
    r=pathlib.Path('/refactor');PP=4;CARDS=32;is_emitter=stage==PP-1 and rank==0
    tag=os.environ['GLM_RUN_R5040'];folder=pathlib.Path(__file__).parent
    from deadline_r290 import monotonic_deadline,check as deadline_check
    from cleanup_owned_r290 import cleanup_owned
    import copy
    deadline=monotonic_deadline()
    # The immutable plan is common source, not an HTTP option. Initial WORLD
    # agreement rejects mismatched owner modules before any Native forward.
    windows=(2,)*102+(1,2)+(1,2,2,1)*4
    agreement=torch.tensor(list(windows),device='cuda',dtype=torch.float32)
    control(agreement)
    assert agreement.cpu().tolist()==[CARDS*w for w in windows], 'All32 paired arm-plan disagreement'
    paired=payload['benchmark']['prefill_pair_r240']
    assert set(paired)=={'request_file','request_SHA256','output_root'}
    assert all(isinstance(v,str) and v for v in paired.values())
    assert len(paired['request_SHA256'])==64 and all(c in '0123456789abcdef' for c in paired['request_SHA256'])
    request_index=0;paired_input=None
    report.update(paired_actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],run=tag,source_manifest_SHA256=os.environ['GLM_SOURCE_MANIFEST_SHA'],paired_run=tag,prefill_control_windows=list(windows),paired_warmup_requests=2,
        paired_measured_requests=16,normal_reset=False,normal_shutdown=False,
        measured_pairing_qualified=False)

    report.update(scope='Latest99 full normal/history/real dualcall/GSM32/MMLU32 then fixed W1/W2 pair; buffered HTTP, no incremental client SLO qualification',service_requests=[],cohort_phases_completed=False,normal_reset_close=False,tool_protocol_commit='7ceb60d656074376b911c2a232196a9457dfa6a6 + scoped-timeout-correction + GLM namedtools preserve-all patch; agent_protocol_SHA256=c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3',model_context=CONTEXT,validation_stream_timeout_s=3600,validation_request_timeout_s=7200,served_model='GLM-5.3',HTTP_qualified=False,tool_closed_loop_qualified=False)
    def save():
        temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(report_path)
    def barrier():
        value=torch.ones(1,device='cuda');control(value);assert value.item()==CARDS
    check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0))
    # Long context retains actual indexer/TopK, not the <=2048 metadata shortcut.
    for name in ['owned_prefix_policy_r1669','feature_bypass_r1406','indexer_policy_r998']:
        check(getattr(lib,'modern_glm_model_'+name)(ctx,0))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
    # Context32768: do not call cap640-only resident/transient setters, even with OFF.
    lib.modern_glm_model_transient_report_r5688.argtypes=[ctypes.c_void_p];lib.modern_glm_model_transient_report_r5688.restype=ctypes.c_char_p
    transient=json.loads(lib.modern_glm_model_transient_report_r5688(ctx));assert transient['rows']==1 and transient['capacity']==CONTEXT and not transient['enabled']and transient['GPU_logical_bytes']==0
    report.update(default_host_KV_path=False,cap640_setters_not_called=True,constructor_rows=1)
    lib.modern_glm_model_batch_cache_policy_r3068.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.glm_reduced_output_policy_r9032.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.glm_attention_cohort_policy_r9054.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.glm_ragged_policy_r9136.argtypes=[ctypes.c_int,ctypes.c_int]
    check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,1));check(lib.glm_ragged_policy_r9136(rank,1))
    os.environ.update(GLM_SHARED_PREFIX_VALIDATE_R5814='0',GLM_CROSSREQUEST_PREFILL_R7328='0',GLM_PREFILL_PREFIX_PLAN_R8703='0',GLM_GROUPED_ROUTED_R5123='1',GLM_BATCH_ALLSELECTED_R5096='1',GLM_DECODE_ROW_DEDUP_R5386='0')
    eos=payload['benchmark']['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list)else[eos]);barrier()
    report.update(plan={'HTTP':121,'Native':120,'phase_Native_end':{'regular':29,'long-output':32,'history-retention':34,'multicall':38,'quality':102,'paired':120}},prepared_entry_identity=prepared['identity'],thinking_template_policy='Actual pinned template always opens think under both requested flags; no toggle qualification',Native_SHA256=report['library_sha256'])
    engine=server=thread=auto_client=auto_log=None;owned_pgid=None;req=None;seq=0
    try:
        if is_emitter:
            import uvicorn
            engine=prepared['engine'];api=prepared['server'];app=prepared['app'];tokenizer=prepared['tokenizer'];FinishReason=prepared['FinishReason']
            assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
            assert set(prepared['identity']['eos_token_ids'])==eos
            server=uvicorn.Server(uvicorn.Config(app,host=api.host,port=api.port,log_level='info',access_log=True));server.install_signal_handlers=lambda:None
            thread=threading.Thread(target=server.run,name='GLM-latest99-HTTP-only',daemon=False)
            deadline_check('after cold load before HTTP startup',120);thread.start();startup=time.monotonic()
            while not server.started:
                deadline_check('HTTP startup',120)
                if not thread.is_alive()or time.monotonic()-startup>40:raise RuntimeError('HTTP startup failed')
                time.sleep(.1)
            ready={'ready':True,'run':tag,'actual_step':report['actual_step'],'TP':8,'PP':4,'cards':32,'context':32768,'capacity':32768,'rows':1,'Native_DSO_SHA256':report['library_sha256'],'Native_SHA256':report['library_sha256'],'source_manifest_SHA256':os.environ['GLM_SOURCE_MANIFEST_SHA'],'prepared_entry_identity':prepared['identity'],'port':api.port,'normal_output_budget':4096,'thinking_output_budget':8192,'P0_text_buffered_until_completion':True,'prefill_pair_windows':[1,2]+[1,2,2,1]*4,'paired_request_SHA256':paired['request_SHA256'],'service_deadline_monotonic':deadline,'thinking_toggle_qualified':False}
            ready_path=r/'results'/('GLM-latest-full-owner-r290-ready-'+tag+'.json')
            with ready_path.open('x')as out:json.dump(ready,out,indent=2)
            report['service_ready']=ready;save()
            child_env=dict(os.environ);child_env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',USE_TORCH='0',USE_TF='0');child_env.pop('LD_PRELOAD',None)
            auto_log=(r/'logs'/('GLM-latest-full-owner-r290-client-'+tag+'.log')).open('x')
            auto_client=subprocess.Popen([sys.executable,'-B','-u',str(folder/'client_driver_r290.py'),'--run',tag,'--handle',ready['actual_step'],'--identity',str(ready_path),'--service-report',str(report_path),'--manifest-sha',os.environ['GLM_SOURCE_MANIFEST_SHA'],'--deadline-unix',str(float(os.environ['GLM_R290_GPUSTART_UNIX'])+27720)],env=child_env,stdout=auto_log,stderr=subprocess.STDOUT,start_new_session=True)
            owned_pgid=auto_client.pid;assert os.getpgid(auto_client.pid)==owned_pgid
            report['automatic_CPU_client_PID']=auto_client.pid;save()
        while True:
            req=None;wire=None
            if is_emitter:
                stopfile=r/'control'/('GLM-latest-full-owner-r290-stop-'+tag+'.json')
                if time.monotonic()>=deadline or server.should_exit or stopfile.exists():
                    wire=encode_wire(seq+1,STOP)
                else:
                    try:req=engine.take(timeout=.5)
                    except queue.Empty:wire=encode_wire(seq+1,WAIT)
                    else:
                        if req is None:wire=encode_wire(seq+1,STOP)
                        elif req.cancel_event.is_set():engine.record_finished(req,0,True);req.finish(req.abort_reason or FinishReason.CANCELED);wire=encode_wire(seq+1,WAIT)
                        else:engine.record_started(req);wire=encode_wire(seq+1,BEGIN,req.prompt_token_ids,req.max_new_tokens)
            seq+=1;tensor=torch.tensor(wire if is_emitter else [0]*WIRE,dtype=torch.float32,device='cuda');control(tensor);op,prompt,limit=decode_wire(tensor.cpu().tolist(),seq,vocab)
            if op==STOP:break
            if is_emitter and auto_client is not None and auto_client.poll() is not None:
                # Covers supervisor preflight failure before it could create its marker.
                stopfile=r/'control'/('GLM-latest-full-owner-r290-stop-'+tag+'.json')
                if not stopfile.exists():
                    with stopfile.open('x')as out:json.dump({'normal_stop':True,'reason':'CPU_supervisor_exited','exit_code':auto_client.returncode},out)
            if op==WAIT:continue
            assert request_index<len(windows), 'Full120 Native plan exhausted'
            if is_emitter:
                with engine._admission_lock:
                    assert req._admission_metadata_ready;admission=copy.deepcopy(req.metadata)
                backend_ticks=[];native_started=time.monotonic()
            current_input=(tuple(prompt),limit)
            if request_index>=102:
                if paired_input is None:paired_input=current_input
                assert current_input==paired_input,'Paired full prompt IDs/output budget changed'
            prefill_window=windows[request_index];pair_index=request_index;request_index+=1
            check(lib.modern_glm_model_reset(ctx));barrier();generated=[];past=0;reason='length';started=time.monotonic();calls=[]
            request_completed_normally=False
            report['normal_reset']=False
            try:
                for step in range(limit):
                    plan=prefill_chunks(prompt)if step==0 else[(past,[generated[-1]])]
                    retained=[None]
                    def vote():
                        if is_emitter and time.monotonic()>=deadline:req.mark_abort(FinishReason.TIMEOUT)
                        flag=torch.tensor([float(is_emitter and req.cancel_event.is_set())],device='cuda');control(flag)
                        return bool(flag.item())
                    def forward(logical_past,values,final):
                        assert retained[0]is None and len(values)<=512 and logical_past+len(values)<=CONTEXT
                        desc=(ctypes.c_size_t*3)(0,logical_past,len(values));ids=torch.tensor(values if stage==0 else[0]*len(values),device='cuda',dtype=torch.int64);positions=torch.arange(logical_past,logical_past+len(values),device='cuda',dtype=torch.int64);logits=torch.full((1,vocab),float('nan'),device='cuda',dtype=torch.bfloat16)
                        torch.cuda.synchronize(rank);tick=time.perf_counter();check(lib.modern_glm_model_forward(ctx,len(values),1,desc,ptr(ids),ptr(positions),ptr(logits),0,stream()));elapsed=time.perf_counter()-tick
                        if stage==PP-1:assert bool(torch.isfinite(logits).all()),'Nonfinite real Native logits'
                        calls.append({'T':len(values),'past':logical_past,'Native_CAPI_s':elapsed,'prefill':step==0,'samples_next_token':False})
                        free,total=torch.cuda.mem_get_info(rank);assert free>=512*1024**2,'Original512MiB GLM floor';report['minimum_driver_free_bytes']=min(report.get('minimum_driver_free_bytes',free),free)
                        if final:retained[0]=logits
                        del ids,positions,desc,logits
                    def discard():
                        retained[0]=None
                    def sample():
                        logits=retained[0];assert logits is not None
                        selected=torch.zeros(2,device='cuda',dtype=torch.float32)
                        if is_emitter:selected[0]=float(logits.argmax(-1).item());selected[1]=1
                        control(selected);raw=selected.cpu().tolist();assert raw[1]==1 and int(raw[0])==raw[0]and 0<=raw[0]<vocab
                        token=int(raw[0]);generated.append(token)
                        if is_emitter:
                            tick=time.monotonic();req.emit_token(token,visible=token not in eos);backend_ticks.append(tick)
                        calls[-1]['samples_next_token']=True
                        discard();del selected,logits
                        return token
                    past,token,canceled=execute_plan(plan,past,prefill=step==0,window=prefill_window if step==0 else 1,forward=forward,vote=vote,sample=sample,discard=discard)
                    if canceled:reason='canceled'
                    if reason=='canceled':break
                    if token in eos:reason='eos_token';break
                request_completed_normally=True
            finally:
                # Exception is NOT cooperative cancellation. Do not enter shared
                # WORLD drain/reset on one rank; guarded() propagates via os._exit(2)
                # and existing job fail-stop. No subsequent request on this owner.
                if request_completed_normally:
                    torch.cuda.synchronize(rank);check(lib.modern_glm_model_reset(ctx));barrier()
                    report['normal_reset']=True
                else:
                    report.update(normal_reset=False,normal_shutdown=False,normal_reset_close=False)
            row={'sequence':seq,'prompt_tokens':len(prompt),'generated_tokens':len(generated),'reason':reason,'wall_s':time.monotonic()-started,'native_calls':calls,'original_reset_all32_completed':True,'prefill_chunk':512,'requested_output_budget':limit}
            row.update(admitted_index=pair_index,pair_index=pair_index-102 if pair_index>=102 else None,prefill_control_window=prefill_window,paired_warmup=102<=pair_index<104,
                generated_token_ids_SHA256=__import__('hashlib').sha256(json.dumps(generated,separators=(',',':')).encode()).hexdigest(),
                generated_natural_EOS=bool(generated and generated[-1] in eos and reason=='eos_token'),
                actual_prompt_token_ids_SHA256=__import__('hashlib').sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest())
            if is_emitter:
                row.update(request_id=req.request_id,admission=admission,max_new_tokens=limit,normal_reset_before=True,normal_reset_after=True,native_finish_reason=reason,generated_token_ids=list(generated),actual_generated_tokens=len(generated),backend_token_monotonic=backend_ticks,started_monotonic=native_started,ended_monotonic=time.monotonic(),Native_continuation=[{'kind':'prefill'if c['prefill']else'decode','tokens':c['T']}for c in calls],Native_forward_host_summary={'prefill_calls':sum(c['prefill']for c in calls),'prefill_s':sum(c['Native_CAPI_s']for c in calls if c['prefill']),'decode_calls':sum(not c['prefill']for c in calls),'decode_s':sum(c['Native_CAPI_s']for c in calls if not c['prefill'])})
            report['service_requests'].append(row);save()
            if is_emitter:
                engine.record_finished(req,len(calls),True)
                report['queue_journal']=engine.journal_snapshot()
                queue_path=r/'results'/('GLM-latest-full-owner-r290-queue-'+tag+'.json');temp=queue_path.with_suffix('.tmp');temp.write_text(json.dumps(report['queue_journal'],indent=2));temp.replace(queue_path)
                req.metadata.update(original_reset_all32_completed=True,backend_sequence=seq)
                req.finish(FinishReason(reason));row['final_metadata']=copy.deepcopy(req.metadata);save()
        barrier()
        if is_emitter:
            engine.cancel_pending_after_global_drain(FinishReason.CANCELED)
            report['queue_journal']=engine.journal_snapshot();report['automatic_CPU_client_group_forced_stop']=finish_CPU_group(auto_client,60);report['automatic_CPU_client_exit_code']=auto_client.returncode
        vote=torch.tensor([float(is_emitter and auto_client.returncode==0 and request_index==120 and not report.get('automatic_CPU_client_group_forced_stop'))],device='cuda');control(vote);assert vote.item()==1,'Full client plan failed or incomplete; stop without retry'
        barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0));report.update(cohort_phases_completed=True,normal_reset_close=True,native_service_loop_completed=True,normal_shutdown=True,full_requests_completed=request_index,measured_pairing_qualified=False);save()
    except BaseException as error:
        report.update(normal_reset=False,normal_shutdown=False,normal_reset_close=False,service_error=repr(error),service_traceback=traceback.format_exc())
        if is_emitter and req is not None:
            try:req.fail(error)
            except BaseException:pass
        save();raise
    finally:
        active_failure=sys.exc_info()[0]is not None
        if is_emitter:
            cleanup=cleanup_owned(auto_client,server,thread,engine,auto_log,owned_pgid=owned_pgid);report['cleanup']=cleanup;report['normal_HTTP_shutdown']=cleanup['normal_server_shutdown'];save()
            if not active_failure and(cleanup['errors']or cleanup['HTTP_thread_alive_after_cleanup']or cleanup['forced_cleanup']):raise RuntimeError('GLM owned cleanup forced/incomplete; normal qualification refused')
