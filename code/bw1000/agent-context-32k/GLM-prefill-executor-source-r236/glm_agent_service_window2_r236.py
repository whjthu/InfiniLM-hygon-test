"""LOCAL SOURCE CANDIDATE ONLY: W2 same-request prefill, sequential per owner. No runtime qualification."""
import ctypes,json,os,pathlib,queue,socket,subprocess,sys,threading,time,traceback
from native_engine_bridge_r143c import existing_server_app
from service_queue_bridge_r204 import ServiceQueueBridge
from api_request_constraints_r203 import make_encode_chat
from service_budget_r204 import NORMAL_CLEANUP_RESERVE

from glm_prefill_execute_r236 import execute_plan
from longcontext_wire_r204 import WIRE,CONTEXT,OUTPUT_MAX,WAIT,BEGIN,STOP,encode_wire,decode_wire,prefill_chunks

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker):
    import torch
    r=pathlib.Path('/refactor');PP=4;CARDS=32;is_emitter=stage==PP-1 and rank==0
    tag=os.environ['GLM_RUN_R5040'];folder=pathlib.Path(__file__).parent
    deadline=time.monotonic()+max(0,report['remaining_process_seconds'])-NORMAL_CLEANUP_RESERVE
    report.update(prefill_control_window=2,normal_reset=False,normal_shutdown=False)
    report.update(scope='UNQUALIFIED W2 candidate; Original GLM Native TP8PP4/32K regular-agent validation;serial FIFO,4K native-default/8K high,complete history,8 tool steps;not performance qualification',service_requests=[],cohort_phases_completed=False,normal_reset_close=False,tool_protocol_commit='7ceb60d656074376b911c2a232196a9457dfa6a6 + scoped-timeout-correction + GLM namedtools preserve-all patch; agent_protocol_SHA256=c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3',model_context=CONTEXT,validation_stream_timeout_s=3600,validation_request_timeout_s=7200,served_model='GLM-5.3',HTTP_qualified=False,tool_closed_loop_qualified=False)
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
    engine=server=thread=auto_client=None
    if is_emitter:
        import uvicorn
        from transformers import PreTrainedTokenizerFast
        from infinilm.llm import FinishReason
        tokenizer=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True)
        assert __import__('hashlib').sha256(tokenizer.get_chat_template().encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
        engine=ServiceQueueBridge(tokenizer,CONTEXT,FinishReason,encode_chat=make_encode_chat(tokenizer),eos_token_ids=sorted(eos),default_max_tokens=4096,max_pending=8)
        api,app=existing_server_app(engine,'/old/models/ZhipuAI-GLM-5.3','glm5',host=os.environ.get('GLM_SERVICE_HOST_R143','127.0.0.1'),port=int(os.environ.get('GLM_SERVICE_PORT_R143','43843')))
        server=uvicorn.Server(uvicorn.Config(app,host=api.host,port=api.port,log_level='info',access_log=True))
        thread=threading.Thread(target=server.run,name='HTTP-only-no-Native-owner-calls',daemon=True);thread.start()
        start=time.monotonic()
        while not server.started:
            if not thread.is_alive()or time.monotonic()-start>40:raise RuntimeError('HTTP startup failed')
            time.sleep(.1)
        ready={'ready':True,'run':tag,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'stage':stage,'rank':rank,'HTTP_bind':api.host,'port':api.port,'model_path':'/old/models/ZhipuAI-GLM-5.3','Native_DSO_SHA256':report['library_sha256'],'context':CONTEXT,'rows':1,'validation_stream_timeout_s':3600,'validation_request_timeout_s':7200,'backend':'serial FIFO, concurrent HTTP entry','TP':8,'PP':PP,'cards':CARDS,'precision':report['numerical_variant'],'eos_ids':sorted(eos),'protocol_commit':report['tool_protocol_commit'],'P0_text_buffered_until_completion':True,'prefill_chunk':512,'output_budget_max':8192,'normal_output_budget':4096,'thinking_output_budget':8192,'host_projected_bank':False,'HTTP_model_qualification':False}
        ready.update(service_deadline_monotonic=deadline,normal_cleanup_reserve_seconds=NORMAL_CLEANUP_RESERVE,process_budget_seconds=21600,budget_is_completion_guarantee=False)
        (r/'results'/('GLM-regular-service-r204-ready-'+tag+'.json')).write_text(json.dumps(ready,indent=2))
        report['service_ready']=ready;save()
        # CPU subprocess only: owner thread remains available for actual HTTP inference.
        child_env=dict(os.environ)
        child_env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
        child_env.pop('LD_PRELOAD',None)
        auto_log=(r/'logs'/('GLM-agent-P0-auto-r204-'+tag+'.log')).open('x')
        auto_client=subprocess.Popen([sys.executable,'-B','-u',str(folder/'glm_auto_client_r204.py'),'--run',tag,'--handle',ready['actual_step']],env=child_env,stdin=subprocess.DEVNULL,stdout=auto_log,stderr=subprocess.STDOUT)
        auto_log.close();report['automatic_CPU_client_PID']=auto_client.pid;save()

    seq=0
    try:
        while True:
            req=None;wire=None
            if is_emitter:
                stopfile=r/'control'/('GLM-regular-service-r204-stop-'+tag+'.json')
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
                stopfile=r/'control'/('GLM-regular-service-r204-stop-'+tag+'.json')
                if not stopfile.exists():
                    with stopfile.open('x')as out:json.dump({'normal_stop':True,'reason':'CPU_supervisor_exited','exit_code':auto_client.returncode},out)
            if op==WAIT:continue
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
                        if is_emitter:req.emit_token(token,visible=token not in eos)
                        calls[-1]['samples_next_token']=True
                        discard();del selected,logits
                        return token
                    past,token,canceled=execute_plan(plan,past,prefill=step==0,window=2 if step==0 else 1,forward=forward,vote=vote,sample=sample,discard=discard)
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
            report['service_requests'].append(row);save()
            if is_emitter:
                engine.record_finished(req,len(calls),True)
                report['queue_journal']=engine.journal_snapshot()
                queue_path=r/'results'/('GLM-regular-service-r204-queue-'+tag+'.json');temp=queue_path.with_suffix('.tmp');temp.write_text(json.dumps(report['queue_journal'],indent=2));temp.replace(queue_path)
                req.metadata.update(original_reset_all32_completed=True,backend_sequence=seq)
                req.finish(FinishReason(reason))
        barrier()
        if is_emitter:
            engine.cancel_pending_after_global_drain(FinishReason.CANCELED)
            report['queue_journal']=engine.journal_snapshot()
            server.should_exit=True;thread.join(40);assert not thread.is_alive(),'Normal HTTP shutdown required';report['normal_HTTP_shutdown']=True
            if auto_client is not None:
                if auto_client.poll() is None:auto_client.terminate()
                auto_client.wait(timeout=20);report['automatic_CPU_client_exit_code']=auto_client.returncode
        barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0));report.update(cohort_phases_completed=True,normal_reset_close=True,native_service_loop_completed=True,normal_shutdown=True);save()
    except BaseException as error:
        report.update(normal_reset=False,normal_shutdown=False,normal_reset_close=False)
        # Best-effort CPU reporting must never replace the original failure.
        try:
            if is_emitter:
                if req is not None:req.fail(error)
                if server is not None:server.should_exit=True
            report['service_error']=repr(error);report['service_traceback']=traceback.format_exc();save()
        except BaseException:
            pass
        raise
