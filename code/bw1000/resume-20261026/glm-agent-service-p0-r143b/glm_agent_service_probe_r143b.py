"""Resident serial Native GLM service. Actual CAPI owners remain on original threads."""
import ctypes,json,os,pathlib,queue,socket,threading,time,traceback
from native_engine_bridge_r143c import NativeEngineBridge,existing_server_app

WIRE=648
MAGIC=743143
WAIT,BEGIN,STOP=0,1,2

def encode_wire(seq,op,ids=(),limit=0):
    if op not in [WAIT,BEGIN,STOP]or len(ids)>640:
        raise ValueError('Native service wire')
    values=[MAGIC,seq,op,len(ids),limit]+list(ids)
    values += [0]*(WIRE-len(values))
    if any(type(x)is not int or not 0<=x<2**24 for x in values):
        raise ValueError('Exact FP32 integer wire bounds')
    return values

def decode_wire(values,seq,vocab):
    if len(values)!=WIRE or any(int(x)!=x for x in values):raise ValueError('Wire shape/type')
    values=list(map(int,values));magic,s,op,count,limit=values[:5]
    if magic!=MAGIC or s!=seq or op not in[WAIT,BEGIN,STOP]or not 0<=count<=640:raise ValueError('Wire identity')
    if any(values[5+count:])or any(not 0<=x<vocab for x in values[5:5+count]):raise ValueError('Wire payload')
    if op==BEGIN and(not count or limit<1 or count+limit>640):raise ValueError('Complete context budget')
    if op!=BEGIN and(count or limit):raise ValueError('Control payload')
    return op,values[5:5+count],limit

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker):
    import torch
    from glm_crossrequest_control_r9059 import CrossrequestControl
    r=pathlib.Path('/refactor');is_emitter=stage==1 and rank==0
    tag=os.environ['GLM_RUN_R5040'];folder=pathlib.Path(__file__).parent
    deadline=time.monotonic()+max(0,report['remaining_process_seconds'])-60
    report.update(scope='Resident original GLM Native + 7ceb+scoped-timeout-correction HTTP/SSE protocol;P0greedy serial/no performance optimization',service_requests=[],cohort_phases_completed=False,normal_reset_close=False,tool_protocol_commit='7ceb60d656074376b911c2a232196a9457dfa6a6',model_context=640,served_model='GLM-5.3',HTTP_qualified=False,tool_closed_loop_qualified=False)
    def save():
        temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(report_path)
    def barrier():
        value=torch.ones(1,device='cuda');control(value);assert value.item()==16
    resident=CrossrequestControl(lib,ctx,stage,rank,control,check,report,save)
    check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_transient_policy_r5688(ctx,0));check(lib.modern_glm_vector_policy_r8540(rank,0))
    for name in ['decode_leases_policy_r1356','indexer_policy_r998','feature_bypass_r1406','owned_prefix_policy_r1669']:
        check(getattr(lib,'modern_glm_model_'+name)(ctx,1))
    resident.apply('R143-resident-validated-mask',True,False)
    lib.modern_glm_model_batch_cache_policy_r3068.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.glm_reduced_output_policy_r9032.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.glm_attention_cohort_policy_r9054.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.glm_ragged_policy_r9136.argtypes=[ctypes.c_int,ctypes.c_int]
    check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,1));check(lib.glm_ragged_policy_r9136(rank,1))
    os.environ.update(GLM_SHARED_PREFIX_VALIDATE_R5814='1',GLM_CROSSREQUEST_PREFILL_R7328='0',GLM_PREFILL_PREFIX_PLAN_R8703='0',GLM_GROUPED_ROUTED_R5123='1',GLM_BATCH_ALLSELECTED_R5096='1',GLM_DECODE_ROW_DEDUP_R5386='0')
    eos=payload['benchmark']['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list)else[eos]);barrier()
    engine=server=thread=None
    if is_emitter:
        import uvicorn
        from transformers import PreTrainedTokenizerFast
        from infinilm.llm import FinishReason
        tokenizer=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True)
        engine=NativeEngineBridge(tokenizer,640,FinishReason,eos_token_ids=sorted(eos),default_max_tokens=96)
        api,app=existing_server_app(engine,'/old/models/ZhipuAI-GLM-5.3','glm5',host=os.environ.get('GLM_SERVICE_HOST_R143','127.0.0.1'),port=int(os.environ.get('GLM_SERVICE_PORT_R143','43843')))
        server=uvicorn.Server(uvicorn.Config(app,host=api.host,port=api.port,log_level='info',access_log=True))
        thread=threading.Thread(target=server.run,name='HTTP-only-no-Native-owner-calls',daemon=True);thread.start()
        start=time.monotonic()
        while not server.started:
            if not thread.is_alive()or time.monotonic()-start>40:raise RuntimeError('HTTP startup failed')
            time.sleep(.1)
        ready={'ready':True,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'stage':stage,'rank':rank,'HTTP_bind':api.host,'port':api.port,'model_path':'/old/models/ZhipuAI-GLM-5.3','Native_DSO_SHA256':report['library_sha256'],'context':640,'TP':8,'PP':2,'cards':16,'precision':report['numerical_variant'],'eos_ids':sorted(eos),'protocol_commit':report['tool_protocol_commit'],'P0_text_buffered_until_completion':True,'HTTP_model_qualification':False}
        (r/'results'/('GLM-agent-service-r143b-ready-'+tag+'.json')).write_text(json.dumps(ready,indent=2))
        report['service_ready']=ready;save()
    seq=0
    try:
        while True:
            req=None;wire=None
            if is_emitter:
                stopfile=r/'control'/('GLM-agent-service-r143b-stop-'+tag+'.json')
                if time.monotonic()>=deadline or server.should_exit or stopfile.exists():
                    wire=encode_wire(seq+1,STOP)
                else:
                    try:req=engine.take(timeout=.5)
                    except queue.Empty:wire=encode_wire(seq+1,WAIT)
                    else:
                        if req is None:wire=encode_wire(seq+1,STOP)
                        elif req.cancel_event.is_set():req.finish(req.abort_reason or FinishReason.CANCELED);wire=encode_wire(seq+1,WAIT)
                        else:wire=encode_wire(seq+1,BEGIN,req.prompt_token_ids,req.max_new_tokens)
            seq+=1;tensor=torch.tensor(wire if is_emitter else [0]*WIRE,dtype=torch.float32,device='cuda');control(tensor);op,prompt,limit=decode_wire(tensor.cpu().tolist(),seq,vocab)
            if op==STOP:break
            if op==WAIT:continue
            check(lib.modern_glm_model_reset(ctx));barrier();generated=[];past=0;reason='length';started=time.monotonic();calls=[]
            try:
                for step in range(limit):
                    flag=torch.tensor([float(is_emitter and req.cancel_event.is_set())],device='cuda');control(flag)
                    if flag.item():reason='canceled';break
                    values=prompt if step==0 else [generated[-1]]
                    desc=(ctypes.c_size_t*3)(0,past,len(values));ids=torch.tensor(values if stage==0 else[0]*len(values),device='cuda',dtype=torch.int64);positions=torch.arange(past,past+len(values),device='cuda',dtype=torch.int64);logits=torch.full((1,vocab),float('nan'),device='cuda',dtype=torch.bfloat16)
                    torch.cuda.synchronize(rank);tick=time.perf_counter();check(lib.modern_glm_model_forward(ctx,len(values),1,desc,ptr(ids),ptr(positions),ptr(logits),0,stream()));elapsed=time.perf_counter()-tick
                    if stage==1:assert bool(torch.isfinite(logits).all()),'Nonfinite real Native logits'
                    selected=torch.zeros(2,device='cuda',dtype=torch.float32)
                    if is_emitter:selected[0]=float(logits.argmax(-1).item());selected[1]=1
                    control(selected);raw=selected.cpu().tolist();assert raw[1]==1 and int(raw[0])==raw[0]and 0<=raw[0]<vocab
                    token=int(raw[0]);generated.append(token);past+=len(values)
                    free,total=torch.cuda.mem_get_info(rank);assert free>=512*1024**2,'Original512MiB GLM floor';report['minimum_driver_free_bytes']=min(report.get('minimum_driver_free_bytes',free),free)
                    calls.append({'T':len(values),'past':past-len(values),'Native_CAPI_s':elapsed,'selected_token':token})
                    if is_emitter:req.emit_token(token,visible=token not in eos)
                    del ids,positions,logits,selected,desc
                    if token in eos:reason='eos_token';break
            finally:
                # Same original context reset + WORLD drain. No next request may
                # finish/enter before every owner has completed this boundary.
                torch.cuda.synchronize(rank);check(lib.modern_glm_model_reset(ctx));barrier()
            row={'sequence':seq,'prompt_tokens':len(prompt),'generated_tokens':len(generated),'reason':reason,'wall_s':time.monotonic()-started,'native_calls':calls,'original_reset_all16_completed':True}
            report['service_requests'].append(row);save()
            if is_emitter:
                req.metadata.update(original_reset_all16_completed=True,backend_sequence=seq)
                req.finish(FinishReason(reason))
        if is_emitter:
            server.should_exit=True;thread.join(40);assert not thread.is_alive(),'Normal HTTP shutdown required';report['normal_HTTP_shutdown']=True
        barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));resident.apply('R143-final-OFF',False,False);check(resident.vector_OFF());report.update(cohort_phases_completed=True,normal_reset_close=True,native_service_loop_completed=True);save()
    except BaseException as error:
        if is_emitter:
            if req is not None:req.fail(error)
            if server is not None:server.should_exit=True
        report['service_error']=repr(error);report['service_traceback']=traceback.format_exc();save();raise
