"""Static r417 Native service loop, without HTTP ownership or source execution."""
import ctypes,copy,hashlib,json,os,queue,time,traceback
from pathlib import Path
from glm_prefill_execute_r236 import execute_plan
from longcontext_wire_r204 import WIRE,CONTEXT,OUTPUT_MAX,WAIT,BEGIN,STOP,prefill_chunks
from service_policy import encode_profile_wire as encode_wire,decode_profile_wire as decode_wire,select

NATIVE_ENVIRONMENT={
 'GLM_SHARED_PREFIX_VALIDATE_R5814':'0','GLM_CROSSREQUEST_PREFILL_R7328':'0',
 'GLM_PREFILL_PREFIX_PLAN_R8703':'0','GLM_GROUPED_ROUTED_R5123':'1',
 'GLM_BATCH_ALLSELECTED_R5096':'1','GLM_DECODE_ROW_DEDUP_R5386':'0'}

def configure_native_environment(environ):
    # Call after real context construction/load, as in r417. Never pretend a
    # detached dict changes Native getenv. Explicit caller-owned environment.
    if environ is not os.environ:raise ValueError('Native policy must configure the actual process environment')
    environ.update(NATIVE_ENVIRONMENT)
    return dict(NATIVE_ENVIRONMENT)

def serve_requests(*,lib,ctx,stage,rank,control,check,ptr,stream,vocab,torch,
                   report,report_path,queue_path,engine,FinishReason,
                   prepared_identity,eos_token_ids,run_identity,default_profile,
                   profile_manifest_SHA256,deadline,should_stop,
                   clock=time.monotonic,event_clock=time.perf_counter):
    # This function borrows an already loaded same-thread ctx. No HTTP lifecycle.
    PP=2;CARDS=16;is_emitter=stage==PP-1 and rank==0
    _,default_window=select({},default_profile)
    policy_sha=profile_manifest_SHA256
    assert 0<=rank<8 and stage in (0,1) and vocab>0
    eos=set(eos_token_ids)
    assert eos and all(type(v)is int and 0<=v<vocab for v in eos)
    assert prepared_identity['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
    assert set(prepared_identity['eos_token_ids'])==eos
    if is_emitter:
        assert engine is not None and engine._r417_bound
    else:
        assert engine is None
    report_path=Path(report_path);queue_path=Path(queue_path)
    request_index=0
    report.update(actual_step=run_identity['actual_step'],run=run_identity['run'],
        source_manifest_SHA256=run_identity['parent_source_manifest_SHA256'],service_profile_manifest_SHA256=policy_sha,
        default_native_profile=default_profile,allowed_native_profiles=['glm-nov16-w2','glm-nov16-w4'],
        scope='Static Native queue owner C1TP8PP2/cap32768;complete admitted histories;no fixture plan or HTTP owner',
        service_requests=[],normal_reset=False,normal_shutdown=False,normal_reset_close=False,
        HTTP_qualified=False,fullquality_qualified=False,online_SLO_qualified=False)
    def save():
        temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(report_path)
    def barrier():
        value=torch.ones(1,device='cuda');control(value);assert value.item()==CARDS
    req=None;seq=0
    try:
        agreement=torch.tensor([2,1,32768,default_window],device='cuda',dtype=torch.float32)
        control(agreement);assert agreement.cpu().tolist()==[16*x for x in [2,1,32768,default_window]]
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
        barrier()
        report.update(planned_HTTP=None,planned_Native=None,attempted_Native=0,completed_Native=0,prepared_entry_identity=prepared_identity,thinking_template_policy='Actual pinned template unchanged; no new toggle qualification',Native_SHA256=report['library_sha256'])
        while True:
            req=None;wire=None
            if is_emitter:
                if clock()>=deadline or should_stop():
                    wire=encode_wire(seq+1,STOP)
                else:
                    try:req=engine.take(timeout=.5)
                    except queue.Empty:wire=encode_wire(seq+1,WAIT)
                    else:
                        if req is None:wire=encode_wire(seq+1,STOP)
                        elif req.cancel_event.is_set():engine.record_finished(req,0,True);req.finish(req.abort_reason or FinishReason.CANCELED);wire=encode_wire(seq+1,WAIT)
                        else:engine.record_started(req);wire=encode_wire(seq+1,BEGIN,req.prompt_token_ids,req.max_new_tokens,profile=req.metadata['native_profile'])
            seq+=1;tensor=torch.tensor(wire if is_emitter else [0]*WIRE,dtype=torch.float32,device='cuda');control(tensor);op,prompt,limit,prefill_window=decode_wire(tensor.cpu().tolist(),seq,vocab)
            if op==STOP:break
            if op==WAIT:continue
            if is_emitter:
                with engine._admission_lock:
                    assert req._admission_metadata_ready;admission=copy.deepcopy(req.metadata)
                backend_ticks=[];native_started=clock()
            # The typed profile is carried in reserved control-wire tail cells;
            # prompt IDs/outputbudget remain byte-for-byte complete.
            assert prefill_window in (2,4)
            if is_emitter:assert prefill_window==req.metadata['prefill_control_window']
            pair_index=request_index;request_index+=1
            report['attempted_Native']=request_index;report['active_request_calls']=[];save()
            check(lib.modern_glm_model_reset(ctx));barrier();generated=[];past=0;reason='length';started=clock();calls=report['active_request_calls']
            request_completed_normally=False
            report['normal_reset']=False
            try:
                for step in range(limit):
                    plan=prefill_chunks(prompt)if step==0 else[(past,[generated[-1]])]
                    retained=[None]
                    def vote():
                        if is_emitter and clock()>=deadline:req.mark_abort(FinishReason.TIMEOUT)
                        flag=torch.tensor([float(is_emitter and req.cancel_event.is_set())],device='cuda');control(flag)
                        return bool(flag.item())
                    def forward(logical_past,values,final):
                        assert retained[0]is None and len(values)<=512 and logical_past+len(values)<=CONTEXT
                        desc=(ctypes.c_size_t*3)(0,logical_past,len(values));ids=torch.tensor(values if stage==0 else[0]*len(values),device='cuda',dtype=torch.int64);positions=torch.arange(logical_past,logical_past+len(values),device='cuda',dtype=torch.int64);logits=torch.full((1,vocab),float('nan'),device='cuda',dtype=torch.bfloat16)
                        row={'T':len(values),'past':logical_past,'prefill':step==0,'samples_next_token':False,'CAPI_returned':False,'finite':None}
                        calls.append(row)
                        torch.cuda.synchronize(rank);tick=event_clock();rc=lib.modern_glm_model_forward(ctx,len(values),1,desc,ptr(ids),ptr(positions),ptr(logits),0,stream())
                        row.update(CAPI_returned=True,CAPI_RC=rc)
                        try:check(rc)
                        finally:row['Native_CAPI_s']=event_clock()-tick
                        if stage==PP-1:
                            row['finite']=bool(torch.isfinite(logits).all())
                            assert row['finite'],'Nonfinite real Native logits'
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
                            tick=clock();req.emit_token(token,visible=token not in eos);backend_ticks.append(tick)
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
            row={'sequence':seq,'prompt_tokens':len(prompt),'generated_tokens':len(generated),'reason':reason,'wall_s':clock()-started,'native_calls':calls,'original_reset_all16_completed':True,'prefill_chunk':512,'requested_output_budget':limit}
            row.update(admitted_index=pair_index,request_index=pair_index,prefill_control_window=prefill_window,native_profile='glm-nov16-w'+str(prefill_window),service_profile_manifest_SHA256=policy_sha,
                generated_token_ids_SHA256=__import__('hashlib').sha256(json.dumps(generated,separators=(',',':')).encode()).hexdigest(),
                generated_natural_EOS=bool(generated and generated[-1] in eos and reason=='eos_token'),
                actual_prompt_token_ids_SHA256=__import__('hashlib').sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest())
            if is_emitter:
                row.update(request_id=req.request_id,admission=admission,max_new_tokens=limit,normal_reset_before=True,normal_reset_after=True,native_finish_reason=reason,generated_token_ids=list(generated),actual_generated_tokens=len(generated),backend_token_monotonic=backend_ticks,started_monotonic=native_started,ended_monotonic=clock(),Native_continuation=[{'kind':'prefill'if c['prefill']else'decode','tokens':c['T']}for c in calls],Native_forward_host_summary={'prefill_calls':sum(c['prefill']for c in calls),'prefill_s':sum(c['Native_CAPI_s']for c in calls if c['prefill']),'decode_calls':sum(not c['prefill']for c in calls),'decode_s':sum(c['Native_CAPI_s']for c in calls if not c['prefill'])})
            report['service_requests'].append(row);report['completed_Native']=len(report['service_requests']);save()
            if is_emitter:
                engine.record_finished(req,len(calls),True)
                report['queue_journal']=engine.journal_snapshot()
                temp=queue_path.with_suffix('.tmp');temp.write_text(json.dumps(report['queue_journal'],indent=2));temp.replace(queue_path)
                req.metadata.update(original_reset_all16_completed=True,backend_sequence=seq)
                req.finish(FinishReason(reason));row['final_metadata']=copy.deepcopy(req.metadata);save()
        barrier()
        if is_emitter:
            engine.cancel_pending_after_global_drain(FinishReason.CANCELED)
            report['queue_journal']=engine.journal_snapshot()
        barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0));report.update(cohort_phases_completed=True,normal_reset_close=True,native_service_loop_completed=True,normal_shutdown=False,native_loop_drained=True,global_Model_READY=False,full_requests_completed=request_index,measured_pairing_qualified=False);save()
    except BaseException as error:
        report.update(normal_reset=False,normal_shutdown=False,normal_reset_close=False,service_error=repr(error),service_traceback=traceback.format_exc())
        if is_emitter and req is not None:
            try:req.fail(error)
            except BaseException:pass
        try:save()
        except BaseException as persistence_error:
            report['report_write_error']=repr(persistence_error)
            if hasattr(error,'add_note'):error.add_note('GLM owner report persistence failed: '+repr(persistence_error))
        raise


def run_resident_owner(resident_context,*,torch,stream,environ,control=None,**service):
    """Compose actual r422 create/load/yield/destroy with this static loop.

    All-rank launch, admission/ready and outer fail-stop remain the caller's
    responsibility. This owner starts/joins no HTTP thread and grants no READY.
    """
    report=resident_context.report
    try:
        with resident_context as resident:
            lib=resident.binding.lib;check=resident.binding.check;ptr=resident.binding.ptr
            ctx=resident.ctx
            configure_native_environment(environ)
            if control is None:
                def control(value):
                    check(lib.modern_glm_model_control(ctx,value.numel(),ptr(value),stream()))
            serve_requests(lib=lib,ctx=ctx,check=check,ptr=ptr,control=control,
                stream=stream,torch=torch,report=report,**service)
    except BaseException as error:
        report['normal_shutdown']=False
        report['owner_scope_error']=repr(error)
        try:
            path=Path(service['report_path']);temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(report,indent=2));temp.replace(path)
        except BaseException as persistence_error:
            if hasattr(error,'add_note'):error.add_note('Owner-scope report persistence failed: '+repr(persistence_error))
        raise
    else:
        # Original OwnedContext's actual record is authoritative. Loop drain is
        # not a claim of destroy/shutdown before the context scope has exited.
        lifecycle=json.loads(Path(resident_context.lifecycle_path).read_text())
        report['resident_context_cleanup']=lifecycle
        report['normal_shutdown']=bool(lifecycle['normal']and report.get('native_loop_drained'))
        assert report['normal_shutdown'],'Original context cleanup did not finish normally'
        path=Path(service['report_path']);temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2));temp.replace(path)
        return report
