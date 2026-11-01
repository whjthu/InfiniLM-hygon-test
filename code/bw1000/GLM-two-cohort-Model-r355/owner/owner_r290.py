"""Finite C2M1 private current038d Model consumer; no row refill or cache changes."""
import ctypes,hashlib,json,os,pathlib,queue,subprocess,sys,threading,time,traceback
from consumer import Row,decode_round
from native_binding import Forward
from glm_prefill_execute_r236 import execute_plan
from workload_r355 import PLAN,load,marker_identity
from deadline_r290 import check as deadline_check,monotonic_deadline
from cleanup_owned_r290 import cleanup_owned


def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker,prepared):
    import torch
    import uvicorn
    D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor')
    tag=os.environ['GLM_RUN_R5040'];mf=os.environ['GLM_SOURCE_MANIFEST_SHA']
    is_emitter=(stage,rank)==(3,0);deadline=monotonic_deadline();fixtures=load(D)
    assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
    eos=set(payload['benchmark']['generation_config']['eos_token_id'])
    assert eos==set(prepared['identity']['eos_token_ids'])
    def save():
        p=report_path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(report_path)
    def barrier():
        x=torch.ones(1,device='cuda');control(x);assert x.item()==32
    def cancel_vote():
        # One agreement per round, OUTSIDE the two-forward window.
        x=torch.tensor([float(is_emitter and (time.monotonic()>=deadline or pathlib.Path(os.environ['GLM_R355_CANCEL_FILE']).exists()))],device='cuda')
        control(x)
        if x.item():raise InterruptedError('Original cohort cancellation; no replay')
    def memory(where):
        free,total=torch.cuda.mem_get_info(rank)
        assert free>=512*1024**2,'Original512MiB floor'
        report.setdefault('memory_checkpoints',[]).append(dict(where=where,free=free,total=total,is_peak=False))
    report.update(run=tag,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],stage=stage,rank=rank,
        source_manifest_SHA256=mf,scope='Static C2M1 TP8PP4 finite pair, two real complete prepared requests; buffered HTTP; no onlineSLO',
        constructor_rows=2,planned_pairs=6,planned_logical_requests=12,pairs=[],normal_reset_close=False,
        normal_shutdown=False,cohort_phases_completed=False,Model_gain='UNKNOWN',HTTP_incremental_qualified=False)
    check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0))
    for name in ['owned_prefix_policy_r1669','feature_bypass_r1406','indexer_policy_r998']:
        check(getattr(lib,'modern_glm_model_'+name)(ctx,0))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
    lib.modern_glm_model_transient_report_r5688.argtypes=[ctypes.c_void_p];lib.modern_glm_model_transient_report_r5688.restype=ctypes.c_char_p
    transient=json.loads(lib.modern_glm_model_transient_report_r5688(ctx))
    assert transient['rows']==2 and transient['capacity']==32768 and not transient['enabled'] and transient['GPU_logical_bytes']==0
    for name in ['glm_reduced_output_policy_r9032','glm_attention_cohort_policy_r9054','glm_ragged_policy_r9136']:
        getattr(lib,name).argtypes=[ctypes.c_int,ctypes.c_int]
    check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0))
    check(lib.glm_attention_cohort_policy_r9054(rank,1));check(lib.glm_ragged_policy_r9136(rank,1))
    os.environ.update(GLM_SHARED_PREFIX_VALIDATE_R5814='0',GLM_CROSSREQUEST_PREFILL_R7328='0',GLM_PREFILL_PREFIX_PLAN_R8703='0',GLM_GROUPED_ROUTED_R5123='1',GLM_BATCH_ALLSELECTED_R5096='1',GLM_DECODE_ROW_DEDUP_R5386='0')
    agreement=torch.tensor([float(x)for x in PLAN],device='cuda');control(agreement)
    assert agreement.cpu().tolist()==[32*float(x)for x in PLAN]
    server=thread=client=log=engine=None;pgid=None;requests=[]
    try:
        if is_emitter:
            engine=prepared['engine'];api=prepared['server']
            server=uvicorn.Server(uvicorn.Config(prepared['app'],host=api.host,port=api.port,log_level='info',access_log=True));server.install_signal_handlers=lambda:None
            thread=threading.Thread(target=server.run,name='GLM355-HTTP-only',daemon=False);thread.start();until=time.monotonic()+40
            while not server.started:
                deadline_check('HTTP startup',120)
                if not thread.is_alive()or time.monotonic()>until:raise RuntimeError('HTTP start failed')
                time.sleep(.1)
            ready=R/'results'/('GLM-r355-ready-'+tag+'.json')
            identity=dict(run=tag,rows=2,context=32768,TP=8,PP=4,cards=32,source_manifest_SHA256=mf,
                Native_SHA256=report['library_sha256'],prepared_entry_identity=prepared['identity'],port=api.port,
                emitter_step=report['actual_step'],static_cohort=True,incremental_HTTP=False)
            with ready.open('x')as out:json.dump(identity,out)
            childenv=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',USE_TORCH='0',USE_TF='0');childenv.pop('LD_PRELOAD',None)
            log=(R/'logs'/('GLM-r355-client-'+tag+'.log')).open('x')
            client=subprocess.Popen([sys.executable,'-B','-u',str(D/'client_r355.py'),'--run',tag,'--ready',str(ready),'--manifest',mf],env=childenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pgid=client.pid;assert os.getpgid(client.pid)==pgid
        barrier();memory('after_loaded_C2_before_firstpair')
        for pair_index,enabled in enumerate(PLAN):
            deadline_check('before pair admission',120);requests=[]
            if is_emitter:
                by_hash={}
                while len(by_hash)<2:
                    deadline_check('two actual admissions',120)
                    if client.poll()is not None:raise RuntimeError('Client exited before pair admission')
                    try:req=engine.take(timeout=.1)
                    except queue.Empty:continue
                    assert req is not None and not req.cancel_event.is_set() and req.max_new_tokens==4096
                    h=hashlib.sha256(json.dumps(list(req.prompt_token_ids),separators=(',',':')).encode()).hexdigest()
                    assert h not in by_hash;by_hash[h]=req
                for f in fixtures:
                    req=by_hash[f['fixture']['current_prepared_prompt_IDs_SHA256']]
                    assert list(req.prompt_token_ids)==f['ids'];engine.record_started(req);requests.append(req)
            barrier();cancel_vote();check(lib.modern_glm_model_reset(ctx));barrier()
            started=time.monotonic();pair=dict(index=pair_index,enabled=enabled,warmup=pair_index<2,requests=[],calls=[],rounds=0)
            generated=[[],[]];ticks=[[],[]];past=[0,0];active_phase=['prefill'];first_admission=[None,None]
            if is_emitter:first_admission=[float(req.metadata['accepted_monotonic'])for req in requests]
            def emit(row,logits):
                chosen=torch.zeros(2,device='cuda',dtype=torch.float32)
                if is_emitter:chosen[0]=float(logits.argmax(-1).item());chosen[1]=1
                control(chosen);assert chosen[1].item()==1
                token=int(chosen[0].item());assert 0<=token<vocab
                generated[row].append(token)
                if is_emitter:
                    requests[row].emit_token(token,visible=token not in eos);ticks[row].append(time.monotonic())
                return token,token in eos
            def forward_values(row,logical_past,values,prefill):
                deadline_check('before actual Native forward',30)
                desc=(ctypes.c_size_t*3)(row,logical_past,len(values))
                ids=torch.tensor(values if stage==0 else [0]*len(values),device='cuda',dtype=torch.int64)
                positions=torch.arange(logical_past,logical_past+len(values),device='cuda',dtype=torch.int64)
                logits=torch.full((1,vocab),float('nan'),device='cuda',dtype=torch.bfloat16)
                torch.cuda.synchronize(rank);tick=time.perf_counter()
                check(lib.modern_glm_model_forward(ctx,len(values),1,desc,ptr(ids),ptr(positions),ptr(logits),0,stream()))
                ended=time.perf_counter()
                if stage==3:assert bool(torch.isfinite(logits).all()),'Nonfinite actual logits'
                free,total=torch.cuda.mem_get_info(rank);assert free>=512*1024**2
                report['minimum_driver_free_bytes']=min(report.get('minimum_driver_free_bytes',free),free)
                pair['calls'].append(dict(row=row,past=logical_past,T=len(values),prefill=prefill,Native_CAPI_s=ended-tick))
                return logits,ended-tick,tick,ended
            for row,f in enumerate(fixtures):
                retained=[None]
                def pf_forward(p,values,final,row=row):
                    out,*_=forward_values(row,p,values,True)
                    if final:retained[0]=out
                def pf_sample(row=row):
                    assert retained[0]is not None
                    token,_=emit(row,retained[0]);retained[0]=None;return token
                plan=[(p,f['ids'][p:p+512])for p in range(0,len(f['ids']),512)]
                def pf_cancel():cancel_vote();return False
                p,token,canceled=execute_plan(plan,0,prefill=True,window=4,forward=pf_forward,vote=pf_cancel,sample=pf_sample,discard=lambda:retained.__setitem__(0,None))
                assert not canceled and p==len(f['ids']) and token==generated[row][-1];past[row]=p
            memory('pair%d-after_serial_prefill'%pair_index)
            states=[Row(row,past[row],generated[row][-1],4096,1,generated[row][-1]in eos)for row in range(2)]
            decode_start=time.monotonic()
            def dec_forward(**kw):
                row,p,n=kw['descriptor'];assert n==1
                return forward_values(row,p,list(kw['input_ids']),False)
            while not all(x.done for x in states):
                cancel_vote() # Never put WORLD agreement inside F0 -> F1.
                decode_round(states,dec_forward,emit,enabled=enabled,check=lambda:deadline_check('local bounded forward window',30))
                pair['rounds']+=1
                if pair['rounds']%64==0:
                    report['progress']=dict(pair_index=pair_index,round=pair['rounds'],generated=[len(x)for x in generated]);save()
            pair['decode_schedule_wall_s']=time.monotonic()-decode_start
            barrier();check(lib.modern_glm_model_reset(ctx));barrier()
            pair['complete_schedule_wall_s']=time.monotonic()-started
            for row,f in enumerate(fixtures):
                ids=generated[row];natural=ids[-1]in eos
                assert natural,'Output budget exhausted; stop at first pair, no replay'
                record=dict(row=row,fixture_id=f['fixture']['id'],prompt_tokens=len(f['ids']),prompt_IDs_SHA256=f['fixture']['current_prepared_prompt_IDs_SHA256'],
                    generated_ids=ids,generated_IDs_SHA256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),tokens=len(ids),natural_EOS=natural,
                    normal_reset_all32=True,ordinary_budget=4096,emitter_token_monotonic=ticks[row]if is_emitter else None,
                    emitter_admission_monotonic=first_admission[row]if is_emitter else None)
                pair['requests'].append(record)
                if is_emitter:
                    req=requests[row];req.metadata.update(r355_pair=pair_index,r355_row=row,r355_enabled=enabled)
                    engine.record_finished(req,sum(c['row']==row for c in pair['calls']),True);req.finish(prepared['FinishReason']('eos_token'))
            report['pairs'].append(pair);save();memory('pair%d-after_reset'%pair_index)
            # Existing original CPU parser/quality remains outside timed schedule.
            vote=torch.zeros(1,device='cuda')
            if is_emitter:
                marker=R/'control'/('GLM-r355-quality-'+tag+'-pair%d.json'%pair_index)
                while not marker.exists():
                    deadline_check('CPU pair quality',90)
                    if client.poll()is not None:raise RuntimeError('CPU quality client exited without pair marker')
                    time.sleep(.1)
                q=json.loads(marker.read_bytes());assert q['identity']==marker_identity(tag,pair_index,mf)
                vote[0]=float(q['passed']is True)
            control(vote);assert vote.item()==1,'Actual pair quality failed; no later pair'
        if is_emitter:
            client.wait(timeout=60);assert client.returncode==0
            report['queue_journal']=engine.journal_snapshot()
        barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0))
        check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_vector_policy_r8540(rank,0))
        report.update(normal_reset_close=True,normal_shutdown=True,cohort_phases_completed=True,completed_pairs=6,completed_logical_requests=12)
        save()
    except BaseException:
        report['service_traceback']=traceback.format_exc();save()
        if is_emitter:
            for req in requests:
                if not req.is_finished():req.fail(RuntimeError('Owned finite cohort failed'))
        raise
    finally:
        if is_emitter:
            cleanup=cleanup_owned(client,server,thread,engine,log,owned_pgid=pgid)
            report['cleanup']=cleanup;report['normal_HTTP_shutdown']=cleanup['normal_server_shutdown'];save()
            if sys.exc_info()[0]is None:assert not cleanup['errors']and not cleanup['forced_cleanup']and not cleanup['HTTP_thread_alive_after_cleanup']
