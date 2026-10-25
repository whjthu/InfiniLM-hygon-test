"""Full-model GLM PP cohort probe. Imported by the qualified latest loader."""
import os, ctypes, hashlib, json, pathlib, time
import torch
from glm_large_schedule_r3627 import groups, accept, metrics
from glm_continuous_feedback_r9030 import FeedbackHub,FeedbackClient,CheckedFeedback,safe_decode,tail_ready_metrics
import socket,secrets
from glm_layer77_policy_r9030 import Layer77Policy
from glm_lowbatch_profile_r8219 import OwnerProfile
from glm_crossrequest_control_r9030 import CrossrequestControl as ResidentControl,CandidateRejected
from glm_crossrequest_contract_r9030 import validate_reference

def run(lib, ctx, stage, rank, payload, control, check, ptr, stream, vocab,
        report, report_path, chat_gate,phase_marker):
    root = pathlib.Path('/refactor')
    eos = set(payload['natural_eos_token_ids']);PP=report['PP'];CARDS=report['all_serving_cards'];ROWS=report['cache_rows'];CAPACITY=report['cache_capacity']
    report.update(scope='ROWS32 TP8PP2/B16M8 fused attention ABBA, capacity640. Same-image OFF/resident, vector fixedON and transientOFF. Short frozen448 anchor and original477 frozen I512 gold. No productionpeak or independent original-math proof.', multi_batch_cases=[],raw_checkpoints=[],single_user15_is_adoption_gate=False)
    report['probe_source_sha256'] = {
        name: hashlib.sha256((root/'incoming'/name).read_bytes()).hexdigest()
        for name in ['glm_large_schedule_r3627.py', 'glm_prefill_micro_probe_r9030.py',
                     'modern_glm_prefill_micro_r9030.py', 'glm_continuous_feedback_r9030.py','kimi_cohort_feedback_idle_r4915.py']}
    def save():
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)
    policy=Layer77Policy(lib,stage,rank,control,check,report,save)
    def public(enabled):
        # No forward until every owner completed reset, transientOFF and the
        # same policy bundle. Do not toggle prerequisites while transient ON.
        def unanimous(codes,label):
            vote=torch.tensor([float(all(code==0 for code in codes))],device='cuda');control(vote)
            assert vote.item()==CARDS,('allowner reset/policy gate',label,codes)
        unanimous([lib.modern_glm_model_reset(ctx)],'reset')
        unanimous([lib.modern_glm_model_transient_policy_r5688(ctx,0)],'transientOFF-before-policy')
        unanimous([resident.vector_OFF()],'vectorOFF-before-policy')
        codes=[lib.modern_glm_model_owned_prefix_policy_r1669(ctx,0),lib.modern_glm_model_feature_bypass_r1406(ctx,0),lib.modern_glm_model_decode_leases_policy_r1356(ctx,0),lib.modern_glm_model_indexer_policy_r998(ctx,0)]
        unanimous(codes,'bundleOFF')
        if enabled:
            codes=[lib.modern_glm_model_decode_leases_policy_r1356(ctx,1),lib.modern_glm_model_indexer_policy_r998(ctx,1),lib.modern_glm_model_feature_bypass_r1406(ctx,1),lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)]
            unanimous(codes,'bundleON')
    def batch_stats():
        a=(ctypes.c_uint64*5)();check(lib.modern_glm_model_batch_cache_stats_r3068(ctx,a,5));return list(a)
    cfg=json.loads((pathlib.Path('/old/models/ZhipuAI-GLM-5.3')/'config.json').read_text());bounds=[0,38,78];stage_layers=bounds[stage+1]-bounds[stage];full_layers=sum(x=='full'for x in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
    def fast_stats():return [int(lib.modern_glm_indexer_metadata_calls_r1406(rank)),int(lib.modern_glm_prefix_calls_r1669(rank,0)),int(lib.modern_glm_prefix_calls_r1669(rank,1))]
    resident=ResidentControl(lib,ctx,stage,rank,control,check,report,save)
    public(0);before=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,2)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,2)!=0 and fast_stats()==before
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0 and fast_stats()==before
    report.update(fastpath_invalid_and_dependency_guards_verified=True,fastpath_stage_layers=stage_layers,fastpath_full_indexer_layers=full_layers)
    lib.glm_fused_entries_policy_r9028.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.glm_fused_entries_split_r9028.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    check(lib.glm_fused_entries_policy_r9028(rank,0))
    def fused_counts():
        v=(ctypes.c_uint64*6)();check(lib.glm_fused_entries_split_r9028(rank,v,6));return list(v)
    report['fused_default_OFF_verified']=True
    report['fused_exact_shape_contract']={'B':8,'H':8,'D':256,'Pi_max':640,'proof':'5555 validates all input and per-row K/V geometry before eligible branch; branch eligibility backend1/B8; same sourceSHA as615'}
    references = {}; geometry_references = {}
    profile = OwnerProfile(lib, report, stage, rank)
    lib.modern_glm_prefill_prefix_plan_counts_r8703.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.modern_glm_prefill_prefix_plan_counts_r8703.restype=ctypes.c_uint64
    def prefix_stats():return [int(lib.modern_glm_prefill_prefix_plan_counts_r8703(rank,k))for k in range(3)]
    gold_path = root/'incoming/GLM-lowbatch-frozen-gold-r8219.json'
    frozen_gold = json.loads(gold_path.read_text())
    assert frozen_gold['source_report_SHA256'] == '92e8a3d16c666896c153d602b05b17d296e09125da7993a0a26ef2735b1d68b2'
    assert len(frozen_gold['input_ids']) == len(frozen_gold['generated_ids']) == 16
    report['lowbatch_frozen_gold_SHA256'] = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    lib.modern_glm_batch_attention_counts_r5096.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def attention_stats():
        a=(ctypes.c_uint64*2)();check(lib.modern_glm_batch_attention_counts_r5096(rank,a,2));return list(a)
    lib.glm_grouped_counts_r5123.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def native_stats():
        v=(ctypes.c_uint64*5)();check(lib.glm_grouped_counts_r5123(rank,v,5));return list(v)
    lib.glm_dedup_scope_r5386.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.glm_dedup_scope_get_r5386.argtypes=[ctypes.c_int]
    lib.glm_dedup_counts_r5386.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def decode_stats():
        v=(ctypes.c_uint64*4)();check(lib.glm_dedup_counts_r5386(rank,v,4));return list(v)
    lib.modern_glm_shared_validation_counts_r5814.argtypes=[ctypes.c_int,ctypes.c_int]
    lib.modern_glm_shared_validation_counts_r5814.restype=ctypes.c_uint64
    def shared_stats():return [int(lib.modern_glm_shared_validation_counts_r5814(rank,k))for k in (0,1)]
    continuous_policy=0
    participant=stage*8+rank;emitter=8;hub=None
    endpoint=root/'control'/('glm-transient-Model-r9030-'+report['step']+'-endpoint.json')
    if participant==emitter:
        assert not endpoint.exists()
        nonce=secrets.token_hex(24);hub=FeedbackHub('0.0.0.0',0,nonce,participants=16,emitter=emitter,timeout=180,idle_timeout=7200)
        e={'host':payload['feedback_host'],'port':hub.address[1],'nonce':nonce,'participants':16,'emitter':emitter}
        temporary=endpoint.with_suffix('.tmp');temporary.write_text(json.dumps(e));temporary.replace(endpoint)
    waiting=time.monotonic()
    while not endpoint.exists():
        assert time.monotonic()-waiting<1200,'feedback endpoint timeout'
        time.sleep(.2)
    e=json.loads(endpoint.read_text());assert e['participants']==16 and e['emitter']==8
    feedback=FeedbackClient(e['host'],e['port'],e['nonce'],participant,timeout=190)
    report['continuous_transport_r5467']={'participants':16,'emitter':8,'dependency_timeout':180,'idle_timeout':7200,'pending_bound':'2*cohorts','HTTP':False}
    def generate(label, prompts, micro, pipelined, limit, answer_eos, qa=False, cache_policy=0, resident_on=False, trace=False, vector_on=True, reference_kind='short',prefill_on=0,diagnostic_only=False,shadow=-1,shared_on=False,prefill_micro=8):
        fused_before=fused_counts()
        _first_io={'prefill':False,'decode':False}
        assert prefill_micro in(4,8)and micro==8
        proof_on=int(reference_kind=='I512'and '-N-'in label);os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']=str(proof_on)
        prefix_before=prefix_stats()
        native_before=native_stats();decode_before=decode_stats();assert lib.glm_dedup_scope_get_r5386(rank)==0
        diagnostic_phase=label in ['smoke8g16m16_routed1','smoke8u16m16_routed1']
        identity_summary=torch.zeros(4,dtype=torch.int64,device='cuda')if diagnostic_phase else None
        if identity_summary is not None:torch.cuda.synchronize(rank)
        os.environ['GLM_ACTUAL_CAPTURE_R5164']='1'if diagnostic_only else'0'
        lib.glm_actual_capture_arm_r5164.argtypes=[ctypes.c_int,ctypes.c_char_p]
        capture_enabled=os.environ.get('GLM_ACTUAL_CAPTURE_R5164')=='1' and label.startswith('diag-')
        capture_prefix=f'/refactor/results/glm-actual-capture-r5164-{report["step"]}-{label}-stage{stage}-rank{rank}' if capture_enabled else ''
        check(lib.glm_actual_capture_arm_r5164(rank,capture_prefix.encode()))

        assert vector_on is True; public(cache_policy)
        resident_before=resident.apply(label,resident_on,trace); batch=len(prompts)
        native_diag=diagnostic_only or label.startswith('validate-')
        policy.switch(label,int(prefill_on),native_diag,shadow)
        os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']='1'if shared_on else'0'
        policy_vote=torch.tensor([float(shared_on),1.],device='cuda');control(policy_vote)
        assert policy_vote.tolist()==[float(CARDS*int(shared_on)),float(CARDS)],'unanimous shared flag before forward'
        shared_before=shared_stats()
        cross_before=int(lib.modern_glm_crossrequest_prefill_calls_r7328(rank))
        check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1))
        before_policy=batch_stats();assert before_policy[0]==1
        before_attention=attention_stats()
        before_fast=fast_stats();report.setdefault("fastpath_phase_flags",{})[label]=cache_policy
        assert batch <= ROWS and all(1 < len(p) and len(p)+limit-1 <= CAPACITY for p in prompts)
        reserve = torch.cuda.mem_get_info(rank)[0]
        assert reserve > 6*1024**3, ('before_phase_reserve', reserve)
        prefill_capi_end=[]
        generated = [[] for _ in prompts]; done = [False]*batch
        completions = [[] for _ in prompts]; hashes = [{} for _ in prompts]
        events = []; feedback_calls = 0; barrier = torch.ones(1, device='cuda');tail_ready=[[]for _ in prompts];root_observer=[[]for _ in prompts]
        case_name=label
        if continuous_policy and participant==emitter:feedback.request('begin',case_name,cohorts=[list(range(a,min(a+micro,batch)))for a in range(0,batch,micro)])
        control(barrier); assert barrier.item() == CARDS
        start = time.perf_counter(); step = 0
        while not all(done):
            if continuous_policy and step==1:
                assert all(len(g)==len(t)==1 for g,t in zip(generated,completions)),'prefill must produce exactly one token per request'
                root_observer=[list(t)for t in completions]
                class ObservedFeedback(CheckedFeedback):
                    def request(self,op,case,**values):
                        packet=super().request(op,case,**values)
                        if op=='take':
                            observed=time.perf_counter()-start
                            for row in packet['rows']:root_observer[row].append(observed)
                        return packet
                checked=ObservedFeedback(feedback,tail_nonemitter=stage==PP-1 and participant!=emitter)
                pasts=[len(p)for p in prompts]
                def local_forward(group,local_step,cohort):
                    nonlocal step,saved_ids
                    step=local_step+1;saved_ids={}
                    descriptions=[]; flat=[]; positions=[]
                    for row in group:
                        assert len(generated[row]) == step
                        p = 0 if step == 0 else len(prompts[row])+step-1
                        current = prompts[row] if step == 0 else [generated[row][-1]]
                        descriptions.extend([row, p, len(current)])
                        flat.extend(current if stage == 0 else [0]*len(current))
                        positions.extend(range(p, p+len(current)))
                    assert len(flat) <= 4096 and max(positions) < 1024
                    ids = torch.tensor(flat, device='cuda', dtype=torch.int64)
                    pos = torch.tensor(positions, device='cuda', dtype=torch.int64)
                    desc = (ctypes.c_size_t*len(descriptions))(*descriptions)
                    logits = torch.full((len(group), vocab), float('nan'),
                                        device='cuda', dtype=torch.bfloat16)
                    modes={'float32_precision':torch.get_float32_matmul_precision(),**{n:getattr(torch.backends.cuda.matmul,n)for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']}};assert modes['float32_precision']=='highest' and not any(v for k,v in modes.items()if k!='float32_precision');forward_native_before=native_stats()
                    decode_scope=int(step>0 and len(flat)==16 and os.environ['GLM_DECODE_ROW_DEDUP_R5386']=='1');forward_decode_before=decode_stats()
                    check(lib.glm_dedup_scope_r5386(rank,decode_scope,ptr(identity_summary)if decode_scope and identity_summary is not None else None));assert lib.glm_dedup_scope_get_r5386(rank)==decode_scope
                    begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()
                    try:
                        _io='prefill'if step==0 else'decode';_mark=not _first_io[_io]
                        if _mark:phase_marker(label+'.first_'+_io+'.enter')
                        check(lib.modern_glm_model_forward(ctx, len(flat), len(group),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()))
                        if _mark:phase_marker(label+'.first_'+_io+'.return');_first_io[_io]=True
                    finally:
                        native_end=time.perf_counter();native_cpu_end=time.thread_time();native_end_realtime_ns=time.time_ns()
                        if step==0 and participant==emitter:prefill_capi_end.append(native_end-start)
                        if step==0 and native_diag:policy.collect(label,step)
                        check(lib.glm_dedup_scope_r5386(rank,0,None));assert lib.glm_dedup_scope_get_r5386(rank)==0
                    native = native_end-begin; native_cpu=native_cpu_end-cpu_begin
                    profile.capture(label,step,group,len(flat),descriptions)
                    if capture_prefix and step==0 and group==batches[0]:
                        manifest=pathlib.Path(capture_prefix+'-manifest.json');audit={'phase':label,'manifest':manifest.name,'available':manifest.exists(),'first_sparse_layer_expected_from_partition':next(i for i in range(bounds[stage],bounds[stage+1])if cfg['mlp_layer_types'][i]=='sparse'),'flat_input_tokens':len(flat),'sample_input_ids_tail':flat[-2:]if stage==0 else None,'sample_positions_tail':positions[-2:],'micro_group_rows':group,'capture_scope':'last2 token /16 route rows plus bounded full-prefill IDs of first eligible sparse prefill, original and native on same frozen prompt/history; source uses original immutable full banks, no weights captured'}
                        if manifest.exists():
                            metadata=json.loads(manifest.read_text());audit.update(metadata=metadata,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),fields_sha256={v['field']:hashlib.sha256(pathlib.Path(v['file']).read_bytes()).hexdigest()for v in metadata['fields']},total_binary_bytes=sum(v['bytes']for v in metadata['fields']))
                        report.setdefault('actual_first_sparse_prefill_captures',[]).append(audit);save()
        
                    if not report.get('fastpath_live_sequence_mutation_rejected'):
                        unchanged=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,cache_policy)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,cache_policy)!=0 and fast_stats()==unchanged
                        report['fastpath_live_sequence_mutation_rejected']=True
                    if stage == PP-1:
                        assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)
                        chosen = logits.argmax(-1)
                        chosen_cpu = [int(v) for v in chosen.cpu().tolist()]
                        if step in [0,1,2,3,63,127,255,383,511]:
                            raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                            if rank == 0:
                                name = f'glm-fastpath-r3627-{report["step"]}-{label}-s{step}-r{group[0]}-bf16.bin'
                                path = root/'results'/name; assert not path.exists(); path.write_bytes(raw)
                                report['raw_checkpoints'].append({'file': name, 'rows': group,
                                    'shape': [len(group), vocab], 'phase': label, 'step': step,
                                    'sha256': hashlib.sha256(raw).hexdigest(), 'GPU_argmax': chosen_cpu})
                            for offset, row in enumerate(group):
                                b = raw[offset*vocab*2:(offset+1)*vocab*2]
                                hashes[row][str(step)] = hashlib.sha256(b).hexdigest()
                        ready_stamp=time.perf_counter()-start
                        if participant==emitter:
                            for row in group:tail_ready[row].append(ready_stamp)
                        for row, token in zip(group, chosen_cpu): saved_ids[row] = token
                    events.append({'round': step, 'rows': group, 'flat_tokens':len(flat), 'native_seconds': native,'native_owner_thread_cpu_seconds':native_cpu,'native_start_realtime_ns':native_start_realtime_ns,'native_end_realtime_ns':native_end_realtime_ns,'native_before':forward_native_before,'native_after':native_stats(),'Torch_matmul_modes':modes,'native_policy':int(os.environ['GLM_GROUPED_ROUTED_R5123']),'decode_dedup_policy':int(os.environ['GLM_DECODE_ROW_DEDUP_R5386']),'decode_scope':decode_scope,'decode_native_before':forward_decode_before,'decode_native_after':decode_stats()})
                    if local_step%32==0 and cohort==0:
                        report['phase_progress']={'label':label,'cohort':cohort,'round':step,'elapsed_seconds':time.perf_counter()-start};save()
                    return(chosen_cpu if stage==PP-1 else None,native)
                safe_decode(feedback=checked,case=case_name,is_emitter=participant==emitter,
                    ids=generated,pasts=pasts,times=completions,micro=micro,limit=limit,eos=answer_eos,
                    forward=local_forward,anchor=start)
                done=[True]*batch
                break
            active = [i for i in range(batch) if not done[i]]
            batches = groups(active, prefill_micro if step==0 else micro, step == 0, [len(p)for p in prompts])
            emitted = torch.zeros(batch, device='cuda', dtype=torch.float32)
            saved_ids = {}
            for group in batches:
                descriptions=[]; flat=[]; positions=[]
                for row in group:
                    assert len(generated[row]) == step
                    p = 0 if step == 0 else len(prompts[row])+step-1
                    current = prompts[row] if step == 0 else [generated[row][-1]]
                    descriptions.extend([row, p, len(current)])
                    flat.extend(current if stage == 0 else [0]*len(current))
                    positions.extend(range(p, p+len(current)))
                assert len(flat) <= 4096 and max(positions) < 1024
                ids = torch.tensor(flat, device='cuda', dtype=torch.int64)
                pos = torch.tensor(positions, device='cuda', dtype=torch.int64)
                desc = (ctypes.c_size_t*len(descriptions))(*descriptions)
                logits = torch.full((len(group), vocab), float('nan'),
                                    device='cuda', dtype=torch.bfloat16)
                modes={'float32_precision':torch.get_float32_matmul_precision(),**{n:getattr(torch.backends.cuda.matmul,n)for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']}};assert modes['float32_precision']=='highest' and not any(v for k,v in modes.items()if k!='float32_precision');forward_native_before=native_stats()
                decode_scope=int(step>0 and len(flat)==16 and os.environ['GLM_DECODE_ROW_DEDUP_R5386']=='1');forward_decode_before=decode_stats()
                check(lib.glm_dedup_scope_r5386(rank,decode_scope,ptr(identity_summary)if decode_scope and identity_summary is not None else None));assert lib.glm_dedup_scope_get_r5386(rank)==decode_scope
                begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()
                try:
                    _io='prefill'if step==0 else'decode';_mark=not _first_io[_io]
                    if _mark:phase_marker(label+'.first_'+_io+'.enter')
                    check(lib.modern_glm_model_forward(ctx, len(flat), len(group),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()))
                    if _mark:phase_marker(label+'.first_'+_io+'.return');_first_io[_io]=True
                finally:
                    native_end=time.perf_counter();native_cpu_end=time.thread_time();native_end_realtime_ns=time.time_ns()
                    if step==0 and participant==emitter:prefill_capi_end.append(native_end-start)
                    if step==0 and native_diag:policy.collect(label,step)
                    check(lib.glm_dedup_scope_r5386(rank,0,None));assert lib.glm_dedup_scope_get_r5386(rank)==0
                native = native_end-begin; native_cpu=native_cpu_end-cpu_begin
                profile.capture(label,step,group,len(flat),descriptions)
                if capture_prefix and step==0 and group==batches[0]:
                    manifest=pathlib.Path(capture_prefix+'-manifest.json');audit={'phase':label,'manifest':manifest.name,'available':manifest.exists(),'first_sparse_layer_expected_from_partition':next(i for i in range(bounds[stage],bounds[stage+1])if cfg['mlp_layer_types'][i]=='sparse'),'flat_input_tokens':len(flat),'sample_input_ids_tail':flat[-2:]if stage==0 else None,'sample_positions_tail':positions[-2:],'micro_group_rows':group,'capture_scope':'last2 token /16 route rows plus bounded full-prefill IDs of first eligible sparse prefill, original and native on same frozen prompt/history; source uses original immutable full banks, no weights captured'}
                    if manifest.exists():
                        metadata=json.loads(manifest.read_text());audit.update(metadata=metadata,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),fields_sha256={v['field']:hashlib.sha256(pathlib.Path(v['file']).read_bytes()).hexdigest()for v in metadata['fields']},total_binary_bytes=sum(v['bytes']for v in metadata['fields']))
                    report.setdefault('actual_first_sparse_prefill_captures',[]).append(audit);save()

                if not report.get('fastpath_live_sequence_mutation_rejected'):
                    unchanged=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,cache_policy)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,cache_policy)!=0 and fast_stats()==unchanged
                    report['fastpath_live_sequence_mutation_rejected']=True
                if stage == PP-1:
                    assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)
                    chosen = logits.argmax(-1)
                    chosen_cpu = [int(v) for v in chosen.cpu().tolist()]
                    if step in [0,1,2,3,63,127,255,383,511]:
                        raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if rank == 0:
                            name = f'glm-fastpath-r3627-{report["step"]}-{label}-s{step}-r{group[0]}-bf16.bin'
                            path = root/'results'/name; assert not path.exists(); path.write_bytes(raw)
                            report['raw_checkpoints'].append({'file': name, 'rows': group,
                                'shape': [len(group), vocab], 'phase': label, 'step': step,
                                'sha256': hashlib.sha256(raw).hexdigest(), 'GPU_argmax': chosen_cpu})
                        for offset, row in enumerate(group):
                            b = raw[offset*vocab*2:(offset+1)*vocab*2]
                            hashes[row][str(step)] = hashlib.sha256(b).hexdigest()
                    ready_stamp=time.perf_counter()-start
                    if participant==emitter:
                        for row in group:tail_ready[row].append(ready_stamp)
                    for row, token in zip(group, chosen_cpu): saved_ids[row] = token
                    if rank == 0: emitted[group] = chosen.float()
                if not pipelined:
                    packet = emitted[group].contiguous(); control(packet); feedback_calls += 1
                    values = [int(v) for v in packet.cpu().tolist()]
                    assert all(0 <= v < vocab for v in values)
                    if stage == PP-1: assert values == [saved_ids[i] for i in group]
                    stamp = time.perf_counter()-start
                    accept(generated, done, group, values, answer_eos, limit)
                    for row in group: completions[row].append(stamp)
                events.append({'round': step, 'rows': group, 'flat_tokens':len(flat), 'native_seconds': native,'native_owner_thread_cpu_seconds':native_cpu,'native_start_realtime_ns':native_start_realtime_ns,'native_end_realtime_ns':native_end_realtime_ns,'native_before':forward_native_before,'native_after':native_stats(),'Torch_matmul_modes':modes,'native_policy':int(os.environ['GLM_GROUPED_ROUTED_R5123']),'decode_dedup_policy':int(os.environ['GLM_DECODE_ROW_DEDUP_R5386']),'decode_scope':decode_scope,'decode_native_before':forward_decode_before,'decode_native_after':decode_stats()})
            if pipelined:
                control(emitted); feedback_calls += 1
                values = [int(v) for v in emitted[active].cpu().tolist()]
                assert all(0 <= v < vocab for v in values)
                if stage == PP-1: assert values == [saved_ids[i] for i in active]
                if reference_kind=='I512'and step==0:
                    expected=[historical_i512['generated_ids'][i][0]for i in active]
                    if values!=expected:
                        report['first_token_failure']={'phase':label,'rows':active,'actual':values,'expected':expected};save();raise CandidateRejected('first_token_gold_mismatch')
                stamp = time.perf_counter()-start
                accept(generated, done, active, values, answer_eos, limit)
                for row in active: completions[row].append(stamp)
            if step % 32 == 0:
                report['phase_progress'] = {'label': label, 'round': step,
                    'active_requests': len(active), 'elapsed_seconds': time.perf_counter()-start}
                save(); print('GLM_COHORT_R3024_PROGRESS', stage, rank,
                              report['phase_progress'], flush=True)
            step += 1
        elapsed = time.perf_counter()-start
        if continuous_policy:
            drained=torch.ones(1,device='cuda');control(drained);assert drained.item()==CARDS
            if participant==emitter:report.setdefault('continuous_transport_cases',{})[label]=feedback.request('end',case_name)
        if not continuous_policy:root_observer=[list(t)for t in completions]
        primary=tail_ready_metrics(generated,tail_ready,answer_eos,elapsed,CARDS,limit)if participant==emitter else None
        identity_diagnostic=None
        if identity_summary is not None:
            torch.cuda.synchronize(rank);identity_diagnostic=identity_summary.cpu().tolist()
        # One EOS or limit terminal token per request; no forward occurs after it.
        assert all(g[-1] in answer_eos or len(g) == limit for g in generated)
        row = {'prefix_plan_enabled':proof_on,'prefix_plan_counter_delta':[b-a for a,b in zip(prefix_before,prefix_stats())],'phase': label, 'prefill_micro':prefill_micro,'prefill_factor_arm':'O8'if prefill_micro==8 else'N4','batch': batch, 'micro': micro, 'timing_includes_actual_capture':capture_enabled,
            'continuous_cohort_policy_r5467':continuous_policy,'tail_ready_times':tail_ready if participant==emitter else None,'tail_ready_metrics':primary,'secondary_phase_elapsed_endpoint':('before final drained control'if continuous_policy else'after last feedback control'),'root_observer_times':root_observer if stage==0 and rank==0 else None,'legacy_step_completion_scope':'baseline rootfeedback-ready; continuous emitter packetstamp, never compare these legacy metrics across arms',
            'reference_kind':reference_kind,'effective_input_tokens':sum(map(len,prompts)),'tail_prefill_CAPI_return_seconds':max(prefill_capi_end)if prefill_capi_end else None,'prefill_input_tokens_per_service_second':sum(map(len,prompts))/max(prefill_capi_end)if prefill_capi_end else None,'prefill_metric_scope':'phase start to last tail prefill CAPI return, includes PP scheduling/wait, excludes subsequent argmax/feedback; not pure kernel time','identity_diagnostic_smoke_only':identity_diagnostic,'resident_on':resident_on,'vector_on':vector_on,'arm':('N'if shared_on else'O'),'shared_validation_enabled':shared_on,'crossrequest_prefill_enabled':prefill_on,'selected_global_layer_count':resident.count,'official_performance_phase':label.startswith('perf-'),'decode_dedup_policy_r5386':int(os.environ['GLM_DECODE_ROW_DEDUP_R5386']),'pipeline_round_feedback': pipelined, 'max_output_tokens': limit,
            'input_lengths': [len(p) for p in prompts], 'generated_ids': generated,
            'raw_hashes_per_request': hashes, 'step_completion_seconds': completions,
            'events': events, 'feedback_calls': feedback_calls, 'elapsed_seconds': elapsed,
            'stopped_at_EOS': [g[-1] in answer_eos for g in generated],
            'no_forward_after_EOS': True, 'qa': qa,
            'memory_free_after_phase': torch.cuda.mem_get_info(rank)[0],
            **metrics(generated, completions, answer_eos, elapsed, CARDS)}
        fused_after=fused_counts();fused_delta=[b-a for a,b in zip(fused_before,fused_after)]
        row['fused_5555_5096_delta']=fused_delta;row['fused_enabled']=fused_enabled
        expected_fused=stage_layers*sum(e['round']>0 and len(e['rows'])==8 for e in events)
        row['fused_expected_5555_calls']=expected_fused
        # Only current Model resident5555 qualifies; standalone5096 is forbidden.
        assert expected_fused>0 and fused_delta==([0,expected_fused,0,0,0,0]if fused_enabled else[expected_fused,0,0,0,0,0]),('actual Model5555 qualification',label,fused_delta,expected_fused)
        report['last_completed_generation']=row;save()
        cross_delta=int(lib.modern_glm_crossrequest_prefill_calls_r7328(rank))-cross_before
        report.setdefault('layer77_dispatch_before_ID_gate',{})[label]={'helper_delta':cross_delta,'mode':int(prefill_on),'diagnostic':diagnostic_only,'shadow':shadow};save()
        if diagnostic_only:
            row.update(diagnostic_only=True,official_performance_phase=False,Model_quality_NOT_RUN=True)
            report.setdefault('layer77_diagnostic_cases',[]).append(row);save();return row

        qa_known=True
        if qa:
            qa_known=all(row['stopped_at_EOS'])
            if stage == PP-1 and rank == 0:
                from glm_chat_template_r20 import prepare_tokenizer, decode_answer
                import re
                tokenizer, kwargs, provenance = prepare_tokenizer()
                assert provenance == payload['chat_template_provenance']
                case = next(c for c in payload['chat_cases'] if label.startswith('QA-'+c['label']+'-'))
                row['answers'] = []
                for g in generated:
                    text, raw_text, closed = decode_answer(tokenizer, g, case['generation_starts_in_thinking'])
                    known = ('北京' in text and ('中国' in text or '首都' in text)) if case['label']=='beijing' else (bool(re.search(r'(?<!\d)3973(?!\d)', text.replace(',',''))) if case['label']=='arithmetic' else '待发货' in text)
                    qa_known=qa_known and known and closed
                    row['answers'].append({'answer': text, 'thinking_closed': closed, 'known_answer': known})
        quality_vote=torch.tensor([float(qa_known)],device='cuda');control(quality_vote)
        if quality_vote.item()!=CARDS:
            if resident_on or vector_on:raise CandidateRejected(('multi_batch_QA',label))
            raise AssertionError(('multi_batch_QA',label))
        same_micro_ids_ok=True;ID_failure_details=[]
        # Identical initial micro cohort and per-position prompts; EOS filtering
        # preserves cohort shape. Compare every sequence and checkpoint byte hash.
        for i, (p, g) in enumerate(zip(prompts, generated)):
            key = (qa, micro, i, hashlib.sha256(json.dumps(p).encode()).hexdigest(), limit)
            if key in references:
                gold = references[key]
                if g != gold['ids']:
                    same_micro_ids_ok=False;ID_failure_details.append({'row':i,'expected':gold['ids'],'actual':g})
                report.setdefault('raw_repeat_diagnostics',[]).append({'phase':label,'row':i,'same_micro_raw':hashes[i]==gold['raw']})if stage==PP-1 else None
            else: references[key] = {'ids': g, 'raw': hashes[i]}
        ID_vote=torch.tensor([float(same_micro_ids_ok)],device='cuda');control(ID_vote)
        if ID_vote.item()!=CARDS:
            report['full_ID_failure']={'phase':label,'details':ID_failure_details};save()
            if resident_on or vector_on:raise CandidateRejected(('same_micro_IDs',label))
            raise AssertionError(('same_micro_IDs',label))
        cross_geometry_ok = True; cross_details = []
        for i, (prompt, ids_) in enumerate(zip(prompts, generated)):
            key = (qa, i, hashlib.sha256(json.dumps(prompt).encode()).hexdigest(), limit)
            if key in geometry_references and geometry_references[key] != ids_:
                cross_geometry_ok = False
                cross_details.append({'row':i,'expected':geometry_references[key],'actual':ids_})
            else: geometry_references[key] = list(ids_)
        ref_ok,ref_detail=validate_reference(reference_kind, 'R'if prefill_on else'O', prompts,generated,row['stopped_at_EOS'],limit,frozen_gold,i512_inputs,independent_references,historical_i512)
        cross_geometry_ok=cross_geometry_ok and ref_ok;cross_details.extend(ref_detail)
        cross_vote = torch.tensor([float(cross_geometry_ok)], device='cuda');control(cross_vote)
        row['cross_geometry_full_IDs_passed'] = cross_vote.item() == CARDS
        if not row['cross_geometry_full_IDs_passed']:
            failure={'phase':label,'details':cross_details,'micro':micro}
            report.setdefault('cross_geometry_ID_failures', []).append(failure)
            report['cross_geometry_ID_failure'] = failure
            row['official_performance_phase']=False
            report['last_completed_generation']=row;save()
            if resident_on or vector_on:raise CandidateRejected(('cross_geometry_IDs',label))
            raise AssertionError(('cross_geometry_IDs',label))
        after_policy=batch_stats();delta=[b-a for a,b in zip(before_policy,after_policy)]
        assert after_policy[0]==1 and delta[4]==0
        after_fast=fast_stats();fast_delta=[b-a for a,b in zip(before_fast,after_fast)]
        decode_calls=sum(e['round']>0 for e in events);decode_requests=sum(len(e['rows'])for e in events if e['round']>0)
        shared_requests=sum(len(e['rows'])for e in events if e['round']>0 and len(e['rows'])==8)
        expected=[full_layers*decode_calls,full_layers*decode_calls,stage_layers*decode_requests-resident.count*shared_requests]if cache_policy else[0,0,0]
        assert fast_delta==expected,('actualfastpathdispatch',label,fast_delta,expected)
        row.update(fastpath_bundle_policy=cache_policy,fastpath_counters_before=before_fast,fastpath_counters_after=after_fast,fastpath_counter_delta=fast_delta,batch_cache_policy=1,batch_cache_before=before_policy,
                   batch_cache_after=after_policy,batch_cache_delta=delta)
        attention_delta=[b-a for a,b in zip(before_attention,attention_stats())]
        enabled=int(os.environ['GLM_BATCH_ALLSELECTED_R5096'])
        eligible=[e for e in events if e['round']>0 and 8<=len(e['rows'])<=128 and len({len(prompts[i]) for i in e['rows']})==1 and len(prompts[e['rows'][0]])+e['round']<=640]
        attention_expected=[(stage_layers-resident.count)*len(eligible),(stage_layers-resident.count)*sum(len(e['rows']) for e in eligible)] if enabled else [0,0]
        assert attention_delta==attention_expected,('actual_batch_attention_dispatch',label,attention_delta,attention_expected)
        row.update(batch_attention_counter_delta=attention_delta,batch_attention_expected=attention_expected,batch_attention_policy_r5096=enabled)
        row['group_commit_policy_r5030']=int(os.environ['GLM_GROUP_PROJECTED_HOST_R5030']);row['async_commit_policy_r5011']=int(os.environ['GLM_ASYNC_PROJECTED_HOST_R5011']);row['same_micro_reference_passed'] = True
        native_after=native_stats();native_delta=[b-a for a,b in zip(native_before,native_after)]
        assert native_delta[4]==0,('native_error',label,native_delta)
        if int(os.environ['GLM_GROUPED_ROUTED_R5123'])==1 and max(map(len,prompts))>=512:assert native_delta[1]>0 and native_delta[2]>0,('native_not_dispatched',label,native_delta)
        row['routed_native_policy_r5123']=int(os.environ['GLM_GROUPED_ROUTED_R5123']);row['native_operator_count_before']=native_before;row['native_operator_count_after']=native_after;row['native_operator_count_delta']=native_delta
        row['decode_native_count_before']=decode_before;row['decode_native_count_after']=decode_stats();row['decode_native_count_delta']=[b-a for a,b in zip(decode_before,decode_stats())];assert lib.glm_dedup_scope_get_r5386(rank)==0
        row['attention_implementation']='5707 ROWS16: resident exact5555 M8/original601 EOS tails; unselected host actual5096 when uniform; transientOFF' 
        cross_delta=int(lib.modern_glm_crossrequest_prefill_calls_r7328(rank))-cross_before
        cross_expected=(stage_layers if prefill_on==1 else int(stage==1)if prefill_on==2 else 0)*sum(e['round']==0 and len(e['rows'])==8 and e['flat_tokens']==4096 and all(len(prompts[i])==512 for i in e['rows'])for e in events)if prefill_on else 0
        no_op=torch.tensor([float(prefill_on==2 and stage==1 and cross_expected>0 and cross_delta==0)],device='cuda');control(no_op)
        if no_op.item()>0:raise CandidateRejected('77 no-op/eligibility')
        assert cross_delta==cross_expected,('actualcrossrequestdispatch',label,cross_delta,cross_expected)
        row.update(crossrequest_call_delta=cross_delta,crossrequest_call_expected=cross_expected)
        shared_delta=[b-a for a,b in zip(shared_before,shared_stats())]
        decode_rows=sum(len(e['rows'])for e in events if e['round']>0)
        total_rows=sum(len(e['rows'])for e in events)
        shared_layers=stage_layers-full_layers
        skipped=shared_layers*decode_rows if shared_on and cache_policy else 0
        expected_shared=[skipped,shared_layers*total_rows-skipped]
        assert shared_delta==expected_shared,('shared validation dispatch',label,shared_delta,expected_shared)
        skipped_bytes=shared_layers*8*sum(len(prompts[i])+e['round']for e in events if e['round']>0 for i in e['rows'])if shared_on and cache_policy else 0
        row.update(shared_validation_counter_before=shared_before,shared_validation_counter_delta=shared_delta,shared_validation_expected=expected_shared,shared_validation_source_exact_avoided_D2H_bytes=skipped_bytes,shared_validation_bytes_scope='Source-derived selection int64 prefix bytes authorized by matches(row,past,count,capacity); not physical bus measurement or exclusive time.')
        resident.finish(label,resident_before,events,prompts)
        row['resident_report_after']=resident.snapshot()
        report['multi_batch_cases'].append(row); save()
        print('GLM_RESIDENT_PHASE',stage,rank,label,row['aggregate_useful_decode_tokens_s'],flush=True)
        return row

    unique_inputs=[c['input_ids']for c in payload['unique16_cases']]
    assert unique_inputs==frozen_gold['input_ids'] and len(unique_inputs)==16
    i512_inputs=payload['I512_input_ids'];assert len(i512_inputs)==32 and all(len(p)==512 for p in i512_inputs)
    frozen477=root/'incoming/GLM477-frozen-gold-r8844.json'
    assert hashlib.sha256(frozen477.read_bytes()).hexdigest()=='ff305cd7c4aa8b414ef1a150bb872422fafed2c02eafb56ba53b4798b4a76176'
    historical_i512=json.loads(frozen477.read_text());assert historical_i512['input_ids']==i512_inputs
    lib.modern_glm_crossrequest_prefill_calls_r7328.argtypes=[ctypes.c_int];lib.modern_glm_crossrequest_prefill_calls_r7328.restype=ctypes.c_uint64
    independent_references={}
    continuous_policy=1
    os.environ['GLM_ASYNC_PROJECTED_HOST_R5011']='0';os.environ['GLM_GROUP_PROJECTED_HOST_R5030']='0';os.environ['GLM_BATCH_ALLSELECTED_R5096']='1';os.environ['GLM_GROUPED_ROUTED_R5123']='1';os.environ['GLM_DECODE_ROW_DEDUP_R5386']='0'
    os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']='0'
    report['prefill_factor_scope']='Same-load current8854 B16/decodeM8/prefillM8 I512O128 fused OFF/ON/ON/OFF; currentvector/prefix/native/SDK/math flags unchanged; ordinary profileOFF; fixed actual38/40 resident mask; full477 finalIDs/EOS/QA.'
    labels=['perf-I512-B16-N-fusedOFF-r0','perf-I512-B16-N-fusedON-r1','perf-I512-B16-N-fusedON-r2','perf-I512-B16-N-fusedOFF-r3'];report['resident_plan']=[{'label':x}for x in labels];rejected=False
    report['performance_adoption']=False;report['ordinary_profileOFF']=True
    try:
        for label in labels:
            fused_enabled=int('-fusedON-'in label)
            check(lib.glm_fused_entries_policy_r9028(rank,fused_enabled))
            prompts=i512_inputs[:16]
            generate(label,prompts,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)
    except CandidateRejected as exc:
        failed=report.get('last_completed_generation')
        if failed:report.setdefault('rejected_generation_records',{})[failed['phase']]=failed
        rejected=True;admission=bool(report.get('resident_admission_rejection'));report.update(resident_candidate_rejected=True,normal_numeric_rejection=not admission,resident_admission_rejected=admission,resident_rejection_detail=str(exc));save()
    finally:
        check(lib.glm_fused_entries_policy_r9028(rank,0));report['fused_final_OFF_restored']=True
        os.environ['GLM_PREFILL_PREFIX_PLAN_R8703']='0'
        public(1);resident.apply('final-before-B1',False,False);policy.switch('final-before-B1',0,False)
        os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']='0'if rejected else'1'
        report['after_select_shared_flag']=int(os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814'])
        qa_vote=torch.tensor([float(report['after_select_shared_flag']),1.],device='cuda');control(qa_vote)
        assert qa_vote.tolist()==[float(CARDS*report['after_select_shared_flag']),float(CARDS)]
        report['after_select_QA_scope']='SKIPPED_REJECTED_CANDIDATE'if rejected else'EXECUTED_REQUIRED'
        if not rejected:assert chat_gate('after_select'),'candidate shared proof frozenB1'
        os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']='0'
        public(0);resident.apply('final-O-empty',False,False);resident.check(resident.vector_OFF())
        final=resident.snapshot();report['resident_final_OFF_empty']=not final['any_enabled']and not final['vector_mode']and not final['transient']['enabled']and not final['transient']['busy']and not final['transient']['poison']
        report['shared_validation_final_OFF']=os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']=='0'
        report['resident_candidate_quality_passed']=not rejected;report['cohort_phases_completed']=not rejected
        report['skipped_rejected_candidate_phases']=[x for x in labels if x not in[c['phase']for c in report['multi_batch_cases']]]
        report['planned_phase_accounting_completed']=len(report['multi_batch_cases'])+len(report['skipped_rejected_candidate_phases'])==len(labels)
        report['continuous_transport_closed']={'all_cases_ended':True,'pending_drained_on_emitter':hub is None or(not hub.pending and hub.case is None)}
        feedback.close()
        if hub is not None:hub.close()
        save()
