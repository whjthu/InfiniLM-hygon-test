"""Full-model GLM PP cohort probe. Imported by the qualified latest loader."""
import os, ctypes, hashlib, json, pathlib, time
import torch
from glm_large_schedule_r3627 import groups, accept, metrics
from glm_continuous_feedback_r5466 import FeedbackHub,FeedbackClient,CheckedFeedback,safe_decode,tail_ready_metrics
import socket,secrets
from glm_lowbatch_profile_r8219 import OwnerProfile
from glm_vector_transient_control_r5694 import VectorTransientControl as ResidentControl,CandidateRejected

def run(lib, ctx, stage, rank, payload, control, check, ptr, stream, vocab,
        report, report_path, chat_gate):
    root = pathlib.Path('/refactor')
    eos = set(payload['natural_eos_token_ids']);PP=report['PP'];CARDS=report['all_serving_cards'];ROWS=report['cache_rows'];CAPACITY=report['cache_capacity']
    report.update(scope='Private5690 transient sameRuntime packedD2H/exactPi staging; same originalRuntime, TP8PP2/B16M8 samegeometry OFF/ON ABBA. OFFM16 QA reference only. FinalIDs/EOS/QA/frozen448/normalclose/CPUargmax/Slurm0 required; no productionpeak or componentmultiplier claim.',
        multi_batch_cases=[],raw_checkpoints=[],single_user15_is_adoption_gate=False)
    report['probe_source_sha256'] = {
        name: hashlib.sha256((root/'incoming'/name).read_bytes()).hexdigest()
        for name in ['glm_large_schedule_r3627.py', 'glm_transient_probe_r5694.py',
                     'modern_glm_transient_r5694.py', 'glm_continuous_feedback_r5466.py','kimi_cohort_feedback_idle_r4915.py']}
    def save():
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)
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
    public(0);resident.contract_guards();before=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,2)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,2)!=0 and fast_stats()==before
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0 and fast_stats()==before
    report.update(fastpath_invalid_and_dependency_guards_verified=True,fastpath_stage_layers=stage_layers,fastpath_full_indexer_layers=full_layers)
    references = {}; geometry_references = {}
    profile = OwnerProfile(lib, report, stage, rank)
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
    continuous_policy=0
    participant=stage*8+rank;emitter=8;hub=None
    endpoint=root/'control'/('glm-transient-Model-r5694-'+report['step']+'-endpoint.json')
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
    def generate(label, prompts, micro, pipelined, limit, answer_eos, qa=False, cache_policy=0, resident_on=False, trace=False, vector_on=False):
        native_before=native_stats();decode_before=decode_stats();assert lib.glm_dedup_scope_get_r5386(rank)==0
        diagnostic_phase=label in ['smoke8g16m16_routed1','smoke8u16m16_routed1']
        identity_summary=torch.zeros(4,dtype=torch.int64,device='cuda')if diagnostic_phase else None
        if identity_summary is not None:torch.cuda.synchronize(rank)
        lib.glm_actual_capture_arm_r5164.argtypes=[ctypes.c_int,ctypes.c_char_p]
        capture_enabled=os.environ.get('GLM_ACTUAL_CAPTURE_R5164')=='1' and label in ['QA-retrieval-routed0-r1','QA-retrieval-routed1-r1','g16m16_routed0_r0','g16m16_routed1_r1']
        capture_prefix=f'/refactor/results/glm-actual-capture-r5164-{report["step"]}-{label}-stage{stage}-rank{rank}' if capture_enabled else ''
        check(lib.glm_actual_capture_arm_r5164(rank,capture_prefix.encode()))

        resident.configure(vector_on);public(cache_policy); resident_before=resident.apply(label,resident_on,trace); batch = len(prompts)
        check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1))
        before_policy=batch_stats();assert before_policy[0]==1
        before_attention=attention_stats()
        before_fast=fast_stats();report.setdefault("fastpath_phase_flags",{})[label]=cache_policy
        assert batch <= ROWS and all(1 < len(p) and len(p)+limit-1 <= CAPACITY for p in prompts)
        reserve = torch.cuda.mem_get_info(rank)[0]
        assert reserve > 6*1024**3, ('before_phase_reserve', reserve)
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
                        check(lib.modern_glm_model_forward(ctx, len(flat), len(group),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()))
                    finally:
                        check(lib.glm_dedup_scope_r5386(rank,0,None));assert lib.glm_dedup_scope_get_r5386(rank)==0
                    native = time.perf_counter()-begin; native_cpu=time.thread_time()-cpu_begin;native_end_realtime_ns=time.time_ns()
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
            batches = groups(active, micro, step == 0, [len(p)for p in prompts])
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
                    check(lib.modern_glm_model_forward(ctx, len(flat), len(group),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()))
                finally:
                    check(lib.glm_dedup_scope_r5386(rank,0,None));assert lib.glm_dedup_scope_get_r5386(rank)==0
                native = time.perf_counter()-begin; native_cpu=time.thread_time()-cpu_begin;native_end_realtime_ns=time.time_ns()
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
        row = {'phase': label, 'batch': batch, 'micro': micro, 'timing_includes_actual_capture':capture_enabled,
            'continuous_cohort_policy_r5467':continuous_policy,'tail_ready_times':tail_ready if participant==emitter else None,'tail_ready_metrics':primary,'secondary_phase_elapsed_endpoint':('before final drained control'if continuous_policy else'after last feedback control'),'root_observer_times':root_observer if stage==0 and rank==0 else None,'legacy_step_completion_scope':'baseline rootfeedback-ready; continuous emitter packetstamp, never compare these legacy metrics across arms',
            'identity_diagnostic_smoke_only':identity_diagnostic,'resident_on':resident_on,'vector_on':vector_on,'arm':('B'if resident_on and vector_on else'T'if resident_on else'V'if vector_on else'O'),'selected_global_layer_count':resident.count,'official_performance_phase':label.startswith('perf-'),'decode_dedup_policy_r5386':int(os.environ['GLM_DECODE_ROW_DEDUP_R5386']),'pipeline_round_feedback': pipelined, 'max_output_tokens': limit,
            'input_lengths': [len(p) for p in prompts], 'generated_ids': generated,
            'raw_hashes_per_request': hashes, 'step_completion_seconds': completions,
            'events': events, 'feedback_calls': feedback_calls, 'elapsed_seconds': elapsed,
            'stopped_at_EOS': [g[-1] in answer_eos for g in generated],
            'no_forward_after_EOS': True, 'qa': qa,
            'memory_free_after_phase': torch.cuda.mem_get_info(rank)[0],
            **metrics(generated, completions, answer_eos, elapsed, CARDS)}
        report['last_completed_generation']=row;save()
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
            if qa and micro == 8 and key not in geometry_references:
                cross_geometry_ok = False
                cross_details.append({'row':i,'missing_M16_logical_row_reference':True})
            if key in geometry_references and geometry_references[key] != ids_:
                cross_geometry_ok = False
                cross_details.append({'row':i,'expected':geometry_references[key],'actual':ids_})
            else: geometry_references[key] = list(ids_)
            if not qa and limit <= 128:
                assert prompts == frozen_gold['input_ids'], 'frozen input identity mismatch'
                if ids_ != frozen_gold['generated_ids'][i][:limit]:
                    cross_geometry_ok = False
                    cross_details.append({'row':i,'historical_reference':True,'expected':frozen_gold['generated_ids'][i],'actual':ids_})
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
        row['attention_implementation']='private5690 transient exact5555 M8 plus originalhost/r601 EOS tails' 
        resident.finish(label,resident_before,events,prompts)
        row['resident_report_after']=resident.snapshot()
        report['multi_batch_cases'].append(row); save()
        print('GLM_RESIDENT_PHASE',stage,rank,label,row['aggregate_useful_decode_tokens_s'],flush=True)
        return row

    natural=[c['input_ids']for c in payload['natural_cases']];unique_inputs=[c['input_ids']for c in payload['unique16_cases']]
    assert len(unique_inputs)==16 and len(set(map(tuple,unique_inputs)))==16
    def scheduling(policy):
        nonlocal continuous_policy
        continuous_policy=policy
        os.environ['GLM_ASYNC_PROJECTED_HOST_R5011']='0';os.environ['GLM_GROUP_PROJECTED_HOST_R5030']='0';os.environ['GLM_BATCH_ALLSELECTED_R5096']='1';os.environ['GLM_GROUPED_ROUTED_R5123']='1';os.environ['GLM_DECODE_ROW_DEDUP_R5386']='0'
    # One image/load. Actual kernel-shape vector dispatch, same frozen B16M8.
    scheduling(1);plan=[]
    arms={'O':(False,False),'T':(True,False),'V':(False,True),'B':(True,True)}
    def add(label,prompts,micro,limit,qa=False,arm='O',trace=False):
        t,v=arms[arm];plan.append(dict(label=label,prompts=prompts,micro=micro,limit=limit,qa=qa,on=t,vector=v,arm=arm,trace=trace))
    for micro in (16,8):
        for case in payload['chat_cases']:add('QA-'+case['label']+'-OFF-m'+str(micro),[case['input_ids']]*16,micro,min(256,CAPACITY-len(case['input_ids'])+1),True)
    add('anchor-O-M8-full128',unique_inputs,8,128)
    add('prof-O-M8-K128',unique_inputs,8,128,trace=True)
    for arm in ('T','V','B'):
        for case in payload['chat_cases']:add('QA-'+case['label']+'-'+arm+'-m8',[case['input_ids']]*16,8,min(256,CAPACITY-len(case['input_ids'])+1),True,arm)
    for i,arm in enumerate(('O','T','V','B','B','V','T','O')):add('perf-'+arm+'-M8-r'+str(i),unique_inputs,8,128,arm=arm)
    add('restore-O-M8-full128',unique_inputs,8,128)
    report['resident_plan']=[{k:v for k,v in c.items()if k!='prompts'}for c in plan]
    report['resident_scope']='Same image/Runtime/provenhost/residentOFF; T sharedScratch, V exact4/8 vectorloads, Bboth. FullIDs/EOS/QA perarm; componentmultipliers not Model claim.'
    rejected=set();b1_done=set();extra_profile_count=0
    for c in plan:
        arm=c['arm']
        if c['label']=='restore-O-M8-full128'and not rejected:
            # Emitter alone chooses; world-gather publishes identical arm.
            scores=None;choice=clear=0
            if participant==8:
                perf=[x for x in report['multi_batch_cases']if x['phase'].startswith('perf-')]
                scores={a:sum(x['tail_ready_metrics']['aggregate_useful_decode_tokens_s']for x in perf if x['arm']==a)/2 for a in arms}
                choice=max(range(4),key=lambda i:scores[('O','T','V','B')[i]])
                clear=int(scores[('O','T','V','B')[choice]]>scores['O']*1.01)
            gathered=resident.all_rows([choice,clear]);winner=('O','T','V','B')[gathered[8][0]]
            report['profile_winner_selection']={'arm':winner,'emitter_scores':scores if participant==8 else None}
            if winner!='O'and gathered[8][1]:
                wt,wv=arms[winner];label='prof-W-'+winner+'-M8-K128'
                report['resident_plan'].insert(-1,dict(label=label,micro=8,limit=128,qa=False,on=wt,vector=wv,arm=winner,trace=True));extra_profile_count=1
                try:generate(label,unique_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,wt,True,wv)
                except CandidateRejected as exc:
                    rejected.add(winner);report.update(resident_candidate_rejected=True,normal_numeric_rejection=True)
                    report.setdefault('rejected_arms',{})[winner]=str(exc)
                    if label not in[x['phase']for x in report['multi_batch_cases']]:report.setdefault('skipped_rejected_candidate_phases',[]).append(label)
                    resident.configure(False);public(0);resident.apply('profile-rejected-reset-O',False,False)
        if arm in rejected or (arm=='B'and rejected&{'T','V'}):
            report.setdefault('skipped_rejected_candidate_phases',[]).append(c['label']);save();continue
        try:
            generate(c['label'],c['prompts'],c['micro'],True,c['limit'],set(payload['chat_eos_token_ids']),c['qa'],1,c['on'],c['trace'],c['vector'])
            if c['qa']and arm!='O'and c['label']==[p['label']for p in plan if p['arm']==arm and p['qa']][-1]:
                resident.configure(c['vector']);public(1);resident.apply('B1-'+arm, c['on'],False)
                ok=chat_gate('candidate_'+arm);b1_done.add(arm)
                if not ok:raise CandidateRejected(('candidateB1 finalIDs/EOS',arm))
        except CandidateRejected as e:
            if arm=='O':raise
            rejected.add(arm);report.update(resident_candidate_rejected=True,normal_numeric_rejection=True)
            report.setdefault('rejected_arms',{})[arm]=str(e)
            if c['label']not in[x['phase']for x in report['multi_batch_cases']]:report.setdefault('skipped_rejected_candidate_phases',[]).append(c['label'])
            resident.configure(False);public(0);resident.apply('rejected-'+arm+'-reset-O',False,False);save()
    resident.configure(False);scheduling(0);public(1);resident.apply('final-O-before-B1',False,False)
    assert chat_gate('after_select'),'finalO frozenB1'
    public(0);resident.apply('final-O-empty',False,False)
    final=resident.snapshot();report['resident_final_OFF_empty']=not final['enabled']and not final['vector_mode']and not final['busy']and not final['poison']
    report['planned_phase_accounting_completed']=len(report['multi_batch_cases'])+len(report.get('skipped_rejected_candidate_phases',[]))==len(plan)+extra_profile_count;assert report['planned_phase_accounting_completed']
    report['cohort_phases_completed']=len(report['multi_batch_cases'])==len(plan)+extra_profile_count
    report['resident_candidate_quality_passed']=not rejected and b1_done=={'T','V','B'}
    report['continuous_transport_closed']={'all_cases_ended':True,'pending_drained_on_emitter':hub is None or(not hub.pending and hub.case is None)}
    feedback.close()
    if hub is not None:hub.close()
    save()
