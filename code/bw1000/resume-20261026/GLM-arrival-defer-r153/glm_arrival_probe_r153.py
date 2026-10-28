"""Finite32 syntheticburst nativearrival emulation; currentce830 unchangedNative ordinaryforward."""
import ctypes,json,pathlib,time,os,hashlib,collections,copy
import torch
from glm_crossrequest_control_r9059 import CrossrequestControl
from quality_metrics_r9132b import paired_quality,serving_metrics,score_case
from arrival_runner_r9258 import run_episode,torch_world_callbacks
from case_plan_r153 import load as load_plan

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker):
 r=pathlib.Path('/refactor');program=payload['benchmark'];requests=program['service_cases'][:32];is_emitter=stage==1 and rank==0;deadline=time.monotonic()+max(0,report['remaining_process_seconds']);native_events=[];plan=load_plan();folder=pathlib.Path(__file__).parent;trace=json.loads((folder/'arrival-trace.json').read_bytes());corpus=json.loads((folder/'corpus-reference.json').read_bytes());assert requests==corpus['requests'];arrivals=[a['arrival_seconds']for a in trace['requests']];assert arrivals==[4*(i//8)for i in range(32)];assert [r['id']for r in requests]==[r['id']for r in trace['requests']];eos=program['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list)else[eos]);report.update(M15_row2_fixed_ON=True,scope='R148 same-load arrival32maxlive32 M16x2 serial/deferred ABBA; currentce830 M15OFF row2ON observerOFF; originalNative/math/drains; notHTTP',episodes=[],native_events=native_events,normal_reset_close=False,benchmark_complete=False,token_exactness_gate=False,arrival_trace_SHA256=hashlib.sha256((folder/'arrival-trace.json').read_bytes()).hexdigest(),core_source_SHA256=hashlib.sha256((folder/'arrival_runner_r9258.py').read_bytes()).hexdigest(),benchmark_program_SHA256=payload['benchmark_SHA256'])
 lib.modern_glm_sparse_prepare_r131.argtypes=[ctypes.c_void_p];lib.modern_glm_sparse_policy_r131.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_sparse_read_r131.argtypes=[ctypes.c_void_p];lib.modern_glm_sparse_read_r131.restype=ctypes.c_char_p
 def sparse_read():
  a=lib.modern_glm_sparse_read_r131(ctx);assert a;return json.loads(a)
 def save():
  tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(report_path)
 def progress(label,seq,op,active,warm):
  if rank!=0:return
  q={'run':os.environ['GLM_RUN_R5040'],'stage':stage,'rank':rank,'phase':label,'seq':seq,'op':op,'warm_active':warm and active,'stall_guard_active':active,'UTC_epoch':time.time(),'stall_limit_seconds':60,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'native_calls':len(native_events)};pp=r/'results'/('GLM-arrival-progress-r153-'+q['run']+'-s'+str(stage)+'-r0.json');tmp=pp.with_suffix('.tmp');tmp.write_text(json.dumps(q));tmp.replace(pp)
 lib.glm_attention_cohort_policy_r9054.argtypes=[ctypes.c_int,ctypes.c_int];lib.glm_attention_cohort_counts_r9054.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.glm_reduced_output_policy_r9032.argtypes=[ctypes.c_int,ctypes.c_int]
 def stats():v=(ctypes.c_uint64*4)();check(lib.glm_attention_cohort_counts_r9054(rank,v,4));return list(v)
 lib.glm_ragged_policy_r9136.argtypes=[ctypes.c_int,ctypes.c_int];lib.glm_ragged_enabled_r9136.argtypes=[ctypes.c_int];lib.glm_ragged_counts_r9136.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.glm_ragged_Bcounts_r9136.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 assert lib.glm_ragged_enabled_r9136(rank)==0
 assert lib.glm_ragged_policy_r9136(rank,2)!=0
 def raggedstats():
  v=(ctypes.c_uint64*4)();check(lib.glm_ragged_counts_r9136(rank,v,4));h=(ctypes.c_uint64*132)();check(lib.glm_ragged_Bcounts_r9136(rank,h,132));assert [sum(h[B*4+j]for B in range(33))for j in range(4)]==list(v);return list(v),list(h)
 lib.modern_glm_down_row2_policy_r9284.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_down_row2_get_r9284.argtypes=[ctypes.c_int];lib.modern_glm_down_row2_counts_r9284.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_glm_vector_counts_r8540.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 config=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_bytes());routed_count=sum(x=='sparse'for x in config['mlp_layer_types'][[0,38][stage]:[38,78][stage]]);assert routed_count==[35,40][stage];report['routed_layer_count']=routed_count;assert lib.modern_glm_down_row2_get_r9284(rank)==0
 lib.modern_glm_model_m15_oproj_policy_r125.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.modern_glm_m15_oproj_stats_r125.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def m15stats():
  a=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,a,17));return list(a)
 assert m15stats()[0]==0,'M15 defaultOFF'
 report['M15_stats_order']=['enabled','eligible','selected','off_fallback','other_M15_rejected','errors','switches','created','destroyed','hits','misses','evictions','active','active_bytes','highwater','runs','invalid']
 def vectorstats():
  a=(ctypes.c_uint64*8)();b=(ctypes.c_uint64*4)();check(lib.modern_glm_vector_counts_r8540(rank,a,8));check(lib.modern_glm_down_row2_counts_r9284(rank,b,4));return {'legacy':list(a),'row2':list(b)}
 def vectorreceipt(before,after,events,enabled):
  delta={k:[b-a for a,b in zip(before[k],after[k])]for k in before};Ts=[e['input_tokens']for e in events];hit=[T for T in Ts if T in [4,8,16]];pair=[T for T in Ts if enabled and T in [8,16]];h=len(hit)*routed_count;fall=(len(Ts)-len(hit))*routed_count;tokens=sum(hit)*routed_count;expected={'legacy':[h,fall,h,fall,0,tokens,tokens,0],'row2':[h-len(pair)*routed_count,len(pair)*routed_count,0,0]};assert delta==expected,('actualperownerprovidercounter',stage,rank,enabled,delta,expected);return {'before':before,'after':after,'delta':delta,'expected':expected,'actual_verified':True,'routed_layer_count':routed_count}
 def barrier():v=torch.ones(1,device='cuda');control(v);assert v.item()==16
 def budget(reserve=90):
  v=torch.tensor([float(time.monotonic()+reserve<deadline)],device='cuda');control(v);return v.item()==16
 resident=CrossrequestControl(lib,ctx,stage,rank,control,check,report,save);check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_transient_policy_r5688(ctx,0));check(lib.modern_glm_vector_policy_r8540(rank,0))
 for name in ['decode_leases_policy_r1356','indexer_policy_r998','feature_bypass_r1406','owned_prefix_policy_r1669']:check(getattr(lib,'modern_glm_model_'+name)(ctx,1))
 resident.apply('R9284B-fixed-resident-mask',True,False);check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0));report['fixed_resident_mask']=resident.chosen;check(lib.modern_glm_sparse_prepare_r131(ctx));assert sparse_read()['enabled']is False
 os.environ.update(GLM_SHARED_PREFIX_VALIDATE_R5814='1',GLM_CROSSREQUEST_PREFILL_R7328='0',GLM_PREFILL_PREFIX_PLAN_R8703='0',GLM_GROUPED_ROUTED_R5123='1',GLM_BATCH_ALLSELECTED_R5096='1',GLM_DECODE_ROW_DEDUP_R5386='0')
 def reset(on):check(lib.modern_glm_model_reset(ctx));check(lib.glm_attention_cohort_policy_r9054(rank,1));check(lib.glm_ragged_policy_r9136(rank,on));assert lib.glm_ragged_enabled_r9136(rank)==on;barrier()
 def forward(rows,tokens,pasts,label,on,prefill):
  assert rows and len(rows)==len(set(rows))and all(0<=i<32 for i in rows)and len(rows)==len(tokens)==len(pasts)and all(tokens);flat=[x for v in tokens for x in v];assert len(flat)<=4096 and len(rows)<=32;positions=[x for past,v in zip(pasts,tokens)for x in range(past,past+len(v))];assert max(positions)<640
  desc=(ctypes.c_size_t*(3*len(rows)))(*[x for row,past,v in zip(rows,pasts,tokens)for x in [row,past,len(v)]]);ids=torch.tensor(flat if stage==0 else [0]*len(flat),device='cuda',dtype=torch.int64);pos=torch.tensor(positions,device='cuda',dtype=torch.int64);logits=torch.full((len(rows),vocab),float('nan'),device='cuda',dtype=torch.bfloat16);t16_control=len(flat)==16 and not prefill;m15_t16_before=m15stats()if t16_control else None;before=stats();rag_before,hist_before=raggedstats();torch.cuda.synchronize();tick=time.perf_counter();check(lib.modern_glm_model_forward(ctx,len(flat),len(rows),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()));ended=time.perf_counter();delta=[b-a for a,b in zip(before,stats())]
  rag_after,hist_after=raggedstats();rag_delta=[b-a for a,b in zip(rag_before,rag_after)];hist_delta=[b-a for a,b in zip(hist_before,hist_after)];expected=[0]*4;B=len(rows);single_decode=len(flat)==B and all(len(t)==1 for t in tokens);equal_p=len(set(pasts))==1
  if single_decode and (B in [8,16]or(on and 2<=B<=32)):
   route=2 if equal_p and B in [8,16]else 1 if on and not equal_p else 0;expected[route]=resident.count
  assert rag_delta==expected,('actualperlayerbranch',label,on,stage,rank,rows,pasts,rag_delta,expected,resident.chosen)
  expected_hist=[0]*132
  for j,n in enumerate(expected):expected_hist[B*4+j]=n
  assert hist_delta==expected_hist,('actualBhist',label,hist_delta,expected_hist)

  if stage==1:assert bool(torch.isfinite(logits).all()),('nonfinite',label)
  free,total=torch.cuda.mem_get_info(rank);assert free>=512*1024**2,('originalGLM512MiBfloor',label,free);report['minimum_driver_free_bytes']=min(report.get('minimum_driver_free_bytes',free),free)
  if t16_control:
   m15_t16_after=m15stats();dd=[b-a for a,b in zip(m15_t16_before,m15_t16_after)];assert dd[1]==dd[2]==dd[3]==dd[5]==dd[15]==dd[16]==0,('T16 no M15 path',stage,rank,dd)
   report.setdefault('M15_T16_same_load_controls',[]).append({'label':label,'rows':len(rows),'input_tokens':len(flat),'native_seconds':ended-tick,'M15_delta17':dd,'default_nonselection_verified':True})
  pipeline_profile=[float(lib.modern_glm_pipeline_last_profile()[j])for j in range(6)];native_events.append({'existing_pipeline_profile':pipeline_profile,'window_profile_omitted':'oldwindowhardB8/pastfixed/round63..65 doesnotcoverEOSvarlen','label':label,'rows':rows,'pasts':pasts,'counts':list(map(len,tokens)),'input_tokens':len(flat),'past_min':min(pasts),'past_max':max(pasts),'native_seconds':ended-tick,'native_return_clock_s':ended,'prefill':prefill,'cohort_counter_delta':delta,'ragged_counter_delta':rag_delta,'ragged_Bhist_delta':hist_delta,'expected_ragged_counter_delta':expected,'ON':on});return logits,ended-tick,tick,ended
 tokenizer=None
 if is_emitter:
  from transformers import PreTrainedTokenizerFast
  tokenizer=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True)
 def episode(label,mode,case_requests,case_arrivals,warm=False,down_row2=True,m15_oproj=False,deferred_feedback=False):
  assert type(down_row2)is bool and mode is False;torch.cuda.synchronize();check(lib.modern_glm_down_row2_policy_r9284(rank,int(down_row2)));assert lib.modern_glm_down_row2_get_r9284(rank)==int(down_row2);barrier();reset(1);check(lib.modern_glm_sparse_policy_r131(ctx,0));check(lib.modern_glm_model_m15_oproj_policy_r125(ctx,int(m15_oproj)));barrier();m15_before=m15stats();assert m15_before[0]==int(m15_oproj);vector_before=vectorstats();phase_marker(label+'.begin');start_index=len(native_events);last={};local_calls=[];cm,tm=torch_world_callbacks(torch,control,is_emitter);progress(label,0,'BEGIN',True,warm)
  def command_exchange(cmd,seq):
   decoded=cm(cmd,seq);last.clear();last.update(decoded)
   progress(label,seq,decoded['op'],decoded['op']!='DONE',warm)
   return decoded
  def bound_forward(rows,tokens,pasts,kind,prefill):
   selected=[x for x in last['items']if x[0]in rows];assert rows==[x[0]for x in selected]and pasts==[x[1]for x in selected]and list(map(len,tokens))==[x[2]for x in selected];assert len(rows)<=(16 if last['op']=='DECODE'else 8)
   index=len(native_events);result=forward(rows,tokens,pasts,label+'.'+kind,1,prefill);local_calls.append({'seq':last['seq'],'event_index':index,'op':last['op'],'rows':rows,'pasts':pasts,'counts':list(map(len,tokens))});return result
  seconds=min(180. if warm else 1200.,max(1.,deadline-time.monotonic()-150))
  data,commands=run_episode(case_requests,case_arrivals,mode,eos,is_emitter,command_exchange,tm,bound_forward,budget_seconds=seconds,deferred_feedback=deferred_feedback)
  complete=last.get('op')=='DONE'and last.get('reason')=='complete';receipt={'label':label,'deferred_feedback':deferred_feedback,'chunked':mode,'down_row2':down_row2,'vector_counters':vectorreceipt(vector_before,vectorstats(),native_events[start_index:],down_row2),'warm':warm,'complete':complete,'command_trace':commands,'command_SHA256':hashlib.sha256(json.dumps(commands,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'native_calls':local_calls,'native_event_start':start_index,'native_event_end':len(native_events),'cap_seconds':seconds,'fixed32':True,'native_ordinary_checks_kept':True}
  m15_after=m15stats();m15_delta=[b-a for a,b in zip(m15_before,m15_after)];eligible=sum(e['input_tokens']==15 for e in native_events[start_index:])*[38,40][stage]
  assert m15_delta[1]==eligible and m15_delta[2]==(eligible if m15_oproj else 0) and m15_delta[3]==(0 if m15_oproj else eligible) and m15_delta[5]==m15_delta[16]==0 and m15_delta[15]==m15_delta[2],('M15 actualownerbranch',stage,rank,m15_delta,eligible)
  receipt['sparse_observer']=sparse_read();receipt['observer_coverage_status']='COMPLETE'if receipt['sparse_observer']['coverage']==[3,3,1,1]else'UNKNOWN';receipt['m15_oproj']=m15_oproj;receipt['m15_provider']={'before':m15_before,'after':m15_after,'delta':m15_delta,'expected_eligible_projection_calls':eligible,'actual_owner_verified':True};receipt['private_cache_added_logical_bytes']=m15_after[13]
  if is_emitter:
   assert data is not None;receipt['serving']=data
   arrived=[z for z in data['requests']if z['arrival']<=data['elapsed_seconds']]
   if arrived and data['elapsed_seconds']>0:receipt['metrics']=serving_metrics(arrived,0.,data['elapsed_seconds'],16)
   if not warm:
    scores=[]
    for req,rec in zip(case_requests,data['requests']):
     text=tokenizer.decode(rec['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False);score=score_case(req,text,rec.get('stop_reason','unfinished'));rec.update(text=text,score=score,input_tokens=len(req['input_ids']));scores.append(score)
    receipt['task_scores']=scores
  report.setdefault('warmups'if warm else'episodes',[]).append(receipt);save();progress(label,last.get('seq',0),'DONE',False,warm);assert complete,('episodeincomplete_saved_no_live_reset',label,last.get('reason'))
  if is_emitter:assert data['all_requests_completed']and data['global_reset_allowed']
  barrier();phase_marker(label+'.returned');return receipt
 warm_requests=[dict(c,input_ids=c['input_ids'][:40],input_tokens=40,max_new_tokens=2)for c in requests]
 episode('warm-serial',False,warm_requests,[0.]*32,True,True,False,False);episode('warm-deferred',False,warm_requests,[0.]*32,True,True,False,True);report['warm_CCL_progress_passed']=True;save()
 report['measurement_start_monotonic']=time.monotonic();deadline=min(deadline,time.monotonic()+2400);report['measurement_budget_seconds']=2400
 for arm in plan['episodes']:
  if not budget(150):report['budget_insufficient_before_episode']=arm['label'];save();raise RuntimeError('reserveinsufficient_saved_noextraepisode')
  episode(arm['label'],arm['chunked'],requests,arrivals,False,True,False,arm['deferred_feedback'])
 if is_emitter:
  report['paired_quality']=[paired_quality(report['episodes'][0]['task_scores'],report['episodes'][1]['task_scores']),paired_quality(report['episodes'][3]['task_scores'],report['episodes'][2]['task_scores'])]
 torch.cuda.synchronize();check(lib.modern_glm_down_row2_policy_r9284(rank,0));assert lib.modern_glm_down_row2_get_r9284(rank)==0;barrier();check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_m15_oproj_policy_r125(ctx,0));check(lib.modern_glm_sparse_policy_r131(ctx,0));report['observer_before_destroy']=sparse_read();assert not report['observer_before_destroy']['enabled'];report['observer_profile_saved_before_OFF']=False;report['observer_deliberately_disabled_all_arms']=all(not a['sparse_observer']['enabled']and not a['sparse_observer']['samples']for a in report['warmups']+report['episodes']);assert report['observer_deliberately_disabled_all_arms'];report['M15_before_destroy']=m15stats();resident.apply('R9284B-final-OFF',False,False);check(resident.vector_OFF());report.update(benchmark_complete=True,quality_benchmark_completed=True,cohort_phases_completed=True,normal_reset_close=True,actual_final_cohort_counts=stats(),actual_final_ragged_counts=raggedstats(),token_exactness_gate=False);save()
