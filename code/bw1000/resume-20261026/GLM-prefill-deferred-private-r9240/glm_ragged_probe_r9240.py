"""G101B actualdistinct C16/C32 cohorts, native emitter metrics. No HTTP/continuous-batching claim."""
import ctypes,json,pathlib,time,os,hashlib,collections
import torch
from glm_crossrequest_control_r9059 import CrossrequestControl
from quality_metrics_r9132b import paired_quality,serving_metrics,score_case
from schedule_r9132b import microbatches,prefix_chunks
from decode_wave_r9225 import decode_wave_r9225
from case_plan_r9226 import load as load_case_plan_r9226

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker,prefill_deferred_r9240=False):
 if type(prefill_deferred_r9240)is not bool:raise TypeError("prefill_deferred strictbool")
 case_plan_r9226=load_case_plan_r9226();deferred_feedback_r9225=False
 r=pathlib.Path('/refactor');program=payload['benchmark'];is_emitter=stage==1 and rank==0;deadline=time.monotonic()+max(0,report['remaining_process_seconds']);native_events=[];report.update(deferred_feedback_r9225=deferred_feedback_r9225,scope='R9225 DEFAULT_OFF two independentmicro forward-before-feedback; R9226 TP8PP2 unchangedM16 native; same32naturalEOS ABBA serial/deferredmax2 plusC16singlemicrocontrol; warmactualCCLqualification; noHTTP',MMLU=[],service=[],native_events=native_events,benchmark_program_SHA256=payload['benchmark_SHA256'],benchmark_coverage={'MMLU_scheduled':program['scheduled_count'],'MMLU_included':[c['id']for c in program['cases']],'MMLU_excluded':program['excluded'],'GSM_distinct32':[c['id']for c in program['service_cases']],'GSM_excluded':program['service_excluded']},normal_reset_close=False,benchmark_complete=False)
 def warm_progress(active,label,**fields):
  q={'run':os.environ['GLM_RUN_R5040'],'stage':stage,'rank':rank,'phase':label,'warm_active':active,'UTC_epoch':time.time(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],**fields};pp=r/'results'/('GLM-warm-r9226-'+q['run']+'-s'+str(stage)+'-r'+str(rank)+'.json');tmp=pp.with_suffix('.tmp');tmp.write_text(json.dumps(q));tmp.replace(pp)
 def save():
  tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(report_path)
 lib.glm_attention_cohort_policy_r9054.argtypes=[ctypes.c_int,ctypes.c_int];lib.glm_attention_cohort_counts_r9054.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.glm_reduced_output_policy_r9032.argtypes=[ctypes.c_int,ctypes.c_int]
 def stats():v=(ctypes.c_uint64*4)();check(lib.glm_attention_cohort_counts_r9054(rank,v,4));return list(v)
 lib.glm_ragged_policy_r9136.argtypes=[ctypes.c_int,ctypes.c_int];lib.glm_ragged_enabled_r9136.argtypes=[ctypes.c_int];lib.glm_ragged_counts_r9136.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.glm_ragged_Bcounts_r9136.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 assert lib.glm_ragged_enabled_r9136(rank)==0
 assert lib.glm_ragged_policy_r9136(rank,2)!=0
 def raggedstats():
  v=(ctypes.c_uint64*4)();check(lib.glm_ragged_counts_r9136(rank,v,4));h=(ctypes.c_uint64*132)();check(lib.glm_ragged_Bcounts_r9136(rank,h,132));assert [sum(h[B*4+j]for B in range(33))for j in range(4)]==list(v);return list(v),list(h)
 def barrier():v=torch.ones(1,device='cuda');control(v);assert v.item()==16
 def budget(reserve=90):
  v=torch.tensor([float(time.monotonic()+reserve<deadline)],device='cuda');control(v);return v.item()==16
 resident=CrossrequestControl(lib,ctx,stage,rank,control,check,report,save);check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_transient_policy_r5688(ctx,0));check(lib.modern_glm_vector_policy_r8540(rank,0))
 for name in ['decode_leases_policy_r1356','indexer_policy_r998','feature_bypass_r1406','owned_prefix_policy_r1669']:check(getattr(lib,'modern_glm_model_'+name)(ctx,1))
 resident.apply('R9226-fixed-resident-mask',True,False);check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0));report['fixed_resident_mask']=resident.chosen
 os.environ.update(GLM_SHARED_PREFIX_VALIDATE_R5814='1',GLM_CROSSREQUEST_PREFILL_R7328='0',GLM_PREFILL_PREFIX_PLAN_R8703='0',GLM_GROUPED_ROUTED_R5123='1',GLM_BATCH_ALLSELECTED_R5096='1',GLM_DECODE_ROW_DEDUP_R5386='0')
 def reset(on):check(lib.modern_glm_model_reset(ctx));check(lib.glm_attention_cohort_policy_r9054(rank,1));check(lib.glm_ragged_policy_r9136(rank,on));assert lib.glm_ragged_enabled_r9136(rank)==on;barrier()
 def forward(rows,tokens,pasts,label,on,prefill):
  assert rows and len(rows)==len(set(rows))and all(0<=i<32 for i in rows)and len(rows)==len(tokens)==len(pasts)and all(tokens);flat=[x for v in tokens for x in v];assert len(flat)<=4096 and len(rows)<=32;positions=[x for past,v in zip(pasts,tokens)for x in range(past,past+len(v))];assert max(positions)<640
  desc=(ctypes.c_size_t*(3*len(rows)))(*[x for row,past,v in zip(rows,pasts,tokens)for x in [row,past,len(v)]]);ids=torch.tensor(flat if stage==0 else [0]*len(flat),device='cuda',dtype=torch.int64);pos=torch.tensor(positions,device='cuda',dtype=torch.int64);logits=torch.full((len(rows),vocab),float('nan'),device='cuda',dtype=torch.bfloat16);before=stats();rag_before,hist_before=raggedstats();torch.cuda.synchronize();tick=time.perf_counter();check(lib.modern_glm_model_forward(ctx,len(flat),len(rows),desc,ptr(ids),ptr(pos),ptr(logits),0,stream()));ended=time.perf_counter();delta=[b-a for a,b in zip(before,stats())]
  rag_after,hist_after=raggedstats();rag_delta=[b-a for a,b in zip(rag_before,rag_after)];hist_delta=[b-a for a,b in zip(hist_before,hist_after)];expected=[0]*4;B=len(rows);single_decode=len(flat)==B and all(len(t)==1 for t in tokens);equal_p=len(set(pasts))==1
  if single_decode and (B in [8,16]or(on and 2<=B<=32)):
   route=2 if equal_p and B in [8,16]else 1 if on and not equal_p else 0;expected[route]=resident.count
  assert rag_delta==expected,('actualperlayerbranch',label,on,stage,rank,rows,pasts,rag_delta,expected,resident.chosen)
  expected_hist=[0]*132
  for j,n in enumerate(expected):expected_hist[B*4+j]=n
  assert hist_delta==expected_hist,('actualBhist',label,hist_delta,expected_hist)

  if stage==1:assert bool(torch.isfinite(logits).all()),('nonfinite',label)
  free,total=torch.cuda.mem_get_info(rank);assert free>=512*1024**2,('originalGLM512MiBfloor',label,free);report['minimum_driver_free_bytes']=min(report.get('minimum_driver_free_bytes',free),free)
  pipeline_profile=[float(lib.modern_glm_pipeline_last_profile()[j])for j in range(3)];native_events.append({'existing_pipeline_profile':pipeline_profile,'window_profile_omitted':'oldwindowhardB8/pastfixed/round63..65 doesnotcoverEOSvarlen','label':label,'rows':rows,'input_tokens':len(flat),'past_min':min(pasts),'past_max':max(pasts),'native_seconds':ended-tick,'native_return_clock_s':ended,'prefill':prefill,'cohort_counter_delta':delta,'ragged_counter_delta':rag_delta,'ragged_Bhist_delta':hist_delta,'expected_ragged_counter_delta':expected,'ON':on});return logits,ended-tick,tick,ended
 def prefix(group,rows,label,on):
  prompts=[c['input_ids'][:-1]for c in group];total=0.;tokens=0
  for rr,tt,pp in prefix_chunks(prompts,rows,8,4096):_,wall,_,_=forward(rr,tt,pp,label+'.prefix',on,True);total+=wall;tokens+=sum(map(len,tt))
  return total,tokens
 continuations=[program['answer_continuations'][x]for x in 'ABCD'];single=all(len(v)==1 for v in continuations);cases=program['cases'];stopped=False
 for on in []:
  arm=[];start=time.perf_counter();phase_marker('MMLU.'+str(on)+'.enter')
  for begin in range(0,len(cases),16):
   if not budget(150):stopped=True;break
   group=cases[begin:begin+16];rows=list(range(len(group)));scores=[[]for _ in group]
   for option in range(1 if single else 4):
    reset(on);prefix(group,rows,'MMLU'+str(on),on);out,_,_,_=forward(rows,[[c['input_ids'][-1]]for c in group],[len(c['input_ids'])-1 for c in group],'MMLU'+str(on)+'.anchor',on,False)
    if is_emitter:
     lp=torch.log_softmax(out.float(),dim=-1)
     if single:scores=lp[:,[v[0]for v in continuations]].cpu().tolist()
     else:total=lp[:,continuations[option][0]].cpu().tolist()
    if not single:
     for j in range(1,len(continuations[option])):
      out,_,_,_=forward(rows,[[continuations[option][j-1]]for _ in group],[len(c['input_ids'])+j-1 for c in group],'MMLU'+str(on)+'.option',on,False)
      if is_emitter:extra=torch.log_softmax(out.float(),dim=-1)[:,continuations[option][j]].cpu().tolist();total=[a+b for a,b in zip(total,extra)]
     if is_emitter:
      for i,v in enumerate(total):scores[i].append(v)
   if is_emitter:
    for i,c in enumerate(group):pred=max(range(4),key=lambda j:scores[i][j]);arm.append({'id':c['id'],'subject':c['subject'],'gold':c['gold'],'prediction':pred,'choice_loglikelihoods':scores[i],'flexible_correct':pred==c['gold'],'hit_output_cap':False})
   report['MMLU_progress']={'ON':on,'completed':begin+len(group),'included':len(cases)};save()
  report['MMLU'].append({'ON':on,'cases':arm,'elapsed_seconds':time.perf_counter()-start,'classification_not_generation':True,'M':16,'forced_empty_thinking':True});save();phase_marker('MMLU.'+str(on)+'.return')
 if is_emitter and len(report['MMLU'])==2 and all(len(a['cases'])==len(cases)for a in report['MMLU']):report['paired_MMLU']=paired_quality(report['MMLU'][0]['cases'],report['MMLU'][1]['cases']);save()
 eos=program['generation_config']['eos_token_id'];eos=set(eos if isinstance(eos,list)else[eos]);allservice=program['service_cases'];tokenizer=None
 if is_emitter:
  from transformers import PreTrainedTokenizerFast
  tokenizer=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True)
 report['warmups']=[]
 serial_wave_seconds=None
 for warm in case_plan_r9226['warmups']:
  warm_label=warm['label'];warm_mode=warm['deferred_feedback'];reset(1);warm_pasts=[0]*32;warm_tokens=[allservice[i]['input_ids'][0]for i in range(32)]
  warm_progress(True,warm_label+'.setup',stall_limit_seconds=60)
  for begin in range(0,32,8):
   rr=list(range(begin,begin+8));out,_,_,_=forward(rr,[[warm_tokens[i]]for i in rr],[0]*len(rr),warm_label+'.prefill',1,True);del out
  warm_pasts=[1]*32;waves=[]
  stall_limit=60 if serial_wave_seconds is None else min(60.,max(2.,10.*serial_wave_seconds))
  for warm_round in range(2):
   warm_progress(True,warm_label+'.wave'+str(warm_round),stall_limit_seconds=stall_limit,deferred_feedback=warm_mode)
   wt=time.perf_counter()
   for rows,out,wall,tick,ended in decode_wave_r9225(microbatches(list(range(32)),16),lambda rr:forward(rr,[[warm_tokens[i]]for i in rr],[warm_pasts[i]for i in rr],warm_label+'.decode',1,False),warm_mode):
    vote=torch.zeros(len(rows),device='cuda',dtype=torch.float32)
    if is_emitter:vote.copy_(out.argmax(-1).float())
    control(vote);values=[int(v)for v in vote.cpu().tolist()]
    for i,token in zip(rows,values):warm_tokens[i]=token;warm_pasts[i]+=1
    del out
   waves.append(time.perf_counter()-wt)
   warm_progress(False,warm_label+'.wave'+str(warm_round)+'.returned',wave_seconds=waves[-1],deferred_feedback=warm_mode)
  if not warm_mode:serial_wave_seconds=max(waves)
  report['warmups'].append({'label':warm_label,'deferred_feedback':warm_mode,'wave_seconds':waves,'synthetic_schedule_not_serving_quality':True,'C':32,'M':16,'rows_independent':True,'stall_limit_seconds':stall_limit});save()
 report['warm_CCL_progress_passed']=True;warm_progress(False,'warm_all_returned',serial_wave_seconds=serial_wave_seconds)
 planned=case_plan_r9226['cohorts']
 for cohort in planned:
  C=cohort['C'];M=cohort['M'];deferred_feedback_r9225=cohort['deferred_feedback']
  for on in [1]:
   if not budget(150):stopped=True;continue
   group=allservice[:C];label=cohort['label'];report['phase']='service:'+label;phase_marker(label+'.enter');began=time.perf_counter();reset(on);generated=[[]for _ in group];done=[False]*C;pasts=[0]*C;events=[[]for _ in group];requests=[{'id':c['id'],'arrival':0.,'service_start':0.,'status':'in_progress','events':events[i]}for i,c in enumerate(group)];npf=ndec=0.;prefill_tokens=0;useful_decode=0;first_decode=None;last_decode=None;counterbefore=stats();raggedbefore,Bhistbefore=raggedstats();rows_hist=collections.Counter();case_budget_stop=False
   def emit(rows,logits,prefill):
    vote=torch.zeros(len(rows),device='cuda',dtype=torch.float32)
    if is_emitter:vote.copy_(logits.argmax(-1).float())
    control(vote);values=[int(v)for v in vote.cpu().tolist()];stamp=time.perf_counter()-began
    for i,token in zip(rows,values):
     assert 0<=token<vocab;generated[i].append(token)
     if token not in eos:events[i].append({'time':stamp,'visible':True,'kind':'committed','EOS':False,'pad':False,'draft':False,'replay':False,'token_id':token,'phase':'prefill'if prefill else'decode','token_ready_wall_utc_ns':time.time_ns()})
     if token in eos or len(generated[i])>=group[i]['max_new_tokens']:done[i]=True;requests[i].update(status='completed',stop_reason='EOS'if token in eos else'max_new_tokens',end_time=stamp)
    return sum(token not in eos for token in values)
   # All requests arrive before reset. Prefill stays M8 for both M8/M16 decode arms.
   def prefill_group_r9240(rows):
    nonlocal npf,prefill_tokens
    micro=[group[i]for i in rows];wall,n=prefix(micro,rows,label,on);npf+=wall;prefill_tokens+=n
    out,wall,tick,ended=forward(rows,[[c['input_ids'][-1]]for c in micro],[len(c['input_ids'])-1 for c in micro],label+'.PFanchor',on,True);npf+=wall;prefill_tokens+=len(rows)
    return out,wall,tick,ended
   for rows,out,wall,tick,ended in decode_wave_r9225(microbatches(list(range(C)),8),prefill_group_r9240,prefill_deferred_r9240):
    emit(rows,out,True)
    for i in rows:pasts[i]=len(group[i]['input_ids'])
    del out
   step=1
   while not all(done):
    if not budget(150):case_budget_stop=stopped=True;break
    active=[i for i in range(C)if not done[i]]
    for rows,out,wall,tick,ended in decode_wave_r9225(microbatches(active,M),lambda rows:forward(rows,[[generated[i][-1]]for i in rows],[pasts[i]for i in rows],label+'.decode',on,False),deferred_feedback_r9225):
     ndec+=wall;first_decode=tick if first_decode is None else first_decode;rows_hist[len(rows)]+=1;useful_decode+=emit(rows,out,False);last_decode=time.perf_counter()
     for i in rows:pasts[i]+=1
     del out
    report['service_progress']={'label':label,'step':step,'finished':sum(done),'C':C,'M':M,'native_decode_seconds':ndec,'useful_decode_tokens':useful_decode}
    if step%16==0:save()
    step+=1
   elapsed=time.perf_counter()-began;delta=[b-a for a,b in zip(counterbefore,stats())];raga,Bha=raggedstats();summary={'label':label,'C':C,'M':M,'ON':on,'deferred_feedback':deferred_feedback_r9225,'prefill_M':8,'native_prefill_seconds':npf,'prefill_completed_input_tokens':prefill_tokens,'native_prefill_input_tok_s':prefill_tokens/npf,'native_decode_seconds':ndec,'native_decode_useful_tokens':useful_decode,'native_decode_useful_tok_s':useful_decode/ndec if ndec else None,'native_decode_useful_tok_s_per_card':useful_decode/ndec/16 if ndec else None,'decode_common_window_seconds':last_decode-first_decode if first_decode is not None else None,'decode_common_window_useful_tok_s':useful_decode/(last_decode-first_decode)if first_decode is not None else None,'elapsed_seconds':elapsed,'cohort_counter_delta':delta,'ragged_counter_delta':[b-a for a,b in zip(raggedbefore,raga)],'ragged_Bhist_delta':[b-a for a,b in zip(Bhistbefore,Bha)],'actual_micro_rows_histogram':dict(rows_hist),'budget_stop':case_budget_stop,'expected_distinct_requests':C,'arrival_protocol':'closedcohort; stablelogicalrows; retireEOS; selected max2 independentmicro forwards before orderedglobalfeedback; noHTTP','minimum_driver_free_bytes':report['minimum_driver_free_bytes']}
   if is_emitter:
    scores=[]
    for i,req in enumerate(requests):
     text=tokenizer.decode(generated[i],skip_special_tokens=True,clean_up_tokenization_spaces=False);score=score_case(group[i],text,req.get('stop_reason','unfinished'));scores.append(score);req.update(generated_ids=generated[i],text=text,input_tokens=len(group[i]['input_ids']),score=score)
    summary.update(requests=requests,task_scores=scores,metrics=serving_metrics(requests,0.,elapsed,16))
   report['service'].append(summary);save();phase_marker(label+'.return')
 if is_emitter:
  report['paired_GSM']={}
  for C in [32,16]:
   aa=[v for v in report['service']if v['C']==C and not v['deferred_feedback']];bb=[v for v in report['service']if v['C']==C and v['deferred_feedback']]
   if aa and bb:report['paired_GSM']['C'+str(C)]=paired_quality(aa[0]['task_scores'],bb[0]['task_scores'])
 check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));resident.apply('R9226-final-OFF',False,False);check(resident.vector_OFF());report.update(benchmark_complete=not stopped and len(report['service'])==len(planned),quality_benchmark_completed=not stopped,cohort_phases_completed=True,normal_reset_close=True,actual_final_cohort_counts=stats(),actual_final_ragged_counts=raggedstats(),token_exactness_gate=False,budget_partial=stopped);save()
