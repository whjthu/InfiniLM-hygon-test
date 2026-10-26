"""G101B actualdistinct C16/C32 cohorts, native emitter metrics. No HTTP/continuous-batching claim."""
import ctypes,json,pathlib,time,os,hashlib,collections
import torch
from glm_crossrequest_control_r9059 import CrossrequestControl
from quality_metrics_r9132b import paired_quality,serving_metrics,score_case
from schedule_r9132b import microbatches,prefix_chunks

def run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,phase_marker):
 r=pathlib.Path('/refactor');program=payload['benchmark'];is_emitter=stage==1 and rank==0;deadline=time.monotonic()+max(0,report['remaining_process_seconds']);native_events=[];report.update(scope='G101B TP8PP2 16cards fresh32naturalGSM C32M16vsC32M32 newraggedON both; oldMMLUnotrerun; serialmicro CAPI+allrankfeedback, noHTTP/continuousarrival or maxpipelineoverlapclaim',MMLU=[],service=[],native_events=native_events,benchmark_program_SHA256=payload['benchmark_SHA256'],benchmark_coverage={'MMLU_scheduled':program['scheduled_count'],'MMLU_included':[c['id']for c in program['cases']],'MMLU_excluded':program['excluded'],'GSM_distinct32':[c['id']for c in program['service_cases']],'GSM_excluded':program['service_excluded']},normal_reset_close=False,benchmark_complete=False)
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
 resident.apply('G101B-fixed-resident-mask',True,False);check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1));check(lib.glm_reduced_output_policy_r9032(rank,0));report['fixed_resident_mask']=resident.chosen
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
   if not budget(180):stopped=True;break
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
 planned=[(32,16),(32,32)]
 for C,M in planned:
  for on in [1]:
   if not budget(180):stopped=True;continue
   group=allservice[:C];label='C%dM%d-ON%d'%(C,M,on);report['phase']='service:'+label;phase_marker(label+'.enter');began=time.perf_counter();reset(on);generated=[[]for _ in group];done=[False]*C;pasts=[0]*C;events=[[]for _ in group];requests=[{'id':c['id'],'arrival':0.,'service_start':0.,'status':'in_progress','events':events[i]}for i,c in enumerate(group)];npf=ndec=0.;prefill_tokens=0;useful_decode=0;first_decode=None;last_decode=None;counterbefore=stats();raggedbefore,Bhistbefore=raggedstats();rows_hist=collections.Counter();case_budget_stop=False
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
   for begin in range(0,C,8):
    micro=group[begin:begin+8];rows=list(range(begin,begin+len(micro)));wall,n=prefix(micro,rows,label,on);npf+=wall;prefill_tokens+=n;out,wall,_,_=forward(rows,[[c['input_ids'][-1]]for c in micro],[len(c['input_ids'])-1 for c in micro],label+'.PFanchor',on,True);npf+=wall;prefill_tokens+=len(rows);emit(rows,out,True)
    for i in rows:pasts[i]=len(group[i]['input_ids'])
   step=1
   while not all(done):
    if not budget(75):case_budget_stop=stopped=True;break
    active=[i for i in range(C)if not done[i]]
    for rows in microbatches(active,M):
     out,wall,tick,ended=forward(rows,[[generated[i][-1]]for i in rows],[pasts[i]for i in rows],label+'.decode',on,False);ndec+=wall;first_decode=tick if first_decode is None else first_decode;rows_hist[len(rows)]+=1;useful_decode+=emit(rows,out,False);last_decode=time.perf_counter()
     for i in rows:pasts[i]+=1
    report['service_progress']={'label':label,'step':step,'finished':sum(done),'C':C,'M':M,'native_decode_seconds':ndec,'useful_decode_tokens':useful_decode}
    if step%16==0:save()
    step+=1
   elapsed=time.perf_counter()-began;delta=[b-a for a,b in zip(counterbefore,stats())];raga,Bha=raggedstats();summary={'label':label,'C':C,'M':M,'ON':on,'prefill_M':8,'native_prefill_seconds':npf,'prefill_completed_input_tokens':prefill_tokens,'native_prefill_input_tok_s':prefill_tokens/npf,'native_decode_seconds':ndec,'native_decode_useful_tokens':useful_decode,'native_decode_useful_tok_s':useful_decode/ndec if ndec else None,'native_decode_useful_tok_s_per_card':useful_decode/ndec/16 if ndec else None,'decode_common_window_seconds':last_decode-first_decode if first_decode is not None else None,'decode_common_window_useful_tok_s':useful_decode/(last_decode-first_decode)if first_decode is not None else None,'elapsed_seconds':elapsed,'cohort_counter_delta':delta,'ragged_counter_delta':[b-a for a,b in zip(raggedbefore,raga)],'ragged_Bhist_delta':[b-a for a,b in zip(Bhistbefore,Bha)],'actual_micro_rows_histogram':dict(rows_hist),'budget_stop':case_budget_stop,'expected_distinct_requests':C,'arrival_protocol':'allC before reset, stablelogicalrows; retireEOS; serialmicro CAPI+globalfeedback; noHTTP','minimum_driver_free_bytes':report['minimum_driver_free_bytes']}
   if is_emitter:
    scores=[]
    for i,req in enumerate(requests):
     text=tokenizer.decode(generated[i],skip_special_tokens=True,clean_up_tokenization_spaces=False);score=score_case(group[i],text,req.get('stop_reason','unfinished'));scores.append(score);req.update(generated_ids=generated[i],text=text,input_tokens=len(group[i]['input_ids']),score=score)
    summary.update(requests=requests,task_scores=scores,metrics=serving_metrics(requests,0.,elapsed,16))
   report['service'].append(summary);save();phase_marker(label+'.return')
 if is_emitter:
  report['paired_GSM_M16_M32']=paired_quality(report['service'][0]['task_scores'],report['service'][1]['task_scores'])if len(report['service'])==2 else None;report['paired_GSM']={}; report['M16_M32_comparison']='samefresh32, bothON, qualitypilot notformalnodrop'
  for C,M in planned:
   arms={a['ON']:a for a in report['service']if a['C']==C and a['M']==M}
   if len(arms)==2 and all(all(q['status']=='completed'for q in a['requests'])for a in arms.values()):report['paired_GSM']['C%dM%d'%(C,M)]=paired_quality(arms[0]['task_scores'],arms[1]['task_scores'])
 check(lib.glm_ragged_policy_r9136(rank,0));check(lib.glm_attention_cohort_policy_r9054(rank,0));check(lib.modern_glm_model_reset(ctx));resident.apply('G101B-final-OFF',False,False);check(resident.vector_OFF());report.update(benchmark_complete=not stopped and len(report['service'])==2,quality_benchmark_completed=not stopped,cohort_phases_completed=True,normal_reset_close=True,actual_final_cohort_counts=stats(),actual_final_ragged_counts=raggedstats(),token_exactness_gate=False,budget_partial=stopped);save()
