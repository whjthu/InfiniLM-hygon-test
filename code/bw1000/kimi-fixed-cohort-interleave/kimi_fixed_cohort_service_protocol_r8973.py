"""Matched ordinary serving tests; all performance compares complete token IDs."""
import ctypes,hashlib,json,os,time,pathlib
import numpy as np
from kimi_cohort_feedback_idle_r4915 import FeedbackHub,FeedbackClient,decode_cohorts

def run(g):
 lib,ctx,control,r,report,save,check=[g[k]for k in['lib','ctx','control','r','report','save','check']];node,tail,emitter,tag,vocab=[g[k]for k in['node','tail','emitter','tag','vocab']];assert(g['tp_nodes'],g['pp'])==(1,4);stage=g['stage'];first=int(g['text']['first_k_dense_replace']);freq=int(g['text']['moe_layer_freq']);moe=sum(i>=first and i%freq==0 for i in range(*g['ranges'][stage]));assert first==1 and freq==1
 counter=lib.modern_kimi_serving_counts_r4665;counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];check(lib.modern_kimi_reduce_policy_r3226(0));lib.modern_kimi_engine_resize_cache_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';frozen=json.loads(frozen_path.read_text());prompts=frozen['natural_input_ids'];P=len(prompts);assert P==2 and len({tuple(v)for v in prompts})==2 and all(len(v)==512 for v in prompts);eos=set(frozen['eos_token_ids']);assert eos==set(chat['eos_token_ids']);seq=0;policy=0;capacity=96;pipeline=False;prefill_pipeline=False
 report['pipeline_scheduler']='Existing540 stage-local prefillM2 and continuous decodeM16, servingpolicy1 grouped4; CPUfeedback/calls included, no concurrentEnginecall, no crosshost clock alignment.';report['local_forward_samples']=[]
 report.update(serving_cases=[],QA=[],counter_dispatch=[],unique_natural_prompts=P,serving_cards=32,cache_capacity_rows=96,decode_output_limit=128,performance_qualified=False)
 report['memory_reserve_r4912']='4GiB collective pregeometry, capacity64 unchanged;4890 clean rejected4.64GiBvs5GiB; runtime512MiB failstop retained, noresize128 planned';report['observer_policy']='All phases share hipprof interposer. Ordinary active trace OFF; instrumented baseline only, not unwrapped636/540 improvement. Separate short decode window uses existing hipProfilerStart/Stop; trace wall excluded from ordinary metrics.'
 report['scope']='Current183f6f/cap64/pool65/workspace64/prefillM2/SpecMath0/CCL0/proj0, servingpolicy1 N16 grouped4; TP8PP4 B64M16. Historical540 full128 IDs/EOS + QA first, ordinary twice active trace OFF, one short hiptrace phase. Two unique natural512 prompts repeated, not production diversity.'
 report['profiling_windows']=[]
 continuous=False
 frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json'
 assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625'
 frozen=json.loads(frozen_path.read_text());assert prompts==frozen['natural_input_ids'] and eos==set(frozen['eos_token_ids'])
 assert 0<g['port']+9000<65536
 if node==0:g['feedback_hub']=FeedbackHub(g['head'],g['port']+9000,g['os'].environ['KIMI_ENGINE_NONCE'],4,emitter,timeout=120,idle_timeout=7200)
 control.collect('continuous-feedback-bound',{})
 feedback=FeedbackClient(g['head'],g['port']+9000,g['os'].environ['KIMI_ENGINE_NONCE'],node,timeout=125);g['feedback']=feedback
 control.collect('continuous-feedback-connected',{})
 report['continuous_feedback_contract']={'source':'kimi_cohort_feedback_idle_r4915.py','server_idle_seconds':7200,'dependency_seconds':120,'client_reply_seconds':125,'failure_latch':True,'default_OFF':True,'only_main_thread_calls_engine':True,'max_pending_packets':'2*cohort_count','fixed_cohort_after_EOS':True,'mixed_EOS_Model_qualified':False,'port':g['port']+9000,'host':g['head'],'IB_link_policy':'Same observed IB address as native Engine head; CPU TCP feedback, GPU PP retains existing IBext_v8.'}
 def counts():
  values=[]
  for device in range(8):
   z=(ctypes.c_uint64*4)();check(counter(device,z,4));values.append(list(z))
  return values
 assert counts()==[[0]*4 for _ in range(8)]
 def select(value,label):
  nonlocal policy
  assert value in[0,1];control.collect('serving-policy-before:'+label,{'policy':value});os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=str(value);policy=value;control.collect('serving-policy-after:'+label,{'policy':value})
 def forward(ins,pasts,rows,label,token_only=False,local_only=False):
  nonlocal seq
  B=len(rows);widths=[len(v)for v in ins];N=sum(widths);assert 0<B<=96 and N<=8192;row=np.asarray(rows,np.int32);past=np.asarray(pasts,np.int32);total=past+np.asarray(widths,np.int32);offset=np.asarray([0]+np.cumsum(widths).tolist(),np.int32);ids=np.ascontiguousarray(np.concatenate(ins),dtype=np.int64);positions=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));logits=None if token_only else np.full((B,vocab),0x7fc0,np.uint16);out=np.full(B,-999,np.int64);flags=np.asarray([-1],np.int32);before=counts();start=time.perf_counter();capi_start_realtime_ns=time.time_ns();arrays=[row,past,total,offset,ids,positions];ptrs=[ctypes.c_void_p(v.ctypes.data)for v in arrays]+[None if logits is None else ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(flags.ctypes.data)];check(lib.modern_kimi_engine_forward_r2475(ctx,N,B,71 if token_only else 67,*ptrs));seconds=time.perf_counter()-start;mem=g['memory']();report['last_forward_memory']=mem;assert all(free>512*1024**2 for free,total in mem),('512MiB_runtime_floor',label,mem);after=counts();expected=[0]*4
  if policy and 8<=N<=64:
   variant=(9 if N>=64 else 4)if policy==1 else policy;off=0 if variant==4 else 2;expected[off]=moe;expected[off+1]=moe*N
  delta=[[b-a for a,b in zip(x,y)]for x,y in zip(before,after)];assert delta==[expected]*8,('actualservingMoEdispatch',label,N,policy,delta,expected);report['counter_dispatch'].append({'label':label,'N':N,'B':B,'policy':policy,'expected':expected,'delta':delta});assert int(flags[0])==((int(tail)if not token_only else 0)+(2 if node==emitter else 0));selected=None;digest=None
  if tail and not token_only:
   f=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();argmax=np.argmax(f,axis=1).tolist();digest=hashlib.sha256(logits.tobytes()).hexdigest()
   if node==emitter:assert out.tolist()==argmax
  if node==emitter:selected=out.tolist();assert all(0<=v<vocab for v in selected)
  report['local_forward_samples'].append({'label':label,'N':N,'B':B,'CAPI_seconds':seconds,'CAPI_start_realtime_ns':capi_start_realtime_ns,'CAPI_end_realtime_ns':capi_start_realtime_ns+int(seconds*1e9),'pipeline':pipeline})
  if local_only:return selected,seconds,digest
  packet=control.collect('serving-call-'+str(seq),{'ids':selected,'seconds':seconds,'full_logits_sha256':digest});seq+=1;selected=packet[emitter]['ids'];assert selected is not None and len(selected)==B;return selected,max(v['seconds']for v in packet),packet[emitter]['full_logits_sha256']
 def reset(label):check(lib.modern_kimi_engine_reset_r2475(ctx));control.collect('serving-reset:'+label,{})
 def generate(inputs,M,label,limit=128,token_only=True,reference=None):
  B=len(inputs);assert B<=capacity and M<=64;report['phase']=label;save();reset(label);ids=[[]for _ in range(B)];pasts=[0]*B;times=[[]for _ in range(B)];ttft=[0.]*B;start=time.perf_counter()
  prefill_parts=[]
  for a in range(0,B,2):
   ii=list(range(a,min(a+2,B)));z,_,_=forward([inputs[i]for i in ii],[0]*len(ii),[i+1 for i in ii],label+'-prefill'+str(a),False,local_only=prefill_pipeline);stamp=time.perf_counter()-start
   if prefill_pipeline:prefill_parts.append({'rows':ii,'ids':z,'stamp':stamp})
   else:
    for i,v in zip(ii,z):ids[i]=[v];pasts[i]=len(inputs[i]);ttft[i]=stamp;times[i]=[0.]
  prefill_local_end=time.perf_counter();prefill_drain_start=prefill_local_end
  if prefill_pipeline:
   packets=control.collect('prefill-drained:'+label,prefill_parts)
   for part in packets[emitter]:
    assert part['ids'] is not None and len(part['ids'])==len(part['rows'])
    for i,v in zip(part['rows'],part['ids']):ids[i]=[v];pasts[i]=len(inputs[i]);ttft[i]=part['stamp'];times[i]=[0.]
  prefill_drain_end=time.perf_counter()
  prefill_end=time.perf_counter()
  if continuous:
   assert pipeline
   if node==emitter:feedback.request('begin',label,cohorts=[list(range(a,min(a+M,B)))for a in range(0,B,M)])
   control.collect('continuous-feedback-case-ready:'+label,{})
  control.collect('serving-anchor:'+label,{});decode_start=time.perf_counter();samples=[]
  if continuous:
   def owner_forward(rows,step,cohort):
    selected,seconds,_=forward([[ids[i][-1]]for i in rows],[pasts[i]for i in rows],[i+1 for i in rows],label+'-decode'+str(step)+'-cohort'+str(cohort),token_only,local_only=True)
    return selected,seconds
   stream_result=decode_cohorts(feedback=feedback,case=label,is_emitter=node==emitter,ids=ids,pasts=pasts,times=times,micro=M,limit=limit,eos=eos,forward=owner_forward,anchor=decode_start)
   samples=stream_result['samples']
   control.collect('continuous-feedback-case-drained:'+label,{'forwards_per_cohort':stream_result['forwards_per_cohort']})
   stream_stats=feedback.request('end',label) if node==emitter else None
   stream_packets=control.collect('continuous-feedback-case-closed:'+label,stream_stats)
   report.setdefault('continuous_feedback_cases',[]).append({'label':label,'stats':stream_packets[emitter],'local_wall_seconds':stream_result['local_wall_seconds'],'forwards_per_cohort':stream_result['forwards_per_cohort']})
  else:
   for step in range(limit-1):
    active=[i for i in range(B)if len(ids[i])<limit and ids[i][-1]not in eos]
    if not active:break
    if pipeline:
     local_parts=[]
     for a in range(0,len(active),M):
      ii=active[a:a+M];z,seconds,_=forward([[ids[i][-1]]for i in ii],[pasts[i]for i in ii],[i+1 for i in ii],label+'-decode'+str(step)+'-part'+str(a),token_only,local_only=True);stamp=time.perf_counter()-decode_start;samples.append(seconds);local_parts.append({'rows':ii,'ids':z,'stamp':stamp,'seconds':seconds})
     packets=control.collect('pipeline-drained:'+label+':'+str(step),local_parts)
     for part in packets[emitter]:
      assert part['ids']is not None and len(part['ids'])==len(part['rows'])
      for i,v in zip(part['rows'],part['ids']):ids[i].append(v);pasts[i]+=1;times[i].append(part['stamp'])
    else:
     for a in range(0,len(active),M):
      ii=active[a:a+M];z,seconds,_=forward([[ids[i][-1]]for i in ii],[pasts[i]for i in ii],[i+1 for i in ii],label+'-decode'+str(step)+'-part'+str(a),token_only);stamp=time.perf_counter()-decode_start;samples.append(seconds)
      for i,v in zip(ii,z):ids[i].append(v);pasts[i]+=1;times[i].append(stamp)
    if step%32==0:report['active_serving']={'label':label,'step':step,'active':len(active),'decode_seconds':time.perf_counter()-decode_start}
  packet=control.collect('serving-finished:'+label,{'ids':ids,'times':times,'ttft':ttft,'prefill_wall_seconds':prefill_end-start,'prefill_CAPI_loop_seconds':prefill_local_end-start,'prefill_final_drain_seconds':prefill_drain_end-prefill_drain_start,'anchor_barrier_seconds':decode_start-prefill_end,'decode_anchor_from_start_seconds':decode_start-start});assert all(z['ids']==ids for z in packet);canonical=packet[emitter];times=canonical['times'];wall=max(z[-1]for z in times);useful=[sum(v not in eos for v in row[1:])for row in ids];same=None if reference is None else ids==reference;assert all((len(z)==limit or z[-1]in eos)and not any(v in eos for v in z[:-1])for z in ids);row={'continuous_cohort_feedback_requested':continuous,'label':label,'B':B,'frontend_cohort_M':M,'policy':policy,'input_lengths':list(map(len,inputs)),'max_output_tokens':limit,'generated_ids':ids,'timestamps_after_common_anchor':times,'TTFT_seconds':canonical['ttft'],'prefill_stage_local_requested':prefill_pipeline,'prefill_CAPI_loop_seconds':canonical['prefill_CAPI_loop_seconds'],'prefill_final_drain_seconds':canonical['prefill_final_drain_seconds'],'prefill_wall_seconds':canonical['prefill_wall_seconds'],'anchor_barrier_seconds':canonical['anchor_barrier_seconds'],'service_wall_seconds':canonical['decode_anchor_from_start_seconds']+wall,'service_boundary_policy':'Emitter perf_counter start beforeprefill through lastdecode ID; includes anchorbarrier; excludes requestarrival/loading; firstprefill IDs and EOS counted separately.', 'useful_all_outputs':sum(sum(v not in eos for v in row)for row in ids),'end_to_end_output_tok_s':sum(sum(v not in eos for v in row)for row in ids)/(canonical['decode_anchor_from_start_seconds']+wall),'useful_decode_outputs':sum(useful),'decode_wall_seconds':wall,'aggregate_decode_tok_s':sum(useful)/wall if wall else None,'decode_tok_s_per_card':sum(useful)/wall/32 if wall else None,'per_request_decode_tok_s':[u/t[-1]if u and t[-1]>0 else None for u,t in zip(useful,times)],'matched_baseline_complete_IDs':same,'timed_decode_token_only':token_only,'sampled_CPU_argmax':'Allprefill; alldecodeonlyinfull-logitreference, token-onlycasescheckedagainstcompletereferenceIDs.','all_service_cards':32,'no_EOS_padding':True,'stage_local_PP_overlap_requested':pipeline,'overlap_measured_by_timeline':False,'native_call_seconds_sum':sum(samples),'scope':'Closedbatchordinarydecode; existingstage-localPP/deferredfeedback acrosscohorts. Decodeexcludesallprefill/anchor/EOS, includesCAPI/IDcopy/CPUcontrol/unhiddencomm. Naturalprompts repeatedfromP uniqueinputs; noproductiontrafficrepresentativenessclaim.'};row['instrumented_baseline']=False
  timeline=[[canonical['ttft'][i]]+[canonical['decode_anchor_from_start_seconds']+v for v in times[i][1:]]for i in range(B)]
  valid=[[stamp for value,stamp in zip(values,stamps)if value not in eos]for values,stamps in zip(ids,timeline)]
  row['per_request_first_last_tok_s']=[(len(v)-1)/(v[-1]-v[0])if len(v)>1 and v[-1]>v[0]else None for v in valid]
  row['per_request_max_ITL_seconds']=[max((b-a for a,b in zip(v,v[1:])),default=None)for v in valid]
  row['token_ready_times_from_phase_start']=timeline
  row['per_request_decode_tok_s_legacy_anchor_metric']=row.pop('per_request_decode_tok_s')
  row['user_clock']='Emitter same perf_counter: first actual prefill decision through last useful token; no cross-host subtraction, not HTTP/client receipt.'
  report['serving_cases'].append(row);save()
  if reference is not None:assert same,('changed_complete_IDs',label)
  return ids
 def qa(value,epoch):
  select(value,'QA-'+epoch+'-'+str(value))
  for task,c in chat['cases'].items():
   z=generate([c['input_ids']],1,f'QA-{epoch}-p{value}-{task}',128,False);key='sort'if task.startswith('sort_probe')else task;gold=[x['generated_ids'][0]for x in known['cases']if x['task']==key and x['grouped']];assert z[0]in gold,(task,z[0]);report['QA'].append({'epoch':epoch,'policy':value,'task':task,'complete_frozen_IDs_exact':True});save()
 report['scope']='Current639f6f TP8PP4 cap96/3pages291/KDA97/workspace64, sameI512O128/twofamilies/prefillM2/decodeM16/serving1 grouped4. OneordinaryB64 thenoneordinaryB96, same32cards, noM32/profile/spec/newnative.'
 report['observer_policy']='No hipprof/interposer/events/draft. Actualemitterclock includingallprefill and deferredfeedback. No timingcomparison against historicalinstrumented639 as optimizationgain.'
 report['memory_reserve_r4912']='Original3GiBafterinitialcap1 and4GiBpostgeometry/preload pluspostQA allowner gate; runtime512MiB; cold666PASS notweight/activationfit.'
 qa(0,'before')
 reset('after-QA-before-cap96-memory')
 packets=control.collect('geometry-memory:96:16',{'memory':g['memory']()})
 minimum=min(free for packet in packets for free,total in packet['memory'])
 report['geometry_memory_gate']={'minimum_free_bytes':minimum,'required_free_bytes':4*1024**3};save()
 assert minimum>4*1024**3,'original4GiB_pregeometry_reserve'
 geometry=next(x for x in frozen['geometries']if (x['B'],x['M'])==(64,16));fixed=geometry['fixed_complete_IDs']
 assert len(fixed)==64 and all(len(v)==128 for v in fixed)
 # B96 gold is an explicit historical same-input/cohort-position mapping;
 # it is never candidate-generated. Any mismatch is a quality failure.
 assert all(fixed[i]==fixed[i%16] for i in range(64)), 'historicalcohortmapping_notuniform_STOP'
 pipeline=True;prefill_pipeline=True;continuous=True;select(1,'current-serving-policy1')
 B=64;inputs=[prompts[i%2]for i in range(B)];reference=[fixed[i%16]for i in range(B)]
 report['scope']='Samecurrent669 f6f/TP8PP4/cap96threepages291/KDA97, B64 fixedM16 cohorts; A allprefill thencontinuousdecode vs C completed16rowcohort admission/M2prefill interleave, no smallM decode or newprovider. Full128 independentgold/EOS/QA; two naturalprompts repeated.'
 baseline=generate(inputs,16,'B64-M16-A-ordinary128',128,True,reference)
 report['fixed_cohort_interleave']={'status':'PENDING_ROOT_RUN_BUDGET'};save()
 def interleave():
  from kimi_fixed_cohort_interleave_r8973 import run as run_fixed,jobs
  label='B64-M16-C-fixed-cohort-interleave128';report['phase']=label;save();reset(label)
  plan_sha=hashlib.sha256(json.dumps(jobs(),separators=(',',':')).encode()).hexdigest()
  plans=control.collect('fixed-cohort-plan:'+label,{'SHA256':plan_sha})
  assert all(x['SHA256']==plan_sha for x in plans)
  if node==emitter:feedback.request('begin',label,cohorts=[list(range(a,a+16))for a in range(0,64,16)])
  control.collect('fixed-cohort-begin:'+label,{})
  def call(kind,rows,ins,pasts,index):
   selected,seconds,_=forward(ins,pasts,[i+1 for i in rows],label+'-'+kind+str(index),kind=='decode',local_only=True)
   return selected,seconds
  actual=run_fixed(inputs=inputs,feedback=feedback,case=label,is_emitter=node==emitter,forward=call,eos=eos)
  packets=control.collect('fixed-cohort-drained:'+label,actual)
  if node==emitter:stats=feedback.request('end',label)
  else:stats=None
  result_packets=control.collect('fixed-cohort-end:'+label,stats)
  canonical=packets[emitter];canonical['feedback']=result_packets[emitter]
  assert all(x['generated_ids']==canonical['generated_ids']for x in packets)
  actual_ids=canonical['generated_ids'];times=canonical['token_ready_times_from_local_emitter_start'];last=max(row[-1]for row in times)
  canonical.update(service_wall_seconds=last,end_to_end_output_tok_s=8192/last,end_to_end_output_tok_s_per_card=8192/last/32,TTFT_seconds=[row[0]for row in times],per_request_first_last_tok_s=[127/(row[-1]-row[0])for row in times],per_request_max_ITL_seconds=[max(b-a for a,b in zip(row,row[1:]))for row in times],all_service_cards=32,scope='All input512/output128, actual emitter return clocks including interleaved prefill/decode. No HTTP arrival/client receipts; mixed service rate not puredecode. No EOS expected on this qualified frozenfixture; first unexpected EOS stops unsupported scope.')
  first=[]
  for i,(expected,got)in enumerate(zip(reference,actual_ids)):
   pos=next((j for j,(x,y)in enumerate(zip(expected,got))if x!=y),None)
   if pos is not None or len(expected)!=len(got):first.append({'request':i,'first_difference':None if pos is None else pos+1,'expected':None if pos is None else expected[pos],'actual':None if pos is None else got[pos]})
  canonical.update(complete_frozen_IDs_EOS_match=not first,first_differences=first,status='PASS'if not first else'FAILED_OBSERVED')
  report['fixed_cohort_interleave']=canonical;save() # Persist outputs BEFORE classify; restore original QA even if full128 differs.
  return not first
 deadline=float(os.environ['KIMI_EXPERIMENT_DEADLINE_EPOCH_R8973'])
 if deadline-time.time()>=400:report['fixed_cohort_qualification_passed']=interleave()
 else:report['fixed_cohort_interleave']={'status':'SKIPPED_ORIGINAL_DEADLINE_BUDGET'};report['fixed_cohort_qualification_passed']=False
 save()
 continuous=False;pipeline=False;prefill_pipeline=False
 qa(0,'after');select(0,'done')
 report['serving_protocol_complete']=True;report['performance_qualified_pending_independent_audit']=report['fixed_cohort_qualification_passed']
 report['comparison_limit']='One same-load A/C full128 pair, no ABBA/causal steadydecode gain. C changes only admission/jobordering, fixedM16 membership and all perrequest inputs/pasts preserved. FullModel finalIDs/EOS+before/restoreQA required. Interleaved service rate separate from puredecode; no15user claim.'
 save()
