def frozen648_reference_r8669zq(r,inputs,eos,actual,report,save):
 p=r/"results/kimi-TP16PP2-I192-r8669zc-TP16PP2-I1928669za-f29b4d4db70b0b8b-node2.json";raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=="5c4c52c8895858be6eed7d5d48a60d2499c0d779c170cf55d11b7d64e11b91f2"
 q=json.loads(raw);assert(q['TP'],q['PP'],q['node'],q['PP_stage'])==(16,2,2,1)
 assert q['cache_geometry']==dict(max_rows=16,pages=68,page_size=256,KDA_pool_rows=17,pages_per_request=4,max_total_tokens=1024)
 cases=[v for v in q['serving_cases']if v['label']=='TP16-B16M16-O-full-reference128'];assert len(cases)==1
 z=cases[0];assert(z['B'],z['frontend_cohort_M'],z['policy'],z['I192_candidate_ON'],z['timed_decode_token_only'])==(16,16,1,False,False)
 assert z['input_lengths']==[512]*16 and len(inputs)==16 and all(len(v)==512 for v in inputs)
 expected=z['generated_ids'];assert len(expected)==16 and all(len(v)==128 for v in expected)
 first=[]
 for i,(a,b)in enumerate(zip(expected,actual)):
  pos=next((j for j,(x,y)in enumerate(zip(a,b))if x!=y),None)
  if pos is not None or len(a)!=len(b):first.append({'request':i,'first_different_ordinal':None if pos is None else pos+1,'expected':None if pos is None else a[pos],'actual':None if pos is None else b[pos],'expected_length':len(a),'actual_length':len(b)})
 report['frozen648_original_reference']={'report_path':str(p),'raw_SHA256':"5c4c52c8895858be6eed7d5d48a60d2499c0d779c170cf55d11b7d64e11b91f2",'label':z['label'],'matched_full_IDs_EOS':actual==expected,'first_differences':first,'input_mapping':'r4917 frozen two natural I512 prompts alternated rows1..16, prefillM2/decodeM16/TP16PP2/flags3/head0/SpecMath0/serving1/I192OFF','input_ids_JSON_SHA256':hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest(),'eos_token_ids':sorted(eos),'scope':'648 original OFF arm completed; 648 candidate/session FAILED. Independent frozen original baseline only, not candidate-generated gold or inherited TPS.'};save()
 assert actual==expected,('baseline_reproducibility_FAILED_OBSERVED',first)
 return expected

def postQA_memory_gate_r8669zq(g,report,save,reset,control):
 before=g['memory']();report['postQA_reset_memory']={'before':before};save()
 reset('after-QA-before-geometry')
 after=g['memory']();report['postQA_reset_memory']['after']=after;save()
 packets=control.collect('geometry-memory:16:16',{'memory':after})
 minimum=min(free for packet in packets for free,total in packet['memory'])
 report['geometry_memory_gate']={'minimum_free_bytes':minimum,'required_free_bytes':4*1024**3};save()
 assert minimum>4*1024**3,'original4GiB_pregeometry_reserve'

"""Matched ordinary serving tests; all performance compares complete token IDs."""
import ctypes,hashlib,json,os,time,pathlib
import numpy as np
from kimi_cohort_feedback_idle_r4915 import FeedbackHub,FeedbackClient,decode_cohorts

def run(g):
 lib,ctx,control,r,report,save,check=[g[k]for k in['lib','ctx','control','r','report','save','check']];node,tail,emitter,tag,vocab=[g[k]for k in['node','tail','emitter','tag','vocab']];assert(g['tp_nodes'],g['pp'])==(2,2);stage=g['stage'];first=int(g['text']['first_k_dense_replace']);freq=int(g['text']['moe_layer_freq']);moe=sum(i>=first and i%freq==0 for i in range(*g['ranges'][stage]));assert first==1 and freq==1
 custom=lib.modern_kimi_I192_counts_r8669y;custom.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];custom.restype=ctypes.c_int
 counter=lib.modern_kimi_serving_counts_r4665;counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];check(lib.modern_kimi_reduce_policy_r3226(0));lib.modern_kimi_engine_resize_cache_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());frozen_path=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';frozen=json.loads(frozen_path.read_text());prompts=frozen['natural_input_ids'];P=len(prompts);assert P==2 and len({tuple(v)for v in prompts})==2 and all(len(v)==512 for v in prompts);eos=set(frozen['eos_token_ids']);assert eos==set(chat['eos_token_ids']);seq=0;policy=0;capacity=16;pipeline=False;prefill_pipeline=False;candidate=False;last_local_ready=0.;events_armed=False
 report['pipeline_scheduler']='Existingstage-localprefillM2/continuousdecodeM16;TP16PP2 all4frontends samejoborder, emitter2. No concurrentEnginecall/crosshostclocksubtraction.';report['local_forward_samples']=[]
 report.update(serving_cases=[],QA=[],counter_dispatch=[],unique_natural_prompts=P,serving_cards=32,cache_capacity_rows=16,decode_output_limit=128,performance_qualified=False)
 report['memory_reserve_r4912']='4GiB collective pregeometry, cap16/pool17, exactTP16I192 compactworkspace4468480, postQA existingownerreset/trim;4890 clean rejected4.64GiBvs5GiB; runtime512MiB failstop retained, noresize128 planned';report['observer_policy']='No hipprof/APIinterposer/capture. Sameprocessoriginalreference3/candidate7 qualification, ordinaryboth7; no new profiling.'
 report['scope']='TP16PP2 currentprivateI192/flags3&7/SpecMath0/cap16/pool17/workspace64/prefillM2. Originalfallbackfull128reference+QA then candidatequalification andordinaryONNO. Two512prompts repeated; nohistoricalcrossTPgold or inheritedspeed.'
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
 def custom_counts():
  result=[]
  for device in range(8):
   z=(ctypes.c_uint64*2)();check(custom(device,z,2));result.append(list(z))
  return result
 def candidate_select(value,label):
  nonlocal candidate
  assert type(value)is bool
  control.collect('I192-before:'+label,{'ON':value});os.environ['INFINILM_KIMI_I192_GROUPED_R8669Y']='1'if value else'0';candidate=value;control.collect('I192-after:'+label,{'ON':value})
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
  nonlocal seq,last_local_ready
  head_policy=int(lib.modern_kimi_vocab_head_get_r8341());assert head_policy==0,('TP16headmustremainreplicated0',label,head_policy);B=len(rows);widths=[len(v)for v in ins];N=sum(widths);assert 0<B<=16 and N<=8192;row=np.asarray(rows,np.int32);past=np.asarray(pasts,np.int32);total=past+np.asarray(widths,np.int32);offset=np.asarray([0]+np.cumsum(widths).tolist(),np.int32);ids=np.ascontiguousarray(np.concatenate(ins),dtype=np.int64);positions=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));logits=None if token_only else np.full((B,vocab),0x7fc0,np.uint16);out=np.full(B,-999,np.int64);flags=np.asarray([-1],np.int32);before=counts();cb=custom_counts();start=time.perf_counter();capi_start_realtime_ns=time.time_ns();arrays=[row,past,total,offset,ids,positions];ptrs=[ctypes.c_void_p(v.ctypes.data)for v in arrays]+[None if logits is None else ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(flags.ctypes.data)];check(lib.modern_kimi_engine_forward_r2475(ctx,N,B,7 if token_only else 3,*ptrs));seconds=time.perf_counter()-start;mem=g['memory']();report['last_forward_memory']=mem;assert all(free>512*1024**2 for free,total in mem),('512MiB_runtime_floor',label,mem);after=counts();expected=[0]*4
  delta=[[b-a for a,b in zip(x,y)]for x,y in zip(before,after)];assert delta==[expected]*8,('actualservingMoEdispatch',label,N,policy,delta,expected);ca=custom_counts();cd=[[b-a for a,b in zip(x,y)]for x,y in zip(cb,ca)];ce=[moe,16*moe]if candidate and policy==1 and N==16 else[0,0];assert cd==[ce]*8,('actualI192dispatch',label,N,candidate,cd,ce);report['counter_dispatch'].append({'label':label,'N':N,'B':B,'policy':policy,'candidate':candidate,'expected':expected,'delta':delta,'I192_delta':cd,'I192_expected':ce,'head_policy':head_policy});assert int(flags[0])==((int(tail)if not token_only else 0)+(2 if node==emitter else 0));selected=None;digest=None
  if tail and not token_only:
   f=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();argmax=np.argmax(f,axis=1).tolist();digest=hashlib.sha256(logits.tobytes()).hexdigest()
   if node==emitter:assert out.tolist()==argmax
  if node==emitter:selected=out.tolist();assert all(0<=v<vocab for v in selected)
  report['local_forward_samples'].append({'label':label,'N':N,'B':B,'CAPI_seconds':seconds,'CAPI_start_realtime_ns':capi_start_realtime_ns,'CAPI_end_realtime_ns':capi_start_realtime_ns+int(seconds*1e9),'pipeline':pipeline})
  last_local_ready=time.perf_counter()
  if local_only:return selected,seconds,digest
  packet=control.collect('serving-call-'+str(seq),{'ids':selected,'seconds':seconds,'full_logits_sha256':digest});seq+=1;selected=packet[emitter]['ids'];assert selected is not None and len(selected)==B;return selected,max(v['seconds']for v in packet),packet[emitter]['full_logits_sha256']
 def reset(label):check(lib.modern_kimi_engine_reset_r2475(ctx));control.collect('serving-reset:'+label,{})
 def generate(inputs,M,label,limit=128,token_only=True,reference=None):
  B=len(inputs);assert B<=capacity and M<=64;report['phase']=label;save();reset(label);ids=[[]for _ in range(B)];pasts=[0]*B;times=[[]for _ in range(B)];ttft=[0.]*B;start=time.perf_counter()
  prefill_parts=[]
  for a in range(0,B,2):
   ii=list(range(a,min(a+2,B)));z,_,_=forward([inputs[i]for i in ii],[0]*len(ii),[i+1 for i in ii],label+'-prefill'+str(a),False,local_only=prefill_pipeline);stamp=last_local_ready-start
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
   stream_result=decode_cohorts(feedback=feedback,case=label,is_emitter=node==emitter,ids=ids,pasts=pasts,times=times,micro=M,limit=limit,eos=eos,forward=owner_forward,anchor=decode_start,clock=lambda:last_local_ready)
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
      ii=active[a:a+M];z,seconds,_=forward([[ids[i][-1]]for i in ii],[pasts[i]for i in ii],[i+1 for i in ii],label+'-decode'+str(step)+'-part'+str(a),token_only,local_only=True);stamp=last_local_ready-decode_start;samples.append(seconds);local_parts.append({'rows':ii,'ids':z,'stamp':stamp,'seconds':seconds})
     packets=control.collect('pipeline-drained:'+label+':'+str(step),local_parts)
     for part in packets[emitter]:
      assert part['ids']is not None and len(part['ids'])==len(part['rows'])
      for i,v in zip(part['rows'],part['ids']):ids[i].append(v);pasts[i]+=1;times[i].append(part['stamp'])
    else:
     for a in range(0,len(active),M):
      ii=active[a:a+M];z,seconds,_=forward([[ids[i][-1]]for i in ii],[pasts[i]for i in ii],[i+1 for i in ii],label+'-decode'+str(step)+'-part'+str(a),token_only);stamp=last_local_ready-decode_start;samples.append(seconds)
      for i,v in zip(ii,z):ids[i].append(v);pasts[i]+=1;times[i].append(stamp)
    if step%32==0:report['active_serving']={'label':label,'step':step,'active':len(active),'decode_seconds':time.perf_counter()-decode_start}
  packet=control.collect('serving-finished:'+label,{'ids':ids,'times':times,'ttft':ttft,'prefill_wall_seconds':prefill_end-start,'prefill_CAPI_loop_seconds':prefill_local_end-start,'prefill_final_drain_seconds':prefill_drain_end-prefill_drain_start,'anchor_barrier_seconds':decode_start-prefill_end,'decode_anchor_from_start_seconds':decode_start-start});assert all(z['ids']==ids for z in packet);canonical=packet[emitter];times=canonical['times'];wall=max(z[-1]for z in times);useful=[sum(v not in eos for v in row[1:])for row in ids];same=None if reference is None else ids==reference;assert all((len(z)==limit or z[-1]in eos)and not any(v in eos for v in z[:-1])for z in ids);row={'continuous_cohort_feedback_requested':continuous,'label':label,'B':B,'frontend_cohort_M':M,'policy':policy,'I192_candidate_ON':candidate,'input_lengths':list(map(len,inputs)),'max_output_tokens':limit,'generated_ids':ids,'timestamps_after_common_anchor':times,'TTFT_seconds':canonical['ttft'],'prefill_stage_local_requested':prefill_pipeline,'prefill_CAPI_loop_seconds':canonical['prefill_CAPI_loop_seconds'],'prefill_final_drain_seconds':canonical['prefill_final_drain_seconds'],'prefill_wall_seconds':canonical['prefill_wall_seconds'],'anchor_barrier_seconds':canonical['anchor_barrier_seconds'],'service_wall_seconds':canonical['decode_anchor_from_start_seconds']+wall,'service_boundary_policy':'Emitter perf_counter start beforeprefill through lastdecode ID; includes anchorbarrier; excludes requestarrival/loading; firstprefill IDs and EOS counted separately.', 'useful_all_outputs':sum(sum(v not in eos for v in row)for row in ids),'end_to_end_output_tok_s':sum(sum(v not in eos for v in row)for row in ids)/(canonical['decode_anchor_from_start_seconds']+wall),'useful_decode_outputs':sum(useful),'decode_wall_seconds':wall,'aggregate_decode_tok_s':sum(useful)/wall if wall else None,'decode_tok_s_per_card':sum(useful)/wall/32 if wall else None,'per_request_decode_tok_s':[u/t[-1]if u and t[-1]>0 else None for u,t in zip(useful,times)],'matched_baseline_complete_IDs':same,'timed_decode_token_only':token_only,'sampled_CPU_argmax':'Allprefill; alldecodeonlyinfull-logitreference, token-onlycasescheckedagainstcompletereferenceIDs.','all_service_cards':32,'no_EOS_padding':True,'stage_local_PP_overlap_requested':pipeline,'overlap_measured_by_timeline':False,'native_call_seconds_sum':sum(samples),'scope':'ClosedbatchTP16PP2 using existing stage-local deferredfeedback/continuouscohorts. Decode excludesprefill/anchor/EOS, includesCAPI/IDcopy/control/communication; no crosshostclock subtraction. Two repeatednaturalprompts, notproductiontraffic.'};row['instrumented_baseline']=events_armed
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
 # No profile/OFF3 arm: current accepted ON token-only A must reproduce
 # the independently frozen original IDs before comparing one mixed geometry.
 report['observer_policy']='MarkersOFF/no profile/hipprof/API interposer. One A complete128, one C complete128, optional one C repeat only within original deadline.'
 report['scope']='TP16PP2 cap16 I192ON flags7 A ordinary M2prefill/M16decode vs one interleaved smallM fallback then M16. Two naturalI512 prompts repeated; no B32/grid/Model quality inherited.'
 candidate_select(False,'beforeQA');qa(0,'before')
 postQA_memory_gate_r8669zq(g,report,save,reset,control)
 inputs=[prompts[i%2]for i in range(16)]
 pipeline=True;prefill_pipeline=True;continuous=True;select(1,'current-serving-policy1')
 candidate_select(True,'accepted-current-I192ON')
 fixed=generate(inputs,16,'TP16-B16M16-A-current-ordinary128',128,True)
 fixed=frozen648_reference_r8669zq(r,inputs,eos,fixed,report,save)
 report['samegeometry_original_reference_scope']='Independent original648 OFF rawgold, already reproduced by accepted654 ON. This fresh A flags7 must independently match every ID/EOS; candidate cannot write gold.';save()
 report['interleaved_cases']=[]
 deadline=float(os.environ['KIMI_EXPERIMENT_DEADLINE_EPOCH_R8669ZQ'])
 def interleaved(label):
  from kimi_interleaved_prefill_decode_r8669zq import run_interleaved
  report['phase']=label;save();reset(label)
  actual={};report['interleaved_cases'].append(actual)
  run_interleaved(inputs=inputs,eos=eos,forward=forward,clock=time.perf_counter,state=actual,label=label,limit=128,token_ready_clock=lambda:last_local_ready)
  packet=control.collect('interleaved-finished:'+label,actual);canonical=packet[emitter]
  assert all(x['generated_ids']==canonical['generated_ids']for x in packet)
  report['interleaved_cases'][-1]=canonical
  first=[]
  for i,(a,b)in enumerate(zip(fixed,canonical['generated_ids'])):
   pos=next((j for j,(x,y)in enumerate(zip(a,b))if x!=y),None)
   if pos is not None or len(a)!=len(b):first.append({'request':i,'ordinal':None if pos is None else pos+1,'expected':None if pos is None else a[pos],'actual':None if pos is None else b[pos],'expected_length':len(a),'actual_length':len(b)})
  canonical['matched_independent_complete_IDs_EOS']=not first;canonical['first_differences']=first
  canonical['timing_scope']='Emitter perf_counter immediately after actual tail ready; start before first prefill. No delayed firsttoken, no crosshost subtraction/client claim. Early decode included in userfirstlast/E2E; postprefill rate is separate, not whole benchmark puredecode.'
  canonical['all_service_cards']=32
  canonical['end_to_end_output_tok_s_per_card']=canonical['end_to_end_output_tok_s']/32
  save() # Persist full actual outputs and clocks BEFORE quality classification.
  return not first
 remaining=deadline-time.time();report['remaining_before_C_seconds']=remaining;save()
 if remaining>=400:
  passed=interleaved('TP16-C-interleaved-qualification128')
  report['interleaved_finaltokens_status']='PASS'if passed else'FAILED_OBSERVED';save()
  if passed and deadline-time.time()>=220:
   repeated=interleaved('TP16-C-interleaved-repeat128')
   report['interleaved_repeat_status']='PASS'if repeated else'FAILED_OBSERVED'
   if not repeated:report['interleaved_finaltokens_status']='FAILED_OBSERVED'
  elif passed:report['interleaved_repeat_status']='SKIPPED_ORIGINAL_DEADLINE_BUDGET'
 else:report['interleaved_finaltokens_status']='SKIPPED_ORIGINAL_DEADLINE_BUDGET'
 continuous=False;pipeline=False;prefill_pipeline=False
 candidate_select(False,'restoreQA');qa(0,'after');select(0,'done')
 report['serving_protocol_complete']=True
 report['performance_qualified_pending_independent_audit']=report['interleaved_finaltokens_status']=='PASS'
 report['comparison_limit']='One A/C pair with optional C repeat, no balancedABBA or fastestserving claim. Geometry changes genericN2/4/6/8/10/12/14 then I192N16; complete128 IDs/EOS and realuser timeline required.'
 save()
