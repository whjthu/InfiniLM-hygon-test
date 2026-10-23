"""Matched ordinary serving tests; all performance compares complete token IDs."""
import ctypes,hashlib,json,os,time,pathlib
import numpy as np
from kimi_cohort_feedback_idle_r4915 import FeedbackHub,FeedbackClient,decode_cohorts
from kimi_metadata_counter_r4935 import validate as validate_metadata

"""Pure CPU timing reconstruction; outputs[0] and EOS never enter decode numerator."""
import statistics,math
def summary(v):
 v=sorted(x for x in v if x is not None)
 if not v:return None
 def q(p):
  a=(len(v)-1)*p;i=int(a);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(a-i)
 return {'mean':statistics.mean(v),'p50':q(.5),'p95':q(.95),'min':v[0],'max':v[-1],'count':len(v)}
def reconstruct(row,eos):
 ids=row['generated_ids'];times=row['timestamps_after_common_anchor'];first=row['TTFT_seconds'];anchor=row['service_wall_seconds']-row['decode_wall_seconds'];rates=[];post=[];nfirst=[];npost=[];gaps=[]
 for seq,ts,f in zip(ids,times,first):
  assert len(seq)==len(ts) and ts[0]==0 and not any(v in eos for v in seq[:-1])
  events=[(anchor+t,v)for t,v in zip(ts[1:],seq[1:])];end=anchor+ts[-1]if len(seq)>1 else f
  a=sum(v not in eos and f<t<=end for t,v in events);b=sum(v not in eos and 0<t-anchor<=end-anchor for t,v in events);nfirst.append(a);npost.append(b)
  rates.append(a/(end-f)if end>f else None);post.append(b/(end-anchor)if end>anchor else None)
  absolute=[f]+[t for t,v in events];gaps.extend(b-a for a,b in zip(absolute,absolute[1:]))
 wall=row['decode_wall_seconds']
 return {'first_ready_request_counts':nfirst,'postanchor_request_counts':npost,'first_ready_request_rates':rates,'postanchor_request_rates':post,'first_ready_mean_summary':summary(rates),'postanchor_mean_summary':summary(post),'aggregate_postanchor_tokens':sum(npost),'aggregate_postanchor_tok_s':sum(npost)/wall if wall>0 else None,'first_ready_undefined':sum(v is None for v in rates),'TBT':summary(gaps),'TTFT':summary(first),'metric_scope':'Internal emitter first-ready and postanchor independent windows; ordinal0/prefill and EOS excluded, EOSdecisiontime retained. No HTTP/qualifiedspeedclaim.'}

def run(g):
 lib,ctx,control,r,report,save,check=[g[k]for k in['lib','ctx','control','r','report','save','check']];node,tail,emitter,tag,vocab=[g[k]for k in['node','tail','emitter','tag','vocab']];assert(g['tp_nodes'],g['pp'])==(1,4);stage=g['stage'];first=int(g['text']['first_k_dense_replace']);freq=int(g['text']['moe_layer_freq']);moe=sum(i>=first and i%freq==0 for i in range(*g['ranges'][stage]));assert first==1 and freq==1
 head_set=lib.modern_kimi_vocab_head_set_r8341;head_set.argtypes=[ctypes.c_int];head_set.restype=ctypes.c_int
 head_get=lib.modern_kimi_vocab_head_get_r8341;head_get.restype=ctypes.c_int
 head_count=lib.modern_kimi_vocab_head_count_r8341;head_count.argtypes=[ctypes.c_int,ctypes.c_int];head_count.restype=ctypes.c_uint64
 head_pending=0
 def head_counts():return [[head_count(d,f)for f in range(3)]for d in range(8)]
 assert head_get()==0 and head_counts()==[[0,0,0]]*8
 def switch_after_owner_reset(label):
  layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert not layout['public_graph_enabled']
  votes=control.collect('head-quiescent:'+label,{'reset_all8_owners_completed':True,'capture_inactive':True,'policy':head_pending});assert len(votes)==4 and all(v==votes[0]for v in votes)
  assert head_set(head_pending)==0 and head_get()==head_pending
  votes=control.collect('head-policy-readback:'+label,{'policy':head_get()});assert len(votes)==4 and all(v['policy']==head_pending for v in votes)
  report.setdefault('head_switches',[]).append({'label':label,'policy':head_pending,'owner_fence':'ExistingEngine.reset_cache dispatches all8owner syncStream before jobdone; worldvote afterreset, graphsdisabled lifetime; no mainthread CUDA fence','world_readback':votes});save()
 counter=lib.modern_kimi_serving_counts_r4665;counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];check(lib.modern_kimi_reduce_policy_r3226(0));lib.modern_kimi_engine_resize_cache_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());natural=json.loads((r/'results'/f'kimi-serving-cases-r1101-{tag}.json').read_text());P=min(4,len(natural['cases']));assert P>=2;prompts=[v['input_ids']for v in natural['cases'][:P]];assert len({tuple(v)for v in prompts})==P and all(480<=len(v)<=512 for v in prompts);eos=set(natural['eos_token_ids']);assert eos==set(chat['eos_token_ids']);seq=0;policy=0;capacity=32;pipeline=False;prefill_pipeline=False
 report['head_candidate_hypothesis']='Actual9001bf dualhead OFF/ON only; N4fixedOFF, sameTP8PP4/cap32/cache4, B1M1 matched-only and B16M4 historicalr4917; no ModelVRAMsaving.'
 report['pipeline_scheduler']='AlltimedABBA use same boundedcohortfeedback and fixedrows; QA originalserialfull-logit. No changedscheduler or GPUtimeline claim.'
 report['local_forward_samples']=[]
 report.update(scope='Dualvocabhead OFF/ON, B1M1 matched-only and B16M4 historical+QA; N4OFF, otherfiveflagsOFF, servingpolicy1 timed.',serving_cases=[],QA=[],counter_dispatch=[],unique_natural_prompts=P,serving_cards=32,cache_capacity_rows=32,decode_output_limit=128,performance_qualified=False)
 report['memory_reserve_r8152']='5GiB collective eachphase/cell; cap32/cache4; perforward512MiB failstop; no trim, resize64 or loweringguard; full+shard additional280MiB/tailowner.'
 report['observer_policy']='ProfileOFF CAPIwall+identical semanticcounters botharms. ActualGPUtimeline NOT_MEASURED; optionalwindow deferred rather than delayingtest.'
 metadata_enabled=False
 bounded_enabled=False
 packed_enabled=False
 packed_counter=g['packed_counter']
 report['packed_W2_dispatch']=[]
 def packed_counts():
  answer=[]
  for device in range(8):
   a=(ctypes.c_uint64*8)();assert packed_counter(device,a,8)==0;answer.append(list(a))
  return answer
 assert packed_counts()==[[0]*8 for _ in range(8)]
 def packed_select(enabled,label):
  nonlocal packed_enabled
  assert enabled in [False,True]
  control.collect('packed-before:'+label,{'enabled':enabled})
  os.environ['INFINILM_KIMI_PACKED_W2_R6122']='0'
  os.environ['INFINILM_KIMI_PACKED_AUTO_R6136']=str(int(enabled));packed_enabled=enabled
  control.collect('packed-after:'+label,{'enabled':enabled})
 report['bounded_BF16_dispatch']=[]
 def bounded_counts():
  values=[]
  for device in range(8):
   a=(ctypes.c_uint64*4)();assert g['bounded_counter'](device,a,4)==0;values.append(list(a))
  return values
 assert bounded_counts()==[[0]*4 for _ in range(8)]
 setter=lib.modern_kimi_engine_set_cache_pages_r4974;setter.argtypes=[ctypes.c_void_p,ctypes.c_size_t];setter.restype=ctypes.c_int
 def cache_select(pages,label):
  control.collect('cache-before:'+label,{'pages':pages})
  check(setter(ctx,pages));geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx))
  assert geometry=={'max_rows':capacity,'pages':pages*(capacity+1),'page_size':256,'KDA_pool_rows':capacity+1,'pages_per_request':pages,'max_total_tokens':256*pages}
  mem=g['memory']();packet=control.collect('cache-after:'+label,{'geometry':geometry,'memory':mem})
  report.setdefault('ready_cache_transitions',[]).append({'label':label,'pages':pages,'all_ranks':packet});save()
 def bounded_select(enabled,label):
  nonlocal bounded_enabled
  assert enabled in [False,True];control.collect('bounded-before:'+label,{'enabled':enabled})
  os.environ['INFINILM_KIMI_BOUNDED_BF16_R6054']=str(int(enabled));bounded_enabled=enabled
  control.collect('bounded-after:'+label,{'enabled':enabled})
 report['metadata_dispatch']=[]
 def metacounts():
  values=[]
  for device in range(8):
   a=(ctypes.c_uint64*7)();assert g['metadata_counter'](device,a,7)==0;values.append(list(a))
  return values
 assert metacounts()==[[0]*7 for _ in range(8)]
 def metaselect(enabled,label):
  nonlocal metadata_enabled
  assert enabled in [False,True]
  control.collect('metadata-before:'+label,{'enabled':enabled})
  os.environ['INFINILM_KIMI_METADATA_PACK_R5858']=str(int(enabled));metadata_enabled=enabled
  control.collect('metadata-after:'+label,{'enabled':enabled})
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
  assert value in[0,1,4,9];control.collect('serving-policy-before:'+label,{'policy':value});os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=str(value);policy=value;control.collect('serving-policy-after:'+label,{'policy':value})
 n4_counter=g['n4_counter_r8167']
 def n4_counts():
  result=[]
  for device in range(8):
   v=(ctypes.c_uint64*2)();check(n4_counter(device,v,2));result.append(list(v))
  return result
 assert n4_counts()==[[0,0]]*8
 def forward(ins,pasts,rows,label,token_only=False,local_only=False):
  nonlocal seq
  assert all(os.environ.get(k)=='0' for k in ['INFINILM_KIMI_PACKED_W2_R6122','INFINILM_KIMI_PACKED_AUTO_R6136','INFINILM_KIMI_METADATA_PACK_R5858','INFINILM_KIMI_BOUNDED_BF16_R6054','INFINILM_KIMI_STATIC_DIV_R6020','INFINILM_KIMI_N4_TILE_R8167'])
  B=len(rows);widths=[len(v)for v in ins];N=sum(widths);assert 0<B<=capacity and N<=8192;row=np.asarray(rows,np.int32);past=np.asarray(pasts,np.int32);total=past+np.asarray(widths,np.int32);offset=np.asarray([0]+np.cumsum(widths).tolist(),np.int32);ids=np.ascontiguousarray(np.concatenate(ins),dtype=np.int64);positions=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));logits=None if token_only else np.full((B,vocab),0x7fc0,np.uint16);out=np.full(B,-999,np.int64);flags=np.asarray([-1],np.int32);head_before=head_counts();n4_before=n4_counts();before=counts();meta_before=metacounts();bounded_before=bounded_counts();packed_before=packed_counts();start=time.perf_counter();capi_start_realtime_ns=time.time_ns();arrays=[row,past,total,offset,ids,positions];ptrs=[ctypes.c_void_p(v.ctypes.data)for v in arrays]+[None if logits is None else ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(flags.ctypes.data)];check(lib.modern_kimi_engine_forward_r2475(ctx,N,B,71 if token_only else 67,*ptrs));seconds=time.perf_counter()-start;mem=g['memory']();report['last_forward_memory']=mem;assert all(free>512*1024**2 for free,total in mem),('512MiB_runtime_floor',label,mem);after=counts();meta_after=metacounts();bounded_after=bounded_counts();packed_after=packed_counts()
  head_after=head_counts();head_delta=[[b-a for a,b in zip(x,y)]for x,y in zip(head_before,head_after)];expected_head=([0,1,B]if head_pending else[1,0,0])if tail else[0,0,0];assert head_delta==[expected_head]*8,('actual_head_dispatch',label,head_delta,expected_head);report.setdefault('head_dispatch',[]).append({'label':label,'B':B,'N':N,'policy':head_pending,'delta':head_delta,'expected':expected_head})
  n4_after=n4_counts();n4_delta=[[b-a for a,b in zip(x,y)]for x,y in zip(n4_before,n4_after)];expected_n4=[moe,4*moe]if os.environ.get('INFINILM_KIMI_N4_TILE_R8167')=='1'and N==4 else[0,0]
  assert n4_delta==[expected_n4]*8,('8167dispatch',label,n4_delta,expected_n4)
  report.setdefault('N4_dispatch_r8167',[]).append({'label':label,'N':N,'B':B,'candidate_requested':os.environ.get('INFINILM_KIMI_N4_TILE_R8167')=='1','delta':n4_delta,'expected':expected_n4})
  meta_verified=validate_metadata(meta_before,meta_after,tokens=N,moe_layers=moe,policy=policy,enabled=metadata_enabled)
  report['metadata_dispatch'].append({'label':label,'N':N,'B':B,'policy':policy,'enabled':metadata_enabled,'before':meta_before,'after':meta_after,'verified':meta_verified})
  bounded_delta=[[b-a for a,b in zip(x,y)]for x,y in zip(bounded_before,bounded_after)]
  for device,delta in enumerate(bounded_delta):
   assert all(v>=0 for v in delta)
   plans=meta_after[device][2]-meta_before[device][2]
   if bounded_enabled and N>=512:
    assert delta[0]==delta[2]==plans and plans>=moe
    assert delta[1]==delta[3] and moe<=delta[1]<=moe*min(896,N*16)
   else:assert delta==[0]*4
  report['bounded_BF16_dispatch'].append({'label':label,'N':N,'B':B,'enabled':bounded_enabled,'before':bounded_before,'after':bounded_after,'delta':bounded_delta,'metadata_plans_delta':[b[2]-a[2]for a,b in zip(meta_before,meta_after)]})
  packed_delta=[[b-a for a,b in zip(x,y)]for x,y in zip(packed_before,packed_after)]
  eligible=packed_enabled and policy==1 and N==16
  expected_packed=[0]*6+([moe,16*moe]if eligible else[0,0])
  assert packed_delta==[expected_packed]*8,('packed_auto_submission_counts',label,N,packed_delta,expected_packed)
  report['packed_W2_dispatch'].append({'label':label,'N':N,'B':B,'policy':policy,'enabled':packed_enabled,'before':packed_before,'after':packed_after,'delta':packed_delta,'expected':expected_packed,'GPU_branch_selection_observed':False})
  expected=[0]*4
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
  B=len(inputs);assert B<=capacity and 0<M<=capacity;report['phase']=label;save();reset(label);switch_after_owner_reset(label);ids=[[]for _ in range(B)];pasts=[0]*B;times=[[]for _ in range(B)];ttft=[0.]*B;start=time.perf_counter()
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
   assert pipeline  # Full-logit untimed reference uses the same fixed cohorts as token-only arms.
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
  packet=control.collect('serving-finished:'+label,{'ids':ids,'times':times,'ttft':ttft,'prefill_wall_seconds':prefill_end-start,'prefill_CAPI_loop_seconds':prefill_local_end-start,'prefill_final_drain_seconds':prefill_drain_end-prefill_drain_start,'anchor_barrier_seconds':decode_start-prefill_end,'decode_anchor_from_start_seconds':decode_start-start});assert all(z['ids']==ids for z in packet);canonical=packet[emitter];times=canonical['times'];wall=max(z[-1]for z in times);useful=[sum(v not in eos for v in row[1:])for row in ids];same=None if reference is None else ids==reference;assert all((len(z)==limit or z[-1]in eos)and not any(v in eos for v in z[:-1])for z in ids);row={'packed_W2_auto_requested':packed_enabled,'bounded_BF16_requested':bounded_enabled,'cache_pages_per_request':json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx))['pages_per_request'],'metadata_pack_requested':metadata_enabled,'continuous_cohort_feedback_requested':continuous,'label':label,'B':B,'frontend_cohort_M':M,'policy':policy,'input_lengths':list(map(len,inputs)),'max_output_tokens':limit,'generated_ids':ids,'timestamps_after_common_anchor':times,'TTFT_seconds':canonical['ttft'],'prefill_stage_local_requested':prefill_pipeline,'prefill_CAPI_loop_seconds':canonical['prefill_CAPI_loop_seconds'],'prefill_final_drain_seconds':canonical['prefill_final_drain_seconds'],'prefill_wall_seconds':canonical['prefill_wall_seconds'],'anchor_barrier_seconds':canonical['anchor_barrier_seconds'],'service_wall_seconds':canonical['decode_anchor_from_start_seconds']+wall,'service_boundary_policy':'Emitter perf_counter start beforeprefill through lastdecode ID; includes anchorbarrier; excludes requestarrival/loading; firstprefill IDs and EOS counted separately.', 'useful_all_outputs':sum(sum(v not in eos for v in row)for row in ids),'end_to_end_output_tok_s':sum(sum(v not in eos for v in row)for row in ids)/(canonical['decode_anchor_from_start_seconds']+wall),'useful_decode_outputs':sum(useful),'decode_wall_seconds':wall,'aggregate_decode_tok_s':sum(useful)/wall if wall else None,'decode_tok_s_per_card':sum(useful)/wall/32 if wall else None,'per_request_decode_tok_s':[u/t[-1]if u and t[-1]>0 else None for u,t in zip(useful,times)],'matched_baseline_complete_IDs':same,'timed_decode_token_only':token_only,'sampled_CPU_argmax':'Allprefill; alldecodeonlyinfull-logitreference, token-onlycasescheckedagainstcompletereferenceIDs.','all_service_cards':32,'no_EOS_padding':True,'stage_local_PP_overlap_requested':pipeline,'overlap_measured_by_timeline':False,'native_call_seconds_sum':sum(samples),'scope':'Closedbatchordinarydecode; front-endcohortscompleteoneforwardbefore next, soB/MdoesnotimplyPPfill. Decodeexcludesallprefill/anchor/EOS, includesCAPI/IDcopy/CPUcontrol/unhiddencomm. Naturalprompts repeatedfromP uniqueinputs; noproductiontrafficrepresentativenessclaim.'};row['head_policy']=head_pending;row['N4_candidate_requested']=os.environ.get('INFINILM_KIMI_N4_TILE_R8167')=='1';report['serving_cases'].append(row);save()
  if reference is not None:assert same,('changed_complete_IDs',label)
  return ids
 def qa(value,epoch):
  select(value,'QA-'+epoch+'-'+str(value));passed=True
  for task,c in chat['cases'].items():
   label=f'QA-{epoch}-p{value}-{task}';z=generate([c['input_ids']],1,label,128,False);key='sort'if task.startswith('sort_probe')else task;gold=[x['generated_ids'][0]for x in known['cases']if x['task']==key and x['grouped']];exact=z[0]in gold
   packet=control.collect('N4-QA-finalIDs:'+label,{'exact':exact,'actual_IDs':z[0]});assert len(packet)==4 and all(p['exact']==exact and p['actual_IDs']==z[0]for p in packet),('crossnode_QA_gate_disagreement',label)
   report['QA'].append({'epoch':epoch,'policy':value,'task':task,'complete_frozen_IDs_exact':exact,'actual_complete_IDs':z[0],'expected_frozen_complete_ID_alternatives':gold,'all_four_node_QA_gate_consistent':True,'N4_candidate_requested':os.environ.get('INFINILM_KIMI_N4_TILE_R8167')=='1'})
   if not exact:report['quality_failure']=True
   passed=passed and exact;save()
  return passed
 os.environ['INFINILM_KIMI_COMPACT_CACHE_R4974']='1'
 os.environ['INFINILM_KIMI_N4_TILE_R8167']='0'
 report.update(scope='Dualhead same-load OFF/ON only; N4fixedOFF,TP8PP4,cap32/cache4',smallB_checks=[],smallB_NOT_RUN=[],quality_failure=False,B1_historical_status='NOT_AVAILABLE_FOR_THIS_EXACT_PROMPT_GEOMETRY',head_switches=[],head_dispatch=[],serving_cards=32)
 def reserve(label):
  rows=control.collect('head-reserve:'+label,{'node':node,'memory':g['memory'](),'domains':g['memory_domains'](g['torch'],label)});ready=all(f>=5*1024**3 for n in rows for f,t in n['memory']);report.setdefault('smallB_reserve_checks',[]).append({'label':label,'minimum_GiB':5,'all_nodes':rows,'ready':ready});save();return ready
 def mode(value,label):
  nonlocal head_pending
  assert value in[0,1];head_pending=value
 gold16=next(z['fixed_complete_IDs']for z in frozen['geometries']if(z['B'],z['M'])==(16,4));baseline1=None
 def cell(B,value,label,freeze=False):
  nonlocal baseline1
  if not reserve(label):report['smallB_NOT_RUN'].append({'label':label,'status':'NOT_RUN_RESOURCE','reason':'5GiB'});save();return False
  mode(value,label);actual=generate([prompts[i%2]for i in range(B)],1 if B==1 else 4,label,128,True,None)
  if freeze:
   assert B==1 and value==0 and baseline1 is None
   baseline1=actual;body={'scope':'Same-load OFF matched-only diagnostic reference; NOT independent qualified historicalgold','run':tag,'B':1,'M':1,'capacity':32,'cache_pages':4,'input_ids':prompts[0],'generated_ids':actual,'eos_token_ids':sorted(eos),'Model_SHA256':g['sha'](g['library'])};path=r/'results'/f'headB1-OFF-baseline-{tag}-node{node}.json';raw=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
   with path.open('xb')as f:f.write(raw)
   digest=hashlib.sha256(raw).hexdigest();votes=control.collect('headB1-freeze-before-anyON',{'SHA256':digest});assert len(votes)==4 and all(v['SHA256']==digest for v in votes);report['B1_baseline']={'path':str(path),'SHA256':digest,'scope':body['scope']};save()
  reference=baseline1 if B==1 else gold16;assert reference is not None;exact=actual==reference
  report['smallB_checks'].append({'label':label,'B':B,'M':1 if B==1 else 4,'limit':128,'warmup':False,'candidate':value,'full_reference_IDs_EOS_exact':exact,'reference_scope':'same-loadOFFmatched-only'if B==1 else'historicalr4917'});report['serving_cases'][-1]['smallB_metrics']=reconstruct(report['serving_cases'][-1],eos)
  if not exact:report['quality_failure']=True
  save();return exact
 def guarded_qa(value,epoch):
  if not reserve('QA-'+epoch):report['smallB_NOT_RUN'].append({'label':'QA-'+epoch,'status':'NOT_RUN_RESOURCE','reason':'5GiB'});save();return False
  mode(value,epoch);outcomes=[qa(policy,epoch)for policy in[0,1]];return all(outcomes)
 before=guarded_qa(0,'OFF-before');cache_select(4,'head-initial4');select(1,'head-fixed-servingpolicy1');pipeline=True;prefill_pipeline=True;continuous=True
 proceed=before and cell(16,0,'head-history-B16M4')
 if proceed:proceed=cell(1,0,'head-freeze-B1M1',True)
 if proceed:proceed=cell(1,1,'head-qualityON-B1M1')
 if proceed:proceed=cell(16,1,'head-qualityON-B16M4')
 continuous=False;pipeline=False;prefill_pipeline=False
 onqa=guarded_qa(1,'ON-before')if proceed else False;proceed=proceed and onqa
 select(1,'head-fixed-servingpolicy1-afterQA');pipeline=True;prefill_pipeline=True;continuous=True
 for B in[1,16]:
  for i,value in enumerate([0,1,1,0]):
   if proceed:proceed=cell(B,value,f'head-ABBA-B{B}-r{i}')
 restored1=cell(1,0,'head-restore-B1M1')if baseline1 is not None else False
 restored16=cell(16,0,'head-restore-B16M4')
 continuous=False;pipeline=False;prefill_pipeline=False
 after=guarded_qa(0,'OFF-after')
 # Final policy reset uses the same existing owner-reset fence, never Flight alone.
 mode(0,'done');reset('head-final-OFF');switch_after_owner_reset('head-final-OFF');select(0,'done')
 complete=len(report['smallB_checks'])==14 and not report['smallB_NOT_RUN']and not report['quality_failure']and before and onqa and after and restored1 and restored16
 report.update(serving_protocol_complete=True,candidate_qualified=False,smallB_quality_passed=bool(complete),smallB_matrix_completed=bool(complete),smallB_status='COMPLETE'if complete else'PARTIAL',smallB_restore_status='PASS'if restored1 and restored16 else'NOT_RUN_OR_NUMERIC_FAILURE_SEE_CHECKS',final_N4_flag=os.environ['INFINILM_KIMI_N4_TILE_R8167'],final_serving_policy=os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665'],final_head_policy=head_get());save()
