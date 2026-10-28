"""Real heldout-task quality; no token-equality acceptance gate. Single resident model."""
import ctypes,hashlib,json,time,statistics
import numpy as np
from Deep_eos_quality_metrics_r9148 import score_case,paired_quality,serving_metrics
from deep_owner_consumer_r168a import bind_inner,inner_owner_forward_once,OwnerWindowStillArmed,OwnerObservationFailed,native_expected_kinds

def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 from pilot_data_contract_r168a import prepare_pilot
 manifest=g['manifest'];prepared,gold,cases,proof=prepare_pilot(r,manifest,__import__('pathlib').Path(__file__).resolve().parent,g['model'],cfg);report['data_preflight_r168a']=proof
 lib.modern_deep_c4_smallt_mode_r9252.argtypes=[ctypes.c_int];lib.modern_deep_c4_smallt_mode_r9252.restype=ctypes.c_int;check(lib.modern_deep_c4_smallt_mode_r9252(0))
 tokenizer=g['tokenizer'];assert tokenizer.bos_token_id==0 and tokenizer.eos_token_id==1;EOS=cfg['eos_token_id'];assert EOS==1
 # Real deployment mode: no unrelated flags/quantization experiments in this comparison.
 g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=16;g['precision_mode']=5
 for name,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',16)]:check(getattr(lib,name)(value))
 g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16';g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'
 for name in ['deep_model_quant1_mode_r8886','deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_sink_broadcast_mode_r9040','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083','deep_mode1_sink_r9207']:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_int
 for name,n in [('deep_reference_packed_batch_counts_r8996b',7),('deep_sink_broadcast_counts_r9040',3),('deep_scalar_cast_counts_r9054',3),('deep_mode1_sink_counts_r9207',5),('deep_prefill_pack_counts_r8767',12),('modern_deep_precision_stats_r3443',6)]:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];f.restype=ctypes.c_int
 lib.deep_packed_varlen_capability_r9101b.argtypes=[];lib.deep_packed_varlen_capability_r9101b.restype=ctypes.c_int;assert lib.deep_packed_varlen_capability_r9101b()==1
 def counters(name,n):
  values=[]
  for d in range(8):
   q=(ctypes.c_uint64*n)();check(getattr(lib,name)(d,q,n));values.append(list(q))
  return values
 def select(mode,sinkarm):
  for name,value in [('deep_model_quant1_mode_r8886',0),('deep_latent_prefill_mode_r8933',0),('deep_reference_scratch_mode_r8953',1),('deep_reference_packed_batch_mode_r8996b',mode),('deep_sink_broadcast_mode_r9040',1),('deep_scalar_cast_mode_r9054',1),('deep_values_reuse_mode_r9083',0),('deep_mode1_sink_r9207',sinkarm)]:check(getattr(lib,name)(value))
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(path)

 bind_inner(lib);check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0));expected_kinds=native_expected_kinds(cfg);report['decode_profiles']=[];report['prefill_diagnostics']=[];captured=set()
 report.update(scope='One resident TP8PP1 8DCU, fresh disjoint realGSM8K64 custom3shot. Same packed prefill micro4 in both arms; decode fixed rows vs retire EOS rows. B32 AB_BA are closedcohorts, not HTTP or continuous serving.',quality_trials=[],grid_complete=False,performance_qualified=False,quality_noninferiority_proven=False,goal_complete=False,no_historical_token_identity_gate=True,quality_dataset=prepared['template_provenance'])
 deadline=g['packet_started']+2200
 def generate(items,compact,label,capture_owner=False,sinkarm=1):
  select(1,sinkarm);began=time.perf_counter();check(lib.modern_native_engine_reset(ctx));B=len(items);assert B==1
  prompts=[v['input_ids']for v in items];caps=[min(512,int(v.get('max_new_tokens',512)))for v in items];generated=[[]for _ in items];done=[False]*B;reasons=[None]*B;windows=[];events=[[]for _ in items];budget=False
  def forward(rows,seq,past,prefill):
   nonlocal budget
   if time.perf_counter()>deadline-120:budget=True;return None
   A=len(rows);sizes=list(map(len,seq));assert A and len(set(rows))==A and all(0<=x<B for x in rows);assert all(p+n<=1024 for p,n in zip(past,sizes))
   off=[0]+np.cumsum(sizes).tolist();pos=sum([list(range(p,p+n))for p,n in zip(past,sizes)],[])
   arrays=[np.ascontiguousarray(v,dtype=np.int64 if j>=4 else np.int32)for j,v in enumerate([rows,past,[p+n for p,n in zip(past,sizes)],off,sum(seq,[]),pos])];inputs=[v.copy()for v in arrays]
   outback=np.full(A*cfg['vocab_size']+64,0x7fc0,np.uint16);outback[-64:]=0x1234;out=outback[:-64].reshape(A,cfg['vocab_size']);ib=np.full(A+8,-999,np.int64);ids=ib[:A]
   g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16'if not prefill and 16<A<=64 else'0'
   capture_kind=None
   if capture_owner and not prefill:
    assert A==1 and sum(sizes)==1
    if len(generated[rows[0]])==8 and label not in captured:capture_kind=label
   geometry={'tokens':sum(sizes),'requests':A,'rows':rows.copy(),'past':past.copy(),'sizes':sizes,'capture':capture_kind}
   report['active_call_packet']=dict(geometry,profile_expected=bool(capture_kind));save()
   before=counters('deep_reference_packed_batch_counts_r8996b',7)if prefill else None
   sink_before=counters('deep_mode1_sink_counts_r9207',5)if prefill else None
   def snapshot():return {'fused':[int(lib.modern_deep_latent_fused_calls_r681(d))for d in range(8)],'reference':[int(lib.modern_deep_latent_fallback_calls_r681(d))for d in range(8)],'MoE':counters('deep_prefill_pack_counts_r8767',12),'precision':counters('modern_deep_precision_stats_r3443',6),'FP32':[int(lib.modern_deep_FP32_calls_r3597(d))for d in range(8)]}
   actual_before=snapshot()if capture_kind else None
   call=lambda:lib.modern_native_engine_forward(ctx,sum(sizes),A,*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data))
   if capture_kind:
    prefix=path.parent/(path.stem+'-'+capture_kind)
    try:rc,ended,tick,profile=inner_owner_forward_once(lib,call,check,1,expected_kinds,geometry,prefix)
    except(OwnerWindowStillArmed,OwnerObservationFailed)as e:
     report['decode_profiles'].append(dict(e.profile,label=label,capture=capture_kind));report['owner_capture_complete']=False;save();raise
    after=snapshot();deltas={k:[[v-u for u,v in zip(x,y)]for x,y in zip(actual_before[k],after[k])]if k in ['MoE','precision']else[v-u for u,v in zip(actual_before[k],after[k])]for k in actual_before}
    profile.update(label=label,capture=capture_kind,actual_counter_before=actual_before,actual_counter_after=after,actual_counter_delta=deltas,policy5={'requested':5,'actual_attention_FP32_SUM_calls':deltas['FP32'],'source':'kind0 BF16 attention SUM -> CastF32/allreduce/CastBF16/originalsync. Otherkindsmode5alsosync; totalreducecalls fromprecisiondelta[0]. NoF64.','cost_boundary':'Inner output_projection_collective combines projection+casts+collective+sync; individualF32comm/syncwall UNKNOWN. OldMoE TPspan separatelyavailable, includesstreamwait; notpureNICtime.','precision_peak_field':'Stats[5] ismonotonic historicalmax includingPF, deltaNOTdecodepeak.'})
    profile['C1_actual_attention_contract']={'tokens':1,'requests':1,'all43_reference_attention_per_owner':deltas['reference'],'FP32_batch_attention_calls':deltas['FP32'],'SWA_HCA_CSA_layers_per_owner':[2,20,21],'source':'Original mode5 C1 does not enter batched FP32 attention SUM path; no new precision candidate'}
    report['decode_profiles'].append(profile);captured.add(capture_kind);save()
    assert deltas['fused']==([43]*8 if A>=8 else[0]*8)and deltas['reference']==([0]*8 if A>=8 else[43]*8),'actualfusedreferencebranchdiffers'
    assert deltas['FP32']==[0]*8,'B1mustretainoriginalnoFP32batchattentionpath'
    assert all(profile['kinds'][str(k)]['layers_per_device']==[n]*8 for k,n in [(0,2),(1,20),(2,21)]),'all43actualattentionframes'
   else:tick=time.perf_counter();rc=call();ended=time.perf_counter();check(rc)
   assert np.all(outback[-64:]==0x1234)and np.all(ib[A:]==-999)and all(np.array_equal(a,b)for a,b in zip(arrays,inputs))
   logits=(out.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(logits).all()and np.array_equal(logits.argmax(-1),ids);assert np.all((ids>=0)&(ids<cfg['vocab_size']))
   delta=None
   if prefill:
    after=counters('deep_reference_packed_batch_counts_r8996b',7);delta=[[y-x for x,y in zip(a,b)]for a,b in zip(before,after)];assert 64<sum(sizes)<=2048 and all(v[0]+v[1]==43 and v[0]>0 and v[3]>0 and v[4]>0 for v in delta)
   sink_delta=None
   if prefill:
    sink_after=counters('deep_mode1_sink_counts_r9207',5);sink_delta=[[y-x for x,y in zip(a,b)]for a,b in zip(sink_before,sink_after)];assert all(v[2]>0 and v[3]>0 and v[0]==v[1]==0 for v in sink_delta)
   useful=0
   for j,i in enumerate(rows):
    if done[i]:continue
    token=int(ids[j]);generated[i].append(token);eos=token==EOS;events[i].append(dict(time=ended,kind='EOS'if eos else'committed',visible=not eos,token_id=token));useful+=int(not eos)
    if eos:done[i]=True;reasons[i]='EOS'
    elif len(generated[i])>=caps[i]:done[i]=True;reasons[i]='max_new_tokens'
   windows.append(dict(prefill=prefill,rows=rows.copy(),physical_rows=A,active_useful_tokens=useful,seconds=ended-tick,return_monotonic=ended,packed_delta=delta,mode1_sink_delta=sink_delta,profiled=bool(capture_kind),performance_eligible=not bool(capture_kind),decode_index_before_call=len(generated[0])-1 if not prefill else None));report['active_call_packet']=None;return True
  # Stage all requests with a proven <=2048-token prefill. Same prefill schedule in both arms.
  for start in range(0,B,4):
   rows=list(range(start,min(B,start+4)))
   if forward(rows,[prompts[i]for i in rows],[0]*len(rows),True)is None:break
   report['live_generation']={'label':label,'phase':'prefill','admitted_rows':min(B,start+4)};save()
  for step in range(1,max(caps)):
   if budget or all(done):break
   rows=[i for i in range(B)if not done[i]]if compact else list(range(B));seq=[[generated[i][-1]if not done[i]else EOS]for i in rows];past=[len(prompts[i])+step-1 for i in rows]
   if forward(rows,seq,past,False)is None:break
   if step%32==0:
    assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=512*1024**2;report['live_generation']={'label':label,'step':step,'active':sum(not x for x in done),'last_physical_rows':len(rows),'generated_lengths':[len(v)for v in generated]};save()
  ended=time.perf_counter();records=[];requests=[]
  for i,item in enumerate(items):
   reason=reasons[i]or'wall_budget';text=tokenizer.decode(generated[i],skip_special_tokens=True).strip();request={'id':item['id'],'arrival':began,'status':'unfinished'if reason=='wall_budget'else'completed','stop_reason':reason,'events':events[i]};requests.append(request);records.append(dict(id=item['id'],generated_ids=generated[i],text=text,stop_reason=reason,input_tokens=len(prompts[i]),score=score_case(gold[item['id']],text,reason),logical_row=i))
  pre=sum(w['seconds']for w in windows if w['prefill']);dec=sum(w['seconds']for w in windows if not w['prefill']);useful=sum(w['active_useful_tokens']for w in windows if not w['prefill']);physical=sum(w['physical_rows']for w in windows if not w['prefill']);outs=sum(w['active_useful_tokens']for w in windows)
  pf=[w for w in windows if w['prefill']];dw=[w for w in windows if not w['prefill']];anchor=max((w['return_monotonic']for w in pf),default=ended);common=max((w['return_monotonic']for w in dw),default=anchor)-anchor
  return dict(common_decode_window_seconds=common,common_decode_useful_tok_s=useful/common if common>0 else None,common_decode_tok_s_per_card=useful/common/8 if common>0 else None,label=label,B=B,sinkarm=sinkarm,compact_EOS_rows=compact,records=records,requests=requests,windows=windows,native_prefill_seconds=pre,native_decode_seconds=dec,prefill_completed_input_tokens=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill']),native_prefill_input_tok_s=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill'])/pre if pre else None,native_decode_useful_tok_s=useful/dec if dec else None,native_decode_useful_tokens=useful,physical_decode_row_steps=physical,driver_wall_seconds=ended-began,driver_output_tok_s=outs/(ended-began),serving_metrics=serving_metrics(requests,began,ended,8),budget_stop=budget,finite_readonly_canary_all_calls=True,all_prefill_packed_hit=True,clock_scope='Single native CAPI return emitter; same closedcohort arrival before reset, includes allprefill staging inTTFT, notHTTP or continuous service')
 report.update(scope='Current ac937 TP8PP1 C1 realQA attribution; 4 fixed questions sequentially, each naturalEOS/cap512; one actual T=B=1 decode8 capture and adjacent OFF7/OFF9. No code or quant candidate; no cleanTPS promotion.',matrix=[1],prefill_microbatch=1,profile_mode_default_OFF=True,owner_capture_complete=False,instrumented_observation_only=True,selected_ids=[x['id']for x in cases],planned_capture_decode_index=8,actual_Native_support='NativeFrame nonzero tokens/count; real344rows/43layer geometry and counters required per capture')
 save()
 for index,item in enumerate(cases):
  label='C1-question%d'%index
  entry=generate([item],True,label,True,1)
  entry['instrumented_observation_only']=True;entry['eligible_for_clean_performance']=False
  dw=[w for w in entry['windows']if not w['prefill']]
  prof=[j for j,w in enumerate(dw)if w['profiled']]
  entry['adjacent_OFF_PROFILE_OFF']=None
  if len(prof)==1:
   j=prof[0]
   if 0<j<len(dw)-1:
    trio=dw[j-1:j+2];assert [w['profiled']for w in trio]==[False,True,False]
    entry['adjacent_OFF_PROFILE_OFF']={'seconds':[w['seconds']for w in trio],'decode_indices':[w['decode_index_before_call']for w in trio],'scope':'Neighbor actual autoregressive positions differ by1; observer perturbation diagnostic, not same-state optimization ABBA','OFF_endpoint_drift':abs(trio[2]['seconds']/trio[0]['seconds']-1),'profile_to_OFFmean':trio[1]['seconds']/((trio[0]['seconds']+trio[2]['seconds'])/2)}
  report['quality_trials'].append(entry);save()
  if entry['budget_stop']:break
 scores=[v['score']for t in report['quality_trials']for v in t['records']]
 report['grid']={'1':{'cases':len(scores),'task_correct':sum(v['flexible_correct']for v in scores),'natural_EOS':sum(v['stop_reason']=='EOS'for t in report['quality_trials']for v in t['records']),'caps':sum(v['stop_reason']=='max_new_tokens'for t in report['quality_trials']for v in t['records']),'complete':len(scores)==4 and not any(t['budget_stop']for t in report['quality_trials'])}}
 report['grid_complete']=report['grid']['1']['complete'];report['performance_qualified']=False;report['quality_noninferiority_proven']=False
 report['owner_capture_complete']=len(captured)==4 and len(report['decode_profiles'])==4 and all(v.get('inner_event_validation_passed')and v.get('native_window_closed')and v['geometry']['tokens']==v['geometry']['requests']==1 for v in report['decode_profiles'])
 report['outcome']='C1_ATTRIBUTION_COMPLETE_NO_OPTIMIZATION_CLAIM'if report['owner_capture_complete']and report['grid_complete']else'PARTIAL_C1_PROFILE_OR_BUDGET'
 check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0));check(lib.modern_deep_c4_smallt_mode_r9252(0));save()
