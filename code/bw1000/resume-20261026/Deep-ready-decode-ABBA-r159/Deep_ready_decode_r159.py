"""Real heldout-task quality; no token-equality acceptance gate. Single resident model."""
import ctypes,hashlib,json,time,statistics,pathlib
import numpy as np
from Deep_eos_quality_metrics_r9148 import score_case,paired_quality,serving_metrics
from deep_owner_consumer_r9254 import bind_inner,inner_owner_forward_once,OwnerWindowStillArmed,native_expected_kinds

def ready_rows_before_next_prefill(total,admitted,done):
 return []if admitted>=total else[i for i in range(admitted)if not done[i]]
def per_row_decode_past(prompt_lengths,generated,rows):
 return[prompt_lengths[i]+len(generated[i])-1 for i in rows]

def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 manifest=g['manifest'];tp=r/manifest['tokenized_path'];assert hashlib.sha256(tp.read_bytes()).hexdigest()==manifest['existing_remote_SHA256'][manifest['tokenized_path']]
 prepared=json.loads(tp.read_bytes());dataset=json.loads(pathlib.Path(__file__).with_name('dataset64.json').read_bytes());gold={v['id']:v for v in dataset['cases']};replica=g['replica'];config=json.loads(pathlib.Path(__file__).with_name('config.json').read_bytes());rcfg=config['replicas'][replica];planned=[v['id']for v in dataset['cases']];all_cases={v['id']:v for v in prepared['cases']};cases=[all_cases[i]for i in planned if i in all_cases];rejected=[v for v in prepared['skipped']if v['id']in planned];assert len(planned)==64 and len(cases)+len(rejected)==64 and len(gold)==64;assert max((v['input_tokens']for v in cases),default=0)<=512;assert prepared['shot_count']==3
 tokenizer=g['tokenizer'];assert tokenizer.bos_token_id==0 and tokenizer.eos_token_id==1;EOS=cfg['eos_token_id'];assert EOS==1
 # Real deployment mode: no unrelated flags/quantization experiments in this comparison.
 g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=16;g['precision_mode']=5
 for name,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',16)]:check(getattr(lib,name)(value))
 g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16';g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'
 for name in ['deep_model_quant1_mode_r8886','deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_sink_broadcast_mode_r9040','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083','deep_mode1_sink_r9207']:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_int
 lib.modern_deep_c4_smallt_mode_r9252.argtypes=[ctypes.c_int];lib.modern_deep_c4_smallt_mode_r9252.restype=ctypes.c_int;lib.modern_deep_c4_smallt_stats_r9252.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.modern_deep_c4_smallt_stats_r9252.restype=ctypes.c_int
 for name,n in [('deep_reference_packed_batch_counts_r8996b',7),('deep_sink_broadcast_counts_r9040',3),('deep_scalar_cast_counts_r9054',3),('deep_mode1_sink_counts_r9207',5)]:
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

 csa_layers=sum(int(v)==4 for v in cfg['compress_ratios'][:cfg['num_hidden_layers']]);assert csa_layers==21
 bind_inner(lib);check(lib.deep_owner_arm_r8900(0));check(lib.deep_attention_inner_arm_r9193(0));expected_kinds=native_expected_kinds(cfg);report['owner_prefill_profiles']=[];report['prefill_diagnostics']=[]
 report.update(scope='One resident TP8PP1 8DCU, fresh disjoint realGSM8K64 custom3shot. Same packed prefill micro4 in both arms; decode fixed rows vs retire EOS rows. B32 AB_BA are closedcohorts, not HTTP or continuous serving.',quality_trials=[],grid_complete=False,performance_qualified=False,quality_noninferiority_proven=False,goal_complete=False,no_historical_token_identity_gate=True,quality_dataset=prepared['template_provenance'])
 deadline=g['packet_started']+3300
 def generate(items,compact,label,capture_owner=False,sinkarm=1,smallarm=0,ready_decode=False):
  select(1,sinkarm);check(lib.modern_deep_c4_smallt_mode_r9252(smallarm));began=time.perf_counter();check(lib.modern_native_engine_reset(ctx));B=len(items);assert 1<=B<=32
  prompts=[v['input_ids']for v in items];caps=[min(512,int(v.get('max_new_tokens',512)))for v in items];generated=[[]for _ in items];done=[False]*B;reasons=[None]*B;windows=[];events=[[]for _ in items];budget=False
  def forward(rows,seq,past,prefill):
   nonlocal budget
   if time.perf_counter()>deadline-120:budget=True;return None
   A=len(rows);sizes=list(map(len,seq));assert A and len(set(rows))==A and all(0<=x<B for x in rows);assert all(p+n<=1024 for p,n in zip(past,sizes))
   off=[0]+np.cumsum(sizes).tolist();pos=sum([list(range(p,p+n))for p,n in zip(past,sizes)],[])
   arrays=[np.ascontiguousarray(v,dtype=np.int64 if j>=4 else np.int32)for j,v in enumerate([rows,past,[p+n for p,n in zip(past,sizes)],off,sum(seq,[]),pos])];inputs=[v.copy()for v in arrays]
   outback=np.full(A*cfg['vocab_size']+64,0x7fc0,np.uint16);outback[-64:]=0x1234;out=outback[:-64].reshape(A,cfg['vocab_size']);ib=np.full(A+8,-999,np.int64);ids=ib[:A]
   g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16'if not prefill and 16<A<=64 else'0'
   short_before=counters('modern_deep_c4_smallt_stats_r9252',8);f32_before=[int(lib.modern_deep_FP32_calls_r3597(d))for d in range(8)];sink_before=counters('deep_mode1_sink_counts_r9207',5);before=counters('deep_reference_packed_batch_counts_r8996b',7)if prefill else None
   call=lambda:lib.modern_native_engine_forward(ctx,sum(sizes),A,*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data))
   if capture_owner:
    assert prefill and B==4 and A==4 and not windows and sum(sizes)==1681,'owner capture must cover exactly one PF call'
    try:rc,ended,tick,profile=inner_owner_forward_once(lib,call,check,1,expected_kinds)
    except OwnerWindowStillArmed as e:
     report['owner_prefill_profiles'].append(dict(e.profile,label=label));report['owner_capture_complete']=False;save();raise
    profile['label']=label;profile['input_tokens']=sum(sizes);report['owner_prefill_profiles'].append(profile)
   else:tick=time.perf_counter();rc=call();ended=time.perf_counter();check(rc)
   assert np.all(outback[-64:]==0x1234)and np.all(ib[A:]==-999)and all(np.array_equal(a,b)for a,b in zip(arrays,inputs))
   logits=(out.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(logits).all()and np.array_equal(logits.argmax(-1),ids);assert np.all((ids>=0)&(ids<cfg['vocab_size']))
   delta=None
   if prefill:
    after=counters('deep_reference_packed_batch_counts_r8996b',7);delta=[[y-x for x,y in zip(a,b)]for a,b in zip(before,after)];assert 64<sum(sizes)<=2048 and all(v[0]+v[1]==43 and v[0]>0 and v[3]>0 and v[4]>0 for v in delta)
   sink_after=counters('deep_mode1_sink_counts_r9207',5);sink_delta=[[y-x for x,y in zip(a,b)]for a,b in zip(sink_before,sink_after)]
   if prefill:
    assert all(v[2]>0 and v[3]>0 and v[0]==v[1]==0 for v in sink_delta)if sinkarm else all(v[0]>0 and v[1]>0 and v[2]==v[3]==v[4]==0 for v in sink_delta)
   else:assert all(not any(v)for v in sink_delta),'PF-only candidate unexpectedly selected decode'
   short_after=counters('modern_deep_c4_smallt_stats_r9252',8);short_delta=[[y-x for x,y in zip(a[:4],b[:4])]for a,b in zip(short_before,short_after)];short_eligible=not prefill and A<=7;expected_short=[csa_layers,csa_layers,csa_layers*smallarm,csa_layers*(1-smallarm)]if short_eligible else[0]*4;assert short_delta==[expected_short]*8,('actualC4shortbranch',A,prefill,smallarm,short_delta);f32_delta=[int(lib.modern_deep_FP32_calls_r3597(d))-f32_before[d]for d in range(8)];assert f32_delta==[43 if A>1 else 0]*8,('B1mode0stillSUM_countercontract',A,f32_delta)
   useful=0
   for j,i in enumerate(rows):
    if done[i]:continue
    token=int(ids[j]);generated[i].append(token);eos=token==EOS;events[i].append(dict(time=ended,kind='EOS'if eos else'committed',visible=not eos,token_id=token,prefill=prefill));useful+=int(not eos)
    if eos:done[i]=True;reasons[i]='EOS'
    elif len(generated[i])>=caps[i]:done[i]=True;reasons[i]='max_new_tokens'
   windows.append(dict(prefill=prefill,rows=rows.copy(),physical_rows=A,active_useful_tokens=useful,seconds=ended-tick,return_monotonic=ended,packed_delta=delta,mode1_sink_delta=sink_delta,c4_short_per_owner_delta=expected_short,all8_C4short_verified=True,fp32_attention_per_owner_delta=f32_delta[0],all8_precision_contract_verified=True));return True
  # Stage all requests with a proven <=2048-token prefill. Same prefill schedule in both arms.
  for start in range(0,B,4):
   rows=list(range(start,min(B,start+4)))
   if forward(rows,[prompts[i]for i in rows],[0]*len(rows),True)is None:break
   report['live_generation']={'label':label,'phase':'prefill','admitted_rows':min(B,start+4)};save()
   if ready_decode:
    admitted=min(B,start+4);ready=ready_rows_before_next_prefill(B,admitted,done)
    if ready:
     seq=[[generated[i][-1]]for i in ready];past=per_row_decode_past([len(v)for v in prompts],generated,ready)
     if forward(ready,seq,past,False)is None:break
     windows[-1]['early_ready_decode']=True;report['live_generation']={'label':label,'phase':'early_ready_decode','admitted_rows':admitted,'physical_rows':len(ready)};save()
  for step in range(1,max(caps)):
   if budget or all(done):break
   rows=[i for i in range(B)if not done[i]]if compact else list(range(B));seq=[[generated[i][-1]if not done[i]else EOS]for i in rows];past=per_row_decode_past([len(v)for v in prompts],generated,rows)
   if forward(rows,seq,past,False)is None:break
   if step%32==0:
    assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=512*1024**2;report['live_generation']={'label':label,'step':step,'active':sum(not x for x in done),'last_physical_rows':len(rows),'generated_lengths':[len(v)for v in generated]};save()
  ended=time.perf_counter();records=[];requests=[]
  for i,item in enumerate(items):
   reason=reasons[i]or'wall_budget';text=tokenizer.decode(generated[i],skip_special_tokens=True).strip();request={'id':item['id'],'arrival':began,'status':'unfinished'if reason=='wall_budget'else'completed','stop_reason':reason,'events':events[i]};requests.append(request);records.append(dict(id=item['id'],generated_ids=generated[i],text=text,stop_reason=reason,input_tokens=len(prompts[i]),score=score_case(gold[item['id']],text,reason),logical_row=i))
  per_user=[]
  for rec,req in zip(records,requests):
   stamps=[e['time']for e in req['events']if e['visible']and e['kind']=='committed'];rate=(len(stamps)-1)/(stamps[-1]-stamps[0])if len(stamps)>1 and stamps[-1]>stamps[0]else None;natural=req['stop_reason']=='EOS';correct=bool(rec['score']['flexible_correct']);per_user.append({'id':rec['id'],'visible_output_tokens':len(stamps),'first_visible_monotonic':stamps[0]if stamps else None,'last_visible_monotonic':stamps[-1]if stamps else None,'TTFT_seconds':stamps[0]-req['arrival']if stamps else None,'first_to_last_decode_tok_s':rate,'first_to_second_visible_seconds':stamps[1]-stamps[0]if len(stamps)>1 else None,'first_visible_token_source':'prefill'if stamps else None,'ITL_seconds':[y-x for x,y in zip(stamps,stamps[1:])],'stop_reason':req['stop_reason'],'answer_correct':correct,'natural_EOS_and_correct':natural and correct,'strict_correct_EOS_speed15':natural and correct and rate is not None and rate>15})
  pre=sum(w['seconds']for w in windows if w['prefill']);dec=sum(w['seconds']for w in windows if not w['prefill']);useful=sum(w['active_useful_tokens']for w in windows if not w['prefill']);physical=sum(w['physical_rows']for w in windows if not w['prefill']);outs=sum(w['active_useful_tokens']for w in windows)
  pf=[w for w in windows if w['prefill']];dw=[w for w in windows if not w['prefill']];anchor=max((w['return_monotonic']for w in pf),default=ended);common=max(0.,max((w['return_monotonic']for w in dw),default=anchor)-anchor);post_useful=sum(w['active_useful_tokens']for w in dw if w['return_monotonic']>anchor);early_calls=sum(bool(w.get('early_ready_decode'))for w in dw);assert early_calls<=max(0,(B+3)//4-1)
  return dict(ready_decode=ready_decode,early_ready_decode_calls=early_calls,first_generated_token_source='prefill_not_decode',common_decode_useful_tokens=post_useful,common_decode_clock_scope='Only decode emissions strictly after final prefill; early ready-decode tokens excluded from this numerator',common_decode_window_seconds=common,common_decode_useful_tok_s=post_useful/common if common>0 else None,common_decode_tok_s_per_card=post_useful/common/8 if common>0 else None,label=label,B=B,sinkarm=sinkarm,smallt_arm=smallarm,user_metrics=per_user,compact_EOS_rows=compact,records=records,requests=requests,windows=windows,native_prefill_seconds=pre,native_decode_seconds=dec,prefill_completed_input_tokens=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill']),native_prefill_input_tok_s=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill'])/pre if pre else None,native_decode_useful_tok_s=useful/dec if dec else None,native_decode_useful_tokens=useful,physical_decode_row_steps=physical,driver_wall_seconds=ended-began,driver_output_tok_s=outs/(ended-began),serving_metrics=serving_metrics(requests,began,ended,8),budget_stop=budget,finite_readonly_canary_all_calls=True,all_prefill_packed_hit=True,clock_scope='Single native CAPI return emitter; same closedcohort arrival before reset, includes allprefill staging inTTFT, notHTTP or continuous service')

 def summarize_trials(ts):
  records=[v for t in ts for v in t['records']];users=[v for t in ts for v in t['user_metrics']];wall=sum(t['driver_wall_seconds']for t in ts);common=sum(t['common_decode_window_seconds']for t in ts);native=sum(t['native_decode_seconds']for t in ts);tokens=sum(t['native_decode_useful_tokens']for t in ts);post_tokens=sum(t['common_decode_useful_tokens']for t in ts);pf=sum(t['native_prefill_seconds']for t in ts);correctEOS=sum(v['score']['flexible_correct']and v['stop_reason']=='EOS'for v in records)
  return dict(cases=len(records),ids=[v['id']for v in records],correct=sum(v['score']['flexible_correct']for v in records),wrong=sum(not v['score']['flexible_correct']for v in records),EOS=sum(v['stop_reason']=='EOS'for v in records),nonEOS=sum(v['stop_reason']!='EOS'for v in records),caps=sum(v['stop_reason']=='max_new_tokens'for v in records),parse_failed=sum(v['score']['parse_failed']for v in records),correct_EOS=correctEOS,driver_task_wall_seconds=wall,correct_EOS_requests_per_task_second=correctEOS/wall if wall else None,useful_decode_tokens=post_tokens,native_decode_useful_tokens=tokens,early_ready_decode_tokens=tokens-post_tokens,early_ready_decode_calls=sum(t['early_ready_decode_calls']for t in ts),driver_visible_output_tok_s=sum(sum(w['active_useful_tokens']for w in t['windows'])for t in ts)/wall if wall else None,first_to_second_visible_seconds=[u['first_to_second_visible_seconds']for u in users if u['first_to_second_visible_seconds']is not None],common_decode_window_seconds=common,common_decode_useful_tok_s=post_tokens/common if common else None,native_decode_useful_tok_s=tokens/native if native else None,decode_tok_s_per_card=post_tokens/common/8 if common else None,native_prefill_seconds=pf,prefill_input_tok_s=sum(t['prefill_completed_input_tokens']for t in ts)/pf if pf else None,user_metrics=users,strict_correct_EOS_speed15_good_requests=sum(u['strict_correct_EOS_speed15']for u in users))
 report.update(scope='B32 fixed-cohort ready-decode scheduling ABBA, not dynamic service. First output token is produced during micro4 prefill; one decode of admitted active rows after each nonfinal micro4 in B, at most7 extra decode calls. Same production ac937/native math and naturalEOS quality.',matrix=[32],schedule_order=['A','B','B','A'],planned_ids=planned,admitted_ids=[v['id']for v in cases],bilateral_capacity_rejections=rejected,planned_count=64,admitted_count=len(cases),prefill_microbatch=4,quality_independent_ids=len(cases),quality_noninferiority_proven=False,measurement_trials=[],warmup_trials=[],by_arm={},grid_complete=False,performance_qualified=False,smallt_arm=0,profile_mode_default_OFF=True,metric_warning='Do not divide all early+late decode output tokens by a window starting at final prefill. Primary user clocks include their actual first prefill token and all subsequent waits; total task/output goodput includes early decode costs.')
 save();stop=False
 warm=generate([dict(v,max_new_tokens=2)for v in cases[:4]],True,'ABBA-warmup-cap2',False,1,0,False);warm['diagnostic_only_excluded_from_quality_performance']=True;report['warmup_trials'].append(warm);save()
 if warm['budget_stop']:stop=True
 for group in range(2):
  if stop:break
  items=cases[group*32:(group+1)*32];assert len(items)==32;seen={'A':0,'B':0}
  for order_index,arm in enumerate(['A','B','B','A']):
   repeat=seen[arm];seen[arm]+=1;label='group%d-%s%d-order%d'%(group,arm,repeat,order_index)
   trial=generate(items,True,label,False,1,0,arm=='B');trial.update(arm=arm,cohort_group=group,repeat=repeat,order_index=order_index,phase='ABBA_measure');report['measurement_trials'].append(trial)
   report['by_arm'][arm]=summarize_trials([v for v in report['measurement_trials']if v['arm']==arm]);save()
   if trial['budget_stop']:stop=True;break
 report['grid_complete']=len(report['measurement_trials'])==8 and all(len(report['by_arm'].get(a,{}).get('ids',[]))==128 for a in ['A','B']);report['performance_qualified']=bool(report['grid_complete']and not any(v['budget_stop']for v in report['measurement_trials']));report['outcome']='COMPLETE_ABBA_MEASUREMENT_NOT_AUTOMATIC_PROMOTION'if report['performance_qualified']else'PARTIAL_BUDGET_RETAINED'
 report['paired_quality']=[]
 for group in range(2):
  for repeat in range(2):
   arms={t['arm']:t for t in report['measurement_trials']if t['cohort_group']==group and t['repeat']==repeat}
   if set(arms)=={'A','B'}:
    aa={v['id']:v for v in arms['A']['records']};bb={v['id']:v for v in arms['B']['records']};assert set(aa)==set(bb)
    report['paired_quality'].append({'group':group,'repeat':repeat,'ids':list(aa),'regressions':[i for i in aa if aa[i]['score']['flexible_correct']and not bb[i]['score']['flexible_correct']],'improvements':[i for i in aa if not aa[i]['score']['flexible_correct']and bb[i]['score']['flexible_correct']],'exact_tokens_required':False})
 check(lib.modern_deep_c4_smallt_mode_r9252(0));save()
