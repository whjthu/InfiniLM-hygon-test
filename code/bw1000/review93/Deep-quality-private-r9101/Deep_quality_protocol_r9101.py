"""Real heldout-task quality; no token-equality acceptance gate. Single resident model."""
import ctypes,hashlib,json,time,statistics
import numpy as np
from quality_bench_r9100 import score_case,paired_quality,serving_metrics

def run(g):
 r,lib,ctx,cfg,check,report,path=[g[k]for k in ['r','lib','ctx','cfg','check','report','path']]
 manifest=g['manifest'];tp=r/manifest['tokenized_path'];assert hashlib.sha256(tp.read_bytes()).hexdigest()==manifest['existing_remote_SHA256'][manifest['tokenized_path']]
 prepared=json.loads(tp.read_bytes());dataset=json.loads((r/'incoming/quality-bench-gsm8k-r9100.json').read_bytes());gold={v['id']:v for v in dataset['cases']};cases=prepared['cases'];assert len(cases)==32 and len(gold)==32 and {v['id']for v in cases}==set(gold)
 assert prepared['accepted_count']==32 and not prepared['skipped'];assert max(v['input_tokens']for v in cases)<=512
 tokenizer=g['tokenizer'];assert tokenizer.bos_token_id==0 and tokenizer.eos_token_id==1;EOS=cfg['eos_token_id'];assert EOS==1
 # Real deployment mode: no unrelated flags/quantization experiments in this comparison.
 g['layout_mode']=g['HC_mode']=g['tail_SUM_mode']=0;g['projection_tile_mode']=g['c4_tile_mode']=16;g['precision_mode']=5
 for name,value in [('modern_deep_precision_policy_r3443',5),('modern_deep_layout_policy_r7052',0),('deep_hc_canonical_policy_r7130',0),('deep_moe_tail_set_r7211',0),('modern_deep_c4_tile_policy_r5920',16)]:check(getattr(lib,name)(value))
 g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16';g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'
 for name in ['deep_model_quant1_mode_r8886','deep_latent_prefill_mode_r8933','deep_reference_scratch_mode_r8953','deep_reference_packed_batch_mode_r8996b','deep_sink_broadcast_mode_r9040','deep_scalar_cast_mode_r9054','deep_values_reuse_mode_r9083']:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_int
 for name,n in [('deep_reference_packed_batch_counts_r8996b',7),('deep_sink_broadcast_counts_r9040',3),('deep_scalar_cast_counts_r9054',3)]:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];f.restype=ctypes.c_int
 lib.deep_packed_varlen_capability_r9101b.argtypes=[];lib.deep_packed_varlen_capability_r9101b.restype=ctypes.c_int;assert lib.deep_packed_varlen_capability_r9101b()==1
 def counters(name,n):
  values=[]
  for d in range(8):
   q=(ctypes.c_uint64*n)();check(getattr(lib,name)(d,q,n));values.append(list(q))
  return values
 def select(mode):
  for name,value in [('deep_model_quant1_mode_r8886',0),('deep_latent_prefill_mode_r8933',0),('deep_reference_scratch_mode_r8953',1),('deep_reference_packed_batch_mode_r8996b',mode),('deep_sink_broadcast_mode_r9040',1),('deep_scalar_cast_mode_r9054',1),('deep_values_reuse_mode_r9083',0)]:check(getattr(lib,name)(value))
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(path)
 report.update(scope='OneModelweightload TP8PP1/8DCU. OfficialGSM8Ktest32 paired customuniformtrainfewshot; max512 newtokens/realEOS. B<=4 fixedphysicalrows untilcohortends, finishedrowEOSpad excluded. Noexacttokenacceptancegate, noquestiontruncation/padding. NewHOST varleneligibility onlymode1; mode2originalT2048scope retained.',reference_numeric_policy='Userallowsdifferenttokens. Actualtaskscore/basicreasonableoutput. NaN/OOB/state/lifecycle remainerrors. Pilot32cannotestablishnoqualitydrop.',quality_policy='actualfinalnumericanswer withstrict/flexibleparse, truncation/parsefail shown; scoresnotforcedgoldtokenmatch',quality_dataset=dict(source_SHA256=hashlib.sha256((r/'incoming/quality-bench-gsm8k-r9100.json').read_bytes()).hexdigest(),tokenized_SHA256=hashlib.sha256(tp.read_bytes()).hexdigest(),shot_count=prepared.get('shot_count',0),shot_source_SHA256=prepared.get('shot_source_SHA256'),input_lengths=prepared['input_lengths'],coverage=prepared['accepted_count'],skipped=prepared['skipped'],template_provenance=prepared['template_provenance'],rendered_example=prepared['rendered_example'],prompt_policy=prepared['prompt_policy']),quality_trials=[],quality_complete=False,actual_candidate_groups_hit=0,actual_candidate_groups_total=0,packed_counter_observability='ExistingcounterupdatesT>64; testsallprefillT>64; decode<=4 remainsoriginalpath andneverclaimsnewhits.',online_service={'state':'PENDING_AFTER_QUALITY_IF_BUDGET'},performance_qualified=False,performance_adopted=False,goal_complete=False)
 save();deadline=g['packet_started']+2450;first_candidate_verified=False
 def generate(items,mode,label,emit=None):
  nonlocal first_candidate_verified
  assert 0<len(items)<=4;select(mode);began=time.perf_counter();check(lib.modern_native_engine_reset(ctx));prompts=[v['input_ids']for v in items];B=len(items);assert all(0<len(v)<=512 for v in prompts);limit=max(int(v.get('max_new_tokens',512))for v in items);assert 0<limit<=512
  requests=[dict(id=v['id'],arrival=began,status='generating',events=[])for v in items];generated=[[]for _ in items];finished=[False]*B;reasons=[None]*B;cache_text=[[]for _ in items];printed=[0]*B;windows=[];decode_useful=0;visible_useful=0;packed_delta=None;budget_stop=False
  outback=np.full(B*cfg['vocab_size']+64,0x7fc0,np.uint16);outback[-64:]=0x1234;out=outback[:-64].reshape(B,cfg['vocab_size']);idsback=np.full(B+8,-999,np.int64);idsout=idsback[:B]
  def stream_piece(i,token,eos,flush=False):
   if emit is None:return
   if not eos:cache_text[i].append(token)
   text=tokenizer.decode(cache_text[i],skip_special_tokens=True);end=len(text)if eos or flush or text.endswith('\n')else text.rfind(' ')+1;piece=text[printed[i]:end];printed[i]=end;emit(i,int(token),piece,eos=bool(eos))
  for step in range(limit):
   if time.perf_counter()>deadline-90:
    budget_stop=True
    for i in range(B):
     if not finished[i]:reasons[i]='wall_budget';requests[i]['status']='unfinished'
    break
   seq=[prompts[i]if step==0 else[EOS if finished[i]else generated[i][-1]]for i in range(B)];sizes=list(map(len,seq));past=[0 if step==0 else len(prompts[i])+step-1 for i in range(B)];assert all(p+n<=1024 for p,n in zip(past,sizes));positions=sum([list(range(p,p+n))for p,n in zip(past,sizes)],[])
   arrays=[np.ascontiguousarray(v,dtype=np.int64 if j>=4 else np.int32)for j,v in enumerate([list(range(B)),past,[p+n for p,n in zip(past,sizes)],[0]+np.cumsum(sizes).tolist(),sum(seq,[]),positions])];input_copies=[v.copy()for v in arrays]
   if step==0:
    before=counters('deep_reference_packed_batch_counts_r8996b',7);g['checkpoint']('quality_prefill_enter_'+label);report['current_group']={'label':label,'mode':mode,'ids':[v['id']for v in items],'input_tokens':sizes};save()
   tick=time.perf_counter();code=lib.modern_native_engine_forward(ctx,sum(sizes),B,*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(idsout.ctypes.data));ended=time.perf_counter();check(code)
   assert np.all(outback[-64:]==0x1234)and np.all(idsback[B:]==-999),'outputcanary';assert all(np.array_equal(a,b)for a,b in zip(arrays,input_copies)),'readonlyinput'
   values=(out.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all(),'nonfinitelogits';assert np.array_equal(values.argmax(-1),idsout),'nativeCPUargmax';assert np.all((idsout>=0)&(idsout<cfg['vocab_size']));del values
   if step==0:
    after=counters('deep_reference_packed_batch_counts_r8996b',7);packed_delta=[[y-x for x,y in zip(a,b)]for a,b in zip(before,after)];assert sum(sizes)>64
    assert all(v[0]+v[1]==43 for v in packed_delta),('all43prefillobserved',packed_delta)
    if mode==1:
     report['actual_candidate_groups_total']+=1;hit=all(v[0]>0 and v[3]>0 and v[4]>0 for v in packed_delta);report['actual_candidate_groups_hit']+=int(hit)
     if not first_candidate_verified:
      assert hit,('first_actual_mode1_hit_required',sizes,packed_delta);first_candidate_verified=True
    else:
     assert all(v[0]==0 for v in packed_delta)if sum(sizes)!=2048 else True
    g['checkpoint']('quality_prefill_return_'+label);save()
   active_visible=0
   for i in range(B):
    if finished[i]:continue
    token=int(idsout[i]);generated[i].append(token);eos=token==EOS
    if eos:
     finished[i]=True;reasons[i]='EOS';requests[i]['status']='completed';requests[i]['events'].append(dict(time=ended,visible=False,kind='EOS',token_id=token));stream_piece(i,token,True)
    else:
     requests[i]['events'].append(dict(time=ended,visible=True,kind='committed',token_id=token));active_visible+=1;stream_piece(i,token,False,flush=len(generated[i])>=int(items[i].get('max_new_tokens',512)))
     if len(generated[i])>=int(items[i].get('max_new_tokens',512)):
      finished[i]=True;reasons[i]='max_new_tokens';requests[i]['status']='completed'
   visible_useful+=active_visible
   if step>0:decode_useful+=active_visible
   windows.append(dict(step=step,prefill=step==0,physical_rows=B,active_visible_tokens=active_visible,seconds=ended-tick,return_monotonic=ended,finished_rows=sum(finished)))
   if step%32==0:
    assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=512*1024**2;report['live_generation']={'label':label,'step':step,'finished':sum(finished),'generated_lengths':[len(v)for v in generated]};save()
   if all(finished):break
  for i in range(B):
   if reasons[i]is None:reasons[i]='max_new_tokens';requests[i]['status']='completed'
  ended=time.perf_counter();records=[]
  for i,item in enumerate(items):
   text=tokenizer.decode(generated[i],skip_special_tokens=True).strip();row=dict(id=item['id'],input_tokens=len(prompts[i]),generated_ids=generated[i],text=text,stop_reason=reasons[i],events=requests[i]['events'],physical_row=i,request_metrics_source=requests[i],no_reference_token_gate=True)
   if item['id']in gold:row['score']=score_case(gold[item['id']],text,reasons[i])
   records.append(row)
  pre=sum(v['seconds']for v in windows if v['prefill']);dec=sum(v['seconds']for v in windows if not v['prefill']);entry=dict(label=label,mode=mode,records=records,requests=requests,window_start=began,window_end=ended,driver_wall_seconds=ended-began,native_prefill_seconds=pre,native_decode_seconds=dec,prefill_input_tokens=sum(map(len,prompts)),native_prefill_input_tok_s=sum(map(len,prompts))/pre,native_decode_useful_tokens=decode_useful,native_decode_useful_tok_s=decode_useful/dec if dec else None,useful_visible_tokens=visible_useful,native_all_output_tok_s=visible_useful/(pre+dec),driver_all_output_tok_s=visible_useful/(ended-began),serving_metrics=serving_metrics(requests,began,ended,8),native_windows=windows,packed_prefill_counter_delta=packed_delta,budget_stop=budget_stop,finite_readonly_canary_all_calls=True,finished_pad_excluded=True,clock_scope='nativeCAPIreturnemitter;notHTTPclient. All4cohortrequestarrivalsbeforectxreset. Intercohortexperimentsarenot32simultaneoususers.')
  return entry
 order=sorted(cases,key=lambda v:(-v['input_tokens'],v['id']));scores={2:[],1:[]};trials={2:[],1:[]}
 for group in range(0,32,4):
  items=order[group:group+4]
  for mode in ([2,1]if(group//4)%2==0 else[1,2]):
   if time.perf_counter()>deadline-180:report['quality_budget_stop_before_group']=group;save();break
   label=('baseline'if mode==2 else'packed')+'-group'+str(group//4);entry=generate(items,mode,label);report['quality_trials'].append(entry);trials[mode].append(entry);scores[mode].extend(v['score']for v in entry['records']);save()
   if entry['budget_stop']:break
  if report.get('quality_budget_stop_before_group')is not None or any(v['budget_stop']for v in report['quality_trials']):break
 ids2={v['id']for v in scores[2]};ids1={v['id']for v in scores[1]};common=ids2&ids1
 if common:report['paired_quality']=paired_quality([v for v in scores[2]if v['id']in common],[v for v in scores[1]if v['id']in common])
 report['quality_complete']=len(scores[2])==len(scores[1])==32 and not any(v['budget_stop']for v in report['quality_trials']);report['candidate_exercised']=first_candidate_verified;report['all_candidate_groups_hit']=report['actual_candidate_groups_hit']==report['actual_candidate_groups_total']and report['actual_candidate_groups_total']>0
 report['aggregate_native_metrics']={}
 for mode,label in [(2,'baseline'),(1,'packed')]:
  ts=trials[mode];pre=sum(v['native_prefill_seconds']for v in ts);dec=sum(v['native_decode_seconds']for v in ts);outs=sum(v['useful_visible_tokens']for v in ts);useful=sum(v['native_decode_useful_tokens']for v in ts);wall=sum(v['driver_wall_seconds']for v in ts)
  report['aggregate_native_metrics'][label]=dict(groups=len(ts),cases=len(scores[mode]),native_prefill_input_tok_s=sum(v['prefill_input_tokens']for v in ts)/pre if pre else None,native_decode_useful_tok_s=useful/dec if dec else None,decode_useful_tokens=useful,native_decode_seconds=dec,output_useful_tokens=outs,driver_output_tok_s=outs/wall if wall else None,native_output_tok_s=outs/(pre+dec)if pre+dec else None,scope='Pooled8closedB4cohorts. VariableEOS/outputlengths;qualitycomparisonrequired. Notfixed512 nor32concurrentrequests.')
 save()
 # Optional actual localhostHTTP/SSE clients reuse this same resident model.
 remaining=deadline-time.perf_counter()-120
 if report['quality_complete']and remaining>=45:
  from resident_service_r9101 import run_local_service_benchmark
  def run_batch(items,emit):
   local=[]
   from deepseek_v4_reference_template_r18_r2 import apply_reference_fallback,encode_reference_single_user
   apply_reference_fallback(tokenizer)
   for item in items:
    ids=encode_reference_single_user(tokenizer,item['prompt']);assert 0<len(ids)<=512
    local.append(dict(id=item['id'],input_ids=ids,max_new_tokens=min(64,int(item['max_new_tokens']))))
   entry=generate(local,1,'actual-localhost-service',emit);report.setdefault('service_native_cohorts',[]).append(entry);save()
  prompts=[v['question']+' Give a brief answer.'for v in cases[:8]];report['online_service']=run_local_service_benchmark(run_batch,prompts,max_batch=4,max_new_tokens=64,request_rate=.5,deadline_seconds=min(300,remaining));save()
 else:report['online_service']={'state':'SKIPPED','reason':'qualityincomplete_or_remainingbudgetbelow45s','remaining_seconds':remaining};save()
 select(2);report.update(outcome='TASK_QUALITY_PILOT_COMPLETE'if report['quality_complete']else'TASK_QUALITY_PILOT_PARTIAL_BUDGET',quality_noninferiority_proven=False,no_full_benchmark_claim=True,performance_qualified=report['quality_complete']and first_candidate_verified,performance_adopted=False,quality_passed='TASK_SCORES_REPORTED_NOT_TOKEN_EQUALITY',owner_event_capture_enabled=False)
 save()
