"""Pure terminal metadata checks, quality/coverage separate from normal shutdown."""
from pathlib import Path
import json,math,hashlib,sys
from support import sha,NODE,actual,PACKAGE
sys.path.insert(0,str(PACKAGE))
from hc_consume import consume as hc_consume

class TerminalNotVisible(RuntimeError):pass

def collect(output,step,epoch):
 output=Path(output);p=output/'entry-report.json';q=json.loads(p.read_bytes())
 if not q.get('completed'):raise TerminalNotVisible('Original terminal JSON still pending')
 assert q['passed']and q['identity']['actual_step']==step and q['identity']['gpu_start_epoch']==epoch
 assert q['HTTP']==0 and not q['Model_READY']and q['loaded_key_count']==67569 and q['normal_Native_shutdown']and q['engine_worker_shutdown_completed']and q['observed_all8_threads_exited']
 assert q['precision_policy']['policy']==0 and not q['precision_policy']['F64_allowed'];assert all(f>=16*1024**3 for f,t in q['GPU_memory_after_load']);assert len(q['owner_thread_observation']['owners'])==8
 rs=q['requests'];assert len(rs)==5 and[r['index']for r in rs]==list(range(5))and[r['armed']for r in rs]==[False,False,False,True,False]
 count=0;scalars=[]
 for r in rs:
  assert r['prompt_tokens']==8489 and r['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and r['output_budget']==128
  assert r['normal_reset_before']and r['normal_reset_after']and r['forward_attempts']==r['forward_completed']and not r.get('error');count+=r['forward_completed'];ids=r['generated_token_ids'];assert 0<len(ids)<=128 and all(type(t)is int and 0<=t<129280 for t in ids)
  assert (ids[-1]==1 and ids.count(1)==1)if r['natural_EOS']else(1 not in ids and len(ids)==128)
  for key in ['GPU_free_total_before','GPU_free_total_after']:assert len(r[key])==8 and all(f>=16*1024**3 for f,t in r[key])
  assert len(r['HC_owner_counts'])==8 and all(not any(v['fullpre'].values())and v['legacy']['f32_t1']==0 for v in r['HC_owner_counts'])
  for w in r['profile_windows']:
   hc=w['HC_pre_split'];p=actual(hc['packet']['path']);b=p.read_bytes();assert len(b)==hc['packet']['bytes']and hashlib.sha256(b).hexdigest()==hc['packet']['SHA256'];owner=actual(w['scalar_packets']['owner']['path']).read_bytes();checked=hc_consume(b,owner);assert checked['rows']==688 and checked['perstage_perowner_ms']==hc['perstage_perowner_ms'];scalars.append(dict(path=str(p),bytes=len(b),SHA256=sha(p)))
   assert w['native_window_closed']and w['owner_event_validation_passed']and w['inner_event_validation_passed']and w['owner_rows']==w['inner_rows']==344
   for f in w['scalar_packets'].values():
    path=actual(f['path']);assert path.resolve().is_relative_to(output.resolve())and path.stat().st_size==f['bytes']and sha(path)==f['SHA256'];scalars.append(dict(path=str(path),bytes=f['bytes'],SHA256=f['SHA256']))
 assert 45<=count<=680 and len(q['actual_single_CAPI_calls'])==count==q['HC_dispatch_checked_forwards']==q['layout_dispatch_checked_forwards']
 assert all(r['rc']==0 and math.isfinite(r['seconds'])and r['seconds']>=0 for r in q['actual_single_CAPI_calls'])
 samples={};base=0;by_request=[]
 for r in rs:
  n=r['forward_completed'];packet=q['actual_single_CAPI_calls'][base:base+n];assert n==9+len(r['generated_token_ids'])-1;assert [v['tokens']for v in packet[:9]]==[1024]*8+[297];assert all(v['count']==1 for v in packet)and all(v['tokens']==1 for v in packet[9:]);by_request.append(packet);base+=n
 for i in [8,87]:
  observed=[]
  for j in [2,3,4]:
   ids=rs[j]['generated_token_ids']
   if len(ids)>i:
    h=hashlib.sha256(json.dumps(ids[:i],separators=(',',':')).encode()).hexdigest();observed.append(dict(request_index=j,past=8489+i-1,generated_prefix_SHA256=h,CAPI_seconds=by_request[j][9+i-1]['seconds']))
  same=len(observed)==3 and len({v['generated_prefix_SHA256']for v in observed})==1
  samples[str(i)]=dict(actual=observed,same_prompt_and_generated_prefix=same,armed_over_adjacent_OFF_mean_CAPI_ratio=observed[1]['CAPI_seconds']/((observed[0]['CAPI_seconds']+observed[2]['CAPI_seconds'])/2)if same else None,causal_overhead_scope='Matchedstate only; otherwise UNKNOWN, no teacher forcing or replacement')
 return dict(entry_report_SHA256=sha(p),normal_Native_shutdown=True,normal8owner_exit=True,logical_requests=5,Native_CAPI_calls=count,natural_EOS=sum(r['natural_EOS']for r in rs),caps=sum(r['cap']for r in rs),profile_coverage_complete=q['profile_coverage_complete'],actual_profile_windows=len(rs[3]['profile_windows']),scalar_event_packets=scalars,quality_qualification=False,exclusive_GPUbusy=False,HTTP=0,state_checked_sample_CAPI_comparison=samples)
