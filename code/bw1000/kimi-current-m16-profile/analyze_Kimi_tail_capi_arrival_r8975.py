"""Reuse completed639 trace; match only complete middle CAPI windows."""
from pathlib import Path
import bisect,hashlib,json,statistics
r=Path.home()/'hygon-inference-refactor-20261014';run='currentprofile8670-2780c25f37791711'
p=r/'results'/f'kimi-current-profile-r8670-{run}-node3.json';raw=p.read_bytes();q=json.loads(raw)
assert q['completed'] and q['all8_owned_workers_shutdown_completed']
assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
w=q['profiling_windows'][0];lo=w['start_realtime_ns'];hi=w['stop_realtime_ns']
calls=[v for v in q['local_forward_samples'] if v['N']==16 and v['B']==16 and lo<=v['CAPI_start_realtime_ns']<v['CAPI_end_realtime_ns']<=hi]
assert len(calls)==8,('complete_trace_CAPI_count',len(calls))
calls=calls[1:-1];events={i:[] for i in range(8)};dec=json.JSONDecoder();files=[]
for f in sorted((r/'results'/f'kimi-current-profile-trace-r8670-{run}-node3').glob('trace*.json')):
 files.append(dict(name=f.name,bytes=f.stat().st_size,SHA256=hashlib.sha256(f.read_bytes()).hexdigest()))
 for line in f.open():
  t=line.strip()
  if t.startswith('{"traceEvents":['):t=t[len('{"traceEvents":['):]
  if not t.startswith('{'):continue
  try:e,_=dec.raw_decode(t)
  except json.JSONDecodeError:continue
  if e.get('ph')!='X' or e.get('cat')!='HIPOPS':continue
  name=e.get('name','');kind='W2' if 'staticW2R326U' in name else 'W13' if 'groupedW134531' in name else 'AR' if 'AllReduce' in name else None
  if kind:
   a=e['args'];events[int(a['devId'])].append((int(a['BeginNs']),int(a['EndNs']),kind))
for v in events.values():v.sort()
def stat(xs):
 xs=sorted(xs)
 return dict(n=len(xs),min=xs[0],median=statistics.median(xs),mean=statistics.mean(xs),p95=xs[int((len(xs)-1)*.95)],max=xs[-1]) if xs else None
out=dict(CPU_only=True,GPU=False,native_build=False,Model_load=False,source_report_SHA256=hashlib.sha256(raw).hexdigest(),source_trace_files=files,actual_original_handle='910375.639',same_host_node3_only=True,complete_middle_CAPI_windows=[],matched_groups=[],performance_gain='NONE; retained trace diagnosis, no perturbation or performance round')
for c in calls:
 a,b=c['CAPI_start_realtime_ns'],c['CAPI_end_realtime_ns'];matched={};counts={}
 for d,v in events.items():
  local=[x for x in v if a<=x[0]<=x[1]<=b];ws=[x for x in local if x[2]=='W2'];ars=[x for x in local if x[2]=='AR'];starts=[x[0]for x in ars];w13=[x[0]for x in local if x[2]=='W13'];pairs=[]
  for x,y,_ in ws:
   i=bisect.bisect_left(starts,y-1000);j=bisect.bisect_right(w13,x)
   if i<len(ars) and (j==len(w13) or ars[i][0]<w13[j]):pairs.append((x,y,*ars[i][:2]))
  matched[d]=pairs;counts[d]=dict(W2=len(ws),paired_next_AR=len(pairs))
 ok=len({len(v)for v in matched.values()})==1 and all(len(matched[d])==counts[d]['W2']>0 for d in range(8))
 out['complete_middle_CAPI_windows'].append(dict(label=c['label'],counts=counts,matched=ok))
 if not ok:continue
 for j in range(len(matched[0])):
  z=[matched[d][j]for d in range(8)];arrival=[x[1]for x in z];start=[x[2]for x in z];end=[x[3]for x in z];dur=[x[3]-x[2]for x in z];early=arrival.index(min(arrival));late=arrival.index(max(arrival))
  wd=[x[1]-x[0]for x in z];wb=[x[0]for x in z]
  out['matched_groups'].append(dict(call=c['label'],MoE_ordinal=j,W2_start_spread_ms=(max(wb)-min(wb))/1e6,W2_end_spread_ms=(max(arrival)-min(arrival))/1e6,W2_duration_spread_ms=(max(wd)-min(wd))/1e6,AR_start_spread_ms=(max(start)-min(start))/1e6,AR_end_spread_ms=(max(end)-min(end))/1e6,AR_max_duration_ms=max(dur)/1e6,W2_max_duration_ms=max(wd)/1e6,earliest_W2_rank_AR_duration_ms=dur[early]/1e6,latest_W2_rank_AR_duration_ms=dur[late]/1e6,W2_to_AR_submit_gap_max_ms=max(x[2]-x[1]for x in z)/1e6,earliest_W2_rank=early,latest_W2_rank=late))
out['all_six_windows_matched']=all(v['matched']for v in out['complete_middle_CAPI_windows'])
if out['all_six_windows_matched']:
 out['statistics_ms']={k:stat([z[k]for z in out['matched_groups']])for k in out['matched_groups'][0]if k.endswith('_ms')}
 out['latest_W2_rank_counts']={d:sum(x['latest_W2_rank']==d for x in out['matched_groups'])for d in range(8)}
 out['decision']='MATCHED_DIAGNOSTIC_ONLY: compare arrival/start/end spread; no pure network cost or removable gain claim'
else:out['decision']='UNKNOWN: window matching still unsupported; stop, no GPU/profile/reload'
out['limits']='Samehost HIPOPS converted clocks only; CCL may include peer wait, host enqueue and queue/dependencies. Source8171 W2->routedAR->norm supports identification but no dedicated runtime label. No crosshost clocks, rank summation, GPUbusy or causal speedup claim.'
print(json.dumps(out))
