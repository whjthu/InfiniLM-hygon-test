"""One CPU backward pass over completed639 node3 trace, no GPU operations."""
from pathlib import Path
import json,hashlib,statistics,collections
r=Path.home()/'hygon-inference-refactor-20261014';run='currentprofile8670-2780c25f37791711';p=r/'results'/f'kimi-current-profile-r8670-{run}-node3.json';raw=p.read_bytes();q=json.loads(raw);assert hashlib.sha256(raw).hexdigest()=='a0f1a014ca13ac2bfa007d1a7e66b74ee3e7e54148f1511cf56165ea54937fa0'
w=q['profiling_windows'][0];calls=[v for v in q['local_forward_samples']if v['N']==v['B']==16 and w['start_realtime_ns']<=v['CAPI_start_realtime_ns']<v['CAPI_end_realtime_ns']<=w['stop_realtime_ns']];assert len(calls)==8;calls=calls[1:-1]
events={d:[]for d in range(8)};dec=json.JSONDecoder();schemas={};files=[]
for f in sorted((r/'results'/f'kimi-current-profile-trace-r8670-{run}-node3').glob('trace*.json')):
 files.append({'file':f.name,'SHA256':hashlib.sha256(f.read_bytes()).hexdigest()})
 for line in f.open():
  s=line.strip()
  if s.startswith('{"traceEvents":['):s=s[len('{"traceEvents":['):]
  if not s.startswith('{'):continue
  try:e,_=dec.raw_decode(s)
  except json.JSONDecodeError:continue
  if e.get('ph')!='X'or e.get('cat')not in('HIPOPS','HIP'):continue
  a=e.get('args',{});cat=e['cat'];schemas.setdefault(cat,sorted(a))
  if cat!='HIPOPS':continue
  if not all(k in a for k in('devId','BeginNs','EndNs')):continue
  events[int(a['devId'])].append((int(a['BeginNs']),int(a['EndNs']),e.get('name',''),a.get('streamId',a.get('stream'))))
for v in events.values():v.sort()
def stat(v):
 v=sorted(v);return {'n':len(v),'median':statistics.median(v),'p95':v[int((len(v)-1)*.95)],'max':v[-1]}
patterns={};bad=[];matched=0;call_counts=[]
for c in calls:
 a,b=c['CAPI_start_realtime_ns'],c['CAPI_end_realtime_ns'];local={d:[e for e in v if a<=e[0]<=e[1]<=b]for d,v in events.items()};pairs={}
 for d,v in local.items():
  loc=[]
  for i,e in enumerate(v):
   if'staticW2R326U'not in e[2]:continue
   wi=[j for j in range(i)if'groupedW134531'in v[j][2]]
   if not wi:continue
   k=wi[-1];prior=[j for j in range(k)if'AllReduce'in v[j][2]]
   if not prior:continue
   loc.append(v[prior[-1]:i+1])
  pairs[d]=loc
 counts={d:len(v)for d,v in pairs.items()};call_counts.append({'call':c['label'],'counts':counts})
 if set(counts.values())!={22}:bad.append({'call':c['label'],'why':'missing W13 or precedingAR','counts':counts});continue
 for j in range(22):
  chains=[pairs[d][j]for d in range(8)];names=[tuple(e[2]for e in z)for z in chains]
  if len(set(names))!=1:bad.append({'call':c['label'],'ordinal':j,'why':'unmatched kernel sequence','lengths':[len(z)for z in chains]});continue
  matched+=1;key=names[0];rows=patterns.setdefault(key,[]);measures=[]
  # Baseline is the preceding collective's END, not its peer-wait interval.
  prev=[z[0][1]for z in chains];measures.append({'boundary':'precedingAR.end','spread_ms':(max(prev)-min(prev))/1e6,'dependency_gap_max_ms':0.0})
  for k in range(1,len(key)):
   starts=[z[k][0]for z in chains];ends=[z[k][1]for z in chains];dur=[x-y for x,y in zip(ends,starts)]
   measures.append({'boundary':f'{k}.start','spread_ms':(max(starts)-min(starts))/1e6,'dependency_gap_max_ms':max(x-y for x,y in zip(starts,prev))/1e6,'duration_spread_ms':(max(dur)-min(dur))/1e6})
   measures.append({'boundary':f'{k}.end','spread_ms':(max(ends)-min(ends))/1e6,'dependency_gap_max_ms':0.0});prev=ends
  rows.append(measures)
out={'CPU_only':True,'GPU':False,'build':False,'original_handle':'910375.639','source_report_SHA256':hashlib.sha256(raw).hexdigest(),'source_trace_files':files,'matched_groups':matched,'unmatched_groups':bad,'call_counts':call_counts,'trace_args_schemas':schemas,'host_enqueue_vs_queue':'UNKNOWN: no presumed equivalence of generic index fields; no validated HIPcorrelation join performed','patterns':[]}
for names,rows in sorted(patterns.items(),key=lambda v:len(v[1]),reverse=True):
 table=[]
 for i in range(len(rows[0])):
  x=rows[0][i];table.append({'boundary':x['boundary'],'spread_ms':stat([z[i]['spread_ms']for z in rows]),'dependency_gap_max_ms':stat([z[i]['dependency_gap_max_ms']for z in rows])})
 out['patterns'].append({'groups':len(rows),'kernel_sequence':names,'boundaries':table})
out['decision']='SOURCE_DIAGNOSTIC_ONLY'if matched==132 and not bad else'UNKNOWN/PARK: unmatched ordinals; do not infer firstboundary or optimize from partialmatches'
out['limits']='Only samehostGPUtimestamps; chronological AllReduce is not an explicit layer tag. No CPUgapcost, rankduration sums, commflags/affinity/stream changes or performance claims. MatchedW13 mayalready includeactivation; inspect actualsource before alleging separateactivation.'
print(json.dumps(out))
