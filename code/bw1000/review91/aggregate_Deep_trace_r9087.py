import pathlib,json,collections,re,statistics,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';raw=r/'results/Deep-trace-r9087-raw-910374.1027';report=json.loads((r/'results/deep-trace-r9087-910374.1027.json').read_bytes());owners={str(x['thread']):x['device']for x in report['normal_owners']};windows={f['kind']:next(x['trace_window']for x in f['measurements']if x['trace_window']['trace'])for f in report['fixtures']};groups=collections.defaultdict(list);clock=collections.defaultdict(list);schemas={};files=[]
for p in sorted(raw.glob('*.json')):
 j=json.loads(p.read_bytes());ev=j.get('traceEvents',[]);labels={str(e['pid']):e.get('args',{}).get('name','')for e in ev if e.get('ph')=='M'and e.get('name')=='process_name'};files.append({'path':str(p),'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'events':len(ev)})
 for e in ev:
  if e.get('ph')!='X' or e.get('cat')not in ['HIP','HIPCOPY','HIPOPS']:continue
  cat=e['cat'];a=e.get('args',{});schemas.setdefault(cat,{'keys':sorted(a),'DurationNs_to_dur_ratio':float(a.get('DurationNs',0))/max(float(e.get('dur',1)),1e-10)})
  if 'BeginNs'not in a or 'EndNs'not in a:continue
  begin=int(a['BeginNs']);end=int(a['EndNs']);assert end>=begin
  if len(clock[cat])<20000:clock[cat].append(begin-float(e['ts'])*1000)
  label=labels.get(str(e['pid']),'');m=re.search(r'Device (\d+)',label);dev=int(m.group(1))if m else owners.get(str(a.get('tid','')));lane=str(e.get('tid'));owner_tid=str(a.get('tid'))if cat=='HIP'else None
  for kind,w in windows.items():
   lo=w['realtime_ns_enter'];hi=w['realtime_ns_exit']
   if end<=lo or begin>=hi:continue
   groups[(kind,cat,dev,lane,owner_tid)].append((max(lo,begin),min(hi,end),e['name']))
def union(seq):
 seq=sorted(seq);total=0;last=None
 for a,b in seq:
  if last is None:last=[a,b]
  elif a<=last[1]:last[1]=max(last[1],b)
  else:total+=last[1]-last[0];last=[a,b]
 if last:total+=last[1]-last[0]
 return total
rows=[]
for key,events in groups.items():
 kind,cat,dev,lane,tid=key;start=min(v[0]for v in events);end=max(v[1]for v in events);u=union((a,b)for a,b,n in events);by=collections.defaultdict(list)
 for a,b,n in events:by[n].append((a,b))
 rows.append(dict(map=kind,category=cat,device=dev,lane=lane,owner_tid=tid,event_count=len(events),interval_union_ms=u/1e6,event_span_ms=(end-start)/1e6,nonactive_inside_span_ms=(end-start-u)/1e6,top_names=sorted([{'name':n,'count':len(z),'sum_ms':sum(b-a for a,b in z)/1e6,'union_ms':union(z)/1e6}for n,z in by.items()],key=lambda x:-x['union_ms'])[:8]))
combined=[]
for kind,w in windows.items():
 for d in range(8):
  seq=[(a,b)for key,v in groups.items()if key[0]==kind and key[2]==d and key[1]in ['HIPOPS','HIPCOPY']for a,b,n in v];kern=[(a,b)for key,v in groups.items()if key[0]==kind and key[2]==d and key[1]=='HIPOPS'for a,b,n in v]
  if seq:
   lo=min(a for a,b in seq);hi=max(b for a,b in seq);u=union(seq);combined.append(dict(map=kind,device=d,kernel_union_ms=union(kern)/1e6,kernel_copy_union_ms=u/1e6,device_activity_span_ms=(hi-lo)/1e6,nonactive_inside_device_span_ms=(hi-lo-u)/1e6,window_ms=(w['realtime_ns_exit']-w['realtime_ns_enter'])/1e6))
overhead=[]
for f in report['fixtures']:
 ms=[x['trace_window']['wall_seconds']for x in f['measurements']if x['phase']=='measure'];overhead.append({'map':f['kind'],'OFF_ON_OFF_seconds':ms,'ON_over_OFF_mean':ms[1]/((ms[0]+ms[2])/2),'qualitative_only':True})
q={'status':'QUALITATIVE_ONLY_TRACER_PERTURBS_SCHEDULING','handle':'910374.1027','raw_remote_only':True,'files':files,'units':'args BeginNs/EndNs/DurationNs ns; Chrome ts/dur us verified ratio','clock_checks':{k:{'samples':len(v),'BeginNs_minus_ts_us_times1000_min':min(v),'max':max(v),'median':statistics.median(v)}for k,v in clock.items()},'schemas':schemas,'host_window_clock':'time.time_ns/realtime; only intersect events with same epoch BeginNs. Device time supplied converted by hipprof; no independent device calibration.','API_owner_map':owners,'rows':rows,'device_unions':combined,'observer_overhead':overhead,'limitations':['Perdevice/perlane union, never8card sum vs1ownerwall.','Gap only inside instrumented device activity span, not proven removable/normal-run idle.','Large observer effect: qualitative ranking only; no extrapolated wall speedup.','SyntheticQKV and actualmaps component, not Modelweights.']};p=r/'results/Deep-trace-aggregate-r9087.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
