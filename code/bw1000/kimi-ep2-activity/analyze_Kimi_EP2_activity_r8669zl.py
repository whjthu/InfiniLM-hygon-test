"""CPU streaming consumer reusing r8673 HIP event schema/interval union semantics."""
import pathlib,json,hashlib,collections,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
def merged(intervals):
 out=[]
 for x,y in sorted(intervals):
  assert y>=x
  if out and x<=out[-1][1]:out[-1]=(out[-1][0],max(y,out[-1][1]))
  else:out.append((x,y))
 return out
def total(intervals):return sum(y-x for x,y in merged(intervals))
def category(name):
 if 'mask_ep2_r8669zj'in name:return 'EP2_route_mask'
 if 'groupedW13I192R8669y'in name or 'groupedW134531'in name:return 'routed_W13'
 if 'staticW2I192R8669y'in name or 'staticW2R326U'in name:return 'routed_W2'
 if any(v in name.lower()for v in ['nccl','rccl','infiniccl']):return 'collective_name_family'
 return 'other_kernel'
def consume(events,windows):
 out=[]
 for w in windows:
  begin,end=w['after_start_realtime_ns'],w['before_stop_realtime_ns'];assert 0<w['begin_realtime_ns']<=begin<end<=w['end_realtime_ns'] and w['start_rc']==w['stop_rc']==0
  out.append({'arm':w['arm'],'EP2':w['EP2'],'begin_ns':begin,'end_ns':end,'profiler_control_ns':(begin-w['begin_realtime_ns'])+(w['end_realtime_ns']-end),'kernel':collections.defaultdict(list),'copy':collections.defaultdict(list),'families':collections.defaultdict(list),'names':collections.defaultdict(lambda:[0,0]),'API':collections.defaultdict(list),'events':0,'straddling_events':0})
 for e in events:
  a=e.get('args',{});cat=e.get('cat');name=e.get('name','UNASSIGNED')
  if e.get('ph')!='X'or cat not in ['HIP','HIPOPS','HIPCOPY']or'BeginNs'not in a or'EndNs'not in a:continue
  x,y=int(a['BeginNs']),int(a['EndNs']);assert y>=x
  for z in out:
   if y<=z['begin_ns']or x>=z['end_ns']:continue
   z['events']+=1;z['straddling_events']+=int(x<z['begin_ns']or y>z['end_ns']);u,v=max(x,z['begin_ns']),min(y,z['end_ns'])
   if cat=='HIP':z['API'][name].append((u,v));continue
   d=int(a['devId']);assert 0<=d<8
   if cat=='HIPCOPY':z['copy'][d].append((u,v))
   else:
    z['kernel'][d].append((u,v));z['families'][d,category(name)].append((u,v));z['names'][d,name][0]+=1;z['names'][d,name][1]+=v-u
 answer=[]
 for z in out:
  per={};span=z['end_ns']-z['begin_ns']
  for d in range(8):
   combined=merged(z['kernel'][d]+z['copy'][d]);last=z['begin_ns'];gaps=[]
   for a,b in combined:
    if a>last:gaps.append(a-last)
    last=max(last,b)
   if last<z['end_ns']:gaps.append(z['end_ns']-last)
   per[d]={'kernel_union_ms':total(z['kernel'][d])/1e6,'copy_union_ms':total(z['copy'][d])/1e6,'activity_union_ms':total(combined)/1e6,'no_recorded_device_activity_ms':(span-total(combined))/1e6,'largest_activity_gaps_ms':sorted([v/1e6 for v in gaps],reverse=True)[:8],'named_family_union_ms':{family:total(z['families'][d,family])/1e6 for family in ['routed_W13','routed_W2','EP2_route_mask','collective_name_family','other_kernel']}}
  answer.append({'arm':z['arm'],'EP2':z['EP2'],'inner_window_ms':span/1e6,'profiler_start_stop_control_ms_excluded':z['profiler_control_ns']/1e6,'events':z['events'],'straddling_events':z['straddling_events'],'all8device_kernels_present':all(z['kernel'][d]for d in range(8)),'per_device':per,'kernel_top':[{'device':d,'name':name,'count':v[0],'duration_sum_ms':v[1]/1e6}for(d,name),v in sorted(z['names'].items(),key=lambda kv:kv[1][1],reverse=True)[:32]],'API_top':[{'name':name,'count':len(v),'interval_union_ms':total(v)/1e6,'duration_sum_ms':sum(b-a for a,b in v)/1e6}for name,v in sorted(z['API'].items(),key=lambda kv:total(kv[1]),reverse=True)[:16]]})
 return answer
def event_stream(files,errors):
 decoder=json.JSONDecoder()
 for f in files:
  with f.open()as stream:
   for line_no,line in enumerate(stream,1):
    s=line.strip()
    if s.startswith('{"traceEvents":['):s=s[len('{"traceEvents":['):]
    if not s.startswith('{'):continue
    try:e,_=decoder.raw_decode(s)
    except json.JSONDecodeError as error:
     if len(errors)<8:errors.append({'file':f.name,'line':line_no,'error':str(error)})
     continue
    yield e
def main(handle):
 assert handle.startswith('910374.')and handle.split('.')[1].isdigit();result={'actual_handle':handle,'nodes':[],'scope':'Instrumented fullMoEL1 actualT16; activityunion notSMutilization/causalexclusivewall, noModelTPS; node-localepoch windows only/no cross-hosttime subtraction. Namedfamilymatchingdiagnostic, unrecognizedkernels retained. API cumulative time notexclusivewall.'}
 for node in range(2):
  p=r/'results'/f'Kimi-EP2-activity-r8669zl-{handle}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators'];w=q['profile_windows'];assert len(w)==2 and [z['EP2']for z in w]==[False,True];b=r/'results'/f'Kimi-EP2-activity-trace-r8669zl-{handle}-node{node}';files=list(b.glob('trace*.json'));assert files;errors=[];windows=consume(event_stream(files,errors),w);assert not errors and all(z['all8device_kernels_present']and z['events']for z in windows),('window_event_schema_or_clockbase_failed',errors)
  result['nodes'].append({'node':node,'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'windows':windows,'raw_trace_files':[{'path':str(f),'bytes':f.stat().st_size,'SHA256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in files],'parse_errors':errors})
 p=r/'results'/f'Kimi-EP2-activity-summary-r8669zl-{handle}.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps({'summary':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'nodes':len(result['nodes'])}))
if __name__=='__main__':main(sys.argv[1])
