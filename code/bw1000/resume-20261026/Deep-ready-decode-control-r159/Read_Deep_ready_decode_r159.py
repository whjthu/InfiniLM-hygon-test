"""Bounded whitelist reader; no Slurm, network, model or GPU calls."""
import pathlib,json,hashlib,re,statistics,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'GPU':False};sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p,limit):
 if not p.is_file():return {'exists':False}
 assert p.stat().st_size<=limit,('Reader bounded file size',p.name);b=p.read_bytes();return {'exists':True,'path':str(p),'SHA256':sha(b),'bytes':len(b),'data':json.loads(b)}
cp=read(R/'candidates/Deep-ready-decode-ABBA-r159/CPU-static.json',256*1024);q['CPU_static']={k:v for k,v in cp.items()if k!='data'}
if cp.get('data'):q['CPU_static']['summary']={k:cp['data'].get(k)for k in ['completed','actual_step','node','source_manifest_SHA256','Native_SHA256','error']}
watch=read(R/'results/Deep-ready-decode-controller-r159.json',256*1024);q['controller']=watch
if watch.get('data'):
 handle=watch['data'].get('actual_handle')
 if handle is not None:
  assert re.fullmatch(r'910374\.[0-9]+',handle),'Only actual owned Deep handle allowed';rp=R/'results'/('Deep-ready-decode-ABBA-r159-'+handle+'.json');result=read(rp,32*1024*1024);q['result']={k:v for k,v in result.items()if k!='data'}
  if result.get('data'):
   d=result['data'];summary={k:d.get(k)for k in ['step','completed','phase','loading_seconds','grid_complete','performance_qualified','outcome','engine_worker_shutdown_completed','error','shutdown_error','live_generation','metric_warning']};summary['trials']=len(d.get('measurement_trials',[]));summary['unique_questions']=len(set(d.get('planned_ids',[])));summary['paired_quality']=d.get('paired_quality',[]);summary['arms']={}
   for arm,row in d.get('by_arm',{}).items():
    out={k:row.get(k)for k in ['cases','correct','EOS','caps','parse_failed','correct_EOS','driver_task_wall_seconds','correct_EOS_requests_per_task_second','driver_visible_output_tok_s','common_decode_useful_tok_s','native_decode_useful_tok_s','decode_tok_s_per_card','prefill_input_tok_s','early_ready_decode_calls','early_ready_decode_tokens','strict_correct_EOS_speed15_good_requests']};users=row.get('user_metrics',[])
    for key,label in [('TTFT_seconds','TTFT'),('first_to_last_decode_tok_s','user_decode_tok_s'),('first_to_second_visible_seconds','first_second_gap_s')]:
     values=sorted(float(v[key])for v in users if isinstance(v.get(key),(int,float)));out[label]={'n':len(values),'min':min(values)if values else None,'p50':statistics.median(values)if values else None,'p95':values[max(0,(95*len(values)+99)//100-1)]if values else None,'max':max(values)if values else None}
    summary['arms'][arm]=out
   q['result']['summary']=summary
  log=R/'logs'/('Deep-ready-decode-ABBA-r159-'+handle+'.log')
  if log.is_file():
   # Read at most the final64 KiB; do not expose weights, environment or traces.
   with log.open('rb')as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read(65536).decode(errors='replace')
   q['log_tail_markers']={'path':str(log),'normal_marker_seen_in_tail':'DEEP_READY_DECODE_R159_NORMAL'in tail,'traceback_seen_in_tail':'Traceback (most recent call last)'in tail}
print(json.dumps(q))
