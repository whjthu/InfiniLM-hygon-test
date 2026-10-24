"""Readonly compact terminal/user-clock extraction for the existing 814 run."""
import pathlib,json,hashlib,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
label='deep-pack-model-r8774'
wp=r/(label+'-watch.json');w=json.loads(wp.read_text())
d={'watch':{k:v for k,v in w.items()if k!='CPU_result_audit_stdout'},'watch_SHA256':hashlib.sha256(wp.read_bytes()).hexdigest()}
h=w.get('actual_handle')
if h:
 p=r/'results'/f'{label}-{h}.json'
 if p.exists():
  q=json.loads(p.read_text());d['report_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
  d['report']={k:v for k,v in q.items()if k not in ['pack_trials_r8774','chat_cases_by_epoch','device_memory_snapshots']and not isinstance(v,(dict,list))}
  d['QA_epochs']={k:{'count':len(v),'explicit_passed_fields':[c.get('passed')for c in v if isinstance(c,dict)]}for k,v in q.get('chat_cases_by_epoch',{}).items()}
  d['memory']=q.get('device_memory_snapshots',[])
  d['trials']=[]
  for c in q.get('pack_trials_r8774',[]):
   t={k:v for k,v in c.items()if k not in ['generated_ids','CAPI_windows','request_completion_offsets_seconds','request_metrics','per_step_completion_seconds']}
   t['request_metric_summary']={}
   for key in ['TTFT_seconds','decode_tok_s','ITL_p95_seconds','max_gap_seconds']:
    values=[v[key]for v in c.get('request_metrics',[])];t['request_metric_summary'][key]={'n':len(values),'min':min(values),'median':statistics.median(values),'mean':statistics.mean(values),'max':max(values)}if values else None
   t['CAPI_window_count']=len(c.get('CAPI_windows',[]))
   d['trials'].append(t)
print(json.dumps(d))
