import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-eos-compaction-r9130d';q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ['Deep-eos-supervisor-r9130d-launch.json',label+'-supervisor.json',label+'-launch.json',label+'-watch.json']:
 p=r/('results/'+name if name.startswith('Deep-eos-supervisor')else name)
 if p.exists():
  d=json.loads(p.read_bytes());q[name]={k:v for k,v in d.items()if k not in ['audit_stdout','CPU_result_audit_stdout','CPU_result_audit_stderr','error_warning_lines']}
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910374,910406','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30).splitlines()];q['actual_jobs']=[x for x in rows if x[1]in [label,'Deep-eos-supervisor-r9130d',label+'-before',label+'-post']]
watch=q.get(label+'-watch.json',{});h=watch.get('actual_handle')
if h:
 p=r/'results'/(label+'-'+h+'.json')
 if p.exists():
  d=json.loads(p.read_bytes());q['model_report']={k:d.get(k)for k in ['step','phase','loading_seconds','completed','error','shutdown_error','live_generation','grid_complete','grid']};q['model_report']['finished_cohorts']=len(d.get('quality_trials',[]))
print(json.dumps(q))
