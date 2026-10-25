import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-scalar-Cast-model-r9060';q={}
for suffix in ['-launch.json','-watch.json']:
 p=r/(label+suffix)
 if p.exists():q[suffix]=json.loads(p.read_bytes())
w=q.get('-watch.json',{});tag=w.get('actual_handle');q['live_squeue']=subprocess.check_output(['squeue','-s','-j','910374','-h','-o','%i|%j|%N|%M'],text=True,timeout=20).splitlines()
if tag:
 p=r/'results'/(label+'-'+tag+'.json')
 if p.exists():
  z=json.loads(p.read_bytes());q['report_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();q['report']={k:v for k,v in z.items()if k not in ['owner_trials_r9060']};q['trial_summaries']=[{k:v for k,v in t.items()if k not in ['generated_ids','request_completion_offsets_seconds','CAPI_windows','request_metrics','per_step_completion_seconds']}for t in z.get('owner_trials_r9060',[])];q['trial_full_ID_SHA256']=[hashlib.sha256(json.dumps(t.get('generated_ids')).encode()).hexdigest()for t in z.get('owner_trials_r9060',[])]
 p=r/'logs'/(label+'-'+tag+'.log')
 if p.exists():q['logtail']=p.read_text(errors='replace')[-5000:]
for n in [label+'-controller.log',label+'-watcher.log']:
 p=r/'logs'/n
 if p.exists():q[n]=p.read_text(errors='replace')[-3000:]
print(json.dumps(q))
