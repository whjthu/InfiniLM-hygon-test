import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Deep-owner-prefill-control-r9210.json';q={'control':json.loads(p.read_text())if p.exists()else None,'accounting':subprocess.check_output(['sacct','-S','2026-10-26','-j','910407,910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed'],text=True)[-5000:]};c=q['control']or{};h=c.get('GPU_handle');q['manifest_SHA256']=hashlib.sha256((r/'incoming/Deep-mode1-sink-model-r9210/Deep-owner-prefill-manifest-r9210.json').read_bytes()).hexdigest()
if h:
 p=r/'results'/('deep-mode1-sink-model-r9210-'+h+'.json')
 if p.exists():
  z=json.loads(p.read_text());q['report']={k:z.get(k)for k in ['step','phase','completed','error','error_phase','collection_finished','load_seconds','load_batches','parameters_loaded','live_generation','grid_complete','grid','outcome','engine_worker_shutdown_completed']};q['PF_completed']=len(z.get('prefill_diagnostics',[]));q['quality_arms_completed']=len(z.get('quality_trials',[]))
 p=r/'logs'/('deep-mode1-sink-model-r9210-'+h+'.log');q['log_tail']=p.read_text(errors='replace')[-5000:]if p.exists()else None
print(json.dumps(q))
