import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=r/'GLM-M15-Model-r129-watch.json';out['GLM']={'watch_exists':p.exists()}
if p.exists():
 w=json.loads(p.read_bytes());out['GLM'].update({k:w.get(k)for k in ['state','actual_handle','heartbeat_UTC','CPU_keeper_step','qualified','ready','Slurm','terminal','warmups']});out['GLM'].update(error_present=bool(w.get('error')or w.get('audit_error')))
ctx=json.loads((r/'incoming/Kimi-ctx2k-Model-r127/launcher/run-context-public.json').read_bytes());run=ctx['run'];p=r/'results'/('Kimi-ctx2k-phase-A-model-r127-'+run+'.json');out['Kimi']={'watch_exists':p.exists(),'run':run}
if p.exists():
 w=json.loads(p.read_bytes());out['Kimi'].update({k:w.get(k)for k in ['heartbeat_UTC','CPU_actual_step','hostname','ready','completed','passed','GPU_submitted','manifest_SHA256','post_KFD_passed','preflight_pins_passed','capacity_passed_before_draft','all_target_terminal']});out['Kimi'].update(error_present=bool(w.get('error')))
 out['Kimi']['targets']=[{k:t.get(k)for k in ['parent','name','actual_handle','actual_row','terminal','signal']}for t in w.get('groups',[])]
 out['Kimi']['draft']={k:w.get('draft',{}).get(k)for k in ['parent','name','actual_handle','actual_row','terminal','status','ready_seen']}
 out['Kimi']['capacity']=[{k:c.get(k)for k in ['node','passed','node_rank','minimum_free_bytes']}for c in w.get('all4_capacity_projections',[])]
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910371,910376,910379,910406,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%110'],text=True,timeout=30).splitlines()
out['Slurm']=[x.split('|')for x in rows if any(n in x for n in ['GLM-M15-Model','kimi-ctx2k-phase-r127-b37fd2ea50','Deep-prefix-'])]
p=r/'candidates/Deep-prefix-native-CPU-metadata-r128b/CPU-build.json';out['Deep_CPU']={'receipt_exists':p.exists()}
if p.exists():out['Deep_CPU'].update(receipt=json.loads(p.read_bytes()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
