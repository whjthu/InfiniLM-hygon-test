import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=r/'GLM-critical-Model-r133-watch.json';out['GLM']={'exists':p.exists()}
if p.exists():
 w=json.loads(p.read_bytes());out['GLM'].update({k:w.get(k)for k in ['state','actual_handle','Slurm','terminal','qualified','error','post_passed','audit_qualified','warm_stall_failure']});out['GLM']['warmups']={k:{n:v.get(n)for n in ['phase','completed','stage','UTC_epoch','stall_guard_active','stall_limit_seconds']}for k,v in w.get('warmups',{}).items()}
p=r/'candidates/Kimi-TP32-construct-r120h/CPU-build.json';out['TP32link']={'receipt_exists':p.exists()}
if p.exists():
 c=json.loads(p.read_bytes());out['TP32link'].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),summary={k:c.get(k)for k in ['completed','producer','library','library_SHA256','HOST_TUs_compiled','GPU_TUs_compiled','elapsed_s','error','provider_mode','normal_CPU_exit']})
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910376,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['GLM-critical-Model-r133','GLM-critical-Model-keeper-r133','Kimi-TP32-constructor-link-r120h'])];print(json.dumps(out))
