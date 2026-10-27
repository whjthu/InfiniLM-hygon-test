import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=r/'Kimi-TP32-construct-once-r120k-watch.json';out['TP32']={'exists':p.exists()}
if p.exists():
 q=json.loads(p.read_bytes());out['TP32'].update({k:q.get(k)for k in ['GPU_submitted','actual_handle','Slurm','terminal','qualified','state','error','post_empty','post_passed','results','audit']});out['TP32']['watch_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'candidates/Kimi-prefill-pool-fixture-r134b/CPU-build.json';out['KimiFixtureCPU']={'exists':p.exists()}
if p.exists():out['KimiFixtureCPU'].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_bytes()))
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910375,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['Kimi-TP32-construct-once-r120k','Kimi-pool-fixture-HOST-r134b'])];print(json.dumps(out))
