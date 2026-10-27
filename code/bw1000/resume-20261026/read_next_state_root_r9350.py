import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU':{}}
for name,folder in [('TP32construct','Kimi-TP32-construct-r120f'),('KimiPool','Kimi-prefill-pool-component-r132')]:
 p=r/'candidates'/folder/'CPU-build.json';v={'receipt_exists':p.exists()}
 if p.exists():v.update(receipt=json.loads(p.read_bytes()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
 out['CPU'][name]=v
p=r/'Deep-prefix-native-once-r129-watch.json';out['Deep_GPU']={'watch_exists':p.exists()}
if p.exists():
 w=json.loads(p.read_bytes());out['Deep_GPU'].update({k:w.get(k)for k in ['qualified','GPU_submitted','actual_handle','state','Slurm','terminal','result_summary','result_SHA256','error','post_empty']})
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910407,910374','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if any(n in s for n in ['TP32-construct-HOST-r9349','Kimi-prefill-pool-HOST-r132','Deep-prefix-native-once-r129','Deep-prefix-native-keeper-r129'])];print(json.dumps(out))
