import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'components':{}}
for name,label in [('GLM','GLM-M15-offset8-r127'),('TP32','Kimi-TP32-I96-once-r120e')]:
 p=r/(label+'-watch.json');q={'watch_exists':p.exists()}
 if p.exists():q.update(watch=json.loads(p.read_bytes()),watch_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
 out['components'][name]=q
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910375,910376,910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines()
out['Slurm']=[x.split('|')for x in rows if any(n in x for n in ['GLM-M15-offset8','Kimi-TP32-I96-once-r120e','Kimi-ctx2048-HOST-r123d'])]
p=r/'candidates/Kimi-ctx2048-r123d/CPU-build.json';out['Kimi_CPU']={'receipt_exists':p.exists()}
if p.exists():out['Kimi_CPU'].update(receipt=json.loads(p.read_bytes()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
