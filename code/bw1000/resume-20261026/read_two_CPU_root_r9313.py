import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU':{}}
for name,folder in [('GLM','GLM-M15-oproj-Model-r125'),('Kimi','Kimi-ctx2048-r123b')]:
 p=r/'candidates'/folder/'CPU-build.json';out['CPU'][name]={'receipt_exists':p.exists()}
 if p.exists():out['CPU'][name].update(receipt=json.loads(p.read_bytes()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910407','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines()
out['Slurm']=[x.split('|')for x in rows if any(n in x for n in ['GLM-M15-oproj-Model-CPU-r125','Kimi-ctx2048-HOST-r123b'])]
print(json.dumps(out))
