import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU':{}}
for name,folder in [('GLM','GLM-M15-component-r121b'),('Kimi','Kimi-TP32-I96-MoE-r120')]:
 p=r/'candidates'/folder/'CPU-build.json';out['CPU'][name]={'receipt_exists':p.exists()}
 if p.exists():out['CPU'][name].update(receipt=json.loads(p.read_bytes()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
rows=subprocess.check_output(['sacct','-S','2026-10-26','-j','910407','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[x.split('|')for x in rows if any(n in x for n in ['GLM-M15-CPU-r121b','Kimi-TP32-I96-CPU-r120b'])]
print(json.dumps(out))
