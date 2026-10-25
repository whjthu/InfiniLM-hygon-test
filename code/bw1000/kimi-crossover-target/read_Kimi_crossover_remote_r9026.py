import pathlib,json,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
for n in ['Kimi-crossover-TP16PP2-r9026-launch.json','Kimi-crossover-TP16PP2-r9026-watch.json']:
 p=r/n;out[n]=json.loads(p.read_text()) if p.exists() else None
launch=out['Kimi-crossover-TP16PP2-r9026-launch.json']
if launch:
 run=launch['run'];out['reports']=[]
 for node in range(4):
  p=r/'results'/f'kimi-crossover-r9026-{run}-node{node}.json'
  if p.exists():
   q=json.loads(p.read_text());out['reports'].append({k:q.get(k) for k in ['node','phase','completed','error','shutdown_error','loaded_keys','loading_elapsed_seconds','loading_seconds','crossover_complete']})
 p=r/'logs/Kimi-crossover-TP16PP2-r9026-controller.log'
 if p.exists():out['controller_tail']=p.read_text()[-3000:]
s=subprocess.run(['sacct','-n','-P','-j','910375','--format=JobIDRaw,JobName%110,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=25)
out['accounting']=[row for row in s.stdout.splitlines() if 'Kimi-crossover-' in row]
print(json.dumps(out))
