import pathlib,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Kimi-W2-INT8-phase-K114B-control.json';out={'control_exists':p.exists(),'control':json.loads(p.read_bytes())if p.exists()else None};out['node_progress']=[]
for p in sorted((r/'results').glob('Kimi-W2-INT8-phase-K114B-*-node*.json')):
 q=json.loads(p.read_bytes());out['node_progress'].append({'file':p.name,'completed':q.get('completed'),'phase':q.get('phase'),'error':q.get('error'),'native_session_s':q.get('native_session_s')})
print(json.dumps(out))
