import pathlib,json,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-W2-INT8-register-v2-r113c';p=r/'results'/(label+'-control.json');w=json.loads(p.read_bytes())if p.exists()else None;out={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'no_Slurm_query':True,'no_GPU_init':True,'health':w,'phases':[],'logs':[]}
if w and w.get('actual_handle'):
 h=w['actual_handle']
 for n in [0,1]:
  p=r/'results'/(label+'-'+h+'-node%d.json'%n)
  if p.exists():
   q=json.loads(p.read_bytes());out['phases'].append({'node':n,'completed':q.get('completed'),'phase':q.get('phase'),'error':q.get('error'),'native_s':q.get('native_session_s')})
  p=r/'logs'/(label+'-'+h+'-%d.log'%n)
  if p.exists():out['logs'].append({'node':n,'tail':p.read_text(errors='replace')[-1200:]})
print(json.dumps(out))
