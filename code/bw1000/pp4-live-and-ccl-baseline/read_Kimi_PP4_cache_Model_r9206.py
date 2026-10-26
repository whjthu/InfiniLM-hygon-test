import pathlib,json,os,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='shard-cache-r9206-1eeada521ff4f579';out={'observedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run};p=r/'results'/('Kimi-shard-PP4-cache-model-r9206-'+run+'.json')
if p.exists():
 w=json.loads(p.read_bytes());out['health']={k:w[k]for k in ['CPU_actual_step','hostname','PID','heartbeat_UTC','preflight_pins_passed','ready','watch_ready_before_GPU','GPU_submitted','capacity_passed_before_draft','completed','passed','error','traceback','observation_errors','explicit_target_failure','post_KFD_passed','audit']if k in w};out['groups']=[{k:v for k,v in g.items()if k!='args'}for g in w.get('groups',[])];out['draft']={k:v for k,v in w.get('draft',{}).items()if k!='args'}
out['capacity']=[]
for n in range(8):
 p=r/'results'/('Kimi-shard-PP4-cache-capacity-r9206-'+run+'-node%d.json'%n)
 if p.exists():out['capacity'].append(json.loads(p.read_bytes()))
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM','PMI','CUDA','HIP','ROCR'))};z=subprocess.run(['sacct','-S','2026-10-25','-j','910407,910375,910381,910371','-n','-P','-o','JobIDRaw,JobName%95,State,ExitCode,Elapsed'],env=env,text=True,capture_output=True,timeout=30);out['sacct']=[s for s in z.stdout.splitlines()if run in s or'Kimi-PP4-cache-Model-keeper-r9206'in s or'PP4-cache-model-before-r9206'in s];print(json.dumps(out,ensure_ascii=False))
