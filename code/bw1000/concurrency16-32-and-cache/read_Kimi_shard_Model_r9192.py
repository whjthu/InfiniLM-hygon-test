import pathlib,json,subprocess,os,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014'); run='shard-r9192-3dfd61b66fd538b3'
p=r/'results'/('Kimi-shard-model-r9192-'+run+'.json')
w=json.loads(p.read_bytes()); keys=['run','hostname','PID','actual_step','CPU_actual_step','heartbeat_UTC','ready','watch_ready_before_GPU','preflight_pins_passed','GPU_submitted','capacity_passed_before_draft','completed','passed','error','explicit_target_failure','observation_errors','audit','post_KFD_passed']
out={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health_mtime':p.stat().st_mtime,'health':{k:w[k]for k in keys if k in w},'groups':[]}
for v in w.get('groups',[]):out['groups'].append({k:v[k]for k in ['parent','nodes','name','actual_handle','actual_row','terminal','client_returncode','native_error','signal']if k in v})
if w.get('draft'):out['draft']={k:w['draft'][k]for k in ['actual_handle','actual_row','terminal','status','ready_seen','report','client_returncode','signal']if k in w['draft']}
out['capacity']=[]
for n in range(4):
 cp=r/'results'/('Kimi-shard-capacity-r9192-'+run+'-node'+str(n)+'.json')
 if cp.exists():
  v=json.loads(cp.read_bytes());out['capacity'].append(v)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM','SBATCH','SRUN','CUDA','HIP','ROCR','GPU_'))}
z=subprocess.run(['sacct','-j','910407,910380,910371','--format=JobIDRaw,JobName%85,State,ExitCode,Elapsed','--noheader','-P'],env=env,text=True,capture_output=True,timeout=30)
out['sacct']=[s for s in z.stdout.splitlines()if run in s or 'Kimi-shard-Model-keeper-r9192' in s or 'Kimi-shard-model-before-r9192' in s or 'Kimi-shard-model-draft-before-r9192'in s];out['sacct_rc']=z.returncode
print(json.dumps(out,ensure_ascii=False))
