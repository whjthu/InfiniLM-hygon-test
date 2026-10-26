import pathlib,json,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Kimi-adaptive-Model-AB-CPUclosure-r9225.json';out={'exists':p.exists()}
if p.exists():out['receipt']=json.loads(p.read_bytes())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM','PMI','CUDA','HIP','ROCR'))};z=subprocess.run(['sacct','-S','2026-10-25','-j','910407,910379,910375,910381','-n','-P','-o','JobIDRaw,JobName%80,State,ExitCode,Elapsed'],env=env,text=True,capture_output=True,timeout=30);out['sacct']=[s for s in z.stdout.splitlines()if'Kimi-AB-CPU'in s];print(json.dumps(out))
