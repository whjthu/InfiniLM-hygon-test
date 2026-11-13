from pathlib import Path
import json,subprocess,hashlib
B=Path(__file__).resolve().parent
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
remote=r'''
import json,os,subprocess
from pathlib import Path
roots=(Path('/home/whj/nvidia-baseline'),Path('/mnt/data/nvidia-baseline-20261010'),Path('/home/whj'))
rel=('env-hunyuan/bin/python','envs/env-hunyuan/bin/python','venvs/env-hunyuan/bin/python','envs/hunyuan/bin/python')
paths=[p/q for p in roots for q in rel]
probe="import json,torch,importlib.metadata as m;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,packages={n:m.version(n) for n in ('transformers','diffusers','peft','accelerate','numpy','safetensors')})))"
results={}
for path in paths:
    if path.is_file():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1')
        r=subprocess.run([str(path),'-B','-c',probe],capture_output=True,env=env,timeout=30)
        results[str(path)]=dict(rc=r.returncode,versions=json.loads(r.stdout) if r.returncode==0 else None)
print(json.dumps(results))
'''
records={}
for host in ('a100-26','a100-27'):
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),host,'timeout 50 python3 -B -u -'],input=remote.encode(),capture_output=True,timeout=65)
    records[host]=dict(rc=r.returncode,result=json.loads(r.stdout) if r.returncode==0 else None,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
with (B/'hunyuan-A100-exact-runtime-v1.json').open('x') as f:json.dump(records,f,indent=2)
print(json.dumps(records))
