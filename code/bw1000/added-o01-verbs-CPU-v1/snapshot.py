"""Observe host and matching ROCm container under one CPU-only Slurm step."""
import json,os,socket,subprocess,sys
from pathlib import Path
from probe import observe
destination=Path(sys.argv[1])
root=destination.parent
intent=json.loads((root/'intent.json').read_bytes())
node=socket.gethostname().split('.')[0]
assert node in intent['actual_nodes'] and os.environ['SLURM_JOB_ID']=='1103606'
parent=root.parent.parent
image=parent/'containers/dtk26-sandbox-r1'
apptainer='/public/software/apps/apptainer/1.3.4/bin/apptainer'
host=observe()
shell='set -euo pipefail; source /opt/dtk/env.sh; export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1; exec python3 -B "$1"'
argv=[apptainer,'exec','--rocm','--no-home','--bind',str(root)+':'+str(root)+':ro',
      '--bind','/opt/hyhal:/opt/hyhal:ro',str(image),'/bin/bash','--noprofile','--norc','-c',shell,
      'verbs',str(root/'src/probe.py')]
run=subprocess.run(argv,capture_output=True,text=True,timeout=30)
assert run.returncode==0
container=json.loads(run.stdout.splitlines()[-1])
value=dict(node=node,step='1103606.'+os.environ['SLURM_STEP_ID'],rank=int(os.environ['SLURM_PROCID']),
           host=host,container=container)
p=destination/(node+'.json')
with p.open('x') as f:
    json.dump(value,f,indent=2)
