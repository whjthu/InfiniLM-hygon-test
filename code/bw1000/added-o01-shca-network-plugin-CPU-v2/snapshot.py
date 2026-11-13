"""Run the installed plugin load check on each assigned CPU node."""
import json,os,socket,subprocess,sys,time
from pathlib import Path
destination=Path(sys.argv[1])
root=destination.parent
i=json.loads((root/'intent.json').read_bytes())
project=root.parent.parent
node=socket.gethostname().split('.')[0]
assert node in i['actual_nodes'] and os.environ['SLURM_JOB_ID']=='1103606'
bundle=project/'candidates/public-shca-layers-r2228/layer3-selected/opt'
stack=project/'build-deps/rdma-host-stack'
aliases=root/'verbs-aliases'
shell=('set -euo pipefail; source /opt/dtk/env.sh; '
       'export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1; '
       'export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:/verbs-aliases:/rdma-provider-stack:$LD_LIBRARY_PATH; '
       'export RDMAV_DRIVERS=/rdma-provider-stack/libshca UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx; '
       'exec python3 -B "$1"')
cmd=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home']
for host,target in [(root,str(root)),(bundle,'/candidate-opt'),(stack,'/rdma-provider-stack'),(aliases,'/verbs-aliases'),(Path('/opt/hyhal'),'/opt/hyhal')]:
    cmd+=['--bind',str(host)+':'+target+':ro']
cmd += [str(project/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'plugin',str(root/'src/probe.py')]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=min(40,i['work_deadline']-time.time()))
plugin=json.loads(r.stdout.splitlines()[-1]) if r.returncode==0 else dict(returncode=r.returncode,error_tail=r.stderr.splitlines()[-4:])
value=dict(node=node,rank=int(os.environ['SLURM_PROCID']),step='1103606.'+os.environ['SLURM_STEP_ID'],plugin=plugin)
with (destination/(node+'.json')).open('x') as f:
    json.dump(value,f,indent=2)
