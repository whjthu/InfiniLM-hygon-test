"""Root once smallsource install then exec originalstyle ownedCPU supervisor; state readonly."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import time

A=Path('/home/whj/nvidia-baseline/alignment-assets');PY='/home/whj/nvidia-baseline/env-vlm/bin/python';NAME='training-USGS-T04-original-saved-state-CPU-v1'


def need(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,b):
    need(not p.exists() and not p.is_symlink(),'Newownedsourcefile exists');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())

def main():
    need(isinstance(INSTALL_PAYLOAD_B64,str) and len(INSTALL_PAYLOAD_B64)<2*1024**2,'Bounded root sourcepayload required')
    payload=json.loads(base64.b64decode(INSTALL_PAYLOAD_B64,validate=True));epoch=payload['epoch'];need(0<epoch<=time.time()<epoch+120,'Original180epoch admission')
    mono=time.monotonic()+epoch+170-time.time()
    def check():
        if min(epoch+170-time.time(),mono-time.monotonic())<=0:raise TimeoutError('Originalsavedstateinstallbudget')
    def expired(*_):raise TimeoutError('Originalsavedstateinstallalarm')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+170-time.time()))
    package={n:base64.b64decode(v,validate=True) for n,v in payload['package'].items()};mf=package['SOURCE_FILES.json'];need(sha(mf)==payload['source_manifest_SHA256'],'SourceMFdiffers');pins=json.loads(mf)['files']
    need(set(package)==set(pins)|{'SOURCE_FILES.json'},'Exactsmallpackage required')
    for n,b in package.items():
        need(Path(n).name==n and n not in {'.','..'},'Flat sourceonly')
        if n!='SOURCE_FILES.json':need(len(b)==pins[n]['bytes'] and sha(b)==pins[n]['SHA256'],'Exactsourcechanged')
    claim=json.loads(package['CLAIM.json']);need(sha(package['CLAIM.json'])==payload['claim_SHA256'] and claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False and claim['training_GO'] is False,'RootCPUadoptionrequired')
    plan=json.loads(package['PLAN.json']);scope=A/NAME;need(str(scope)==plan['remote_namespace'] and not scope.exists() and not scope.is_symlink(),'Exactnewnamespace exists/reconcile only');scope.mkdir(mode=0o700)
    write(scope/'install-intent.json',(json.dumps(dict(request_id=payload['request_id'],publication=payload['publication'],epoch=epoch,deadline=epoch+180,planned_state_files=29,source_manifest_SHA256=payload['source_manifest_SHA256'],claim_SHA256=payload['claim_SHA256'],GPU=0,actual_updates=0))+'\n').encode())
    for n,b in package.items():check();write(scope/'src'/n,b)
    allowed=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(allowed)>=4,'CPU4excluding0/1 unavailable');os.sched_setaffinity(0,set(allowed[:4]))
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(scope/'src')+':'+plan['framework_root'],WANDB_MODE='disabled',WANDB_DISABLED='true')
    check();need(time.time()<epoch+120,'Originalnativeworkreserve exhausted');(scope/'run').mkdir(mode=0o700)
    command=[PY,'-B','-X','utf8',str(scope/'src/run_CPU_once.py'),'--case','T04.sft','--framework',plan['framework_root'],'--scope',str(scope/'run'),'--request-id',payload['request_id'],'--publication',payload['publication'],'--epoch',str(epoch)]
    os.execve(PY,command,env)

if __name__=='__main__':main()
