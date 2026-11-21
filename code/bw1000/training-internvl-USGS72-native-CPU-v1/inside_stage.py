"""Actual CPU Slurm stage, readonly HIP runtime and metadata-only model view."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import time
from common import D,P,RUN,admission,clock,verify,save,read,need,require
from bounded import identity,device_fds
from step_control import run

def runtime_bind_source(host):
    if host==Path('/opt/hyhal'):
        target=Path('/usr/local/hyhal')
        require(host.resolve(strict=True)==target and target.is_dir() and not target.is_symlink(),
                'system hyhal alias identity changed')
        return target
    require(host.is_dir() and not host.is_symlink(),'required read-only runtime bind missing')
    return host

def main(epoch,step_deadline):
    c=clock(epoch);admission();verify()
    need(str(D)==str(RUN/'source') and os.environ.get('SLURM_JOB_ID')==P['parent']
        and os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)'),'actual CPU parent/0GRES namespace')
    need(time.time()<step_deadline<=c['deadline']-40,'actual Slurm step deadline')
    worker_end=min(c['work_deadline'],step_deadline-35)
    need(time.time()+10<worker_end,'insufficient original worker budget')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original stage alarm')))
    signal.setitimer(signal.ITIMER_REAL,step_deadline-time.time()-2)
    before=dict(owner=identity(),device_FDs=device_fds(os.getpid()),GPU_visibility=os.environ.get('SLURM_STEP_GPUS'),step=P['parent']+'.'+os.environ['SLURM_STEP_ID'])
    save(RUN/'stage-before.json',before)
    for name in ('cache','tmp','home'):(RUN/name).mkdir(mode=0o700)
    binds=[(str(RUN),str(RUN),'rw'),(P['assets_root'],P['assets_root'],'ro'),(P['model_metadata_root'],P['model_metadata_root'],'ro'),
        (P['runtime'],'/swift-runtime','ro'),(P['framework_root'],P['framework_root'],'ro'),('/opt/hyhal','/opt/hyhal','ro')]
    launcher=Path(P['container_launcher']);before_exe=launcher.stat()
    executable=dict(path=str(launcher),resolved_path=str(launcher.resolve()),X_OK=os.access(launcher,os.X_OK),
        stat_identity={k:getattr(before_exe,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_uid','st_gid')},
        regular=stat.S_ISREG(before_exe.st_mode),observed_at=time.time())
    after_exe=launcher.stat()
    executable['identity_stable']=all(getattr(before_exe,k)==getattr(after_exe,k) for k in executable['stat_identity'])
    save(RUN/'launcher-executable-identity.json',executable)
    need(executable['regular'] and executable['X_OK'] and executable['identity_stable'],'qualified absolute Apptainer executable stat/X_OK unavailable')
    argv=[str(launcher),'exec','--contain','--no-home','--pwd','/'];resolved_binds=[]
    for a,b,mode in binds:
        actual=runtime_bind_source(Path(a));resolved_binds.append([str(actual),b,mode])
        argv+=['--bind',str(actual)+':'+b+':'+mode]
    exports=dict(PYTHONPATH='/swift-runtime/site',CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HOME=str(RUN/'home'),TMPDIR=str(RUN/'tmp'),XDG_CACHE_HOME=str(RUN/'cache'),
        HF_HOME=str(RUN/'cache/hf'),HF_MODULES_CACHE=str(RUN/'cache/modules'))
    shell='set -e; source /opt/dtk/env.sh >&2; '+''.join('export '+k+'='+shlex.quote(v)+'; ' for k,v in exports.items())
    shell+='exec /usr/bin/python3 -I -S -B '+shlex.quote(str(D/'native_release.py'))+' '+shlex.quote(str(RUN))+' '+str(epoch)+' '+str(c['deadline'])+' '+str(worker_end)
    argv += [P['container'],'/bin/bash','--noprofile','--norc','-c',shell]
    save(RUN/'container-intent.json',dict(c,step_deadline=step_deadline,work_deadline=worker_end,binds=binds,resolved_binds=resolved_binds,command=argv,CPU=8,GRES=0))
    outer_env=os.environ.copy();outer_env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
    controlled=run(argv,outer_env,RUN/'container-control',epoch,worker_end,worker_end+2,worker_end+12,worker_end+17)
    worker=read(RUN/'worker-result.json') if (RUN/'worker-result.json').exists() else None
    passed=bool(controlled['passed'] and worker and worker['passed'] and worker['completed_rows']==72 and worker['own_device_FDs_after']==[])
    result=dict(passed=passed,**c,step=before['step'],node=P['node'],allocated_GPU_cards=0,planned_rows=72,
        completed_rows=worker['completed_rows'] if worker else 0,worker=worker,ownership_pre_post_verified=controlled['passed'],controller=controlled)
    save(RUN/'stage-outcome.json',result);signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:result[k] for k in ('passed','completed_rows','planned_rows','allocated_GPU_cards')}));return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True);p.add_argument('--step-deadline',type=float,required=True)
    a=p.parse_args();raise SystemExit(main(a.epoch,a.step_deadline))
