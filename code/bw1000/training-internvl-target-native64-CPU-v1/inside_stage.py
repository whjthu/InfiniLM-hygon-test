"""Single0GPU stage with pre/post ownership and separate bootstrap stderr."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from scope import D,P,RUN,require,read,save,pin,admission,verify
from ownership import capture,validate_snapshot,validate_pair,container_exit_identity


def main(epoch,deadline,work_deadline):
    require(deadline==epoch+900 and time.time()<work_deadline<=deadline-90,'original stage epoch differs')
    admission(read(D/'CLAIM.json'),read(D/'PREREQUISITES.json'))
    require(str(D)==str(RUN/'source') and os.environ.get('SLURM_JOB_ID')==P['parent']
            and os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)'), 'exact CPU namespace/parent/GRES')
    verify();stage=RUN/'stage';stage.mkdir()
    before=capture(P['parent'],P['node'],work_deadline);save(stage/'ownership-pre.json',before)
    child=None;owner=None;rc=None;error=None
    try:
        validate_snapshot(before,P['parent'],P['node'])
        seconds=int(work_deadline-time.time()-30);require(seconds>0,'no budget after pre/post reserve')
        for name in ('cache','tmp','home'):(RUN/name).mkdir(exist_ok=False)
        binds=[(str(RUN),str(RUN),'rw'),(P['assets_root'],P['assets_root'],'ro'),
               (P['model_metadata_root'],P['model_metadata_root'],'ro'),(P['runtime'],'/swift-runtime','ro')]
        pre=read(D/'PREREQUISITES.json');binds.append((pre['framework_root'],pre['framework_root'],'ro'))
        command=['apptainer','exec','--contain','--rocm','--no-home','--pwd','/']
        for source,target,mode in binds:command+=['--bind',source+':'+target+':'+mode]
        # Same shell retains DTK environment; bootstrap diagnostics cannot contaminate worker JSON.
        shell='source /opt/dtk/env.sh >&2; export PYTHONPATH=/swift-runtime/site; '
        shell+='export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
        shell+='export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1; '
        shell+='export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; '
        shell+='export HOME='+str(RUN/'home')+' TMPDIR='+str(RUN/'tmp')+' XDG_CACHE_HOME='+str(RUN/'cache')+'; '
        shell+='export HF_HOME='+str(RUN/'cache/hf')+' HF_MODULES_CACHE='+str(RUN/'cache/modules')+'; '
        shell+='exec /usr/bin/python3 -B '+str(D/'worker.py')+' --epoch '+str(epoch)+' --deadline '+str(deadline)+' --work-deadline '+str(work_deadline-20)
        command += [P['container'],'/usr/bin/timeout','--kill-after=10s',str(seconds)+'s','/bin/bash','--noprofile','--norc','-c',shell]
        save(stage/'container-intent.json',dict(original_epoch=epoch,deadline=deadline,work_deadline=work_deadline,
             timeout_seconds=seconds,binds=binds,bootstrap_stdout_redirected_to_stderr=True,command=command,GPU=0))
        with (stage/'worker.stdout').open('xb') as out,(stage/'container.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            owner=dict(PID=child.pid)
            try:
                fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
                owner.update(PGID=int(fields[2]),startticks=int(fields[19]))
            except FileNotFoundError:owner['exited_before_identity_observation']=True
            save(stage/'container-owner.json',owner)
            try:rc=child.wait(timeout=max(.01,work_deadline-time.time()-10))
            except subprocess.TimeoutExpired:error='ORIGINAL_CONTAINER_OUTCOME_UNKNOWN'
    except (OSError,ValueError,subprocess.SubprocessError) as exc:error=type(exc).__name__
    after=capture(P['parent'],P['node'],work_deadline);save(stage/'ownership-post.json',after)
    exited=container_exit_identity(owner,work_deadline);clean=False
    try:clean=validate_pair(before,after,P['parent'],P['node'],exited)
    except (ValueError,KeyError,TypeError) as exc:error=error or type(exc).__name__
    save(stage/'ownership-comparison.json',dict(passed=clean,container_exit=exited,original_work_deadline=work_deadline))
    worker=read(RUN/'worker-result.json') if (RUN/'worker-result.json').exists() else None
    passed=bool(rc==0 and error is None and clean and worker and worker['passed'] and worker['completed_rows']==64)
    value=dict(passed=passed,rc=rc,error=error,epoch=epoch,deadline=deadline,work_deadline=work_deadline,
               step=P['parent']+'.'+os.environ['SLURM_STEP_ID'],node=P['node'],allocated_GPU_cards=0,
               ownership_pre_post_verified=clean,worker=worker,completed_rows=worker['completed_rows'] if worker else 0,
               planned_rows=64,missing_rows=64-(worker['completed_rows'] if worker else 0),CPU_SOURCE_MF=verify(),
               original_container_owner=owner,GPU_GO=False,Model_GO=False,training_numerical_GO=False)
    save(stage/'outcome.json',value);print(__import__('json').dumps({k:value[k] for k in ('passed','completed_rows','planned_rows','allocated_GPU_cards')}))
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--work-deadline',type=float,required=True)
    a=parser.parse_args();raise SystemExit(main(a.epoch,a.deadline,a.work_deadline))
