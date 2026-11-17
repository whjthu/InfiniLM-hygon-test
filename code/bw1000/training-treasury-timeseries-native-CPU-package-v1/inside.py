"""Native CPU wrapper with exact providers, no weight opens, and same original epoch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

SOURCE=Path(__file__).resolve().parent
RUN=SOURCE.parent


def require(ok,why):
    if not ok:raise ValueError(why)


def save(name,value):
    with (RUN/'work'/name).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def fd_snapshot():
    rows=[]
    for path in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(path)
        except FileNotFoundError:continue
        rows.append(dict(fd=int(path.name),target=target))
    return dict(full_fd_inventory=True,fd_count=len(rows),
                KFD_DRM_holders=[x for x in rows if x['target']=='/dev/kfd' or x['target'].startswith('/dev/dri/')])


def placement(environ,hostname):
    require(environ.get('SLURM_JOB_ID')=='1103606' and hostname=='f13r1n11'
            and environ.get('SLURM_STEP_GPUS') in (None,'','(null)'),"unexpected CPU placement")
    return dict(parent='1103606',node=hostname,step=environ['SLURM_STEP_ID'])


def main(deadline,work_deadline):
    started=time.monotonic()
    require(all(math.isfinite(x) for x in (deadline,work_deadline))
            and 0<work_deadline-time.time()<=450 and work_deadline<=deadline-75,"invalid original CPU work epoch")
    def expired(*_):raise TimeoutError('original CPU work deadline exhausted')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,work_deadline-time.time())
    actual_placement=placement(os.environ,socket.gethostname().split('.')[0])
    require(all(os.environ.get(key)=='' for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),
            "CPU visibility not empty")
    attempts=[]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if name.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
                attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest())
                raise RuntimeError('weight opening forbidden in CPU processor scope')
    sys.addaudithook(audit)
    before=fd_snapshot();require(not before['KFD_DRM_holders'],'CPU scope already owns device FD')
    save('CPU-owner.json',dict(PID=os.getpid(),PGID=os.getpgrp(),
        startticks=int(Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]),
        **actual_placement,
        deadline=deadline,work_deadline=work_deadline,fd_before=before))
    result=None;failure=None;providers={}
    try:
        import importlib
        import torch
        require(torch.version.hip is not None and not torch.cuda.is_available() and not torch.cuda.is_initialized(),
                'unexpected DTK CPU Torch provider')
        expected={'swift':'/swift-runtime/site/swift/','transformers':'/minicpm-transformers/transformers/',
                  'qwen_vl_utils':'/vision/qwen_vl_utils/','torch':'/usr/local/lib/python3.10/dist-packages/torch/'}
        for name,prefix in expected.items():
            module=importlib.import_module(name);path=Path(module.__file__)
            require(path.as_posix().startswith(prefix) and path.is_file() and not path.is_symlink(),
                    'actual module provider differs')
            raw=path.read_bytes();providers[name]=dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
        metadata=json.loads((SOURCE/'RUNTIME_PINS.json').read_bytes())
        for name,pin in metadata['processor'].items():
            path=RUN/'model'/name
            raw=path.read_bytes()
            require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'container processor pin differs')
        sys.path.insert(0,str(SOURCE/'native'))
        import check_native
        result=check_native.check(RUN/'data',RUN/'model',RUN/'work/native',deadline,
                                  time.monotonic()+max(0,work_deadline-time.time()))
    except BaseException as error:
        failure=type(error).__name__
    finally:
        after=fd_snapshot()
        passed=bool(result and result.get('state')=='NATIVE_CPU_INPUT_AND_TRAIN_COLLATOR_PASS'
                    and not after['KFD_DRM_holders'] and not attempts and failure is None)
        receipt=dict(passed=passed,deadline=deadline,work_deadline=work_deadline,
            actual_GPU_cards=0,actual_updates=0,weight_open_attempt_hashes=attempts,
            planned_rows={s:144 for s in ('train','dev','blind')},planned_native_encode_calls=576,
            model_weight_values_read=False,actual_providers=providers,failure_type=failure,
            fd_before=before,fd_after=after,full_fd_inventory_performed=True,
            native_progress_exists=(RUN/'work/native/progress.json').is_file(),
            native_progress_SHA256=hashlib.sha256((RUN/'work/native/progress.json').read_bytes()).hexdigest()
                if (RUN/'work/native/progress.json').is_file() else None,
            wall_seconds=time.monotonic()-started,model_qualification=False,
            semantic_readability='UNKNOWN')
        save('CPU-receipt.json',receipt)
        print(json.dumps(dict(state='NATIVE_TIMESERIES_CPU_FINISHED',passed=passed,failure_type=failure,
                              actual_GPU_cards=0,actual_updates=0)),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--work-deadline',type=float,required=True);a=p.parse_args()
    raise SystemExit(main(a.deadline,a.work_deadline))
