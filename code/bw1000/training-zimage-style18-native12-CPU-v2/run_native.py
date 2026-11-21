"""One new native12 image/prompt CPU scope; never run meta186 or load weights."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

REQUEST='1268f832fa01473a8f523dc9b80d51ae'
ENTRY='source/tools/preflight_zimage_style18_train_input.py'
ENTRY_SHA='af9c8ce526b4a1ba5c83ee6ae2ff142ac41acdda7eb0935f757158ffa556d897'
BINDING='source/coordination/training/zimage-style18-target-train-inputs-v1.json'
BINDING_SHA='a06c29c8af3d56ed8192c39f63063db82bf67a14b75ba1fb2913202155d1ea09'
PROVIDERS_SHA='cc0aa3793e84644810376367063f2c836a2607b0ba8d2e4309b9d2afb03c3ccc'


def need(ok,why):
    if not ok:raise ValueError(why)


def read_pinned(path,expected,maximum):
    need(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Source alias')
    with path.open('rb') as f:raw=f.read(maximum+1)
    need(len(raw)<=maximum and hashlib.sha256(raw).hexdigest()==expected,'Source identity differs')
    return raw


def save(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();phase='CLOCK';start=time.monotonic();root=Path(__file__).resolve().parent
    need(math.isfinite(args.epoch) and math.isfinite(args.deadline) and args.deadline==args.epoch+300
         and 0<args.epoch<=time.time()<args.epoch+220,'Original300/work220 epoch differs')
    need(os.environ.get('T07_NATIVE12_REQUEST')==REQUEST,'Wrong new request')
    need(socket.gethostname().split('.')[0]=='f11r4n14' and os.environ.get('SLURM_JOB_ID')=='1087656'
         and os.environ.get('SLURM_CPUS_PER_TASK')=='4' and os.environ.get('SLURM_STEP_ID','').isdigit(),
         'Outside new CPU step')
    need(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
         and not os.environ.get('SLURM_STEP_GPUS'),'Need0GRES/hiddenGPU')
    claim=json.loads((root/'CLAIM.json').read_bytes())
    need(claim.get('request_id')==REQUEST and claim.get('root_GO') is True and claim.get('CPU_GO') is True
         and claim.get('GPU_GO') is False and claim.get('training_GO') is False,'New CPU scope not adopted')
    allowed=sorted(os.sched_getaffinity(0));need(len(allowed)>=4,'Fewer than4 available CPUs')
    os.sched_setaffinity(0,set(allowed[:4]));need(len(os.sched_getaffinity(0))==4,'CPU affinity not applied')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original220s work deadline')))
    signal.setitimer(signal.ITIMER_REAL,args.epoch+220-time.time())
    args.output.mkdir(mode=0o700)
    owner={'request_id':REQUEST,'epoch':args.epoch,'deadline':args.deadline,'PID':os.getpid(),
           'startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
           'host':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'CPU_affinity':sorted(os.sched_getaffinity(0)),
           'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
    save(args.output/'owner.json',owner)
    result={'request_id':REQUEST,'planned_train_rows':12,'native_dataset_rows':0,'passed_rows':0,
            'failed_or_unobserved_rows':12,'native_entry_calls':0,'model_loads':0,'optimizer_updates':0,
            'training_READY':False,'read_development_or_blind_rows':0,
            'implicit_GPU_initialization_proof':'UNKNOWN_IMPORT_SIDE_EFFECTS_NOT_SDK_AUDITED'}
    forbidden=[]
    def deny_tensor_files(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            name=os.fsdecode(values[0]).lower()
            if name.endswith(('.safetensors','.pt','.pth','.ckpt','.bin','.npy','.npz')):
                if len(forbidden)<8:forbidden.append(True)
                raise RuntimeError('Tensor/model file access forbidden in native12 CPU scope')
    sys.addaudithook(deny_tensor_files)
    try:
        phase='PUBLISHED_ENTRY_AND_PROVIDER_IDENTITIES'
        read_pinned(root/ENTRY,ENTRY_SHA,16384);read_pinned(root/BINDING,BINDING_SHA,16384)
        providers=json.loads(read_pinned(root/'ORIGINAL_PROVIDER_FILES.json',PROVIDERS_SHA,16384))
        need(len(providers)==40,'Original40 source closure differs')
        for row in providers.values():
            name=row['relative_path'];need('..' not in Path(name).parts and not Path(name).is_absolute(),'Bad source member')
            read_pinned(Path('/native-source')/name,row['SHA256'],row['bytes'])
        phase='ORIGINAL_NATIVE12_ENTRY'
        save(args.output/'caller-progress.json',{'request_id':REQUEST,'phase':phase,'planned_rows':12,'entry_import_attempted':True})
        spec=importlib.util.spec_from_file_location('original_published_T07_native12',root/ENTRY)
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        save(args.output/'native-call-intent.json',{'request_id':REQUEST,'phase':phase,'planned_rows':12,
                                                  'native_entry_call_attempted':True})
        result['native_entry_calls']=1
        native=entry.run(Path('/zimage-input/train'),Path('/native-source'))
        save(args.output/'native-result.json',native)
        result.update(native_state=native['state'],native_dataset_rows=native.get('native_dataset_rows',0),
                      passed_rows=native.get('passed',0),failed_or_unobserved_rows=12-native.get('passed',0))
        need(native['state']=='NATIVE_TRAIN_INPUT_PASS' and native['passed']==12
             and native['failed']==0 and native['native_dataset_rows']==12,'Native input qualification failed')
        result.update(passed=True,state='T07_STYLE18_ACTUAL_NATIVE12_INPUT_PASS')
    except BaseException as error:
        save(args.output/'failure-private.json',{'phase':phase,'error_type':type(error).__name__,
              'message':str(error)[:1024],'traceback':traceback.format_exc()[-4096:]})
        result.update(passed=False,state='T07_STYLE18_NATIVE12_INPUT_FAILED',phase=phase,error_type=type(error).__name__)
    result.update(wall_seconds=time.monotonic()-start,owner=owner,
                  forbidden_tensor_file_attempts=len(forbidden))
    save(args.output/'result.json',result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result.get('passed') is True else 1


if __name__=='__main__':raise SystemExit(main())
