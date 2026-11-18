"""Exactly one execution-mount Transformer whole SHA; purestdlib/0GPU/0tensor decode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import sys
import time
SOURCE = Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from original_hash import file_fingerprint, hash_on_execution_mount

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LABEL = 'training-hunyuanvideo-execution-mount-SHA-CPU-v1'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser()
    for name in ('request-id','publication','source-manifest-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--work-deadline', type=float, required=True)
    args = parser.parse_args()
    need(args.epoch <= time.time() < args.work_deadline <= args.epoch+780, 'ORIGINAL_WORK_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, max(.001,args.work_deadline-time.time()))
    source = Path(__file__).resolve().parent
    need(source == R/LABEL/'source', 'EXACT_NEW_SOURCE_NAMESPACE')
    plan = json.loads((source/'PLAN.json').read_bytes())
    node = socket.gethostname().split('.',1)[0]
    need(node == plan['node'] and os.environ.get('SLURM_JOB_ID') == plan['parent']
         and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid() == 154459, 'ACTUAL_NODE_PARENT_UID')
    need(len(os.sched_getaffinity(0)) == 2, 'ACTUAL_TWO_CPU')
    os.nice(10); need(os.getpriority(os.PRIO_PROCESS,0) == 10,'ACTUAL_NICE10')
    path = R/plan['weight_relative']
    for item in [*reversed(path.parents),path]:
        need(not item.is_symlink(),'EXACT_WEIGHT_PATH_SYMLINK')
    fields = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    identity = dict(schema='T09_EXECUTION_MOUNT_WHOLE_SHA_RECEIPT_V1',request_id=args.request_id,
        publication=args.publication,source_manifest_SHA256=args.source_manifest_sha256,
        epoch=args.epoch,deadline=args.epoch+900,work_deadline=args.work_deadline,
        node=node,parent=plan['parent'],CPU_step=plan['parent']+'.'+os.environ['SLURM_STEP_ID'],
        host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        PID=os.getpid(),startticks=int(fields[19]),uid=os.getuid(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        nice=10,GPU_cards=0,relative_weight=plan['weight_relative'],Torch_SDK_imports=0)
    root = R/LABEL
    with (root/'worker-intent.json').open('x') as stream:
        json.dump(identity,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    started = time.monotonic(); result = dict(identity,passed=False,payload_bytes_read=None,planned_payload_bytes=plan['weight_bytes'])
    try:
        before = path.lstat(); need(stat.S_ISREG(before.st_mode) and before.st_size == plan['weight_bytes'],'EXACT_WEIGHT_SIZE')
        stamp = file_fingerprint(before)
        digest, count = hash_on_execution_mount(path,stamp,args.work_deadline)
        result.update(payload_bytes_read=count,weight_SHA256=digest,file_identity=stamp,file_mode=before.st_mode)
        need(count == plan['weight_bytes'] and digest == plan['weight_SHA256'],'COMPLETE_TRANSFORMER_SHA_MISMATCH')
        result['passed'] = True
    except BaseException as error:
        result.update(error_type=type(error).__name__,error_reason=str(error)[:180],partial_read_bytes='UNKNOWN_IF_HASH_RAISED')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(finished_epoch=time.time(),wall_seconds=time.monotonic()-started,
        CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,self_maxrss_bytes=usage.ru_maxrss*1024,
        physical_IO_wire_bytes='UNKNOWN_NOT_MEASURED',tensor_decodes=0,original_replay_inputs_read=False,
        GPU_GO=False,independent_terminal_accounting_required=True)
    with (root/'weight-receipt.json').open('x') as stream:
        json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({k:result[k] for k in ('request_id','passed','payload_bytes_read','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
