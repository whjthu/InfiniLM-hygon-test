"""One source-only installation then exec existing CPU launcher, same root origin epoch.

Root injects INSTALL_PAYLOAD_B64 through the existing ordinary SSH stdin body.
No network setup, weight transfer, Torch/SDK or Slurm call occurs in this installer.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import sys
import time

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LABEL='training-hunyuanvideo-execution-mount-SHA-CPU-v1'
PYTHON='/usr/local/python3.12/bin/python3'


def need(ok,reason):
    if not ok:raise ValueError(reason)


def tick(epoch,mono_end,reserve=0):
    left=min(epoch+900-time.time(),mono_end-time.monotonic())-reserve
    need(math.isfinite(left) and left>0,'ORIGINAL_INSTALL_EPOCH_EXHAUSTED')
    return left


def no_links(path):
    need(path.is_absolute(),'ABSOLUTE_OWNED_PATH')
    for item in [*reversed(path.parents),path]:need(not item.is_symlink(),'OWNED_NAMESPACE_ALIAS_OR_SYMLINK')
    need(path.resolve(strict=False)==path,'OWNED_NAMESPACE_ALIAS')
    return path


def write(path,raw,epoch,mono_end):
    tick(epoch,mono_end,120)
    no_links(path.parent)
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    need(path.parent.stat().st_uid==os.getuid(),'SOURCE_DIRECTORY_OWNER')
    with path.open('xb') as stream:
        need(stream.write(raw)==len(raw),'SOURCE_SHORT_WRITE')
        stream.flush();os.fsync(stream.fileno())
    tick(epoch,mono_end,120)


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def validate_payload(payload):
    epoch=payload['epoch']
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+60,
         'ROOT_ORIGINAL_INSTALL_EPOCH')
    need(re.fullmatch('[0-9a-f]{32}',payload['request_id']) and re.fullmatch('[0-9a-f]{40}',payload['publication'])
         and re.fullmatch('[0-9a-f]{64}',payload['source_manifest_SHA256']),'ORIGINAL_PUBLISHED_CONTEXT')
    members=payload['package']
    need(type(members) is dict and 1<=len(members)<=64,'SOURCE_MEMBER_COUNT')
    package={}
    for name,data in members.items():
        need(isinstance(name,str) and isinstance(data,str),'SOURCE_MEMBER_SCHEMA')
        rel=PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
             and rel.as_posix()==name and name not in ('','.','..'),'SAFE_RELATIVE_SOURCE')
        need(len(data)<=2*1024**2,'ONE_ENCODED_SOURCE_BOUND')
        package[name]=base64.b64decode(data,validate=True)
    need(sum(len(raw) for raw in package.values())<=2*1024**2,'WHOLE_SOURCE_PAYLOAD_BOUND')
    mf=package['SOURCE_FILES.json']
    need(hashlib.sha256(mf).hexdigest()==payload['source_manifest_SHA256'],'PUBLISHED_EXACT_SOURCE_MF')
    pins=json.loads(mf)['files']
    need(set(package)==set(pins)|{'SOURCE_FILES.json'},'EXACT_ALL_SOURCE_MEMBERS')
    for name,pin in pins.items():
        raw=package[name]
        need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'SOURCE_BYTES_SHA')
    need({'install_and_run.py','launch_once.py','worker.py','PLAN.json','CLAIM.json','held_child.py',
          'slurm_accounting.py','original_hash.py'}<=set(package),'REQUIRED_ACTUAL_CPU_ENTRY')
    claim=json.loads(package['CLAIM.json']);plan=json.loads(package['PLAN.json'])
    need(claim['root_GO'] is claim['CPU_GO'] is True and claim['GPU_GO'] is claim['Model_GO'] is False
         and claim['max_invocations']==1 and claim['case']=='T09.sft','ROOT_CPU_ONLY_ONCE_ADOPTION')
    need((claim['parent'],claim['node'],claim['CPU'],claim['memory_MiB'],claim['GPU_cards'],claim['single_epoch_seconds'])
         ==('1103606','f13r3n13',2,4096,0,900),'EXACT_CPU_RESOURCE_SCOPE')
    need(plan['root']==(ROOT/LABEL).as_posix() and plan['label']==LABEL and plan['parent']=='1103606'
         and plan['node']=='f13r3n13' and plan['CPU']==2 and plan['memory_MiB']==4096 and plan['GPU']==0
         and plan['epoch_seconds']==900 and plan['weight_bytes']==claim['exact_weight_bytes']==33306632192
         and plan['weight_SHA256']==claim['exact_weight_SHA256']=='71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86',
         'ONLY_EXACT_TRANSFORMER_WHOLE_SHA_SCOPE')
    return package


def main():
    need(isinstance(INSTALL_PAYLOAD_B64,str) and len(INSTALL_PAYLOAD_B64)<=4*1024**2,'BOUNDED_ROOT_STDIN_SOURCE')
    payload=json.loads(base64.b64decode(INSTALL_PAYLOAD_B64,validate=True))
    package=validate_payload(payload);epoch=payload['epoch']
    mono_end=time.monotonic()+epoch+900-time.time()
    def hard_exit(*_):os._exit(124)
    def stop_install(*_):
        signal.signal(signal.SIGALRM,hard_exit)
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(epoch+900-time.time(),mono_end-time.monotonic())))
        raise TimeoutError('ORIGINAL_INSTALL_LIMIT_NO_EXTENSION')
    signal.signal(signal.SIGALRM,stop_install)
    signal.setitimer(signal.ITIMER_REAL,tick(epoch,mono_end,120))
    need(os.getuid()==154459,'ORIGINAL_PROJECT_UID')
    root=no_links(ROOT/LABEL);no_links(ROOT)
    need(ROOT.is_dir() and ROOT.stat().st_uid==os.getuid(),'OWN_PROJECT_RESULTS_PARENT')
    need(not root.exists(),'EXISTING_OR_UNKNOWN_ORIGINAL_NAMESPACE_NO_REPLAY')
    root.mkdir(mode=0o700)  # Never append a new nonce inside a previous/unknown installation.
    try:
        fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        identity=dict(request_id=payload['request_id'],publication=payload['publication'],epoch=epoch,deadline=epoch+900,
            source_manifest_SHA256=payload['source_manifest_SHA256'],uid=os.getuid(),PID=os.getpid(),
            startticks=int(fields[19]),host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            phase='SOURCE_INSTALL_ONLY_NO_SLURM_NO_GPU',payload_bytes=sum(len(raw) for raw in package.values()),
            source_member_count=len(package),Torch_SDK_GPU_calls=0,replay_allowed=False)
        write(root/'install-intent.json',encode(identity),epoch,mono_end)
        source=root/'source';source.mkdir(mode=0o700)
        for name,raw in package.items():
            tick(epoch,mono_end,120)
            path=source/PurePosixPath(name)
            write(path,raw,epoch,mono_end)
        fd=os.open(source,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        write(root/'install-ready.json',encode(dict(identity,phase='SOURCE_READY_LAUNCHER_NOT_EXECUTED',
            completed_source_members=len(package),finished_epoch=time.time())),epoch,mono_end)
        tick(epoch,mono_end,120)
        command=[PYTHON,'-I','-S','-B',str(source/'launch_once.py'),'--publication',payload['publication'],
            '--source-manifest-sha256',payload['source_manifest_SHA256'],'--request-id',payload['request_id'],'--epoch',str(epoch)]
        # ITIMER_REAL survives exec: hand off only the original hard deadline,
        # never leave the install-work alarm to kill the future cleanup window.
        signal.signal(signal.SIGALRM,hard_exit)
        signal.setitimer(signal.ITIMER_REAL,max(.001,tick(epoch,mono_end)))
        os.execv(PYTHON,command)  # Same process/epoch; installer has created zero children.
    except BaseException as error:
        failure=dict(request_id=payload['request_id'],publication=payload['publication'],epoch=epoch,deadline=epoch+900,
            state='KNOWN_INSTALL_FAILURE_OR_LAUNCHER_NOT_EXECUTED',error_type=type(error).__name__,
            error_reason=str(error)[:160],installed_source_only=True,installer_child_processes=0,
            installer_Torch_SDK_GPU_calls=0,replay_allowed=False)
        # Best effort in the remaining original tail, exclusive write only. Original hard alarm remains armed.
        try:
            raw=encode(failure)
            with (root/'install-failure.json').open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
        except BaseException:
            pass
        print(json.dumps(failure),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
