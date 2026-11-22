"""Common Swift exec for control25; ROOT's audited held/RSS/Slurm supervisor must own it."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from control_contract import *


def read(path,expected,cap=2<<20):
    path=Path(path);before=path.lstat()
    need(path.is_file() and not path.is_symlink() and 0<before.st_size<=cap,'bounded regular worker metadata')
    with path.open('rb') as f:
        need(identity(os.fstat(f.fileno()))==identity(before),'worker metadata open lifetime differs')
        raw=f.read(cap+1)
        need(identity(os.fstat(f.fileno()))==identity(before),'worker metadata changed')
    need(identity(path.lstat())==identity(before) and
         dict(bytes=len(raw),SHA256=sha(raw))==expected,'worker exact file pin differs')
    return raw


def check_stage_identity(stage):
    for path,expected in stage['reads'].items():
        need(identity(Path(path).lstat())==expected['file_identity'],
             'original same-compute asset lifetime changed since stage full SHA')


def execute(run_path,run_sha,admission_path,admission_sha):
    source=verify('GPU')
    raw=run_path.read_bytes();need(sha(raw)==run_sha,'ROOT original run intent SHA differs');run=json.loads(raw)
    need(run['request_id']==TRAINING_REQUEST and run['root']==ROOT.as_posix() and
         run['parent']==PARENT and run['node']==NODE and run['source_manifest']==source and
         run['deadline']-run['epoch']==2400 and
         run['epoch']<run['native_deadline']<=min(run['deadline']-120,run['epoch']+1980),
         'exact original control2400 intent required')
    clock(run['epoch'],run['native_deadline'],time.time(),2400)
    need(not any(n in os.environ for n in ('RANK','LOCAL_RANK','WORLD_SIZE')),'worker must start unranked')
    step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    need(re.fullmatch(PARENT+r'\.\d+',step) and socket.gethostname().split('.')[0]==NODE,'exact assigned control step required')
    raw=admission_path.read_bytes();need(sha(raw)==admission_sha,'fresh admission SHA differs');before=json.loads(raw)
    need(before['request_id']==TRAINING_REQUEST and before['node']==NODE and before['step']!=step and
         before['kfd_holder_count']==0 and before['full_fd_inventory_performed'] is True and
         len(set(before['pci']))==8 and before['master_port']==PORT and before['master_port_available'] is True and
         before['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
         0<=time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'fresh eight-PCI/KFD/port admission missing')
    stage_raw=read(ROOT/'stage-complete.json',run['stage_complete']);stage=json.loads(stage_raw)
    need(stage['passed'] is True and stage['source_manifest']==source and stage['training_request']==TRAINING_REQUEST and
         stage['node']==NODE and stage['boot_id']==before['boot_id'],'actual same-compute stage/source/boot required')
    check_stage_identity(stage)
    clock(run['epoch'],run['native_deadline'],time.time(),2400)
    manifest=json.loads(read(ROOT/'fresh-framework-files.json',stage['framework_manifest']))
    need(len(manifest['files'])==396,'exact control framework396 required')
    for name,pin in manifest['files'].items():
        relative(name);read(ROOT/'framework'/name,pin)
    read(ROOT/'view/train800-swift.jsonl',stage['view'])
    plan=json.loads(read(ROOT/'continuous-plan.json',stage['plan']))
    runtime=json.loads(read(ROOT/'runtime-binding.json',stage['runtime_binding']))
    read(Path(runtime['trainer']['path']),{k:runtime['trainer'][k] for k in ('bytes','SHA256')})
    scope={'__name__':'control_snapshot'}
    exec(compile((D/'snapshot.py').read_bytes(),str(D/'snapshot.py'),'exec'),scope)
    scope['NODE']=NODE;scope['STEP']=re.compile(PARENT+r'\.\d+')
    current=scope['snapshot'](True)
    need(current['pci']==before['pci'] and current['boot_id']==before['boot_id'] and current['kfd_holder_count']==0,
         'actual assigned kernel devices changed')
    with socket.socket() as probe:probe.bind(('127.0.0.1',PORT))
    hip=ctypes.CDLL('libamdhip64.so');count=ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8,'actual HIP card allocation differs')
    pcis=[]
    for index in range(8):
        bus=ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(bus,32,index)==0,'actual HIP PCI mapping failed')
        value=bus.value.decode().lower();pcis.append('0000:'+value if len(value)==7 else value)
    need(sorted(pcis)==current['pci'],'HIP/runtime PCI is outside exact lease')
    arm=ROOT/'continuous';arm.mkdir(mode=0o700)
    cfg=dict(case='T02.sft',dataset_SHA256=stage['view']['SHA256'],binding=str(ROOT/'structural-binding.json'),
        initialization_mode='TARGET_CREATED_DOMAIN_INITIAL_REUSE',seed=42,data_seed=42,
        control_stage=str(ROOT/'stage-complete.json'),control_stage_SHA256=sha(stage_raw))
    raw=canonical(cfg);config=arm/'witness-config.json';config.write_bytes(raw)
    env=dict(os.environ,**plan['environment_overrides'])
    # The external callback imports only this source-pinned helper, not a sibling supervisor's module.
    env['PYTHONPATH']=D.as_posix()+os.pathsep+env.get('PYTHONPATH','')
    env.update(MASTER_ADDR='127.0.0.1',NATIVE_CONTINUOUS_CONFIG=str(config),
        NATIVE_CONTINUOUS_CONFIG_SHA256=sha(raw),NATIVE_CONTINUOUS_EPOCH=str(run['epoch']),
        NATIVE_CONTINUOUS_DEADLINE=str(run['native_deadline']),NATIVE_CONTINUOUS_MAX_SECONDS='2400')
    entry=dict(request_id=TRAINING_REQUEST,arm='continuous',epoch=run['epoch'],deadline=run['deadline'],
        step=step,native_deadline=run['native_deadline'],source_manifest=source,plan=stage['plan'],
        actual_updates=None,output_directory_initially_absent=True,admission_SHA256=admission_sha,
        actual_visible_PCI=pcis,source_rows=72,physical_exposures=800,original_A100_initial_read=False)
    (arm/'entry.json').write_bytes(canonical(entry))
    need(plan['command'][0]==str(ROOT/'framework/vlm/runtime/swift_entry.py'),'isolated common entry must be used')
    need(time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'admission expired before common exec')
    os.execve(sys.executable,[sys.executable,'-B',*plan['command']],env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('continuous',))
    for name in ('run','admission'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('run-sha','admission-sha'):p.add_argument('--'+name,required=True)
    a=p.parse_args();execute(a.run,a.run_sha,a.admission,a.admission_sha)
