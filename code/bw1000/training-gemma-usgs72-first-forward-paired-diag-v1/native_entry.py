"""Exec one independently restored native5, with a derived output/plugin path only."""
import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import socket
import sys
import time

from contract import *


def derive_plan(base,root,plugin,port):
    plan=copy.deepcopy(base);argv=plan['command']
    root_posix=root.as_posix();plugin_posix=plugin.as_posix()
    need(PurePosixPath(root_posix).is_absolute() and PurePosixPath(plugin_posix).is_absolute()
         and '\\' not in root_posix+plugin_posix,'canonical POSIX native output/plugin required')
    need(argv.count('--output_dir')==argv.count('--external_plugins')==1,'original CLI shape changed')
    need(argv[argv.index('--resume_from_checkpoint')+1]==(ASSET/'continuous/train/checkpoint-20').as_posix()
         and argv[argv.index('--callbacks')+1]=='gemma_domain_resume_witness' and 'nvidia_baseline' not in argv,
         'original checkpoint/callback changed')
    index=argv.index('--external_plugins')
    need(index+2<len(argv) and argv[index+2].startswith('--'),'expected one exact original plugin')
    argv[argv.index('--output_dir')+1]=(root/'train').as_posix();argv[index+1]=plugin_posix
    plan['environment_overrides'].update(OUTPUT_DIR=(root/'train').as_posix(),MASTER_PORT=str(port))
    return plan


def main():
    source=verify('GPU');need(ARM in ('untraced','traced'),'explicit native arm required')
    intent,_=stable_json(ROOT/'intent.json');clock=Clock(intent['epoch'])
    prior=prior_continuous()
    stage,_=stable_json(ASSET/'stage-complete.json',STAGE_PIN)
    need(stage['passed'] is True and stage['source_manifest']['SHA256']==CORE_MF_SHA,'original845 stage changed')
    need(pin(ASSET/'fresh-framework-files.json')['SHA256']==FRAMEWORK_SHA
         and pin(ASSET/'runtime-binding.json')['SHA256']==RUNTIME_SHA,'original framework/runtime changed')
    need(pin(ASSET/'view/train800-swift.jsonl')==dict(bytes=602794,SHA256='74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc'),
         'actual original800 training view changed')
    for path,p in stage['reads'].items():
        s=Path(path).lstat()
        need([getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]==p['file_identity'],
             'original staged source/model/data identity changed')
    mf,_=stable_json(ASSET/'fresh-framework-files.json')
    for relative,p in mf['files'].items():
        clock.left(INSIDE);need(pin(ASSET/'framework'/relative)==p,'original396 source changed')
    before,bpin=stable_json(ROOT/'lease-admission.json')
    external,_=stable_json(PAIR_ROOT/'before.json')
    validate_lease(before,external,ARM,time.monotonic_ns(),PARENT+'.'+os.environ['SLURM_STEP_ID'])
    need(socket.gethostname().split('.')[0]==NODE and os.environ.get('SLURM_JOB_ID')==PARENT,'wrong native target')
    hip=ctypes.CDLL('libamdhip64.so');count=ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==8,'actual visible HIP allocation differs')
    pci=[]
    for index in range(8):
        bus=ctypes.create_string_buffer(32);need(hip.hipDeviceGetPCIBusId(bus,32,index)==0,'actual HIP PCI query failed')
        text=bus.value.decode().lower();pci.append('0000:'+text if len(text)==7 else text)
    need(sorted(pci)==before['pci'],'actual HIP PCI differs from fresh HOST gate')
    plan=derive_plan(json.loads((D/'BASE_RESUMED_PLAN.json').read_bytes()),ROOT,D/'diagnostic_plugin.py',PORT)
    argv=plan['command']
    need(not (ROOT/'train').exists(),'new native output already exists')
    runtime,_=stable_json(ASSET/'runtime-binding.json')
    config=dict(case='T02.sft',dataset_SHA256='74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc',
        binding=str(ASSET/'structural-binding.json'),checkpoint_step=20,
        checkpoint=str(ASSET/'continuous/train/checkpoint-20'),checkpoint_files=prior['checkpoint20_files'],
        trainer_SHA256=runtime['trainer']['SHA256'])
    config_pin=save('witness-config.json',config)
    for name in ('common-rng','light-trace','diagnostic-hooks'):(ROOT/name).mkdir(mode=0o700)
    trace=dict(enabled=ARM=='traced',schema='VLM_FIRST_FORWARD_LIGHT_TRACE_V1',root_GO=True,trace_GO=True,
        diagnostic_only=True,request_id=DISPATCH_REQUEST,ranks=[0,1],phase='resumed' if ARM=='traced' else 'independent_control',
        start_ordinal=1,count=4,scope_root=str(ROOT),output=str(ROOT/'light-trace'),
        framework_root=str(ASSET/'framework'),device_type='cuda',already_admitted_native_device=True)
    diag_pin=save('diagnostic-config.json',dict(arm=ARM,trace=trace,source_manifest=source,prior_gate=CONTINUOUS_GATE_PIN))
    save('native-plan.json',dict(command=argv,arm=ARM,prior_stage=STAGE_PIN,admission=bpin,actual_PCI=pci,
        planned_updates=5,only_native_recipe_changes=['new output directory','additive diagnostic plugin'],
        original_A100_initial_read=False))
    env=dict(os.environ,**plan['environment_overrides'])
    env.update(OUTPUT_DIR=str(ROOT/'train'),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(PORT),NATIVE_RESUME_CONFIG=str(ROOT/'witness-config.json'),
        NATIVE_RESUME_CONFIG_SHA256=config_pin['SHA256'],NATIVE_RESUME_EPOCH=str(intent['epoch']),
        NATIVE_RESUME_DEADLINE=str(intent['epoch']+INSIDE),NATIVE_RESUME_MAX_SECONDS='2400',
        GEMMA_DIAGNOSTIC_CONFIG_SHA256=diag_pin['SHA256'])
    clock.left(INSIDE)
    need(time.monotonic_ns()-before['monotonic_ns']<=45_000_000_000,'lease admission stale before native exec')
    os.execve(sys.executable,[sys.executable,'-B',*argv],env)


if __name__=='__main__':main()
