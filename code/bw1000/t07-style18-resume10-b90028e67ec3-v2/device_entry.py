"""Actual new scheduler and compute gate. ROOT supplies one explicit current placement."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import time

_DIR=Path(__file__).resolve().parent
_POLICY=_DIR/'resume10_policy.py'
if (_POLICY.resolve(strict=True)!=_POLICY or
        sha256(_POLICY.read_bytes()).hexdigest()!='a6d3da45aa8b13ba27971d6d289fa555a6ce9e4fea21f7c7df128c9d5c128209'):
    raise RuntimeError('EXACT_SIBLING_RESUME10_POLICY_BEFORE_ISOLATED_IMPORT')
_PRIVATE=_DIR/'private_state.py'
if (_PRIVATE.resolve(strict=True)!=_PRIVATE or
        sha256(_PRIVATE.read_bytes()).hexdigest()!='0f077d38e2a6e61946664d71873e02ea19f0780f61c3a81436f01f7706512c56'):
    raise RuntimeError('EXACT_SIBLING_PRIVATE_STATE_BEFORE_ISOLATED_IMPORT')
sys.path.insert(0,str(_DIR))

import private_state

from resume10_policy import (CLAIM, LABEL, OLD, ORIGINAL_FULL_READ, PACKAGE, REQUEST, SOURCE, WORK,
                             compare, exact, mount_only, need, pin, write)


BASE=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ORIGINAL=BASE/'results/training-zimage-style18-native25-resume20-v3'
STAGE=BASE/'incoming'/LABEL
PYTHON='/usr/local/python3.12/bin/python3'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
MAX_GPU=650
PARENT='1103606'
NODE='f13r2n07'
INV_SHA='dfb047f40a1f21e9ff606afeb95e60026f9d350b5bb35e3b447d8e5b86fa7efc'
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','DEADLINE','BOOT_FAIL'}
PHASES=('pre','gpu','post')
PRIOR_TERMINAL_SHA='956f3ccfe7108eae770fd7c8f1380fa700ad2b4abd01cd83b2615f11e3ca2d16'
PRIOR_CAUSE_SHA='83ac2df71dcd01e61de51ca188a2e05097b3fffc31e7e511bf6e5537e08bb73c'


def manifest(root,want):
    raw=exact(root/'SOURCE_FILES.json',16384,want)
    info=json.loads(raw)
    need(info['request']==REQUEST and info['original_SOURCE_FILES_SHA256']==SOURCE
         and set(info['files'])=={'training_loop.py','worker.py','resume10_policy.py','device_entry.py','private_state.py'},
         'EXACT_NEW_RESUME_SOURCE_CLOSURE')
    for name,expected in info['files'].items():
        body=exact(root/name,65536,expected['SHA256'])
        need(len(body)==expected['bytes'],'EXACT_NEW_RESUME_SOURCE_EXTENT')
    return pin(raw)


def proof(body,pin_sha,before,pre_sha,now):
    a=json.loads(body)
    p=json.loads(before)
    need(pin(body)['SHA256']==pin_sha and pin(before)['SHA256']==pre_sha==a['independent_PRE_SHA256']
         and a['request']==REQUEST and a['label']==LABEL and a['original_full_read_SHA256']==ORIGINAL_FULL_READ
         and (a['parent'],a['node'])==(PARENT,NODE)
         and a['prior_terminal_SHA256']==PRIOR_TERMINAL_SHA and a['prior_miopen_cause_SHA256']==PRIOR_CAUSE_SHA
         and a['root_GPU_GO'] is a['root_sole_owner'] is a['ordinary_exclusion_until_independent_post'] is True
         and a['one_DCU_once'] is a['no_prior_unknown_conflict'] is True
         and a['execution_deadline']<=a['epoch']+900 and a['epoch']<=now<a['execution_deadline']
         and a['execution_deadline']-a['epoch']>=MAX_GPU+30,'EXACT_NEW_ROOT_APPROVED_RESUME10_PLACEMENT_AND_EPOCH')
    need(p['request']==REQUEST and p['phase']=='pre' and p['parent']==a['parent'] and p['node']==a['node']
         and p['boot_id_SHA256']==a['boot_id_SHA256'] and p['PCIs']==a['PCIs']
         and len(p['PCIs'])==len(set(p['PCIs']))==8 and p['kfd_process_count']==p['own_device_fd_process_count']==0
         and len(p['accounting'])==9 and p['accounting'][0:4]==[p['step'],LABEL+'-pre','COMPLETED','0:0']
         and p['accounting'][5]==a['node'] and 0<=now-p['observed_epoch']<=90
         and live_fields(p['accounting'][6],False)
         and re.fullmatch(a['parent']+r'\.[0-9]+',p['step']),
         'EXACT_CURRENT_NODE_INDEPENDENT_0GRES_PHYSICAL_PRE')
    return a,p


def paths():
    return Path(STAGE),Path(STAGE/'work-once')


def live_fields(value,gpu):
    pieces=value.split(',')
    fields=set(pieces)
    if gpu:
        return len(pieces)==4 and fields in (
            {'cpu=8','gres/dcu=1','mem=96G','node=1'},{'cpu=8','gres/dcu=1','mem=98304M','node=1'})
    return len(pieces)==3 and fields in ({'cpu=2','mem=4G','node=1'},{'cpu=2','mem=4096M','node=1'})


def allocated(parent,node,environ,cgroups,controller):
    digit=environ.get('SLURM_STEP_ID','')
    step=parent+'.'+digit
    need(environ.get('SLURM_JOB_ID')==parent and digit.isdecimal() and
         environ.get('SLURM_CPUS_PER_TASK')=='8' and
         re.fullmatch(r'0|[1-9][0-9]*',environ.get('CUDA_VISIBLE_DEVICES','')) is not None
         and any({f'job_{parent}',f'step_{digit}'}<=set(row.split(':',2)[-1].split('/')) for row in cgroups),
         'EXACT_ACTUAL_SINGLE_DCU_SLURM_CGROUP_AND_SELECTOR')
    actual=[]
    for line in controller.splitlines():
        atoms=[item.split('=',1) for item in line.split() if '=' in item]
        row=dict(atoms)
        if row.get('StepId')==step:
            keys=[key for key,_ in atoms if key in ('StepId','State','Name','JobName','NodeList','TRES','AllocTRES')]
            need(len(keys)==len(set(keys)),'EXACT_NO_DUPLICATE_STEP_FIELDS')
            actual.append(row)
    need(len(actual)==1,'EXACT_ONE_ACTUAL_NUMERIC_RESUME10_STEP')
    row=actual[0]
    names=[row[k] for k in ('Name','JobName') if k in row]
    tres=[row[k] for k in ('TRES','AllocTRES') if k in row]
    need(names and set(names)=={LABEL+'-gpu'} and row['State']=='RUNNING' and row['NodeList']==node
         and tres and all(live_fields(value,True) for value in tres),'ACTUAL_8CPU_96GIB_1DCU_AND_EXACT_NODE')
    return step


def srun_argv(authority,manifest_sha,authority_sha,pre_sha):
    stage,work=paths()
    return ['srun','--jobid='+authority['parent'],'--job-name='+LABEL+'-gpu','--exact','--nodes=1','--ntasks=1',
            '--nodelist='+authority['node'],'--immediate=5','--cpus-per-task=8','--mem=98304M',
            '--gres=dcu:1','--time=00:10:50',PYTHON,'-I','-S','-B','-u',str(stage/'device_entry.py'),
            'gpu-host','--source-sha',manifest_sha,'--authority-sha',authority_sha,'--pre-sha',pre_sha,
            '--node',authority['node']]


def image_argv(stage,work):
    results=BASE/'results'
    bindings=(
        (ORIGINAL,'/fusion-run','ro'),(stage,'/resume10-src','ro'),(work,'/resume10-work','rw'),
        (results/'training-zimage-native-checkout-import-v1/source','/native-source','ro'),
        (results/'training-qwen-a68952c-swift-initial-CPU-v1/framework','/common-framework','ro'),
        (results/'training-zimage-style18-native12-CPU-v2/data','/zimage-input','ro'),
        (results/'training-gemma-usgs-render72-native-CPU-v1/data/train/media','/zimage-input/train/images','ro'),
        (results/'training-zimage-native-checkout-import-v2/site','/zimage-compat','ro'),
        (results/'training-zimage-target-native-meta-CPU-v2/dependencies/site','/zimage-runtime','ro'),
        (results/'training-t06-sdxl-runtime-v1','/training-runtime','ro'),
        (results/'training-zimage-modelscope-whole-runtime-v2/runtime','/sealed-runtime','ro'),
        (PurePosixPath('/usr/local/hyhal'),'/opt/hyhal','ro'),
    )
    argv=[APPTAINER,'exec','--rocm','--no-home','--pwd','/']
    for source,destination,mode in bindings:
        argv+=['--bind',f'{source}:{destination}:{mode}']
    env=('set -euo pipefail; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; '
         + private_state.shell_environment() +
         'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 '
         'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
         'PYTHONPATH=/resume10-src:/fusion-run:/native-source:/zimage-compat:/zimage-runtime:/training-runtime/site:'
         '/common-framework/generation/src; exec python3 -B -u /resume10-src/worker.py resume10')
    return argv+[str(BASE/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',env]


def query(args):
    value=subprocess.check_output(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S',*args],timeout=7)
    need(0<len(value)<=2<<20,'BOUNDED_CURRENT_ROOT_SCHEDULER_VIEW')
    return value.decode('utf8')


def gpu_host(args):
    stage,work=paths()
    need(Path(__file__).resolve().parent==stage and socket.gethostname().split('.')[0]==args.node,
         'EXACT_NEW_PACKAGE_EXPECTED_COMPUTE_NODE')
    manifest(stage,args.source_sha)
    a,p=proof(exact(stage/'authority.json',16384),args.authority_sha,exact(stage/'pre.json',16384),
              args.pre_sha,time.time())
    numeric=a['parent']+'.'+os.environ.get('SLURM_STEP_ID','')
    need(a['node']==args.node and 0<=time.time()-p['observed_epoch']<=90 and
         not any(os.getenv(k) for k in ('APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH')),
         'EXACT_NODE_FRESH_NATIVE_PRE_AND_NO_BIND_OVERRIDES')
    rows=query(['scontrol','show','step',numeric,'--oneliner'])
    step=allocated(a['parent'],a['node'],os.environ,Path('/proc/self/cgroup').read_text().splitlines(),rows)
    inventory=Path(ORIGINAL)/'node_inventory.py'
    actuator=exact(inventory,16384,INV_SHA)
    space={'__name__':'resume10_device_inventory'}
    exec(compile(actuator,str(inventory),'exec'),space)
    physical=space['device_status']([])
    boot=sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()
    need(physical['PCIs']==p['PCIs'] and boot==p['boot_id_SHA256']
         and physical['kfd_process_count']==physical['own_device_fd_process_count']==0,
         'SAME_STEP_BOOT_EIGHT_PCI_AND_NO_PRIOR_HOLDERS')
    work.mkdir(mode=0o700)
    need(stat.S_IMODE(work.stat().st_mode)==0o700 and work.stat().st_uid==os.getuid(),'EXACT_NEW_PRIVATE_WORK_NAMESPACE')
    state_identities=private_state.create(work)
    write(work/'admission.json',dict(a,physical=physical,actual_step=step,pre_SHA256=args.pre_sha,
          native_state_DirIdentities=state_identities,
          within_actual_GPU_deadline=min(a['execution_deadline']-15,time.time()+MAX_GPU-8),
          authority_SHA256=args.authority_sha,source_SHA256=args.source_sha,monotonic_ns=time.monotonic_ns(),actual_1DCU=True))
    os.chdir('/')
    argv=image_argv(stage,work)
    os.execv(argv[0],argv)


def current_worker():
    mount_only()
    gate=json.loads(exact(WORK/'admission.json',16384))
    manifest(PACKAGE,gate['source_SHA256'])
    actual=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    need(gate['request']==REQUEST and gate['actual_1DCU'] is True and gate['actual_step']==actual
         and socket.gethostname().split('.')[0]==gate['node'] and os.environ.get('SLURM_CPUS_PER_TASK')=='8'
         and gate['boot_id_SHA256']==sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()
         and 0<=time.monotonic_ns()-gate['monotonic_ns']<=50_000_000_000,
         'EXACT_FRESH_SAME_REAL_GPU_STEP_WORKER')
    private_state.check_environment(os.environ)
    state=private_state.directory_identities(WORK,gate['native_state_DirIdentities'])
    write(WORK/'private-state-ingress.json',dict(request=REQUEST,actual_step=actual,
          verified_directories=len(state),owner_0700_RW_same_host_and_container=True,
          private_home_db_cache_lock_tmp=True,observed_MIOpen_library_access=False))
    return gate


def independent_terminal(args):
    stage,work=paths()
    established=json.loads(exact(work/'admission.json',16384))
    out=work/'resume10'
    runs=json.loads(exact(out/'worker-result.json',16384))
    observed=json.loads(exact(stage/'post.json',16384,args.post_sha))
    actual=query(['sacct','-nP','-j',established['parent'],'-o',
        'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'])
    rows=[r.split('|') for r in actual.splitlines() if r]
    need(all(len(r)==9 for r in rows),'EXACT_NINE_COLUMNS')
    select=lambda k:[r for r in rows if r[0]==k]
    found=select(established['actual_step'])
    post=select(observed['step'])
    need(len(found)==len(post)==1,'EXACT_SEPARATE_GPU_AND_POST_ACCOUNTING')
    gp,after=found[0],post[0]
    need(gp[1]==LABEL+'-gpu' and gp[2] in TERMINAL and gp[5]==after[5]==established['node']
         and live_fields(gp[6],True) and live_fields(after[6],False) and gp[8]<=after[7]
         and after[2:4]==['COMPLETED','0:0']
         and observed['request']==REQUEST and observed['phase']=='post' and observed['accounting']==after
         and observed['boot_id_SHA256']==established['boot_id_SHA256'] and observed['PCIs']==established['PCIs']
         and observed['kfd_process_count']==observed['own_device_fd_process_count']==0,
         'ACTUAL_TERMINAL_AND_INDEPENDENT_SAME_BOOT_POST')
    completed=runs['actual_optimizer_updates']
    out_proof=dict(request=REQUEST,actual_GPU_step=gp,actual_post_step=after,post_clean=True,
               planned_updates=15,completed_updates=completed,missing_updates=15-completed,passed=False)
    need(0<=completed<=15 and runs['request']==REQUEST and runs['step']==gp[0],
         'EXACT_ACTUAL_WORKER_PROGRESS_DENOMINATOR')
    if gp[2:4]==['COMPLETED','0:0'] and runs['passed'] is True:
        out_proof['comparison']=compare(Path(ORIGINAL),out)
        out_proof['passed']=out_proof['comparison']['passed'] and completed==15
    write(work/'independent-terminal.json',out_proof)
    return out_proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('argv','gpu-host','independent-terminal'))
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--authority-sha')
    parser.add_argument('--pre-sha')
    parser.add_argument('--post-sha')
    parser.add_argument('--node')
    args=parser.parse_args()
    stage,_=paths()
    manifest(stage,args.source_sha)
    if args.mode=='gpu-host':
        gpu_host(args)
    elif args.mode=='argv':
        a,_=proof(exact(stage/'authority.json',16384),args.authority_sha,
                  exact(stage/'pre.json',16384),args.pre_sha,time.time())
        print(json.dumps(srun_argv(a,args.source_sha,args.authority_sha,args.pre_sha)))
    else:
        print(json.dumps(independent_terminal(args),sort_keys=True))


if __name__=='__main__':
    main()
