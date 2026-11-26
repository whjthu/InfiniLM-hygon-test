"""Install the exact T09 profiling override and qualify it in one CPU-only image step."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import sys
import time


REQUEST='d2b34ae9bc5244baa79ba28bcebf2769'
LABEL='t09-backward67-overlay-cpu-'+REQUEST[:12]
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
SCOPE=R/'incoming'/LABEL
GUEST_SCOPE=Path('/t09-profile-scope')
PARENT,NODE='1103606','f13r3n13'
WHOLE=120
PYTHON='/usr/local/python3.12/bin/python3'
IMAGE=R/'containers/dtk26-sandbox-r1'
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
OLD=R/'results/training-hunyuanvideo-t09-sft-repeat2-ro393-miopen-untraced-v3/src'
OLD_MF='77be1e9870853872bc4930f0d60f80c292f57e0c223d7f91fc92af206e03d971'
FRAMEWORK=R/'results/training-qwen-a68952c-swift-initial-CPU-v1/framework'
MF=FRAMEWORK.parent/'source/FRAMEWORK_FILES.json'
STAGE=R/'incoming/t09-private-framework393-ingress-v1/b1a692f520b94308a36818d2e1db252'
BOUNDS=(('hunyuanvideo_training_walltime','backend.py',31974,'4eb03a95f76d0affb74f2589afced199d991c894299b36c8c399c09a8ea70d2e'),
        ('t09_backward67_observer','observer.py',7889,'4cbea11835e72de19d222e8953c1be78e56f7286db10db20532d0a2455ae79e0'))
PACKAGE='gencuda.backends'
OVERLAYS={f'{PACKAGE}.{module}':(Path('/t09-profile-'+name),length,sha) for module,name,length,sha in BOUNDS}
MEMBERS={'bridge.py',*(name for _,name,_,_ in BOUNDS)}
ORIGINAL_BACKEND_SHA='2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae'
BIND_ENV=('APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH')
VISIBILITY=('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')
ZERO_TRES=({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
FMT='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
PREFIX='T09_OVERLAY_CPU_RESULT '


def need(flag,code):
    if not flag:
        raise ValueError(code)


def digest(data):
    return sha256(data).hexdigest()


def stamp(info):
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def read(path,cap,expected=None,length=None):
    path=Path(path)
    need(path.resolve(strict=True)==path,'CANONICAL_BOUNDED_PATH')
    before=path.lstat()
    with path.open('rb') as stream:
        live=os.fstat(stream.fileno())
        need(stat.S_ISREG(live.st_mode) and 0<live.st_size<=cap,'REGULAR_BOUNDED_PINNED_FILE')
        raw=stream.read(cap+1)
        after=os.fstat(stream.fileno())
    need(stamp(before)==stamp(path.lstat()) and stamp(live)==stamp(after) and before.st_size==len(raw)==live.st_size
         and (os.name=='nt' or stamp(live)==stamp(before)) and (length is None or len(raw)==length)
         and (expected is None or digest(raw)==expected),'EXACT_STABLE_FILE_BYTES')
    return raw


def encode(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def write(path,value):
    data=value if isinstance(value,bytes) else encode(value)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def closure(folder,mf_sha):
    raw=read(folder/'SOURCE_FILES.json',16384,mf_sha)
    mf=json.loads(raw)
    need(mf['request']==REQUEST and set(mf['files'])==MEMBERS
         and {p.name for p in folder.iterdir()}==MEMBERS|{'SOURCE_FILES.json'},'EXACT_PROFILER_PACKET_MEMBER_SET')
    blobs={name:read(folder/name,65536,value['SHA256'],value['bytes']) for name,value in mf['files'].items()}
    need(all(len(blobs[name])==size and digest(blobs[name])==sha for _,name,size,sha in BOUNDS),
         'EXACT_FROZEN_TWO_OVERLAY_FILES')
    return dict(blobs,**{'SOURCE_FILES.json':raw})


def install(packet,mf_sha):
    blobs=closure(packet,mf_sha)
    need(not SCOPE.exists() and SCOPE.parent.is_dir(),'NEW_PRIVATE_ONE_ONCE_INSTALL_TARGET')
    pending=SCOPE.with_name('.'+LABEL+'-'+secrets.token_hex(6))
    pending.mkdir(mode=0o700)
    src=pending/'src'
    src.mkdir(mode=0o700)
    for name,body in blobs.items():
        path=src/name
        write(path,body)
        path.chmod(0o400)
    closure(src,mf_sha)
    write(pending/'installation.json',dict(request=REQUEST,source_SHA256=mf_sha,verified_members=len(blobs),
        original_framework_changed=False,original_gpu_command_changed=False))
    for p in (src,pending):
        fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
    # Do not replace an existing owner directory, even if another install won the race.
    SCOPE.mkdir(mode=0o700)
    pending.rename(SCOPE/'sealed')
    return dict(request=REQUEST,verified_members=len(blobs),source_SHA256=mf_sha,installed=True)


def original_helpers():
    mf=json.loads(read(OLD/'SOURCE_FILES.json',16384,OLD_MF))
    need({p.name for p in OLD.iterdir()} >= set(mf['files'])|{'SOURCE_FILES.json'},'ORIGINAL_3640_HELPERS_PRESENT')
    for name,pin in mf['files'].items():
        read(OLD/name,65536,pin['SHA256'],pin['bytes'])
    sys.path.insert(0,str(OLD))
    return importlib.import_module('miopen_wall')


def tres(value):
    fields=value.split(',')
    return len(fields)==3 and set(fields) in ZERO_TRES


def controller(raw,step):
    need(len(raw)<=2<<20,'BOUNDED_NUMERIC_CONTROLLER_OUTPUT')
    records=[]
    for line in raw.splitlines():
        fields=[atom.split('=',1) for atom in line.split() if '=' in atom]
        row=dict(fields)
        if row.get('StepId')!=step:
            continue
        keep=[k for k,_ in fields if k in {'StepId','Name','JobName','State','TRES','AllocTRES','NodeList'}]
        need(len(keep)==len(set(keep)),'DUPLICATE_EXACT_CONTROLLER_FIELDS')
        records.append(row)
    need(len(records)==1,'ONE_EXACT_NUMERIC_CONTROLLER_STEP')
    row=records[0]
    names=[row[key] for key in ('Name','JobName') if key in row]
    resources=[row[key] for key in ('TRES','AllocTRES') if key in row]
    need(names and set(names)=={LABEL} and row['State']=='RUNNING' and row['NodeList']==NODE
         and resources and all(tres(resource) for resource in resources), 'ACTUAL_RUNNING_ZERO_GRES_CONTROLLER_TRES')
    return digest(raw.encode())


def deadline(auth):
    need(auth['request']==REQUEST and auth['node']==NODE and auth['parent']==PARENT and auth['label']==LABEL
         and auth['CPU_GO'] is True and auth['GPU_GO'] is False and auth['deadline']==auth['epoch']+WHOLE
         and 0<=time.time()-auth['epoch']<WHOLE, 'EXACT_ROOT_APPROVED_PRIVATE_CPU_EPOCH')
    return auth['deadline']-time.time()


class Override(MetaPathFinder,Loader):
    """Handle only the two exact import names, with private read-only bind origins."""
    def __init__(self,expected):
        self.expected=expected
        self.loaded={}

    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.expected:
            need(target is None and fullname not in self.loaded,'NO_OVERRIDE_RELOAD')
            return spec_from_file_location(fullname,self.expected[fullname][0],loader=self)

    def create_module(self,spec):
        return None

    def exec_module(self,module):
        path,length,sha=self.expected[module.__name__]
        raw=read(path,65536,sha,length)
        exec(compile(raw,str(path),'exec'),module.__dict__)
        self.loaded[module.__name__]=dict(bytes=length,SHA256=sha)


def load_override(paths):
    need(not any(name in sys.modules for name in paths),'NO_PREEXISTING_OVERRIDE_MODULES')
    finder=Override(paths)
    sys.meta_path.insert(0,finder)
    try:
        loaded=[importlib.import_module(name) for name in paths]
        need(set(finder.loaded)==set(paths)
             and all(Path(m.__file__)==paths[m.__name__][0] for m in loaded),'EXACT_TWO_REAL_OVERRIDE_IMPORTS')
        return finder.loaded
    finally:
        sys.meta_path.remove(finder)


def cpu_profiler(torch):
    profiler=torch.profiler
    root=Path(torch.__file__).resolve(strict=True).parent
    files={}
    for name in ('profile','record_function','supported_activities'):
        callable_=getattr(profiler,name)
        source=inspect.getsourcefile(callable_)
        need(source is not None,'PROFILER_API_SOURCE_FILE_REQUIRED')
        path=Path(source).resolve(strict=True)
        raw=read(path,1<<20)
        files[name]=dict(relative=path.relative_to(root).as_posix(),bytes=len(raw),SHA256=digest(raw),
                         resolved_callable=callable_.__qualname__)
    activities=profiler.supported_activities()
    cpu=profiler.ProfilerActivity.CPU
    need(cpu in activities and hasattr(profiler.ProfilerActivity,'CUDA') and not torch.cuda.is_initialized(),
         'ACTUAL_CPU_AND_DEVICE_ENUM_WITHOUT_GPU_INIT')
    marker='T09/backward67/cpu-overlay-preflight'
    with profiler.profile(activities=[cpu],record_shapes=False,profile_memory=False,with_stack=False,with_flops=False) as cap:
        with profiler.record_function(marker):
            value=(torch.ones(3,device='cpu')+2).sum().item()
    names=[event.name for event in cap.events()]
    need(value==9 and names.count(marker)==1 and 'aten::sum' in names and not torch.cuda.is_initialized(),
         'ACTUAL_CPU_KINETO_MARKER_AND_TENSOR_OP')
    return dict(api_files=files,activity_names=sorted(x.name for x in activities),
                CPU_marker=True,CPU_tensor_operation=True,GPU_runtime_initialized=False)


def guest(auth_sha):
    base=dict(request=REQUEST,phase='HOST_RECEIPT',passed=False,framework_before=0,framework_after=0,
              trainer_called=False,model_loaded=False,GPU_calls=0,profile_GPU_events=0)
    try:
        auth=json.loads(read(GUEST_SCOPE/'authority.json',16384,auth_sha))
        deadline(auth)
        host=json.loads(read(GUEST_SCOPE/'host-step.json',16384))
        step=actual_step()
        base.update(step=step)
        need(host['step']==step and host['request']==REQUEST and host['authority_SHA256']==auth_sha
             and host['exact_zero_GRES'] is True and not any(os.environ.get(k) for k in VISIBILITY)
             and sys.dont_write_bytecode and sys.pycache_prefix is None,
             'SAME_GPU_HIDDEN_NUMERIC_CPU_STEP')
        wall=original_helpers()
        from framework_mount import before_model
        from mount_observation import explicit_readonly,observe
        base['phase']='FRAMEWORK_393_BEFORE'
        fresh=before_model({})
        need(fresh['passed'] and fresh['actual_verified']==393,'ORIGINAL_393_ROOT_AND_MANIFEST_BEFORE')
        base['framework_before']=393
        base['phase']='MIOPEN_TORCH_IMPORT'
        torch, mapped=wall.actual_process_import(host['miopen'],auth['deadline'])
        v=Path(torch.__file__).resolve(strict=True).parent/'version.py'
        need(torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113'
             and digest(read(v,65536))=='d324ffd50535bca68131043b76605e36d588a85ec41b2aed9d798ece471aa534',
             'EXACT_ORIGINAL_TORCH_IMAGE_VERSION')
        base['miopen_same_readonly_mapped_fd']=mapped['mapped_exact_same_FD_verified']
        base['phase']='CPU_KINETO'
        base['profiler']=cpu_profiler(torch)
        base['phase']='READONLY_TWO_FILE_OVERRIDE'
        for fullname,(path,size,sha) in OVERLAYS.items():
            target=observe(path)
            item=host['override'][fullname]
            need(explicit_readonly(target) and target['object_type']=='REGULAR'
                 and target['dev_inode']==item['dev_inode'] and target['owner']==os.getuid()
                 and target['permissions']==0o400 and len(read(path,65536,sha,size))==size,
                 'EXACT_EFFECTIVE_READONLY_OVERLAY_FILE_BIND')
        canonical=FRAMEWORK/'generation/src/gencuda/backends/hunyuanvideo_training_walltime.py'
        read(canonical,65536,ORIGINAL_BACKEND_SHA,31603)
        base['imports']=load_override(OVERLAYS)
        need(callable(sys.modules[f'{PACKAGE}.t09_backward67_observer'].fresh),'OBSERVER_ENTRY_RESOLVED')
        need(not torch.cuda.is_initialized(),'NO_GPU_RUNTIME_INIT_AFTER_OVERRIDE_IMPORT')
        base['phase']='FRAMEWORK_393_AFTER'
        after=before_model({})
        need(after['passed'] and after['framework']==fresh['framework'] and after['actual_verified']==393,
             'ORIGINAL_393_CANONICAL_INVENTORY_UNCHANGED')
        deadline(auth)
        base.update(framework_after=393,passed=True,phase='DONE')
    except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError) as exc:
        base.update(error_type=type(exc).__name__,errno=getattr(exc,'errno',None),error_SHA256=digest(str(exc).encode()))
        if type(exc) is ValueError and re.fullmatch('[A-Z0-9_]{3,96}',str(exc)):
            base['reason']=str(exc)
    print(PREFIX+json.dumps(base,sort_keys=True),flush=True)
    return 0 if base['passed'] else 2


def actual_step():
    n=os.environ.get('SLURM_STEP_ID','')
    need(os.environ.get('SLURM_JOB_ID')==PARENT and n.isdecimal() and socket.gethostname().split('.')[0]==NODE
         and os.environ.get('SLURM_CPUS_PER_TASK')=='4' and not os.environ.get('SLURM_STEP_GPUS')
         and any({f'job_{PARENT}',f'step_{n}'}<=set(line.split(':',2)[-1].split('/'))
                 for line in Path('/proc/self/cgroup').read_text().splitlines()),'EXACT_NUMERIC_ZERO_GRES_CGROUP')
    return PARENT+'.'+n


def image_argv(auth_sha):
    private=SCOPE/'miopen'
    src=SCOPE/'sealed/src'
    body=('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
          'unset PYTHONPYCACHEPREFIX LD_PRELOAD LD_AUDIT LD_PROFILE XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME '
          'MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_DIR MIOPEN_LOCKFILE_PATH; '
          'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= '
          'ROCR_VISIBLE_DEVICES= GPU_DEVICE_ORDINAL= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
          'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/miopen-private-home-v4 TMPDIR=/tmp TEMP=/tmp TMP=/tmp '
          'MIOPEN_USER_DB_PATH=/miopen-private-home-v4/.config/miopen/user-db '
          'MIOPEN_CUSTOM_CACHE_DIR=/miopen-private-home-v4/.cache/miopen '
          f'PYTHONPATH=/training-runtime/site:{FRAMEWORK}/generation/src; '
          'exec python3 -B -u /t09-profile-scope/sealed/src/bridge.py guest "$@"')
    bindings=[(R,R,'ro'),(STAGE/'framework',FRAMEWORK,'ro'),(STAGE/'FRAMEWORK_FILES.json',MF,'ro'),
        (STAGE,'/t09-stage-ro','ro'),(SCOPE,GUEST_SCOPE,'ro'),
        (R/'results/training-t06-sdxl-runtime-v1','/training-runtime','ro'),('/opt/hyhal','/opt/hyhal','ro'),
        (private/'provider/libMIOpen.so.1.0','/opt/dtk/lib/libMIOpen.so.1.0','ro'),
        (private/'private-home','/miopen-private-home-v4','rw'),(SCOPE/'tmp','/tmp','rw'),(SCOPE/'var-tmp','/var/tmp','rw')]
    bindings.extend((src/name,'/t09-profile-'+name,'ro') for _,name,_,_ in BOUNDS)
    cmd=[APPTAINER,'exec','--rocm','--no-mount','home,cwd','--pwd','/']
    for src,dest,mode in bindings:
        cmd+=['--bind',f'{src}:{dest}:{mode}']
    return cmd+[str(IMAGE),'/bin/bash','--noprofile','--norc','-c',body,LABEL,auth_sha]


def host(auth_sha):
    auth=json.loads(read(SCOPE/'authority.json',16384,auth_sha))
    deadline(auth)
    step=actual_step()
    need(not any(os.environ.get(k) for k in BIND_ENV),'NO_INHERITED_APPTAINER_BIND_OVERRIDES')
    raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],text=True,timeout=6)
    controller_sha=controller(raw,step)
    closure(SCOPE/'sealed/src',auth['source_SHA256'])
    wall=original_helpers()
    sealed=wall.stage(SCOPE/'miopen',auth['deadline'])
    for name in ('tmp','var-tmp'):
        (SCOPE/name).mkdir(mode=0o700)
    overrides={}
    for module,name,_,_ in BOUNDS:
        entry=(SCOPE/'sealed/src'/name).stat()
        need(entry.st_uid==os.getuid() and stat.S_IMODE(entry.st_mode)==0o400,'PRIVATE_OVERLAY_OWNER_MODE')
        overrides[f'{PACKAGE}.{module}']=dict(dev_inode=[entry.st_dev,entry.st_ino])
    write(SCOPE/'host-step.json',dict(request=REQUEST,step=step,authority_SHA256=auth_sha,
        controller_SHA256=controller_sha,miopen=sealed,override=overrides,exact_zero_GRES=True))
    deadline(auth)
    os.chdir('/')
    cmd=image_argv(auth_sha)
    os.execv(cmd[0],cmd)


def owner_argv(auth_sha, seconds):
    need(type(seconds) is int and 1<=seconds<=90,'BOUNDED_NUMERIC_ZERO_GRES_LIMIT')
    return ['srun','--jobid='+PARENT,'--job-name='+LABEL,'--exact','--immediate=5','--nodes=1','--ntasks=1',
            '--nodelist='+NODE,'--cpus-per-task=4','--mem=8192M','--gres=none',f'--time=00:{seconds//60:02}:{seconds%60:02}',
            PYTHON,'-I','-S','-B','-u',str(SCOPE/'sealed/src/bridge.py'),'host',auth_sha]


def once(approved,source_sha):
    need(Path(__file__).resolve()==SCOPE/'sealed/src/bridge.py','EXACT_INSTALLED_ONCE_OWNER')
    closure(SCOPE/'sealed/src',source_sha)
    raw=read(approved,16384)
    auth=json.loads(raw)
    need(auth['source_SHA256']==source_sha and auth['sole_authorized_CPU_attempt'] is True and deadline(auth)>100,
         'ROOT_ONE_EXACT_ZERO_GRES_DISPATCH')
    p=SCOPE/'authority.json'
    auth_sha=write(p,raw)
    cmd=owner_argv(auth_sha,90)
    write(SCOPE/'dispatch-intent.json',dict(request=REQUEST,authority_SHA256=auth_sha,argv_SHA256=digest(encode(cmd))))
    command=['sacct','-nP','-j',PARENT,'--format',FMT]
    need(not accounting(command),'NO_ORIGINAL_LABEL_IN_ACCOUNTING_BEFORE_ONCE')
    with (SCOPE/'private.stdout').open('xb') as fo,(SCOPE/'private.stderr').open('xb') as fe:
        result=subprocess.run(cmd,stdout=fo,stderr=fe,timeout=min(98,deadline(auth)))
    return dict(request=REQUEST,rc=result.returncode,independent_accounting_required=True)


def accounting(command):
    raw=subprocess.check_output(command,env=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S'),
                                text=True,timeout=8)
    need(len(raw)<=2<<20,'BOUNDED_ACCOUNTING')
    rows=[line.split('|') for line in raw.splitlines() if line]
    need(all(len(row)==9 for row in rows),'EXACT_NINE_ACCOUNTING_COLUMNS')
    return [row for row in rows if row[1]==LABEL]


def terminal(row,auth):
    utc=lambda s: datetime.strptime(s,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    need(len(row)==9 and re.fullmatch(PARENT+r'\.[0-9]+',row[0]) and row[1]==LABEL and row[5]==NODE
         and tres(row[6]) and row[2] in ('COMPLETED','FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED','NODE_FAIL','DEADLINE')
         and re.fullmatch(r'[0-9]+:[0-9]+',row[3]) and row[4].isdecimal()
         and 0<=int(row[4])<=90 and auth['epoch']-1<=utc(row[7])<=utc(row[8])<=auth['deadline']+1
         and abs(utc(row[8])-utc(row[7])-int(row[4]))<=1,'INDEPENDENT_EXACT_ZERO_GRES_STEP_TERMINAL')
    return row


def reconciled():
    auth_raw=read(SCOPE/'authority.json',16384)
    auth=json.loads(auth_raw)
    intent=json.loads(read(SCOPE/'dispatch-intent.json',16384))
    need(intent['request']==auth['request']==REQUEST and intent['authority_SHA256']==digest(auth_raw),
         'EXACT_PERSISTED_ZERO_GRES_INTENT')
    rows=accounting(['sacct','-nP','-j',PARENT,'--format',FMT])
    need(len(rows)==1,'ONE_ORIGINAL_NUMERIC_TERMINAL')
    row=terminal(rows[0],auth)
    report=dict(request=REQUEST,accounting=row,passed=False,original_gpu_command_changed=False,profile_GPU_events=0)
    stderr=SCOPE/'private.stderr'
    if stderr.is_file() and stderr.stat().st_size:
        report['stderr_SHA256']=digest(read(stderr,1<<20))
    stdout=SCOPE/'private.stdout'
    if stdout.is_file() and stdout.stat().st_size:
        raw=read(stdout,1<<20)
        report['stdout_SHA256']=digest(raw)
        results=[json.loads(line[len(PREFIX):]) for line in raw.decode().splitlines() if line.startswith(PREFIX)]
        need(len(results)<=1,'AT_MOST_ONE_PRIVATE_GUEST_RECORD')
        if results:
            result=results[0]
            need(result['request']==REQUEST and result['step']==row[0] and result['GPU_calls']==0
                 and result['trainer_called'] is False and result['model_loaded'] is False,'BOUND_GUEST_RECORD')
            report['guest']=result
            report['passed']=row[2:4]==['COMPLETED','0:0'] and result['passed'] is True
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('install'); a.add_argument('packet',type=lambda x:Path(x).resolve(strict=True)); a.add_argument('source_sha')
    a=s.add_parser('once'); a.add_argument('authority',type=lambda x:Path(x).resolve(strict=True)); a.add_argument('source_sha')
    for name in ('host','guest'):
        a=s.add_parser(name); a.add_argument('authority_sha')
    s.add_parser('terminal')
    a=p.parse_args()
    if a.mode=='host':
        host(a.authority_sha)
        return 2
    if a.mode=='guest':
        return guest(a.authority_sha)
    val=install(a.packet,a.source_sha) if a.mode=='install' else once(a.authority,a.source_sha) if a.mode=='once' else reconciled()
    print(json.dumps(val,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
