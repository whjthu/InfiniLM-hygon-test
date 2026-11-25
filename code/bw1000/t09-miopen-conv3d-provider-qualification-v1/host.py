"""Project the exact four published launchers into scoped model-free mount operations."""
import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys
import time

from scope import ARCHIVE_SHA, SEALED_3583, REQUEST, LABEL, IMAGE, NODE, PARENT, ROLES, STAGE, phase_label, require

APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
GUEST='/miopen-mount-qualification'
GUEST_STATE='/miopen-private-state'
MEMBERS={'scope.py','host.py','inside.py','mount_observation.py','smoke.py','runtime_pci.py'}


def digest(raw):return sha256(raw).hexdigest()


def source(root,wanted):
    manifest=root/'SOURCE_FILES.json'
    raw=manifest.read_bytes()
    require(digest(raw)==wanted,'EXACT_DEPLOYED_SOURCE_MANIFEST')
    data=json.loads(raw)['files']
    for relative,pin in data.items():
        path=root/relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
                and path.resolve(strict=True)==path and path.is_file()
                and path.stat().st_size==pin['bytes'] and digest(path.read_bytes())==pin['SHA256'],
                'ORIGINAL_OR_DIAGNOSTIC_SOURCE_MEMBER_DIFFERS')
    return data


HOME_MEMBERS=('', '.config','.config/miopen','.config/miopen/user-db',
              '.config/miopen/miopen-lockfiles','.cache','.cache/miopen')


def private_home(stage):
    info=stage.stat()
    require(stage.resolve(strict=True)==stage and info.st_uid==os.getuid()
            and stat.S_IMODE(info.st_mode)==0o700,'STAGE_REQUIRES_OWNER_MODE_0700')
    home=stage/'private-home-once'
    home.mkdir(mode=0o700)
    for relative in HOME_MEMBERS[1:]:
        (home/relative).mkdir(mode=0o700)
    sampled={}
    for relative in HOME_MEMBERS:
        path=home/relative
        item=path.lstat()
        require(path.resolve(strict=True)==path and item.st_uid==os.getuid()
                and stat.S_ISDIR(item.st_mode) and stat.S_IMODE(item.st_mode)==0o700,
                'EXACT_NEW_PRIVATE_PATHS_REQUIRED')
        sampled[relative]=[item.st_dev,item.st_ino]
    return home,sampled


STAGE_KIND='T09_MIOPEN_SYNTHETIC_CONV3D'
PHASES=('pre','gpu','post')


def exclusive(path,record):
    raw=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<=65536,'BOUNDED_PRIVATE_RECEIPT')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return digest(raw)


def qualifier(raw,step,phase):
    require(0<len(raw)<=2<<20 and len(raw.splitlines())<=8192,'BOUNDED_LIVE_STEP')
    found=[]
    for line in raw.splitlines():
        values=[item.split('=',1) for item in line.split() if '=' in item]
        row=dict(values)
        if row.get('StepId')==step:
            keys=[k for k,v in values if k in ('StepId','Name','JobName','NodeList','State','TRES','AllocTRES')]
            require(len(keys)==len(set(keys)),'DUPLICATE_LIVE_SCOPE_FIELDS')
            found.append(row)
    require(len(found)==1,'ONE_NUMERIC_SCOPE')
    item=found[0]
    tags=[item[key] for key in ('Name','JobName') if key in item]
    views=[item[key].split(',') for key in ('TRES','AllocTRES') if key in item]
    base={'cpu=4','node=1'}|({'gres/dcu=1'} if phase=='gpu' else set())
    combinations=[base|{f'mem={mem}'} for mem in ('8192M','8G')]
    require(item.get('NodeList')==NODE and item.get('State')=='RUNNING'
         and tags and set(tags)=={phase_label(phase)} and views
         and all(len(view)==len(base)+1 and set(view) in combinations for view in views),'EXACT_PHASE_RESOURCE_BINDING')


def terminal(row,phase,epoch,dline,require_C0=False):
    if len(row)!=9:return False
    states=('COMPLETED',) if require_C0 else ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')
    base={'cpu=4','node=1'}|({'gres/dcu=1'} if phase=='gpu' else set())
    values=row[6].split(',')
    from datetime import datetime,timezone
    try:
        formats=r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d'
        if not all(re.fullmatch(formats,key) for key in row[7:9]):return False
        start,end=[datetime.fromisoformat(key).replace(tzinfo=timezone.utc).timestamp() for key in row[7:9]]
    except (TypeError,ValueError):
        return False
    return bool(re.fullmatch(PARENT+r'\.\d+',row[0]) and row[1]==phase_label(phase) and row[2] in states
            and re.fullmatch(r'\d+:\d+',row[3]) and (not require_C0 or row[3]=='0:0')
            and row[4].isdigit() and row[5]==NODE and len(values)==len(base)+1
            and set(values) in (base|{'mem=8G'},base|{'mem=8192M'})
            and epoch-1<=start<=end<=dline+1 and abs((end-start)-int(row[4]))<=1)


def actual_terminal(phase,step,epoch,dline,require_C0=False):
    text=subprocess.check_output(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-nP',
        '-j',step,'-o','JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'],
        timeout=6,text=True)
    require(0<len(text)<=2<<20,'BOUNDED_INDEPENDENT_ACCOUNTING')
    rows=[line.split('|') for line in text.splitlines() if line]
    require(len(rows)<=64 and all(len(row)==9 for row in rows),'EXACT_NINE_ACCOUNTING_COLUMNS')
    result=[row for row in rows if row[0]==step]
    require(len(result)==1 and terminal(result[0],phase,epoch,dline,require_C0),'EXACT_TERMINAL_SCOPE_STEP')
    return result[0]


def projection(base,context,stage,home):
    start=base.index(APPTAINER)
    stop=base.index(str(IMAGE))
    flags=['exec','--rocm','--no-mount','home,cwd','--pwd','/']
    require(base[start+1:start+7]==flags,'ORIGINAL_T09_IMAGE_FLAGS')
    bindings=base[start+7:stop]
    require(len(bindings)%2==0 and all(bindings[i]=='--bind' and bindings[i+1].endswith((':ro',':rw'))
            for i in range(0,len(bindings),2)),'EXACT_ORIGINAL_ASSET_ARGV')
    readonly=[value[:-2]+'ro' if value.endswith(':rw') else value for value in bindings]
    destinations=sorted({item.rsplit(':',2)[1] for item in readonly[1::2]})
    require(len(destinations)==10 and all(Path(item).is_absolute() for item in destinations),'TEN_ORIGINAL_EXPLICIT_RO_MOUNTS')
    require(base[stop+1:stop+5]==['/bin/bash','--noprofile','--norc','-c'],'ORIGINAL_T09_IMAGE_SHELL')
    readonly+=['--bind',f'{home}:/miopen-private-home-v3:rw']
    for name in ('scope.py','inside.py','smoke.py','mount_observation.py','runtime_pci.py'):
        readonly+=['--bind',f'{stage/name}:/t09-conv3d-probe/{name}:ro']
    script=base[stop+5]
    cutoff=script.index('exec python3 -B -u ')
    shell=script[:cutoff]+('test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; '
            'export HOME=/miopen-private-home-v3 '
            'MIOPEN_USER_DB_PATH=/miopen-private-home-v3/.config/miopen/user-db '
            'MIOPEN_CUSTOM_CACHE_DIR=/miopen-private-home-v3/.cache/miopen; exec python3 -B -u "$@"')
    return [APPTAINER,*flags,*readonly,*base[stop:stop+5],shell,phase_label('gpu'),
            '/t09-conv3d-probe/inside.py','--context-json',json.dumps(dict(context,asset_mounts=destinations),sort_keys=True)]


def pinned(path,expected):
    require(path.resolve(strict=True)==path and path.is_file(),'EXACT_ORIGINAL_SCOPE_AUTHORITY_PATH')
    with path.open('rb') as stream:raw=stream.read(65537)
    require(len(raw)<=65536 and digest(raw)==expected,'EXACT_PRIVATE_REQUEST_PIN')
    return json.loads(raw)


def authority(phase,args):
    record=pinned(STAGE.parent/(LABEL+'.'+phase+'.authority.json'),args.authority_sha)
    require(record['request']==REQUEST and record['phase']==phase and record['manifest_SHA256']==args.manifest_sha
            and record['parent']==PARENT and record['node']==NODE and record['root_approved'] is True
            and record['private_proof_archive_SHA256']==ARCHIVE_SHA and record['no_actual_model_gold_inputs'] is True
            and record['physical_and_scheduler_clearance'] is True and record['deadline']==record['epoch']+240
            and record['epoch']<=time.time()<record['deadline'],'EXACT_NEW_240S_ROOT_AUTHORITY')
    if phase=='gpu':
        require(record['single_DCU_GO'] is True and isinstance(record['current_route_SHA256'],str)
                 and re.fullmatch('[0-9a-f]{64}',record['current_route_SHA256'])
                 and record['sealed_3583_independent_SHA256']==SEALED_3583,'FRESH_EXPLICIT_ONE_DCU_APPROVAL')
    else:
        require(record['single_DCU_GO'] is False,'ZERO_GRES_PHASE_ONLY')
    return record


def snapshot():
    import snapshot as module
    value=module.snapshot(False)
    require(value['node']==NODE and value['kfd_holder_count']==0 and len(set(value['pci']))==8,
            'SAME_N13_EIGHT_DEVICE_HOST_KFD_FD_CLEAR')
    return value


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=PHASES);parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--authority-sha',required=True)
    a=parser.parse_args()
    result=dict(request=REQUEST,phase=a.phase,passed=False,actual_GPU_sdk_calls=0)
    try:
        require(Path(__file__).resolve().parent==STAGE and set(source(STAGE,a.manifest_sha))==MEMBERS,
                'EXACT_NEW_INSTALLED_SOURCE')
        require(socket.gethostname().split('.')[0]==NODE,'EXACT_N13_COMPUTE_HOST')
        decision=authority(a.phase,a)
        current=ROLES['t09']['current']
        source(current,ROLES['t09']['current_sha'])
        sys.path.insert(0,str(current))
        from placement import numeric_cgroup
        step=numeric_cgroup()
        result['step']=step
        raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],text=True,timeout=5)
        qualifier(raw,step,a.phase)
        require(not any(os.environ.get(key) for key in ('LD_PRELOAD','LD_AUDIT','LD_PROFILE')),
                'UNEXPECTED_EXISTING_HOST_LOADER_ENV')
        actual=snapshot()
        require(actual['step']==step,'SAME_CURRENT_PASSIVE_NUMERIC_STEP')
        result['physical']=actual
        result['epoch']=decision['epoch']
        result['authority_SHA256']=a.authority_sha
        if a.phase=='pre':
            result.update(passed=True,snapshot_is_current_scope=True)
            exclusive(STAGE/'pre.json',result)
        else:
            recorded=pinned(STAGE/'pre.json',decision['pre_SHA256'])
            require(recorded['request']==REQUEST and recorded['phase']=='pre' and recorded['passed'] is True
                 and recorded['epoch']==decision['epoch'] and actual['boot_id']==recorded['physical']['boot_id']
                 and actual['pci']==recorded['physical']['pci'],'BOUND_PRE_AND_CURRENT_SAME_BOOT_PCI')
            prereq=actual_terminal('pre',recorded['step'],decision['epoch'],decision['deadline'],True)
            result['pre_accounting']=prereq
            if a.phase=='gpu':
                require(0<=actual['monotonic_ns']-recorded['physical']['monotonic_ns']<=90_000_000_000,
                        'CURRENT_PRE_SAME_BOOT_MONOTONIC_EXPIRY')
                from commands import argv
                base=argv('gpu',1200,decision['epoch'],'a'*64,ROLES['t09']['current_sha'])
                home,expected=private_home(STAGE)
                context=dict(deadline=decision['deadline'],step=step,host_uid=os.getuid(),
                    host_private_identity=expected,host_boot_id=actual['boot_id'],host_PCI=actual['pci'],
                    authority_SHA256=a.authority_sha,asset_runtime_manifest_SHA256=ROLES['t09']['current_sha'])
                chosen=projection(base,context,STAGE,home)
                exclusive(STAGE/'gpu-intent.json',dict(request=REQUEST,step=step,epoch=decision['epoch'],
                          authority_SHA256=a.authority_sha,pre_SHA256=decision['pre_SHA256'],
                          argv_SHA256=digest(json.dumps(chosen).encode())))
                os.chdir('/')
                require(os.getcwd()=='/' and time.time()+25<decision['deadline'],'NO_CWD_BIND_AND_FINITE_GPU_BUDGET')
                print(json.dumps(dict(result,phase='GPU_HOST_QUALIFIED',actual_cgroup_and_controller=True),sort_keys=True),flush=True)
                os.execv(chosen[0],chosen)
            else:
                intent=pinned(STAGE/'gpu-intent.json',decision['gpu_intent_SHA256'])
                require(intent['request']==REQUEST and intent['epoch']==decision['epoch']
                        and intent['step']==decision['gpu_step'] and intent['pre_SHA256']==decision['pre_SHA256'],
                        'EXACT_ORIGINAL_GPU_HOST_INTENT')
                receipt=actual_terminal('gpu',intent['step'],decision['epoch'],decision['deadline'])
                result.update(passed=True,gpu_accounting=receipt,independent_physical_clearance=True)
                exclusive(STAGE/'post.json',result)
    except (OSError,TypeError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        code=str(error)
        result.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                      error_code=code if re.fullmatch('[A-Z0-9_]{1,96}',code) else None)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
