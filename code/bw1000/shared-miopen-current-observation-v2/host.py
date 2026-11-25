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
import subprocess
import sys
import time

from scope import ARCHIVE_SHA, HOME, IDS, IMAGE, LIBRARY_SHA, PARENT, ROLES, R, label, require, staged

APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
GUEST='/miopen-mount-qualification'
GUEST_STATE='/miopen-private-state'
MEMBERS={'scope.py','host.py','inside.py','mount_observation.py'}


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


def live_tres(raw,step,role,mode):
    require(0<len(raw)<=2<<20 and len(raw.splitlines())<=8192,'BOUNDED_NUMBERED_STEP_RESPONSE')
    matched=[]
    for line in raw.splitlines():
        atoms=[entry.split('=',1) for entry in line.split() if '=' in entry]
        row=dict(atoms)
        if row.get('StepId')==step:
            keys=[key for key,_ in atoms if key in ('StepId','Name','JobName','NodeList','State','AllocTRES','TRES')]
            require(len(keys)==len(set(keys)),'DUPLICATE_NUMBERED_STEP_FIELDS')
            matched.append(row)
    require(len(matched)==1,'ONE_EXACT_SCOPE_NUMBERED_STEP')
    row=matched[0]
    names=[row[key] for key in ('Name','JobName') if key in row]
    tres=[row[key].split(',') for key in ('TRES','AllocTRES') if key in row]
    expected=[{'cpu=4','mem='+mem,'node=1'} for mem in ('8G','8192M')]
    require(names and set(names)=={label(role,mode)} and row.get('NodeList')==ROLES[role]['node']
            and row.get('State')=='RUNNING' and tres
            and all(len(item)==3 and set(item) in expected for item in tres),
            'NUMBERED_CPU_MEMORY_AND_DCU_SCOPE_DIFFERS')


def original(role,mode):
    variant='current'
    details=ROLES[role]
    directory=details[variant]
    declared=details[variant+'_sha']
    source(directory,declared)
    sys.path.insert(0,str(directory))
    if role=='gemma':
        argv=importlib.import_module('controller').container_command()
    else:
        argv=importlib.import_module('commands').argv('gpu',1200,time.time(),'a'*64,declared)
    return argv,declared


def command(base,context,stage,role):
    first=base.index(APPTAINER)
    last=base.index(str(IMAGE))
    prefix=base[first:last]
    flags=['exec','--rocm','--no-mount','home,cwd','--pwd','/']
    require(prefix[1:1+len(flags)]==flags,'ORIGINAL_CURRENT_CONTAINER_FLAGS_DIFFERS')
    bindings=prefix[1+len(flags):]
    require(len(bindings)%2==0 and all(bindings[i]=='--bind' and bindings[i+1].endswith((':ro',':rw'))
            for i in range(0,len(bindings),2)),'ORIGINAL_EXPLICIT_BIND_LAYOUT_DIFFERS')
    require(base[last+1:last+5]==['/bin/bash','--noprofile','--norc','-c'],'ORIGINAL_IMAGE_SHELL_DIFFERS')
    shell=base[last+5]
    cutoff=shell.index('exec python3 -B -u ')
    shell=shell[:cutoff].rstrip()
    environment=('test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; '
                 'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; ')
    readonly=[item[:-2]+'ro' if item.endswith(':rw') else item for item in bindings]
    destinations=[item.rsplit(':',2)[1] for item in readonly[1::2]]
    require(all(Path(path).is_absolute() for path in destinations),'EXPECTED_EXPLICIT_MOUNTPOINTS')
    context=dict(context,asset_mounts=sorted(set(destinations)))
    for name in ('scope.py','inside.py','mount_observation.py'):
        readonly+=['--bind',f'{stage/name}:{GUEST}/{name}:ro']
    divider=' ' if shell.endswith(';') else '\n'
    return [APPTAINER,*flags,*readonly,*base[last:last+5],shell+divider+environment+'exec python3 -B -u "$@"',
            label(role,'current'),f'{GUEST}/inside.py',role,'--context-json',json.dumps(context,sort_keys=True)]


def authority(role,args,stage):
    path=stage.parent/(stage.name+'.authority.json')
    require(path.resolve(strict=True)==path and path.is_file(),'EXACT_READONLY_OWNER_AUTHORITY_PATH')
    raw=path.read_bytes()
    require(0<len(raw)<=65536 and digest(raw)==args.authority_sha,'EXACT_READONLY_OWNER_AUTHORITY')
    body=json.loads(raw)
    require(body['role']==role and body['mode']=='current-v2' and body['request']==IDS[role]
            and body['source_manifest_SHA256']==args.manifest_sha and body['library_archive_SHA256']==ARCHIVE_SHA
            and body['parent']==PARENT and body['node']==ROLES[role]['node'] and body['root_approved'] is True
            and body['no_gold_no_models'] is True and body['readonly_only'] is True
            and 0<body['deadline']-time.time()<=120,'FRESH_EXACT_READONLY_ROOT_AUTHORITY')
    return body


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=ROLES)
    p.add_argument('--manifest-sha',required=True);p.add_argument('--authority-sha',required=True)
    args=p.parse_args()
    role=args.role
    stage=staged(role,'current')
    try:
        require(Path(__file__).resolve().parent==stage
                and {path.name for path in stage.iterdir()}==MEMBERS|{'SOURCE_FILES.json'}
                and set(source(stage,args.manifest_sha))==MEMBERS,'EXACT_READONLY_INSTALLED_SOURCE')
        host=socket.gethostname().split('.')[0]
        number=os.environ.get('SLURM_STEP_ID','')
        require(host==ROLES[role]['node'] and os.environ.get('SLURM_JOB_ID')==PARENT and number.isdecimal()
                and not os.environ.get('SLURM_STEP_GPUS'),'EXACT_ZERO_GRES_BUSINESS_NODE')
        step=PARENT+'.'+number
        raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],timeout=5,text=True)
        live_tres(raw,step,role,'current')
        decision=authority(role,args,stage)
        original_argv,current=original(role,'current')
        context=dict(deadline=decision['deadline'],asset_runtime_pins=dict(source_manifest_SHA256=current,
                      previous_step=ROLES[role]['current_step']))
        selected=command(original_argv,context,stage,role)
        require(not any(os.environ.get(key) for key in ('LD_PRELOAD','LD_AUDIT','LD_PROFILE')),'UNEXPECTED_HOST_LOADER_ENV')
        os.chdir('/')
        require(os.getcwd()=='/','ACTUAL_ROOT_CWD_FOR_CURRENT_PRODUCTION')
        print(json.dumps(dict(scope_request=IDS[role],phase='HOST_QUALIFIED',step=step,
                              argv_SHA256=digest(json.dumps(selected).encode()),all_explicit_binds_ro=True),
                         sort_keys=True),flush=True)
        os.execv(selected[0],selected)
    except (OSError,TypeError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        code=str(error)
        print(json.dumps(dict(request=IDS[role],phase='HOST_BLOCKED',actual_models_loaded=0,
            errno=getattr(error,'errno',None),error_type=type(error).__name__,
            error_code=code if re.fullmatch('[A-Z0-9_]{1,120}',code) else None),sort_keys=True),flush=True)
    return 2


if __name__=='__main__':
    raise SystemExit(main())
