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

from scope import ARCHIVE_SHA, HOME, IDS, IMAGE, LIBRARY_SHA, PARENT, ROLES, R, label, layout_for, require, staged

APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
GUEST='/miopen-mount-qualification'
GUEST_STATE='/miopen-private-state'
MEMBERS={'scope.py','host.py','inside.py'}


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
    gpus=mode=='conv'
    expected=[{'cpu=4','mem='+mem,'node=1'}|({'gres/dcu=1'} if gpus else set()) for mem in ('8G','8192M')]
    require(names and set(names)=={label(role,mode)} and row.get('NodeList')==ROLES[role]['node']
            and row.get('State')=='RUNNING' and tres
            and all(len(item)==3+gpus and set(item) in expected for item in tres),
            'NUMBERED_CPU_MEMORY_AND_DCU_SCOPE_DIFFERS')


def original(role,mode):
    variant='historical' if mode=='legacy' else 'current'
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


def prepare_private(path):
    path.mkdir(mode=0o700)
    for relative in ('home','home/.config','home/.config/miopen','home/.config/miopen/user-db',
          'home/.config/miopen/miopen-lockfiles','home/.cache','home/.cache/miopen'):
        (path/relative).mkdir(mode=0o700)


def command(base,context,stage,state,role,mode):
    first=base.index(APPTAINER)
    last=base.index(str(IMAGE))
    prefix=base[first:last]
    old_flags=['exec','--rocm','--no-home','--pwd','/']
    new_flags=['exec','--rocm','--no-mount','home,cwd','--pwd','/']
    flags=old_flags if mode=='legacy' else new_flags
    require(prefix[1:1+len(flags)]==flags,'ORIGINAL_CONTAINER_FLAGS_DIFFER')
    bindings=prefix[1+len(flags):]
    require(len(bindings)%2==0 and all(bindings[i]=='--bind' and bindings[i+1].endswith((':ro',':rw'))
            for i in range(0,len(bindings),2)),'ORIGINAL_EXPLICIT_BIND_LAYOUT_DIFFERS')
    require(base[last+1:last+5]==['/bin/bash','--noprofile','--norc','-c'],'ORIGINAL_IMAGE_SHELL_DIFFERS')
    shell=base[last+5]
    old=shell.index('exec python3 -B -u ')
    shell=shell[:old].rstrip()
    env='test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; '
    if mode!='conv':
        env+='export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; '
    simplified=[item[:-2]+'ro' if item.endswith(':rw') else item for item in bindings]
    destination=[item.rsplit(':',2)[1] for item in simplified[1::2]]
    require(all(Path(path).is_absolute() for path in destination),'EXPECTED_EXPLICIT_MOUNTPOINTS')
    context=dict(context,asset_mounts=sorted(set(destination)))
    if state is not None:
        simplified+=['--bind',f'{state}:{GUEST_STATE}:rw']
        private_home=Path(GUEST_STATE)/'home'
        if context['layout']=='NARROW_MIOPEN_SUBTREES':
            for parent in ('.config','.cache'):
                relative=parent+'/miopen'
                simplified+=['--bind',f'{state/"home"/relative}:{HOME/relative}:rw']
            selected=HOME
        else:
            selected=private_home
            env+='export HOME='+shlex.quote(str(private_home))+'; '
        selectors={'MIOPEN_USER_DB_PATH':selected/'.config/miopen/user-db',
                   'MIOPEN_CUSTOM_CACHE_DIR':selected/'.cache/miopen'}
        env+='export '+' '.join(key+'='+shlex.quote(str(value)) for key,value in selectors.items())+'; '
    for name in ('scope.py','inside.py'):
        simplified+=['--bind',f'{stage/name}:{GUEST}/{name}:ro']
    py='exec python3 -B -u "$@"'
    # The two originals end their setup with either a newline or semicolon.
    divider=' ' if shell.endswith(';') else '\n'
    return [APPTAINER,*flags,*simplified,*base[last:last+5],shell+divider+env+py,label(role,mode),
            f'{GUEST}/inside.py',role,mode,'--context-json',json.dumps(context,sort_keys=True)]


def independent_cpu(row,role,mode,step):
    require(len(row)==9 and row[0]==step and re.fullmatch(PARENT+r'\.[0-9]+',step)
            and row[1]==label(role,mode) and row[2:4]==['COMPLETED','0:0'] and row[4].isdigit()
            and int(row[4])<=120 and row[5]==ROLES[role]['node']
            and len(row[6].split(','))==3 and set(row[6].split(',')) in
                ({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
            and all(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',key) for key in row[7:9])
            and row[7]<=row[8], 'INDEPENDENT_EXACT_C0_ZERO_GRES_REQUIRED')


def approval(role,mode,path,wanted,source_pin):
    raw=path.read_bytes()
    require(0<len(raw)<=65536 and digest(raw)==wanted,'EXACT_ROOT_AUTHORITY')
    decision=json.loads(raw)
    require(decision['role']==role and decision['mode']==mode and decision['request']==IDS[role,mode]
            and decision['source_manifest_SHA256']==source_pin and decision['library_archive_SHA256']==ARCHIVE_SHA
            and decision['parent']==PARENT and decision['node']==ROLES[role]['node']
            and decision['root_approved'] is True and decision['no_gold_no_models'] is True
            and 0<decision['deadline']-time.time()<=180,'FRESH_EXACT_ROOT_AUTHORITY')
    context=dict(deadline=decision['deadline'])
    if mode in ('qualify','conv'):
        before=decision['current_observation']
        require(before['role']==role and before['request']==IDS[role,'current'] and before['mode']=='current'
                and before['passed'] is True and before['library']['SHA256']==LIBRARY_SHA
                and re.fullmatch('[0-9a-f]{64}',decision['current_observation_receipt_SHA256']),
                'FRESH_SAME_ROLE_READONLY_CURRENT_OBSERVATION')
        independent_cpu(decision['current_terminal'],role,'current',before['step'])
        context.update(layout=layout_for(before),variables=before['variables'],
            original_observation=decision['current_observation_receipt_SHA256'])
        if mode=='conv':
            qualified=decision['completed_qualification']
            require(qualified['role']==role and qualified['request']==IDS[role,'qualify'] and qualified['passed'] is True
                    and qualified['layout']==context['layout'] and qualified['model_free_mount_only'] is True
                    and decision['exclusive_single_DCU'] is True
                    and re.fullmatch('[0-9a-f]{64}',decision['qualification_receipt_SHA256']),
                    'INDEPENDENT_BOUNDED_PREVIOUS_MOUNT_QUALIFICATION_AND_ONE_DCU_REQUIRED')
            independent_cpu(decision['qualification_terminal'],role,'qualify',qualified['step'])
    return decision,context


def main():
    parse=argparse.ArgumentParser(description=__doc__)
    parse.add_argument('role',choices=ROLES)
    parse.add_argument('mode',choices=('legacy','current','qualify','conv'))
    parse.add_argument('--manifest-sha',required=True);parse.add_argument('--authority-sha',required=True)
    a=parse.parse_args()
    stage=staged(a.role,a.mode)
    try:
        require(Path(__file__).resolve().parent==stage,'EXACT_INSTALLED_DIAGNOSTIC_SOURCE_ROOT')
        require(set(source(stage,a.manifest_sha))==MEMBERS,'EXACT_SOURCE_MEMBER_SET')
        real=socket.gethostname().split('.')[0]
        step=os.environ.get('SLURM_STEP_ID','')
        require(real==ROLES[a.role]['node'] and os.environ.get('SLURM_JOB_ID')==PARENT and step.isdecimal(),
                'EXACT_BUSINESS_NODE_AND_PARENT')
        step=PARENT+'.'+step
        raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],timeout=5,text=True)
        live_tres(raw,step,a.role,a.mode)
        decision,context=approval(a.role,a.mode,stage.parent/(stage.name+'.authority.json'),a.authority_sha,a.manifest_sha)
        base,current=original(a.role,a.mode)
        context['asset_runtime_pins']=dict(source_manifest_SHA256=current,previous_step=ROLES[a.role][
            'historical_step' if a.mode=='legacy' else 'current_step'])
        cwd='/'
        history='CURRENT_SOURCE_NO_AUTO_HOME_CWD'
        if a.mode=='legacy':
            cwd=decision['historical_host_cwd']
            evidence=decision.get('historical_host_cwd_proof_SHA256')
            require(cwd in (str(HOME),'/') and (evidence is None or re.fullmatch('[0-9a-f]{64}',evidence)),
                    'BOUNDED_LEGACY_CWD_PROJECTION')
            history='RECONSTRUCTED_CWD_WITH_PIN' if evidence else 'RECONSTRUCTED_CWD_HYPOTHESIS'
        context['history_projection']=history
        state=None
        if a.mode in ('qualify','conv'):
            state=stage.parent/(stage.name+'.private-state')
            prepare_private(state)
        selected=command(base,context,stage,state,a.role,a.mode)
        os.chdir(cwd)
        require(os.getcwd()==cwd,'ACTUAL_REQUESTED_COMPUTE_HOST_CWD')
        print(json.dumps(dict(scope_request=IDS[a.role,a.mode],phase='HOST_QUALIFIED',step=step,
                              actual_host_cwd_mode=history,selected_layout=context.get('layout'),
                              argv_SHA256=digest(json.dumps(selected).encode())),sort_keys=True),flush=True)
        os.execv(selected[0],selected)
    except (OSError,TypeError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        code=str(error)
        print(json.dumps(dict(request=IDS[a.role,a.mode],phase='HOST_BLOCKED',actual_model_reads=0,
            errno=getattr(error,'errno',None),error_type=type(error).__name__,
            error_code=code if re.fullmatch('[A-Z0-9_]{1,120}',code) else None),sort_keys=True),flush=True)
    return 2


if __name__=='__main__':
    raise SystemExit(main())
