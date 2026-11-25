"""Zero-GRES compute HOST gate and exact T09 production-image command projection."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

REQUEST='1a5d558b403a416eb234a9a56aaeee52'
LABEL='t09-miopen-import-maps-1a5d558b403a'
PARENT,NODE='1103606','f13r3n13'
PROJECT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PREVIOUS=PROJECT/'results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2/src'
PREVIOUS_SHA='1afde5eb02dc21eed84584d933f1ee0c58253700e5e90533b4c1ec409fe85279'
IMAGE=PROJECT/'containers/dtk26-sandbox-r1'
GUEST='/t09-miopen-probe/inside.py'


def need(ok,code):
    if not ok:raise ValueError(code)


def qualify(raw,step):
    need(0<len(raw)<=2<<20 and len(raw.splitlines())<=8192,'BOUNDED_NUMERIC_CONTROLLER_RESPONSE')
    exact=[]
    for line in raw.splitlines():
        tokens=[word.split('=',1) for word in line.split() if '=' in word]
        row=dict(tokens)
        if row.get('StepId')==step:
            keys=[key for key,_ in tokens if key in {'StepId','Name','JobName','NodeList','State','AllocTRES','TRES'}]
            need(len(set(keys))==len(keys),'DUPLICATE_STEP_FIELDS')
            exact.append(row)
    need(len(exact)==1,'ONE_EXACT_NUMBERED_CPU_STEP')
    row=exact[0]
    labels=[row[key] for key in ('Name','JobName') if key in row]
    values=[row[key].split(',') for key in ('TRES','AllocTRES') if key in row]
    need(labels and all(label==LABEL for label in labels) and row.get('NodeList')==NODE and row.get('State')=='RUNNING'
         and values and all(len(val)==3 and set(val) in ({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
                            for val in values),'ACTUAL_ZERO_GRES_4CPU8G_ONLY')
    return True


def check_source(root,want):
    raw=(root/'SOURCE_FILES.json').read_bytes()
    need(sha256(raw).hexdigest()==want,'SCOPED_DIAGNOSTIC_MANIFEST_DIFFERS')
    value=json.loads(raw)
    need({path.name for path in root.iterdir()}=={'SOURCE_FILES.json','host.py','inside.py','maps_probe.py'},'EXACT_READONLY_PROBE_SOURCE_LAYOUT')
    need(value['request']==REQUEST and set(value['files'])=={'host.py','inside.py','maps_probe.py'},'EXACT_THREE_SOURCE_ONLY_MEMBERS')
    for name,pin in value['files'].items():
        path=root/name
        payload=path.read_bytes()
        need(path.resolve(strict=True)==path and len(payload)==pin['bytes'] and sha256(payload).hexdigest()==pin['SHA256'],
             'SCOPED_DIAGNOSTIC_SOURCE_DIFFERS')


def image_projection(argv,inside,deadline):
    start=argv.index('/public/software/apps/apptainer/1.3.4/bin/apptainer')
    cut=argv.index(str(IMAGE))
    shell=cut+5
    need(argv[start+1:start+9]==['exec','--rocm','--no-mount','home,cwd','--pwd','/','--bind',f'{PROJECT}:{PROJECT}:ro']
         and argv[cut+1:shell]==['/bin/bash','--noprofile','--norc','-c'],'FROZEN_PRODUCTION_IMAGE_ENTRY_DIFFERS')
    bindings=argv[start+7:cut]
    need(len(bindings)%2==0 and all(bindings[i]=='--bind' and bindings[i+1].endswith((':ro',':rw'))
                                   for i in range(0,len(bindings),2)),'ORIGINAL_BIND_SEQUENCE_DIFFERS')
    readonly=[(item[:-2]+'ro' if item.endswith(':rw') else item) for item in bindings]
    script=argv[shell]
    old='exec python3 -B -u "$@"'
    need(script.count(old)==1,'EXACT_ORIGINAL_SHELL_ENTRY')
    guard=('test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"\\n'
           'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\\n')
    # Device selection remains empty. The inherited --rocm option is used
    # solely for identical library injection in an accounting-zero GRES step.
    return [*argv[start:start+7],*readonly,'--bind',f'{inside}:{GUEST}:ro','--bind',f'{inside.with_name("maps_probe.py")}:/t09-miopen-probe/maps_probe.py:ro',*argv[cut:shell],
            script.replace(old,guard.replace('\\n','\n')+old),LABEL,GUEST,'--deadline',str(deadline)]


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--manifest-sha',required=True)
    args.add_argument('--deadline',type=float,required=True)
    args.add_argument('--root-approved-link-injection',action='store_true')
    a=args.parse_args()
    local=Path(__file__).resolve().parent
    try:
        need(a.root_approved_link_injection and local.name==LABEL and 0<a.deadline-time.time()<=120,
             'ROOT_APPROVED_READONLY_LINK_INJECTION_REQUIRED')
        check_source(local,a.manifest_sha)
        node=socket.gethostname().split('.')[0]
        step=os.environ.get('SLURM_STEP_ID','')
        need(os.environ.get('SLURM_JOB_ID')==PARENT and step.isdecimal() and node==NODE
             and not os.environ.get('SLURM_STEP_GPUS'),'EXACT_COMPUTE_HOST_ZERO_GRES_ENTRY')
        number=PARENT+'.'+step
        raw=subprocess.check_output(['scontrol','show','step',number,'--oneliner'],text=True,timeout=5)
        qualify(raw,number)
        need(not any(os.environ.get(key) for key in ('LD_PRELOAD','LD_AUDIT','LD_PROFILE')),
             'UNEXPECTED_HOST_LOADER_SIDE_EFFECT_ENV')
        manifest=(PREVIOUS/'SOURCE_FILES.json').read_bytes()
        need(sha256(manifest).hexdigest()==PREVIOUS_SHA,'TARGET_ORIGINAL_T09_SOURCE_MANIFEST')
        for name,record in json.loads(manifest)['files'].items():
            path=PREVIOUS/name
            need(Path(name).name==name and path.resolve(strict=True)==path and path.is_file()
                 and path.stat().st_size==record['bytes'] and sha256(path.read_bytes()).hexdigest()==record['SHA256'],
                 'TARGET_ORIGINAL_T09_SOURCE_CLOSURE')
        sys.path.insert(0,str(PREVIOUS))
        from commands import argv
        original=argv('gpu',1200,a.deadline-120,'0'*64,PREVIOUS_SHA)
        selected=image_projection(original,local/'inside.py',a.deadline)
        os.chdir('/')
        need(os.getcwd()=='/','ROOT_CWD_BEFORE_PRODUCTION_IMAGE')
        print(json.dumps(dict(phase='HOST_QUALIFIED',scope_request=REQUEST,actual_step=number,
                              exact_zero_GRES=True,controller_stdout_SHA256=sha256(raw.encode()).hexdigest(),
                              argv_SHA256=sha256(json.dumps(selected).encode()).hexdigest(),all_binds_readonly=True),
                         sort_keys=True),flush=True)
        os.execv(selected[0],selected)
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as error:
        value=str(error)
        print(json.dumps(dict(request=REQUEST,phase='HOST_BLOCKED',actual_GPU_SDK_calls=0,
                              actual_db_cache_lock_writes=0,error_type=type(error).__name__,
                              errno=getattr(error,'errno',None),
                              error_code=value if type(error) is ValueError and re.fullmatch('[A-Z0-9_]{1,100}',value)
                              else None),sort_keys=True),flush=True)
    return 2


if __name__=='__main__':
    raise SystemExit(main())
