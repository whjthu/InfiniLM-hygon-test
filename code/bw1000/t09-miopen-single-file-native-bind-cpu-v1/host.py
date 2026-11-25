"""One zero-GRES compute owner for copy, independent readback, and single-file bind."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

from scope import (ARCHIVE_SHA, EXPECTED, EXPECTED_PIN, IMAGE, LABEL, NODE, PARENT, PREVIOUS_ROOT,
                   PREVIOUS_SHA, PRIVATE_LEAVES, REQUEST, STAGE, need)

APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
FILES={'host.py','inside.py','maps_probe.py','mount_observation.py','scope.py'}
GUEST_HOME='/miopen-private-home-v3'
GUEST_CODE='/t09-bound-miopen-probe'
WORK='work-once'


def digest(raw):return sha256(raw).hexdigest()


def exact_json(path,want=None,cap=128<<10):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and path.resolve(strict=True)==path and 0<before.st_size<=cap,
         'EXACT_BOUNDED_READONLY_RECORD')
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno());raw=stream.read(cap+1)
        need(identity(opened)==identity(before)==identity(os.fstat(stream.fileno())),'RECORD_CHANGED_WHILE_OPEN')
    need(identity(path.lstat())==identity(before) and len(raw)==before.st_size,'RECORD_PATH_CHANGED_WHILE_OPEN')
    sha=digest(raw)
    need(want is None or sha==want,'EXACT_METADATA_SHA256')
    return json.loads(raw),sha


def identity(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def source(root,pin):
    manifest,_=exact_json(root/'SOURCE_FILES.json',pin)
    for relative,wanted in manifest['files'].items():
        path=root/relative
        need(Path(relative).name==relative and path.resolve(strict=True)==path and path.is_file()
             and path.stat().st_size==wanted['bytes'] and digest(path.read_bytes())==wanted['SHA256'],
             'EXACT_ORIGINAL_SOURCE_MEMBER')
    return manifest


def record(path,body):
    raw=(json.dumps(body,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<=128<<10,'BOUNDED_PRIVATE_OWNER_PHASE_RECEIPT')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return digest(raw)


def quote_projection(base,stage,witness,mode,context):
    start=base.index(APPTAINER)
    stop=base.index(str(IMAGE))
    flags=['exec','--rocm','--no-mount','home,cwd','--pwd','/']
    need(base[start+1:start+7]==flags,'EXACT_ORIGINAL_T09_IMAGE_AND_SEARCH_FLAGS')
    items=base[start+7:stop]
    need(len(items)%2==0 and all(items[i]=='--bind' and items[i+1].endswith((':ro',':rw'))
            for i in range(0,len(items),2)),'EXACT_ORIGINAL_EXPLICIT_BIND_SEQUENCE')
    readonly=[item[:-2]+'ro' if item.endswith(':rw') else item for item in items]
    destinations=[item.rsplit(':',2)[1] for item in readonly[1::2]]
    need(len(set(destinations))==10 and all(Path(item).is_absolute() for item in destinations)
         and str(EXPECTED) not in destinations,'TEN_EXISTING_ASSETS_AND_NO_CANONICAL_BIND_CONFLICT')
    need(base[stop+1:stop+5]==['/bin/bash','--noprofile','--norc','-c'],'ORIGINAL_IMAGE_SHELL_ENTRY')
    for name in ('scope.py','inside.py','mount_observation.py','maps_probe.py'):
        readonly+=['--bind',f'{stage/name}:{GUEST_CODE}/{name}:ro']
    if mode=='copy':
        readonly+=['--bind',f'{witness/"provider"}:/t09-bound-lib-staging:rw']
    else:
        readonly+=['--bind',f'{witness/"private-home"}:{GUEST_HOME}:rw',
                   '--bind',f'{witness/"provider"/EXPECTED.name}:{EXPECTED}:ro']
    old=base[stop+5]
    command='exec python3 -B -u '
    need(old.count(command)==1,'UNCHANGED_PRODUCTION_PYTHON_ENVIRONMENT')
    setup=old[:old.index(command)]
    setup+=('test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; '
            'export ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=; ')
    if mode=='observe':
        setup+=(f'export HOME={GUEST_HOME} MIOPEN_USER_DB_PATH={GUEST_HOME}/.config/miopen/user-db '
                f'MIOPEN_CUSTOM_CACHE_DIR={GUEST_HOME}/.cache/miopen; ')
    setup+='exec python3 -B -u "$@"'
    return [APPTAINER,*flags,*readonly,*base[stop:stop+5],setup,LABEL,f'{GUEST_CODE}/inside.py',mode,
            '--context-json',json.dumps(dict(context,asset_mounts=sorted(set(destinations))),sort_keys=True)]


def direct_copy_readback(path,deadline):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode)==0o400 and before.st_uid==os.getuid()
         and before.st_nlink==1 and before.st_size==EXPECTED_PIN['bytes'] and path.resolve(strict=True)==path,
         'OWNER_READONLY_COPY_REQUIRED_BEFORE_PRODUCTION_BIND')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    total=0
    sha=sha256()
    try:
        need(identity(before)==identity(os.fstat(fd)),'COMPUTE_HOST_COPY_READBACK_FD_IDENTITY')
        for item in iter(lambda:os.read(fd,1<<20),b''):
            total+=len(item)
            need(time.time()<deadline and total<=EXPECTED_PIN['bytes'],'BOUND_COMPUTE_HOST_COPY_READBACK')
            sha.update(item)
        need(identity(before)==identity(os.fstat(fd))==identity(path.lstat()),'COMPUTE_HOST_COPY_STABLE')
    finally:os.close(fd)
    current=dict(bytes=total,SHA256=sha.hexdigest())
    need(current==EXPECTED_PIN,'COMPUTE_HOST_EXACT_REREAD_FULL_SHA256')
    return dict(pin=current,dev_inode=[before.st_dev,before.st_ino],owner_mode_octal='0400')


def projection_run(argv,phase,context):
    left=context['deadline']-time.time()
    need(left>2,'SINGLE_ORIGINAL_EPOCH_EXHAUSTED')
    done=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        timeout=min(left-1,55 if phase=='copy' else 65))
    need(len(done.stdout)<=2<<20 and len(done.stderr)<=2<<20,'ORIGINAL_IMAGE_BOUNDED_OUTPUT')
    found=[]
    for line in done.stdout.splitlines():
        try:item=json.loads(line)
        except (ValueError,UnicodeError):continue
        if isinstance(item,dict) and item.get('request')==REQUEST and item.get('phase')==phase:
            found.append(item)
    need(len(found)==1 and found[0].get('step')==context['step'],'ONE_SCOPE_COMPLETE_IMAGE_REPORT')
    return dict(rc=done.returncode,stdout_SHA256=digest(done.stdout),stderr_SHA256=digest(done.stderr),
                report=found[0])


def qualify_live(raw,step):
    need(len(raw)<=2<<20 and len(raw.splitlines())<=8192,'BOUNDED_LIVE_OBSERVATION')
    rows=[]
    for line in raw.splitlines():
        atoms=[item.split('=',1) for item in line.split() if '=' in item]
        current=dict(atoms)
        if current.get('StepId')==step:
            keys=[key for key,_ in atoms if key in ('StepId','State','NodeList','Name','JobName','TRES','AllocTRES')]
            need(len(keys)==len(set(keys)),'EXACT_NUMERIC_STEP_NO_DUPLICATE_FIELDS')
            rows.append(current)
    need(len(rows)==1,'EXACT_NUMERIC_STEP_SCOPE')
    value=rows[0]
    names=[value[key] for key in ('Name','JobName') if key in value]
    tres=[value[key].split(',') for key in ('AllocTRES','TRES') if key in value]
    need(names and set(names)=={LABEL} and tres and value['NodeList']==NODE and value['State']=='RUNNING'
         and all(len(row)==3 and set(row) in ({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
                 for row in tres),'LIVE_EXACT_NO_GRES_FOUR_CPU_EIGHT_GIB')


def authority(args):
    body,pin=exact_json(STAGE.parent/(LABEL+'.authority.json'),args.authority_sha)
    need(body['request']==REQUEST and body['manifest_SHA256']==args.manifest_sha and body['archive_SHA256']==ARCHIVE_SHA
         and body['node']==NODE and body['parent']==PARENT and body['root_approved_zero_GRES'] is True
         and body['root_approved_single_file_bind'] is True and body['deadline']==body['epoch']+120
         and body['epoch']<=time.time()<body['deadline'],'FRESH_EXACT_BOUND_ROOT_CPU_AUTHORITY')
    return body,pin


def run(args):
    need(Path(__file__).resolve().parent==STAGE and set(source(STAGE,args.manifest_sha)['files'])==FILES
         and {path.name for path in STAGE.iterdir()}==FILES|{'SOURCE_FILES.json'}
         and STAGE.stat().st_uid==os.getuid() and stat.S_IMODE(STAGE.stat().st_mode)==0o700,
         'EXACT_ONE_UNUSED_PRIVATE_SOURCE_STAGE')
    decision,auth=authority(args)
    source(PREVIOUS_ROOT,PREVIOUS_SHA)
    sys.path.insert(0,str(PREVIOUS_ROOT))
    from placement import numeric_cgroup
    from commands import argv
    step=numeric_cgroup()
    need(socket.gethostname().split('.')[0]==NODE and not os.environ.get('SLURM_STEP_GPUS')
         and not any(os.environ.get(key) for key in ('LD_PRELOAD','LD_AUDIT','LD_PROFILE')),
         'EXACT_NO_GPU_HOST_LOADER_AND_NODE')
    live=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],text=True,timeout=5)
    qualify_live(live,step)
    work=STAGE/WORK
    work.mkdir(mode=0o700)
    context=dict(request=REQUEST,step=step,epoch=decision['epoch'],deadline=decision['deadline'],
                 authority_SHA256=auth,source_manifest_SHA256=args.manifest_sha)
    record(work/'intent.json',context)
    result=dict(context,passed=False,mapped_copy_identity_verified=False,formal_performance_ready=False)
    try:
        (work/'provider').mkdir(mode=0o700)
        home=work/'private-home'
        home.mkdir(mode=0o700)
        for part in ('.config','.config/miopen','.config/miopen/user-db','.config/miopen/miopen-lockfiles','.cache','.cache/miopen'):
            (home/part).mkdir(mode=0o700)
        production=argv('gpu',1200,decision['epoch'],'0'*64,PREVIOUS_SHA)
        os.chdir('/')
        need(os.getcwd()=='/','PRESERVED_ROOT_CWD_NO_HOME_CWD_AUTO_MOUNTS')
        cp_cmd=quote_projection(production,STAGE,work,'copy',context)
        cp=projection_run(cp_cmd,'copy',context)
        result['copy_phase_SHA256']=record(work/'copy.json',cp)
        need(cp['rc']==0 and cp['report']['passed'] is True and cp['report']['copy_pin']==EXPECTED_PIN,
             'ORIGINAL_IMAGE_PROVIDER_COPY_DID_NOT_QUALIFY')
        check=direct_copy_readback(work/'provider'/EXPECTED.name,decision['deadline'])
        need(check['dev_inode']==cp['report']['copy_dev_inode'],'FIRST_AND_SECOND_READ_BIND_SOURCE_IDENTITY')
        result['compute_host_copy_readback_SHA256']=record(work/'compute-host-copy-readback.json',check)
        final=dict(context,host_private_copy_dev_inode=check['dev_inode'],
            host_private_home_dev_inode=[home.stat().st_dev,home.stat().st_ino])
        observe_cmd=quote_projection(production,STAGE,work,'observe',final)
        seen=projection_run(observe_cmd,'observe',final)
        result['import_phase_SHA256']=record(work/'import.json',seen)
        result['mapped_copy_identity_verified']=seen['report'].get('mapped_copy_identity_verified') is True
        result['mapping_outcome']=seen['report'].get('mapping_outcome')
        result['passed']=seen['rc']==0 and seen['report']['passed'] is True
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
        text=str(error)
        result.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
            error_code=text if re.fullmatch('[A-Z0-9_]{1,96}',text) else None)
    record(work/'outcome.json',result)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result['passed'] else 2


def terminal(args):
    work=STAGE/WORK
    out,_=exact_json(work/'outcome.json',args.outcome_sha)
    source(STAGE,out['source_manifest_SHA256'])
    intent,_=exact_json(work/'intent.json')
    need(out['request']==intent['request']==REQUEST and all(out[key]==intent[key] for key in (
        'step','epoch','deadline','authority_SHA256','source_manifest_SHA256')),'ORIGINAL_ONE_OWNER_INTENT_AND_OUTCOME')
    raw=subprocess.check_output(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-nP','-j',out['step'],
        '-o','JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'],timeout=6,text=True)
    rows=[line.split('|') for line in raw.splitlines() if line]
    need(len(raw)<=2<<20 and len(rows)<=64 and all(len(row)==9 for row in rows),'EXACT_NINE_BOUNDED_ACCOUNTING_COLUMNS')
    row,=[row for row in rows if row[0]==out['step']]
    start,end=[datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() for value in row[7:9]]
    need(re.fullmatch(PARENT+r'\.\d+',row[0]) and row[1]==LABEL and row[2:4]==['COMPLETED','0:0']
         and row[4].isdigit() and row[5]==NODE and len(row[6].split(','))==3
         and set(row[6].split(',')) in ({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
         and out['epoch']-1<=start<=end<=out['deadline']+1 and abs(end-start-int(row[4]))<=1
         and out['passed'] is True,'ACTUAL_EXACT_INDEPENDENT_C0_ZERO_GRES_TERMINAL')
    cp,_=exact_json(work/'copy.json',out['copy_phase_SHA256'])
    host,_=exact_json(work/'compute-host-copy-readback.json',out['compute_host_copy_readback_SHA256'])
    imp,_=exact_json(work/'import.json',out['import_phase_SHA256'])
    report=imp['report']
    from inside import select_bound
    dev,inode=host['dev_inode']
    bound_identity=[os.major(dev),os.minor(dev),inode]
    mapping=select_bound(report['after_import'],bound_identity)
    verified=mapping=='ONE_EXECUTABLE_GROUP_MATCHES_BOUND_FD' and report['bound_FD_lifetime_stable'] is True
    need(cp['report']['copy_pin']==host['pin']==EXPECTED_PIN and cp['report']['copy_dev_inode']==host['dev_inode']
         and cp['report']['file_and_directory_fsync'] is True and cp['rc']==imp['rc']==0
         and report['bound_FD_before']==report['bound_FD_after']==EXPECTED_PIN
         and report['canonical_single_file_mount']['dev_inode']==host['dev_inode']
         and report['bound_FD_identity']==bound_identity
         and not report['before_import']['groups'] and not report['before_import']['malformed']
         and out['mapping_outcome']==report['mapping_outcome']==mapping
         and out['mapped_copy_identity_verified'] is report['mapped_copy_identity_verified'] is verified
         and report['explicit_device_API_calls']==report['actual_GPU']==report['actual_conv_calls']==0,
         'EXACT_TWO_ROUNDTRIP_IMAGE_REPORTS_AND_HOST_REREAD')
    print(json.dumps(dict(request=REQUEST,actual_accounting=row,optional_mapped_identity=verified,
           mapping_outcome=mapping,fresh_private_copy=EXPECTED_PIN,formal_performance_ready=False),
           sort_keys=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    run_args=sub.add_parser('run-once');run_args.add_argument('--manifest-sha',required=True)
    run_args.add_argument('--authority-sha',required=True)
    verify=sub.add_parser('verify-terminal');verify.add_argument('--outcome-sha',required=True)
    ns=parser.parse_args()
    raise SystemExit(run(ns) if ns.mode=='run-once' else terminal(ns))
