SOURCE_SHA='b6b403b8130b38607532e9807d8e072ec1c7ff0be836e2d69ac98fdf1ef61684'
CHECKPOINT_SHA='720b4adb7eea6173dd3fb2b9b834862fc371126031932fbe24e084efbeb352f0'
HASHES={'model_SHA256': '80e2f455942fadd91de23311c99500bc823636d1290e3be1ae0f5302a93e060b', 'optimizer_SHA256': 'b07973fed1b7c9a13105f822e2598c1c5248899bb83dc25a24618a3aeb308ea3', 'placements_SHA256': '683c1e25e5ab7ba0d6e1eaa5824167282bebe17157ea8b4e18c3e55fe3cc5069', 'rng_SHA256': '36bea4350d2e7d5c0f86eeffebf1fc308943bfb2cd478fbdcd5ff36878af7756', 'full_boundary_SHA256': '27e3e628dbb163cc96e8f0ca17d6978a1453dbd88c8f776d8e4973faa7302d6c'}
"""One exact stage, zero-GRES reload owner, or independent read of this scope."""
from base64 import b64decode
from contextlib import contextmanager
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
CP_PARENT=ROOT/'results/training-vggt-r45-original1318-resume10-n19-v3'
CP=CP_PARENT/'arms/resume10/train/checkpoint-25.pt'
FRAMEWORK=ROOT/'results/training-vggt-tum-eight-pairs25-recovery-v2/framework'
UPSTREAM=ROOT/'results/training-vggt-native-provider-HIP-CPU-v1/upstream'
PARENT,NODE='1103606','f13r1n12'
PYTHON='/usr/local/python3.12/bin/python3'
LABEL='vggt-3493-cp25-cpu-94b06a02507a'
STAGE='4fc55371ec05460a8d18015cdb438f3b'
REQUEST='94b06a02507a43a19f9276819b3e9578'
OBSERVE='9a2d3a884b7e453f9bff264702d03532'
RUN=ROOT/'incoming'/LABEL
SRC=RUN/'source'
BASE_CAP=415
CPU_CAP=360
PREFIX=b'VGGT_CP25_CPU '
COLUMNS='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'


def need(ok,reason):
    if not ok:raise ValueError(reason)


def pin(raw):
    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def encode(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()


def put(p,raw):
    with p.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return pin(raw)


def sync(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def read(path,max_):
    need(path.resolve(strict=True)==path,'CANONICAL_PRIVATE_ARTIFACT')
    with path.open('rb') as f:
        before=os.fstat(f.fileno())
        raw=f.read(max_+1)
        keys=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid)
        need(stat.S_ISREG(before.st_mode) and 0<len(raw)==before.st_size<=max_ and
             keys(before)==keys(os.fstat(f.fileno()))==keys(path.lstat()),'STABLE_BOUNDED_ARTIFACT')
    return raw


def source_check(raw,members):
    need(pin(raw)['SHA256']==SOURCE_SHA,'EXACT_PUBLISHED_SOURCE_MANIFEST')
    want=json.loads(raw)
    need(want['request']==REQUEST and set(want['files'])==set(members)==
         {'reload_native.py','UPSTREAM_FILES.json','CLAIM.json'},'SOURCE_ONLY_CLOSURE')
    need({name:pin(value) for name,value in members.items()}==want['files'],'EXACT_STAGED_SOURCE_MEMBERS')
    return want


def stage(a):
    need(a['request']==STAGE and re.fullmatch('[0-9a-f]{40}',a['publication']),
         'EXACT_NEW_SOURCE_STAGE')
    raw=b64decode(a['manifest'],validate=True)
    members={k:b64decode(v,validate=True) for k,v in a['files'].items()}
    source_check(raw,members)
    RUN.mkdir(mode=0o700);SRC.mkdir(mode=0o700)
    for k,v in members.items():
        put(SRC/k,v)
    put(SRC/'SOURCE_FILES.json',raw)
    carrier=b64decode(a['carrier'],validate=True)
    need(0<len(carrier)<=65536 and pin(carrier)['SHA256']==a['carrier_SHA256'],'EXACT_PRIVATE_CARRIER_BYTES')
    compile(carrier,'staged_original_carrier','exec')
    put(RUN/'original-carrier.py',carrier)
    sync(SRC)
    source_check(read(SRC/'SOURCE_FILES.json',16384),{p.name:read(p,65536) for p in SRC.iterdir() if p.name!='SOURCE_FILES.json'})
    value=dict(request=STAGE,scope=REQUEST,publication=a['publication'],source=pin(raw),
               carrier=pin(carrier),staged_objects=4,no_checkpoint_transfer=True,actual_GPU=0,passed=True)
    done=put(RUN/'installed.json',encode(value));sync(RUN)
    return dict(value,installed=done)


def installed(sha,carrier_sha):
    raw=read(RUN/'installed.json',16384);rec=json.loads(raw)
    need(pin(raw)['SHA256']==sha and rec['scope']==REQUEST and rec['source']['SHA256']==SOURCE_SHA
         and rec['carrier']==pin(read(RUN/'original-carrier.py',65536)) and rec['carrier']['SHA256']==carrier_sha,
         'EXACT_INSTALLED_SCOPE_AND_SELF')
    source_check(read(SRC/'SOURCE_FILES.json',16384),{p.name:read(p,65536) for p in SRC.iterdir() if p.name!='SOURCE_FILES.json'})
    return rec


def query(argv,seconds=8):
    done=subprocess.run(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S',*argv],
         stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=seconds)
    need(done.returncode==0 and len(done.stdout)<=2<<20 and len(done.stderr)<=65536,'BOUNDED_SCHEDULER_READ')
    return done.stdout.decode('utf-8').splitlines()


def fields(row):
    return dict(atom.split('=',1) for atom in row.split() if '=' in atom)


def hosts(text):
    return set(query(['scontrol','show','hostnames',text]))


def accounting():
    rows=[line.split('|') for line in query(['sacct','-nP','-j',PARENT,'-o',COLUMNS]) if line]
    need(all(len(row)==9 for row in rows),'EXACT_NINE_COL_ACCOUNTING')
    return [row for row in rows if row[1]==LABEL]


def fresh():
    parent=query(['squeue','-h','-j',PARENT,'-o','%T|%N'])
    need(len(parent)==1,'ONE_REAL_PARENT')
    status,nodes=parent[0].split('|')
    need(status=='RUNNING' and NODE in hosts(nodes),'EXACT_CURRENT_PARENT_NODE')
    controllers=[fields(line) for line in query(['scontrol','show','job',PARENT,'--oneliner'])]
    exact=[x for x in controllers if x.get('JobId')==PARENT]
    need(len(exact)==1 and exact[0]['JobState']=='RUNNING' and NODE in hosts(exact[0]['NodeList']),'REAL_DUAL_PARENT')
    details=query(['scontrol','show','job','-dd',PARENT])
    slots=[d for line in details if (d:=fields(line)).get('Nodes') and NODE in hosts(d['Nodes'])]
    need(len(slots)==1 and sum(1+int(r.split('-')[-1])-int(r.split('-')[0])
           for r in slots[0]['CPU_IDs'].split(','))>=4
         and slots[0]['Mem'].isdecimal() and int(slots[0]['Mem'])>=49152,'CURRENT_CPU_MEMORY_EXTENT')
    for line in query(['squeue','--steps','-h','-j',PARENT,'-o','%i|%N']):
        handle,nodes=line.split('|')
        if re.fullmatch(PARENT+r'\.[0-9]+',handle):
            need(NODE not in hosts(nodes),'ACTIVE_QUEUE_N12_EXCLUSION')
    for line in query(['scontrol','show','step',PARENT,'--oneliner']):
        x=fields(line)
        if re.fullmatch(PARENT+r'\.[0-9]+',x.get('StepId','')):
            need(x['State'] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY')
                 or NODE not in hosts(x['NodeList']),'ACTIVE_CONTROLLER_N12_EXCLUSION')
    need(not accounting(),'NEW_NUMERIC_LABEL_NEVER_SUBMITTED')


def tres(value):
    tokens=value.split(',')
    return len(tokens)==3 and set(tokens) in ({'cpu=4','mem=48G','node=1'},{'cpu=4','mem=49152M','node=1'})


def mapped():
    return [(SRC,'/audit-source'),(CP,'/checkpoint-25.pt'),(FRAMEWORK,'/pilot/framework'),
            (UPSTREAM,'/vggt-source'),(ROOT/'results/training-t06-sdxl-runtime-v1','/training-runtime'),
            (Path('/opt/hyhal'),'/opt/hyhal')]


def worker():
    import socket
    step=PARENT+'.'+os.environ['SLURM_STEP_ID']
    need(socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']==PARENT
         and not os.environ.get('SLURM_STEP_GPUS') and re.fullmatch(PARENT+r'\.[0-9]+',step)
         and any({'job_'+PARENT,'step_'+step.split('.')[1]}<=set(line.split(':',2)[-1].split('/'))
                 for line in Path('/proc/self/cgroup').read_text().splitlines()),'REAL_ZERO_GRES_ONE_NUMERIC_CGROUP')
    match=[r for line in query(['scontrol','show','step',step,'--oneliner'])
           if (r:=fields(line)).get('StepId')==step]
    need(len(match)==1 and match[0]['NodeList']==NODE and match[0]['State']=='RUNNING'
         and match[0].get('Name',match[0].get('JobName'))==LABEL and
         tres(match[0].get('TRES',match[0].get('AllocTRES',''))),'LIVE_4CPU_48G_ZERO_GRES')
    os.chdir('/')
    shell=('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; unset PYTHONPYCACHEPREFIX; '
           'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 '
           'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
           'PYTHONPATH=/training-runtime/site:/pilot/framework/generation/src:/pilot/framework; exec '+
           shlex.join(['python3','-B','-u','/audit-source/reload_native.py',step]))
    args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/']
    for source,target in mapped():
        args+=['--bind',f'{source}:{target}:ro']
    args += [str(ROOT/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
    os.execv(args[0],args)


def command():
    return ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none','-N1','-n1',
            '-c4','--mem=49152M','--gres=none','-w',NODE,'-t','00:06:00','--job-name='+LABEL,
            PYTHON,'-I','-S','-B','-u',str(RUN/'original-carrier.py'),'worker']


def row_valid(row,intent):
    def date(val):
        return datetime.fromisoformat(val).replace(tzinfo=timezone.utc).timestamp()
    return (len(row)==9 and re.fullmatch(PARENT+r'\.[0-9]+',row[0]) is not None and row[1]==LABEL
            and row[2:4]==['COMPLETED','0:0'] and row[4].isdecimal() and int(row[4])<=CPU_CAP
            and row[5]==NODE and tres(row[6]) and intent['epoch']-1<=date(row[7])<=date(row[8])<=intent['deadline'])


def native_stdout(raw):
    lines=[r[len(PREFIX):] for r in raw.splitlines() if r.startswith(PREFIX)]
    need(len(lines)==1,'ONE_ORIGINAL_CPU_REPORT')
    return json.loads(lines[0])


def report_ok(value,row):
    return (value['passed'] is True and value['step']==row[0] and value['actual_GPU']==value['optimizer_updates']==value['forward_calls']==0
        and (value['warmup_steps'],value['max_steps'],value['global_step'],value['next_batch_cursor'])==(5,20,25,1)
        and value['no_step26'] is value['original_GPU_rng_and_placements_record_verified'] is True
        and value['device_restore_verified'] is value['original_frozen_weight_payload_read'] is False and value['upstream']==69
        and value['checkpoint_bytes']==2594191443 and value['checkpoint_SHA256']==CHECKPOINT_SHA
        and value['saved_hashes']==HASHES and value['live_model_SHA256']==HASHES['model_SHA256']
        and value['live_optimizer_SHA256']==HASHES['optimizer_SHA256']
        and value['camera_named_tensors']>=value['optimizer_parameters']>0
        and value['optimizer_slot_count']==3*value['optimizer_parameters'])


def supervise():
    intent=json.loads(read(RUN/'launch-intent.json',16384))
    cmd=command()
    need(intent['argv']==pin(encode(cmd)) and 0<=time.time()-intent['epoch']<=12,'EXACT_FRESH_ONE_SHOT_OWNER')
    state=dict(request=REQUEST,passed=False,rc=None,stdout=None,stderr=None,actual_GPU=0)
    try:
        with (RUN/'step.stdout.private').open('xb') as out,(RUN/'step.stderr.private').open('xb') as err:
            proc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=min(367,intent['deadline']-time.time()-25))
        state['rc']=proc.returncode
        for name in ('stdout','stderr'):
            path=RUN/f'step.{name}.private'
            state[name]=pin(path.read_bytes()) if path.stat().st_size<=65536 else None
        state['native']=native_stdout(read(RUN/'step.stdout.private',65536))
        state['passed']=state['rc']==0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        state['error_type']=type(error).__name__
    put(RUN/'supervisor-result.json',encode(state));sync(RUN)


@contextmanager
def lock():
    import fcntl
    fd=os.open(ROOT/'results/.training80-dcu-standby.lock',os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        need(stat.S_ISREG(os.fstat(fd).st_mode),'ACTUAL_STANDBY_LOCK')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def launch(a):
    need(a['request']==REQUEST,'EXACT_ONE_LAUNCH_REQUEST')
    rec=installed(a['installed_SHA256'],a['carrier_SHA256'])
    authority=a['authority']
    now=time.time()
    need(authority['request']==REQUEST and authority['publication']==rec['publication'] and
         authority['parent']==PARENT and authority['node']==NODE and authority['one_zero_GRES_step'] is True
         and authority['fresh_custody_verified'] is True and 0<=now-authority['issued_epoch']<=90,
         'EXACT_FRESH_ROOT_N12_AUTHORITY')
    with lock():
        fresh()
        epoch=time.time()
        need(0<=epoch-authority['issued_epoch']<=90 and authority['valid_through_epoch']>=epoch+BASE_CAP,
             'ROOT_CURRENT_BOUNDED_CLOCK')
        obj=dict(request=REQUEST,epoch=epoch,deadline=epoch+BASE_CAP,publication=rec['publication'],
                 authority=pin(encode(authority)),installed_SHA256=a['installed_SHA256'],argv=pin(encode(command())))
        intent=put(RUN/'launch-intent.json',encode(obj));sync(RUN)
        with (RUN/'owner.stdout.private').open('xb') as out,(RUN/'owner.stderr.private').open('xb') as err:
            child=subprocess.Popen([PYTHON,'-I','-S','-B','-u',str(RUN/'original-carrier.py'),'supervise'],
                       stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19])
        started=put(RUN/'owner-started.json',encode(dict(pid=child.pid,startticks=ticks,intent=intent)))
        sync(RUN)
    return dict(request=REQUEST,passed=True,epoch=epoch,intent=intent,owner=started,once=True,actual_GPU=0)


def observe(a):
    need(a['request']==OBSERVE,'DISTINCT_INDEPENDENT_READ')
    installed(a['installed_SHA256'],a['carrier_SHA256'])
    raw=read(RUN/'launch-intent.json',16384);intent=json.loads(raw)
    need(pin(raw)['SHA256']==a['intent_SHA256'] and intent['installed_SHA256']==a['installed_SHA256']
         and intent['request']==REQUEST,'ORIGINAL_INTENT_BOUNDARY')
    rows=accounting()
    state_path=RUN/'supervisor-result.json'
    if not state_path.is_file():
        return dict(request=OBSERVE,passed=False,read_only=True,accounting=rows,supervisor_result_present=False,
                    planned_reload=1,qualified_reload=0,actual_GPU=0)
    state_raw=read(state_path,65536);state=json.loads(state_raw)
    passed=False
    if len(rows)==1 and row_valid(rows[0],intent) and state['request']==REQUEST and state['passed'] is True:
        stdout=read(RUN/'step.stdout.private',65536)
        passed=pin(stdout)==state['stdout'] and native_stdout(stdout)==state['native'] and report_ok(state['native'],rows[0])
    return dict(request=OBSERVE,passed=passed,read_only=True,accounting=rows,native=state.get('native'),supervisor=pin(state_raw),
                planned_reload=1,qualified_reload=int(passed),actual_GPU=0,device_restore_verified=False)


def main():
    need(os.getuid()==154459,'ORIGINAL_PROJECT_UID')
    os.umask(0o077)
    if sys.argv[1:]==['worker']:worker()
    elif sys.argv[1:]==['supervise']:supervise()
    else:
        a=json.load(sys.stdin)
        try:result={'stage':stage,'launch':launch,'observe':observe}[a['mode']](a)
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
            result=dict(request=a['request'],passed=False,error_type=type(error).__name__,reason_SHA256=pin(str(error).encode())['SHA256'])
        print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
