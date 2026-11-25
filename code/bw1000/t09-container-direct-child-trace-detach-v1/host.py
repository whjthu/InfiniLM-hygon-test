"""One ordinary n13 compute step and a signal confined to its own strace process."""
import argparse
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import subprocess
import sys
import time


REQUEST='9eb24484e3c846a48b87f662bf3b8e65'
LABEL='t09-trace-detach-'+REQUEST[:12]
NODE='f13r3n13'
PARENT='1103606'
ROOT=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
STAGE=ROOT/'incoming'/LABEL
IMAGE=ROOT/'containers/dtk26-sandbox-r1'
ORIGINAL=ROOT/'results/training-hunyuanvideo-t09-sft-repeat2-ro393-v2/src'
OLD_STAGE=ROOT/'incoming/t09-miopen-single-bind-c6025678606b'
ELF=OLD_STAGE/'work-once/provider/libMIOpen.so.1.0'
ELF_SHA='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba'
PRODUCTION_SOURCE_SHA='1afde5eb02dc21eed84584d933f1ee0c58253700e5e90533b4c1ec409fe85279'
QUALIFICATION_SOURCE_SHA='ceae819436335cb5370f2bbc83669ab7bb209d4d9c80aa6cedfb85635596970e'
ARCHIVES={'3600':'d356addd3ef90db2dbe92dc1d86831126962260bf97ac5378dd863c03e7a8228',
          '3605':'b16aaa3df85d75479345cac12d0788853818f4ca7b3175c1e8509bf802d50783'}
CAP=256*1024
APPTAINER='/public/software/apps/apptainer/1.3.4/bin/apptainer'
TRACER='/usr/bin/strace'
CONTAINER_WORK='/t09-detach-work'


def need(flag,code):
    if not flag:
        raise ValueError(code)


def stable_read(path,cap):
    path=Path(path)
    need(path.resolve(strict=True)==path,'EXACT_EXPECTED_CANONICAL_FILE_PATH')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'BOUNDED_EXACT_FILE_SIZE')
        raw=stream.read(cap+1)
        after=os.fstat(fd)
    identity=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mode,row.st_mtime_ns,row.st_ctime_ns)
    need(len(raw)==before.st_size and identity(before)==identity(after)==identity(path.stat()),
         'EXACT_STABLE_READ_DESCRIPTOR')
    return raw,before


def value(path):
    st=path.stat()
    return [st.st_dev,st.st_ino]


def save(path,row):
    raw=(json.dumps(row,sort_keys=True)+'\n').encode()
    need(len(raw)<=32768,'BOUNDED_PRIVATE_JSON')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def atomic_allow(root,row):
    path=root/'allow-after-detach.json'
    pin=save(root/'allow.pending',row)
    os.link(root/'allow.pending',path,follow_symlinks=False)
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return pin


def verify_source(directory,want):
    raw,_=stable_read(directory/'SOURCE_FILES.json',32768)
    need(sha256(raw).hexdigest()==want,'EXACT_SOURCE_MANIFEST')
    manifest=json.loads(raw)
    for member,pin in manifest['files'].items():
        need(PurePosixPath(member).name==member,'EXACT_LOCAL_MEMBER_NAME')
        body,_=stable_read(directory/member,256*1024)
        need(len(body)==pin['bytes'] and sha256(body).hexdigest()==pin['SHA256'],'EXACT_FROZEN_SOURCE_MEMBER')
    return manifest


def projected_container(base,stage,work,context):
    start=base.index(APPTAINER)
    stop=base.index(str(IMAGE))
    flags=['exec','--rocm','--no-mount','home,cwd','--pwd','/']
    need(base[start+1:start+7]==flags,'EXACT_3600_ORIGINAL_PRODUCTION_IMAGE_LAYOUT')
    roots=base[start+7:stop]
    need(len(roots)==20 and all(roots[i]=='--bind' for i in range(0,20,2))
         and len({item.rsplit(':',2)[1] for item in roots[1::2]})==10,'EXACT_TEN_ORIGINAL_BIND_DESTINATIONS')
    readonly=[item[:-2]+'ro' if item.endswith(':rw') else item for item in roots]
    need(all(item.endswith(':ro') for item in readonly[1::2]),'EXACT_ORIGINAL_READONLY_TRAINING_ASSETS')
    need(base[stop+1:stop+5]==['/bin/bash','--noprofile','--norc','-c'],'ORIGINAL_BOUND_SHELL')
    old=base[stop+5]
    command='exec python3 -B -u "$@"'
    need(old.count(command)==1,'ORIGINAL_BOUND_TORCH_SEARCH_PROJECTION')
    setup=old.replace(command,('test -z "${LD_PRELOAD:-}${LD_AUDIT:-}${LD_PROFILE:-}"; '
        'export ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= '
        'HOME=/miopen-private-home-v3 '
        'MIOPEN_USER_DB_PATH=/miopen-private-home-v3/.config/miopen/user-db '
        'MIOPEN_CUSTOM_CACHE_DIR=/miopen-private-home-v3/.cache/miopen; '+command))
    bind=readonly+['--bind',f'{ELF}:/opt/dtk/lib/libMIOpen.so.1.0:ro',
         '--bind',f'{work/"private-home"}:/miopen-private-home-v3:rw',
         '--bind',f'{work/"guest-work"}:{CONTAINER_WORK}:rw']
    for file in ('guest.py','mount_observation.py'):
        bind+=['--bind',f'{stage/file}:/t09-detach-src/{file}:ro']
    return [APPTAINER,*flags,*bind,*base[stop:stop+5],setup,LABEL,'/t09-detach-src/guest.py',
            json.dumps(context,sort_keys=True)]


def trace_match(raw,row):
    """Match the first marker to the same real PID and open FD; paths stay private."""
    count=0
    after=0
    for line in raw.decode('utf8','surrogateescape').splitlines():
        if 'after.marker' in line and ('O_CREAT' in line or 'O_EXCL' in line):
            after+=1
        if f'"{CONTAINER_WORK}/before.marker.pending"' not in line or 'O_CREAT' not in line or 'O_EXCL' not in line:
            continue
        first=re.match(r'\s*(?:\[pid\s+)?(\d+)',line)
        opened=re.search(r'=\s+(\d+)<[^>]*?/before\.marker\.pending>\s*$',line)
        if first is not None and opened is not None and int(first[1])==row['pid'] and int(opened[1])==row['fd']:
            count+=1
    need(count==1 and after==0,'EXACT_TRACED_GUEST_FIRST_PID_FD_AND_NO_SECOND_OPEN')
    return dict(marker_pid_fd_matches=count,traced_second_marker=after)


def proc(tid):
    base=Path('/proc')/str(tid)
    raw=(base/'stat').read_text()
    fields=raw[raw.rfind(')')+2:].split()
    attr=dict(line.split(':',1) for line in (base/'status').read_text().splitlines() if ':' in line)
    need(len(fields)>=20 and len(attr['Uid'].split())==4 and
         {int(item) for item in attr['Uid'].split()}=={os.getuid()},'EXACT_PROBE_UID_IN_OWN_TREE')
    return dict(tid=tid,startticks=int(fields[19]),state=fields[0],ppid=int(fields[1]),
                tracer=int(attr['TracerPid']),tgid=int(attr['Tgid']))


def descendants(top):
    scoped={}
    todo=[top]
    while todo:
        pid=todo.pop()
        if pid in scoped:
            continue
        rows=[]
        for entry in (Path('/proc')/str(pid)/'task').iterdir():
            tid=int(entry.name)
            data=proc(tid)
            need(data['tgid']==pid and data['state'] not in ('Z','X'),'LIVE_IN_SCOPE_ABSENCE_OF_ZOMBIE')
            rows.append(data)
            todo.extend(map(int,(entry/'children').read_text().split()))
        need(rows and len(todo)+len(scoped)+len(rows)<=64,'BOUNDED_OWN_LIVE_CHILD_TREE')
        scoped[pid]=sorted(rows,key=lambda item:item['tid'])
    return scoped


def proc_identity(tree):
    return {(row['tgid'],row['tid'],row['startticks']) for rows in tree.values() for row in rows}


def detached(before,after,guest):
    need(proc_identity(before)==proc_identity(after) and guest in after,'EXACT_SAME_LIVE_TREE_BEFORE_AND_AFTER_DETACH')
    need(all(row['tracer']==0 for rows in after.values() for row in rows),'EVERY_LIVE_SCOPE_TASK_TRACER_PID_ZERO')
    return [row['tid'] for row in after[guest]]


def wait_until(action,deadline):
    while time.time()<deadline:
        try:
            if found:=action():
                return found
        except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):
            pass
        time.sleep(.03)
    raise ValueError('BOUNDED_HANDSHAKE_DEADLINE')


def inspect_marker(path,ctx,phase):
    raw,st=stable_read(path,8192)
    row=json.loads(raw)
    need(st.st_uid==os.getuid() and stat.S_IMODE(st.st_mode)==0o600 and st.st_nlink==1
         and row['request']==REQUEST and row['step']==ctx['step'] and row['uid']==os.getuid()
         and row['dev_inode']==[st.st_dev,st.st_ino] and row['pid']!=row['thread_tid'],
         'EXACT_PRIVATE_PERSISTENT_GUEST_MARKER')
    if phase=='after':
        need(row['genuine_marker_after_detach'] is True,'EXACT_AFTER_DETACH_MARKER')
    return row,sha256(raw).hexdigest()


def live(step):
    raw=subprocess.check_output(['scontrol','show','step',step,'--oneliner'],timeout=5,text=True)
    need(len(raw)<=2<<20,'BOUNDED_CURRENT_CONTROLLER')
    matches=[]
    for line in raw.splitlines():
        pairs=[part.split('=',1) for part in line.split() if '=' in part]
        row=dict(pairs)
        if row.get('StepId')==step:
            keys=[key for key,_ in pairs if key in ('StepId','State','NodeList','Name','JobName','TRES','AllocTRES')]
            need(len(keys)==len(set(keys)),'REPEATED_CONTROLLER_FIELD')
            matches.append(row)
    need(len(matches)==1,'ONE_REAL_CURRENT_NUMERIC_STEP')
    row=matches[0]
    fields=[row[key] for key in ('TRES','AllocTRES') if key in row]
    names=[row[key] for key in ('Name','JobName') if key in row]
    need(row['State']=='RUNNING' and row['NodeList']==NODE and names and set(names)=={LABEL}
         and fields and all(tres_allowed(item) for item in fields),'EXACT_TWO_CPU_TWO_GIB_NO_GRES')


def tres_allowed(value):
    fields=value.split(',')
    return len(fields)==3 and set(fields) in ({'cpu=2','mem=2G','node=1'},{'cpu=2','mem=2048M','node=1'})


def initial_tree(tracer,row):
    owner=os.getpid()
    overall=descendants(owner)
    if tracer not in overall:
        return None
    tree={pid:records for pid,records in overall.items() if pid!=owner}
    guest=row['pid']
    if guest not in tree:
        return None
    direct=[pid for pid,rows in tree.items() if pid!=tracer
            and next(item for item in rows if item['tid']==pid)['ppid']==tracer]
    need(next(item for item in tree[tracer] if item['tid']==tracer)['ppid']==owner
         and all(next(item for item in group if item['tid']==pid)['ppid'] in tree
                 for pid,group in tree.items() if pid!=tracer),'NO_INDEPENDENT_ADOPTED_CHILD_WHILE_TRACING')
    need(len(direct)==1 and len(tree[guest])>=2,'EXACT_ONE_TRACER_CHILD_AND_LIVE_GUEST_THREAD')
    target={item['tid']:item for item in tree[guest]}
    need(target[guest]['startticks']==row['startticks']
         and target[row['thread_tid']]['startticks']==row['thread_startticks']
         and all(item['tracer']==tracer for item in target.values())
         and all(item['tracer']==tracer for p,rows in tree.items() if p!=tracer for item in rows),
         'GUEST_PID_STARTTICKS_AND_ALL_TASKS_TRACED_BY_EXACT_TOOL')
    return direct[0],{pid:rows for pid,rows in tree.items() if pid!=tracer}


def read_live_trace(path):
    with path.open('rb') as stream:
        raw=stream.read(CAP+1)
    need(len(raw)<CAP,'EXACT_BOUNDED_PRIVATE_TRACE')
    return raw


def observation(work,ctx,argv,result):
    from resource import RLIMIT_FSIZE, setrlimit

    need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'ORDINARY_CHILD_SUBREAPER_UNAVAILABLE')
    setrlimit(RLIMIT_FSIZE,(CAP,CAP))
    guest_work=work/'guest-work'
    trace=work/'trace-private.log'
    tracer_path=Path(TRACER)
    tracer_stat=tracer_path.stat()
    need(stat.S_ISREG(tracer_stat.st_mode) and os.access(tracer_path,os.X_OK),'ACTUAL_N13_STRACE_EXECUTABLE')
    cmd=[TRACER,'-I','1','-f','-qq','-yy','-s','256','-e','trace=execve,open,openat,openat2,close',
         '-o',str(trace),*argv]
    deadline=ctx['deadline']
    need(deadline-time.time()>15,'SUFFICIENT_ORIGINAL_REMAINING_EPOCH')
    with (work/'launcher-stdout-private.log').open('xb') as stdout, (work/'launcher-stderr-private.log').open('xb') as stderr:
        tracer=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    tracer_fd=os.pidfd_open(tracer.pid)
    try:
        before,pin=wait_until(lambda:inspect_marker(guest_work/'before.marker',ctx,'before'),min(deadline-9,time.time()+35))
        result['before_marker_SHA256']=pin
        probe=wait_until(lambda:initial_tree(tracer.pid,before),min(deadline-8,time.time()+5))
        launcher,tree=probe
        fd=Path('/proc')/str(before['pid'])/'fd'/str(before['fd'])
        need(value(fd)==before['dev_inode'],'HOST_READONLY_FD_IDENTITY_MATCHES_FIRST_PERSISTENT_MARKER')
        result['first_trace_match']=trace_match(read_live_trace(trace),before)
        result['first_tree_SHA256']=save(work/'traced-tree.json',dict(tree=tree,launcher=launcher))
        need(tracer.poll() is None,'OWN_TRACER_EXITED_BEFORE_DIRECTED_SIGNAL')
        signal.pidfd_send_signal(tracer_fd,signal.SIGINT)
        rc=tracer.wait(timeout=max(.01,min(6,deadline-time.time()-5)))
        result['tracer_exit_rc']=rc
        need(rc in (0,130,-signal.SIGINT),'TRACE_DETACH_PROCESS_DID_NOT_FINISH_WITH_EXPECTED_INTERRUPT')
        owner=os.getpid()
        untraced={pid:group for pid,group in descendants(owner).items() if pid!=owner}
        tids=detached(tree,untraced,before['pid'])
        need(proc(launcher)['ppid']==os.getpid() and value(fd)==before['dev_inode'],
             'OWN_SUBREAPER_ADOPTED_THE_EXACT_LIVE_LAUNCHER_AND_FD')
        raw=read_live_trace(trace)
        result['final_trace_match']=trace_match(raw,before)
        result['trace_pin']=dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())
        result['detached_tree_SHA256']=save(work/'detached-tree.json',dict(tree=untraced,launcher=launcher))
        need(time.time()<deadline-3,'NO_TIME_LEFT_FOR_GENUINE_POST_DETACH_NORMAL_EXIT')
        permit=dict(ctx,pid=before['pid'],startticks=before['startticks'],dev_inode=before['dev_inode'],
                    guest_live_tids=tids,time_ns=time.time_ns())
        result['permission_SHA256']=atomic_allow(guest_work,permit)
        after,pin=wait_until(lambda:inspect_marker(guest_work/'after.marker',ctx,'after'),min(deadline-1,time.time()+10))
        result['after_marker_SHA256']=pin
        need(all(before[key]==after[key] for key in ('pid','startticks','thread_tid','thread_startticks'))
             and after['first_dev_inode']==before['dev_inode'] and after['dev_inode']!=before['dev_inode'],
             'SECOND_PERSISTENT_MARKER_ONLY_FROM_SAME_GUEST_AFTER_DETACH')
        def reap():
            got,status=os.waitpid(launcher,os.WNOHANG)
            return (got,status) if got else None
        _,status=wait_until(reap,deadline)
        need(os.WIFEXITED(status) and os.WEXITSTATUS(status)==0 and set(descendants(owner))=={owner},
             'ACTUAL_ADOPTED_CONTAINER_LAUNCHER_EXIT_NOT_C0_OR_LIVE_REMAINDER')
        result.update(actual_container_launcher_exit=0,traced_tasks=len(proc_identity(tree)),
                      detached_tasks=len(proc_identity(untraced)),actual_gpu=0,actual_model_calls=0,
                      genuine_second_marker=True,passed=True)
    finally:
        os.close(tracer_fd)


def run(args):
    stage=Path(STAGE)
    need(Path(__file__).resolve().parent==stage and stage.stat().st_uid==os.getuid()
        and stat.S_IMODE(stage.stat().st_mode)==0o700,'EXACT_PRIVATE_STAGED_SOURCE')
    manifest=verify_source(stage,args.manifest_sha)
    need(set(manifest['files'])=={'host.py','guest.py','mount_observation.py'}
         and {item.name for item in stage.iterdir()}==set(manifest['files'])|{'SOURCE_FILES.json'},
         'EXACT_STAGE_NO_PREVIOUS_WORK_ONCE')
    auth_raw,_=stable_read(stage.parent/(LABEL+'.authority.json'),8192)
    auth=json.loads(auth_raw)
    need(sha256(auth_raw).hexdigest()==args.authority_sha and auth['request']==REQUEST and auth['label']==LABEL
         and auth['archives']==ARCHIVES and auth['source_manifest_SHA256']==args.manifest_sha
         and auth['node']==NODE and auth['parent']==PARENT and auth['ordinary_zero_GRES_GO'] is True
         and auth['deadline']==auth['epoch']+120 and auth['epoch']<=time.time()<auth['deadline'],
         'EXACT_NEW_AUTHORITY_120S_EPOCH')
    verify_source(Path(ORIGINAL),PRODUCTION_SOURCE_SHA)
    verify_source(Path(OLD_STAGE),QUALIFICATION_SOURCE_SHA)
    sys.path.insert(0,str(ORIGINAL))
    from commands import argv
    from placement import numeric_cgroup

    step=numeric_cgroup()
    need(socket.gethostname().split('.')[0]==NODE and not os.environ.get('SLURM_STEP_GPUS')
         and not any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','LD_PROFILE')),'N13_NO_GPU_OR_LOADER_OVERRIDE')
    live(step)
    work=stage/'work-once'
    work.mkdir(mode=0o700)
    ctx=dict(request=REQUEST,label=LABEL,step=step,epoch=auth['epoch'],deadline=auth['deadline'],
             source_manifest_SHA256=args.manifest_sha,authority_SHA256=args.authority_sha)
    result=dict(ctx,passed=False,actual_gpu=0,actual_model_calls=0,actual_optimizer_updates=0)
    save(work/'intent.json',ctx)
    try:
        for part in ('guest-work','private-home','private-home/.config','private-home/.config/miopen',
                     'private-home/.config/miopen/user-db','private-home/.config/miopen/miopen-lockfiles',
                     'private-home/.cache','private-home/.cache/miopen'):
            (work/part).mkdir(mode=0o700)
        raw,info=stable_read(Path(ELF),18705784)
        need(len(raw)==18705784 and sha256(raw).hexdigest()==ELF_SHA and info.st_uid==os.getuid()
             and stat.S_IMODE(info.st_mode)==0o400,'EXACT_ORIGINAL_3600_PRIVATE_READONLY_ELF')
        del raw
        ctx.update(elf_dev_inode=[info.st_dev,info.st_ino],home_dev_inode=value(work/'private-home'),
                   work_dev_inode=value(work/'guest-work'))
        os.chdir('/')
        need(os.getcwd()=='/','ORIGINAL_CONTAINER_ROOT_CWD')
        docker=projected_container(argv('gpu',1200,auth['epoch'],'0'*64,PRODUCTION_SOURCE_SHA),stage,work,ctx)
        observation(work,ctx,docker,result)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as error:
        text=str(error)
        result.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                      error_code=text if re.fullmatch('[A-Z0-9_]{1,96}',text) else None)
    save(work/'outcome.json',result)
    print(json.dumps(result,sort_keys=True))
    return 0 if result['passed'] else 2


def terminal(args):
    stage=Path(STAGE)
    work=stage/'work-once'
    raw,_=stable_read(work/'outcome.json',32768)
    out=json.loads(raw)
    verify_source(stage,out['source_manifest_SHA256'])
    need(sha256(raw).hexdigest()==args.outcome_sha and out['request']==REQUEST and out['label']==LABEL
         and re.fullmatch(PARENT+r'\.\d+',out['step']) is not None,'EXACT_ONE_PRIOR_COMPUTE_OUTCOME')
    iraw,_=stable_read(work/'intent.json',32768)
    intent=json.loads(iraw)
    need(all(out[key]==value for key,value in intent.items()),'EXACT_COMPUTE_INTENT_BINDING')
    ledger=subprocess.check_output(['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-nP',
        '-j',out['step'],'-o','JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'],
        timeout=5,text=True)
    rows=[line.split('|') for line in ledger.splitlines() if line]
    need(len(ledger)<=2<<20 and len(rows)<=64 and all(len(row)==9 for row in rows),'EXACT_NINE_SLURM_ACCOUNTING_COLUMNS')
    exact=[row for row in rows if row[0]==out['step']]
    need(len(exact)==1,'ONE_EXACT_ACCOUNTING_STEP')
    row=exact[0]
    start,end=[datetime.strptime(item,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp() for item in row[7:9]]
    need(row[1]==LABEL and row[4].isdigit() and row[5]==NODE and tres_allowed(row[6])
         and intent['epoch']-1<=start<=end<=intent['deadline']+1
         and abs(end-start-int(row[4]))<=1,'ACTUAL_TERMINAL_SINGLE_EPOCH_AND_ZERO_GRES')
    if out['passed']:
        files=(('traced-tree.json','first_tree_SHA256'),('detached-tree.json','detached_tree_SHA256'),
               ('guest-work/allow-after-detach.json','permission_SHA256'))
        records=[]
        for name,key in files:
            body,_=stable_read(work/name,32768)
            need(sha256(body).hexdigest()==out[key],'INDEPENDENT_PHASE_JSON_SHA256')
            records.append(json.loads(body))
        before,pin=inspect_marker(work/'guest-work/before.marker',out,'before')
        after,pin_after=inspect_marker(work/'guest-work/after.marker',out,'after')
        old,new,permit=records
        int_keys=lambda tree:{int(k):v for k,v in tree.items()}
        threads=detached(int_keys(old['tree']),int_keys(new['tree']),before['pid'])
        tr,_=stable_read(work/'trace-private.log',CAP-1)
        fd_proof=trace_match(tr,before)
        need(out['trace_pin']==dict(bytes=len(tr),SHA256=sha256(tr).hexdigest())
             and out['first_trace_match']==out['final_trace_match']==fd_proof and
             pin==out['before_marker_SHA256'] and pin_after==out['after_marker_SHA256']
             and permit['pid']==before['pid']==after['pid'] and permit['startticks']==before['startticks']==after['startticks']
             and permit['dev_inode']==before['dev_inode']==after['first_dev_inode'] and permit['guest_live_tids']==threads
             and old['launcher']==new['launcher'] and out['actual_container_launcher_exit']==0
             and row[2:4]==['COMPLETED','0:0'],'ACTUAL_INDEPENDENT_ROUNDTRIP_AND_CONTAINER_C0')
    else:
        need(row[2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')
             and row[2:4]!=['COMPLETED','0:0'],'ACTUAL_FAILED_TERMINAL_RETAINED')
    print(json.dumps(dict(request=REQUEST,accounting=row,container_detach_qualified=out['passed'],
                          outcome_SHA256=args.outcome_sha,actual_GPU=0,formal_training_credit=0,
                          original_training_observer_ready=False),sort_keys=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    start=sub.add_parser('run-once')
    start.add_argument('--manifest-sha',required=True)
    start.add_argument('--authority-sha',required=True)
    verify=sub.add_parser('verify-terminal')
    verify.add_argument('--outcome-sha',required=True)
    args=parser.parse_args()
    raise SystemExit(run(args) if args.mode=='run-once' else terminal(args))
