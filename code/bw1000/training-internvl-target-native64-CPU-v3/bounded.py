"""Qualified RSS-v2 and held-child/owned-tree cleanup on each actual process boundary."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from common import save,need
from owned_processes import snapshot,refresh,cleanup
from held_child import cleanup_held
from owned_rss import sample_group_rss,RSS_SOFT_BYTES

def identity():
    f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(f[19]),state=f[0],UID=os.getuid(),
        host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def device_fds(pid):
    rows=[]
    for item in (Path('/proc')/str(pid)/'fd').iterdir():
        try:target=os.readlink(item)
        except FileNotFoundError:continue
        if target=='/dev/kfd' or target.startswith(('/dev/nvidia','/dev/dri/renderD')):rows.append(item.name)
    return rows

def run(argv,env,outdir,scope,stop,held_deadline,registry_deadline):
    need(time.time()<stop<held_deadline<registry_deadline,'shrinking child/cleanup clock')
    outdir.mkdir(mode=0o700);owner=identity();need(device_fds(owner['PID'])==[],'controller owns device FDs before worker')
    save(outdir/'owner.json',owner)
    child=None;anchor=None;registry={};failure=None;held=None;tree=None;post=None;errors={};phase='open_owned_logs'
    resources=dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_bytes=RSS_SOFT_BYTES,soft_limit_exceeded=False,hard_limit=False,last_sample=None)
    mono=time.monotonic()+stop-time.time()
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(*_):raise TimeoutError('original bounded owned-child deadline/termination')
    for s in handlers:signal.signal(s,interrupted)
    try:
        with (outdir/'stdout.private').open('xb') as out,(outdir/'stderr.private').open('xb') as err:
            phase='Popen_owned_child'
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            phase='save_held_anchor_and_actual_child_identity'
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner)
            save(outdir/'held-child-anchor.json',anchor)
            row=snapshot().get(child.pid);need(row is not None,'child identity unavailable; held cleanup remains mandatory')
            registry[child.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')}
            save(outdir/'child-owner.json',registry[child.pid])
            while child.poll() is None:
                phase='owned_tree_RSS_deadline_observation'
                refresh(registry,scope)
                checked=dict(registry);checked[owner['PID']]=owner
                good=sample_group_rss(checked,resources)
                save(outdir/'owned-registry.json',registry,replace=True)
                need(all((outdir/name).stat().st_size<=16*1024**2 for name in ('stdout.private','stderr.private')),'bounded child log exceeded16MiB')
                left=min(stop-time.time(),mono-time.monotonic())
                need(good and left>0,'RSS soft threshold or original worker deadline')
                try:child.wait(timeout=min(.25,left))
                except subprocess.TimeoutExpired:pass
    except BaseException as error:
        frames=traceback.extract_tb(error.__traceback__)
        own=[f for f in frames if Path(f.filename).resolve()==Path(__file__).resolve()]
        frame=own[-1] if own else None
        failure=dict(error_type=type(error).__name__,reason=str(error)[:256],phase=phase,
            source_linecode=dict(file='bounded.py',line=frame.lineno,function=frame.name) if frame else None)
    finally:
        for s in handlers:signal.signal(s,signal.SIG_IGN)
        try:held=cleanup_held(child,anchor,held_deadline)
        except BaseException as error:
            errors['held_cleanup']=type(error).__name__;held=dict(state='UNKNOWN',reaped=child is None or child.returncode is not None,termination_needed=True)
        try:tree=cleanup(registry,scope,grace=max(0,min(1,(registry_deadline-time.time())/2)))
        except BaseException as error:
            errors['registry_cleanup']=type(error).__name__;tree=dict(state='UNKNOWN',owned_processes_remaining=None,termination_needed=True)
        try:
            live=refresh(registry,scope)
            post=dict(same_host=socket.gethostname()==owner['host'],same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'],
                controller_device_FDs=device_fds(owner['PID']),held_child_reaped=child is None or child.returncode is not None,
                owned_descendants_remaining=len(live),owned_device_FD_holders={str(pid):device_fds(pid) for pid in live})
        except BaseException as error:post=dict(state='UNKNOWN',error_type=type(error).__name__)
        for s,h in handlers.items():signal.signal(s,h)
    rc=child.returncode if child else None
    passed=bool(rc==0 and failure is None and not errors and held.get('reaped') and not held.get('termination_needed')
        and tree.get('owned_processes_remaining')==0 and not tree.get('termination_needed') and post.get('same_host') and post.get('same_boot')
        and post.get('controller_device_FDs')==[] and post.get('owned_descendants_remaining')==0)
    value=dict(passed=passed,returncode=rc,owner=owner,child=anchor,failure=failure,resources=resources,
        RSS_is_sampled_soft_limit=True,RSS_unknown_not_zero=True,held_cleanup=held,registry_cleanup=tree,cleanup_errors=errors,post=post,
        stop=stop,held_deadline=held_deadline,registry_deadline=registry_deadline)
    save(outdir/'resources.json',resources)
    save(outdir/'outcome.json',value);return value
