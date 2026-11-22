"""HOST-only keeper on one 8-DCU lease; fresh full HOST inventory before each arm."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import owned_processes as owned
from held_child import cleanup_held
from contract import *
from controller import container_command
from pair_compare import compare_pair


def write_at(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def lease_snapshot(arm):
    spec=importlib.util.spec_from_file_location('paired_HOST_snapshot',D/'snapshot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.NODE=NODE
    value=module.snapshot(False)
    port=PORTS[arm]
    with socket.socket() as s:s.bind(('127.0.0.1',port))
    value.update(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,arm=arm,
                 master_port=port,master_port_available=True)
    need(value['kfd_holder_count']==0 and value['full_fd_inventory_performed'] is True,'full fresh HOST FD/KFD admission failed')
    before,_=stable_json(PAIR_ROOT/'before.json')
    validate_lease(value,before,arm,time.monotonic_ns(),PARENT+'.'+os.environ['SLURM_STEP_ID'])
    write_at(PAIR_ROOT/arm/'lease-admission.json',value)
    return value


def main():
    verify('GPU');need(ARM is None,'HOST keeper cannot inherit a native arm')
    intent,ipin=stable_json(PAIR_ROOT/'intent.json');clock=Clock(intent['epoch']);prior_continuous()
    def stop(*_):raise TimeoutError('original paired lease stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,clock.left(CLEANUP))
    current=owned.snapshot().get(os.getpid());need(current is not None,'strong HOST lease owner missing')
    supervisor=dict(PID=os.getpid(),startticks=current['startticks'],uid=os.getuid(),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope_request_id=REQUEST)
    step=PARENT+'.'+os.environ['SLURM_STEP_ID']
    save('gpu-step-entry.json',dict(request_id=REQUEST,epoch=intent['epoch'],intent=ipin,node=NODE,step=step,supervisor=supervisor))
    arms={};allowners={};held_rows=[];failure=None;comparison=None;attempted=[]
    try:
        for arm in ('untraced','traced'):
            clock.left(INSIDE,60)
            folder=PAIR_ROOT/arm;folder.mkdir(mode=0o700)
            write_at(folder/'intent.json',dict(intent,arm=arm))
            lease_snapshot(arm) # new full HOST scan even before the second native process
            child=None;anchor={};registry={};held={};tree={};error=None
            try:
                env=dict(os.environ,GEMMA_LIGHT_DIAG_ARM=arm)
                with (folder/'container.stdout').open('xb') as out,(folder/'container.stderr').open('xb') as err:
                    attempted.append(arm)
                    child=subprocess.Popen(container_command(arm),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=env,start_new_session=True)
                    anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
                    row=owned.snapshot().get(child.pid);need(row is not None,'held container identity missing')
                    registry[child.pid]={k:row[k] for k in ('PID','PPID','PGID','startticks','state')}
                    write_at(folder/'container-owner.json',dict(**anchor,startticks=row['startticks'],host=supervisor['host'],boot_id=supervisor['boot_id']))
                    while child.poll() is None:
                        clock.left(INSIDE);owned.refresh(registry,folder)
                        need((folder/'container.stdout').stat().st_size<=2*1024**2 and (folder/'container.stderr').stat().st_size<=2*1024**2,'container log bound')
                        time.sleep(min(.25,clock.left(INSIDE)))
            except BaseException as e:error=dict(type=type(e).__name__,reason=str(e)[:512])
            finally:
                old=signal.signal(signal.SIGTERM,signal.SIG_IGN)
                try:held=cleanup_held(child,anchor,intent['epoch']+CLEANUP)
                except BaseException as e:error=dict(stage='held',type=type(e).__name__,previous=error)
                try:tree=owned.cleanup(registry,folder,grace=5)
                except BaseException as e:error=dict(stage='tree',type=type(e).__name__,previous=error)
                signal.signal(signal.SIGTERM,old)
                held_rows.append(held)
                for pid,row in registry.items():allowners[arm+':container:'+str(pid)]=row
            native,npin=stable_json(folder/'native-outcome.json')
            for pid,row in native['owners'].items():allowners[arm+':native:'+str(pid)]=row
            passed=bool(child and child.returncode==0 and error is None and native['passed'] is True
                and native['actual_updates']==5 and held.get('reaped') is True and not held.get('termination_needed')
                and tree.get('owned_processes_remaining')==0 and tree.get('termination_needed') is False)
            arms[arm]=dict(passed=passed,native_outcome=npin,actual_updates=native.get('actual_updates'),
                held=held,tree=tree,error=error,resources=native['resources'])
            need(passed,'independent native5 arm failed; do not start later arm')
        comparison=compare_pair(PAIR_ROOT);save('pair-comparison.json',comparison)
    except BaseException as e:failure=dict(type=type(e).__name__,reason=str(e)[:512])
    resources=dict(sampled_group_peak_RSS_bytes=max((r['resources']['sampled_group_peak_RSS_bytes'] for r in arms.values()),default=None),
                   arm_profiles_separate=True,RSS_peak_not_sum_for_sequential_arms=True)
    passed=failure is None and len(arms)==2 and all(r['passed'] for r in arms.values())
    lower_bound=sum(r['actual_updates'] for r in arms.values() if type(r.get('actual_updates'))is int)
    actual=10 if passed else None
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.setitimer(signal.ITIMER_REAL,0)
    save('native-outcome.json',dict(request_id=REQUEST,epoch=intent['epoch'],passed=passed,actual_updates=actual,
        planned_updates=10,planned_each_arm=5,observed_completed_updates_lower_bound=lower_bound,
        container_calls_attempted=attempted,native_arms_completed=sum(r['passed'] for r in arms.values()),failure=failure,arms=arms,
        supervisor=supervisor,owners=allowners,resources=resources,
        held_cleanup=dict(reaped=all(r.get('reaped') is True for r in held_rows),termination_needed=any(r.get('termination_needed') for r in held_rows)),
        paired_mechanism_passed=comparison.get('observed_noninterference_exact') if comparison else None,
        original860_recovery_requalified=False,Model_GO=False,formal17_GO=False))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
