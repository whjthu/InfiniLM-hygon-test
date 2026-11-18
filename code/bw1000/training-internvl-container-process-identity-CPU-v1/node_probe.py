import argparse,os,signal,subprocess,time,shlex
from pathlib import Path
from common import D,P,RUN,need,save,load,own,budget
from held_child import cleanup_held
from identity_probe import observe

def main(epoch,step_end):
    left=budget(epoch,90);need(os.environ.get('SLURM_JOB_ID')==P['parent'] and os.environ.get('SLURM_STEP_GPUS') in (None,'','(null)'),'CPU_STEP_CONTEXT')
    need(time.time()<step_end<=epoch+90,'CPU2_STEP_DEADLINE')
    cpus=sorted(os.sched_getaffinity(0));need(len(cpus)>=2,'INHERITED_CPU_AFFINITY');os.sched_setaffinity(0,cpus[:2])
    owner=own();owner['CPU_affinity']=sorted(os.sched_getaffinity(0));save(RUN/'node-owner.json',owner)
    need(Path('/opt/hyhal').resolve(strict=True)==Path('/usr/local/hyhal'),'KNOWN_HYHAL_ALIAS')
    exe=Path(P['container_launcher']);st=exe.stat();need(exe.is_file() and os.access(exe,os.X_OK),'ABSOLUTE_LAUNCHER_ACCESS')
    save(RUN/'launcher.json',dict(path=str(exe),resolved=str(exe.resolve()),stat={k:getattr(st,k) for k in ('st_dev','st_ino','st_uid','st_mode','st_size','st_mtime_ns','st_ctime_ns')}))
    argv=[str(exe),'exec','--contain','--no-home','--pwd','/','--bind',str(RUN)+':'+str(RUN)+':rw']
    for source,target in P['readonly_binds']:
        path=Path(source);need(path.is_dir() and not path.is_symlink(),'EXACT_READONLY_BIND');argv+=['--bind',source+':'+target+':ro']
    end=min(time.time()+20,step_end-15,epoch+75);need(end>time.time()+3,'HANDSHAKE_BUDGET')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
    shell='set -e; source /opt/dtk/env.sh >&2; export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; exec /usr/bin/python3 -I -S -B '+shlex.quote(str(D/'inside_handshake.py'))+' '+shlex.quote(str(RUN))+' '+str(end)
    argv += [P['container'],'/bin/bash','--noprofile','--norc','-c',shell]
    save(RUN/'container-intent.json',dict(epoch=epoch,deadline=epoch+120,handshake_deadline=end,command=argv,owner=owner))
    records=[];registry={};child=None;anchor=None;failure=None;held=None;samples=0
    def persist(row):
        need(len(records)<512,'IDENTITY_RECORD_BOUND');records.append(row)
        save(RUN/'identity-observations.json',dict(owner=owner,records=records,samples=samples,planned_root_processes=1,no_RSS=True,no_global_proc_scan=True))
    def interrupt(*_):raise TimeoutError('OWNED_FRONTEND_DEADLINE')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupt)
    signal.signal(signal.SIGALRM,interrupt);signal.setitimer(signal.ITIMER_REAL,max(.001,end-time.time()))
    mono=time.monotonic()+end-time.time()
    try:
        with (RUN/'container.stdout.private').open('xb') as out,(RUN/'container.stderr.private').open('xb') as err:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True,supervisor=owner);save(RUN/'held-child.json',anchor)
            while True:
                if child.poll() is not None:break
                samples+=1;need(samples<=400 and time.monotonic()<mono,'OBSERVATION_BOUND')
                queue=[(child.pid,None)];seen=set()
                while queue:
                    pid,parent=queue.pop(0)
                    if pid in seen:continue
                    seen.add(pid);need(len(seen)<=32,'OWNED_TREE_BOUND')
                    row,identity,children=observe(pid,registry.get(pid),parent,child,os.getuid(),persist)
                    if identity is not None:registry[pid]=identity
                    save(RUN/'owned-identities.json',dict(owner=owner,identities=registry))
                    need(row['accepted'],'IDENTITY_DIAGNOSTIC_REJECTED')
                    queue.extend((v,identity) for v in children)
                need(all((RUN/n).stat().st_size<=65536 for n in ('container.stdout.private','container.stderr.private')),'LOG_BOUND')
                if child.poll() is not None:break
                time.sleep(min(.05,max(0,mono-time.monotonic())))
    except BaseException as error:failure=dict(type=type(error).__name__,reason=str(error)[:256])
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,min(epoch+88,step_end-1)-time.time()))
        try:held=cleanup_held(child,anchor,min(end+12,step_end-2,epoch+87))
        except BaseException as error:held=dict(reaped=False,error_type=type(error).__name__)
    remaining=[];unknown=[]
    for pid,saved in registry.items():
        try:
            raw=Path('/proc',str(pid),'stat').read_text();v=raw.rsplit(') ',1)[1].split()
            if int(v[19])==saved['startticks'] and v[0]!='Z':remaining.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
        except OSError:unknown.append(pid)
    complete=bool(records and held and held.get('reaped') and not remaining and not unknown)
    handshake=load(RUN/'handshake-done.json') if (RUN/'handshake-done.json').exists() else None
    result=dict(epoch=epoch,deadline=epoch+120,step=P['parent']+'.'+os.environ['SLURM_STEP_ID'],owner=owner,diagnostic_complete=complete,
        frontend_ready=bool(complete and failure is None and child.returncode==0 and handshake and handshake['Torch_imported'] is False),
        diagnostic_rejection=failure,observed_records=len(records),samples=samples,held_cleanup=held,remaining_known_owned=remaining,unknown_owned_liveness=unknown,
        tree_discovery_complete_at_every_sample=all(r['accepted'] for r in records),unobserved_descendants='UNKNOWN_ON_REJECTION',
        worker_calls=0,Torch_calls=0,model_calls=0,weight_body_reads=0,RSS_reads=0,GPU_GO=False,Model_GO=False)
    save(RUN/'node-result.json',result);signal.setitimer(signal.ITIMER_REAL,0);return 0 if complete else 1
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--epoch',type=float,required=True);a.add_argument('--step-end',type=float,required=True);v=a.parse_args();raise SystemExit(main(v.epoch,v.step_end))
