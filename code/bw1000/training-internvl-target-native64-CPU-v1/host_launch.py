"""Future installed HOST controller, one CPU step and bounded actual accounting."""
import argparse
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from scope import D,P,RUN,COLS,require,read,save,pin,verify,admission,remaining,limits
from slurm_accounting import wait_terminal,is_terminal


def main(publication,request,epoch):
    deadline=epoch+900;mono_end=time.monotonic()+deadline-time.time()
    admission(read(D/'CLAIM.json'),read(D/'PREREQUISITES.json'))
    verify();require(D==RUN/'source','exact installed source')
    require(not (RUN/'intent.json').exists() and not (RUN/'outcome.json').exists(),'original namespace no replay')
    def query(argv,own_end=None):
        seconds=min(12,remaining(deadline,mono_end,5))
        if own_end is not None:seconds=min(seconds,own_end-time.monotonic())
        require(seconds>0,'original accounting tail exhausted')
        value=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=seconds)
        require(value.returncode==0 and len(value.stdout)<=1024**2,'bounded HOST query failed')
        return value.stdout.strip().splitlines()
    def accounting(own_end=None):
        rows=[parts for line in query(['sacct','-n','-P','-j',P['parent'],'-o',COLS],own_end)
              if len(parts:=line.split('|'))==7 and parts[1]==P['label']]
        require(len(rows)<=1,'duplicate original CPU label')
        return rows[0] if rows else None
    rows=query(['squeue','-h','-j',P['parent'],'-o','%T|%D|%N']);require(len(rows)==1,'one original parent')
    state,count,nodes=[part.strip() for part in rows[0].split('|')]
    require((state,count)==('RUNNING','40') and P['node'] in query(['scontrol','show','hostnames',nodes]),'parent/node reservation differs')
    for line in query(['scontrol','show','job',P['parent'],'--oneliner']):
        found=re.search(r'\bUserId=\S+\((\d+)\)',line)
        require(found and int(found[1])==os.getuid(),'numeric parent owner differs')
    require(accounting() is None,'same label already exists; original request reconcile only')
    for line in query(['scontrol','show','step',P['parent'],'--oneliner']):
        fields=dict(word.split('=',1) for word in line.split() if '=' in word)
        if fields.get('StepId','').rsplit('.',1)[-1].isdigit():
            require(P['node'] not in query(['scontrol','show','hostnames',fields['NodeList']]),'target node has original numbered step')
    identity=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                  UID=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(RUN/'intent.json',dict(publication=publication,request_id=request,epoch=epoch,deadline=deadline,owner=identity,
                              source_manifest=verify(),case='T03.sft',planned_rows=64,CPU=8,GPU=0))
    bounds=limits(remaining(deadline,mono_end));work_deadline=min(deadline-90,time.time()+bounds['work_seconds'])
    command=['srun','--jobid='+P['parent'],'--job-name='+P['label'],'--exact','--immediate=5','--cpu-bind=cores',
             '-N1','-n1','-w',P['node'],'-c8','--mem=12288M','--gres=none',
             '--time=00:%02d:00'%(bounds['Slurm_seconds']//60),'/usr/bin/timeout','--kill-after=10s',str(bounds['work_seconds'])+'s',
             '/usr/local/python3.12/bin/python3','-B',str(D/'inside_stage.py'),'--epoch',str(epoch),'--deadline',str(deadline),'--work-deadline',str(work_deadline)]
    require(bounds['Slurm_seconds']+85<remaining(deadline,mono_end),'preparation used original reserve')
    save(RUN/'step-intent.json',dict(epoch=epoch,deadline=deadline,work_deadline=work_deadline,bounds=bounds,command=command))
    rc=None;error=None
    with (RUN/'stage.stdout').open('xb') as out,(RUN/'Slurm.stderr').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
        save(RUN/'owned-srun.json',dict(host=identity['host'],PID=child.pid,startticks=int(fields[19]),PGID=int(fields[2]),epoch=epoch,deadline=deadline))
        try:rc=child.wait(timeout=min(bounds['client_seconds'],remaining(deadline,mono_end,45)))
        except subprocess.TimeoutExpired:error='ORIGINAL_STEP_OUTCOME_UNKNOWN_NO_REPLAY'
    own_end=min(mono_end-5,time.monotonic()+35)
    errors=[]
    def observe():
        try:return accounting(own_end)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:errors.append(type(exc).__name__);return None
    visibility=wait_terminal(observe,own_end,interval=2,max_observations=17)
    row=visibility['row'];inside=read(RUN/'stage/outcome.json') if (RUN/'stage/outcome.json').exists() else None
    passed=bool(rc==0 and error is None and is_terminal(row) and row[2:4]==['COMPLETED','0:0']
                and row[5]==P['node'] and set(row[6].split(','))=={'cpu=8','mem=12G','node=1'}
                and inside and inside['passed'] and inside['step']==row[0] and inside['ownership_pre_post_verified'])
    result=dict(publication=publication,request_id=request,epoch=epoch,deadline=deadline,passed=passed,rc=rc,error=error,
                accounting_columns=COLS,accounting=visibility,accounting_errors=errors,inside=inside,
                actual_CPU_steps=1,actual_GPU_steps=0,planned_rows=64,completed_rows=inside['completed_rows'] if inside else 0,
                execution_state='TERMINAL' if is_terminal(row) else 'UNKNOWN',GPU_GO=False,Model_GO=False,replay_allowed=False)
    save(RUN/'outcome.json',result)
    print(__import__('json').dumps({key:result[key] for key in ('request_id','passed','execution_state','completed_rows','planned_rows','actual_GPU_steps')}))
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);parser.add_argument('--request',required=True);parser.add_argument('--epoch',type=float,required=True)
    a=parser.parse_args();raise SystemExit(main(a.publication,a.request,a.epoch))
