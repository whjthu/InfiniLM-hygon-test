"""One unique ordinary CPU0GRES dispatch and bounded actual terminal accounting."""
import argparse
import json
import os
import re
import subprocess
import time
from common import D,P,RUN,COLS,admission,verify,clock,remaining,save,read,need
from bounded import run,identity
from slurm_accounting import wait_terminal,is_terminal
from accounting_contract import select_original,actual_step_count

def main(publication,request,epoch):
    c=clock(epoch);mono=time.monotonic()+c['deadline']-time.time();admission();mf=verify()
    need(D==RUN/'source' and not (RUN/'intent.json').exists(),'unique installed namespace: UNKNOWN never replay')
    def query(argv,end=None):
        limit=min(10,remaining(c['deadline'],mono,8))
        if end is not None:limit=min(limit,end-time.monotonic())
        need(limit>0,'query tail budget exhausted')
        r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=limit)
        need(r.returncode==0 and len(r.stdout)<=1024**2,'bounded CPU allocation query failed');return r.stdout.strip().splitlines()
    observed_accounting_parent=None
    def accounting(end=None):
        nonlocal observed_accounting_parent
        snapshot=select_original(query(['sacct','-n','-P','-j',P['parent'],'-o',COLS],end),P['parent'],P['label'],P['node'])
        observed_accounting_parent=snapshot['parent']
        return snapshot['step']
    parent=query(['squeue','-h','-j',P['parent'],'-o','%T|%D|%N']);need(len(parent)==1,'one actual parent')
    state,count,nodes=parent[0].split('|')
    need(state=='RUNNING' and int(count)>0 and P['node'] in query(['scontrol','show','hostnames',nodes]),'current allocated CPU node unavailable')
    owner_rows=query(['scontrol','show','job',P['parent'],'--oneliner'])
    need(len(owner_rows)==1 and (u:=re.search(r'\bUserId=\S+\((\d+)\)',owner_rows[0])) and int(u[1])==os.getuid(),'actual numeric allocation owner')
    need(accounting() is None,'original label already exists: reconcile only')
    for line in query(['scontrol','show','step',P['parent'],'--oneliner']):
        f=dict(v.split('=',1) for v in line.split() if '=' in v)
        if f.get('StepId','').rsplit('.',1)[-1].isdigit():
            need(P['node'] not in query(['scontrol','show','hostnames',f['NodeList']]),'selected CPU node has active numbered step')
    slurm=min(240,60*int(remaining(c['deadline'],mono,50)//60));need(slurm>=120,'whole-minute Slurm budget exhausted')
    step_end=time.time()+slurm;need(step_end<=c['deadline']-45,'Slurm time would exceed original epoch')
    command=['srun','--jobid='+P['parent'],'--job-name='+P['label'],'--exact','--immediate=5','--cpu-bind=cores','-N1','-n1','-w',P['node'],
        '-c8','--mem=12288M','--gres=none','--time=00:%02d:00'%(slurm//60),'/usr/local/python3.12/bin/python3','-B',str(D/'inside_stage.py'),
        '--epoch',str(epoch),'--step-deadline',str(step_end)]
    save(RUN/'intent.json',dict(publication=publication,request_id=request,**c,owner=identity(),source_manifest=mf,planned_rows=72,CPU=8,GPU=0))
    save(RUN/'step-intent.json',dict(**c,step_deadline=step_end,Slurm_seconds=slurm,command=command))
    controlled=run(command,os.environ.copy(),RUN/'srun-control',RUN,min(c['deadline']-35,step_end+3),c['deadline']-23,c['deadline']-18)
    end=min(mono-5,time.monotonic()+12);errors=[]
    def observe():
        try:return accounting(end)
        except (OSError,ValueError,subprocess.SubprocessError) as e:errors.append(type(e).__name__);return None
    visible=wait_terminal(observe,end,interval=1,max_observations=10);row=visible['row']
    stage=read(RUN/'stage-outcome.json') if (RUN/'stage-outcome.json').exists() else None
    passed=bool(controlled['passed'] and is_terminal(row) and row[2:4]==['COMPLETED','0:0'] and row[5]==P['node']
        and set(row[6].split(','))=={'cpu=8','mem=12G','node=1'} and stage and stage['passed'] and stage['step']==row[0])
    value=dict(publication=publication,request_id=request,**c,passed=passed,controller=controlled,accounting_columns=COLS,accounting=visible,
        accounting_errors=errors,inside=stage,execution_state='TERMINAL' if is_terminal(row) else 'UNKNOWN',srun_invoked=True,srun_invoked_definition='bounded srun command invocation attempt; not proven Slurm allocation',actual_CPU_steps=actual_step_count(row,P['parent'],P['node']),actual_GPU_steps=0,accounting_parent=observed_accounting_parent,
        planned_rows=72,completed_rows=stage['completed_rows'] if stage else 0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    save(RUN/'outcome.json',value);print(json.dumps({k:value[k] for k in ('request_id','passed','execution_state','completed_rows','planned_rows','actual_GPU_steps')}))
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--request',required=True);p.add_argument('--epoch',type=float,required=True)
    a=p.parse_args();raise SystemExit(main(a.publication,a.request,a.epoch))
