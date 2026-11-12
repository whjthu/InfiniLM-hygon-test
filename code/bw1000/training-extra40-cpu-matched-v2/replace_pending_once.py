"""Replace only known unallocated1103563;preserve1087656 and any running replacement."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/training-extra40-cpu-matched-v2';LABEL='training-extra40-cpu-matched-v2'
epoch=time.time();anchor=time.monotonic();deadline=epoch+180
assert not W.exists()
def q(args,timeout=20):
    assert time.time()<deadline and time.monotonic()-anchor<180
    r=subprocess.run(args,capture_output=True,text=True,timeout=min(timeout,max(1,deadline-time.time())))
    assert len(r.stdout)<2*1024**2 and len(r.stderr)<65536
    return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
def fields(raw):return {x.split('=',1)[0]:x.split('=',1)[1] for x in raw.split() if '=' in x}
def job(n):
    r=q(['scontrol','show','job','-o',n]);assert r['rc']==0
    return fields(r['stdout'])
def dump(n,v):
    with (W/n).open('x') as f:json.dump(v,f,indent=2)
old=job('1087656');assert old['JobState']=='RUNNING' and old['NumNodes']=='40' and 'gres/dcu=320' in old['AllocTRES']
old_nodes=q(['scontrol','show','hostnames',old['NodeList']]);assert old_nodes['rc']==0
old_set=set(old_nodes['stdout'].split());assert len(old_set)==40
prior=job('1103563')
assert prior['JobName']=='training-extra40-long-20261112-v1' and prior['Comment']=='extra40-c5f4a0eef2ee4aca8879dbb1cafd667c'
if prior['JobState']=='RUNNING':
    print(json.dumps(dict(claim=CLAIM,request_id=REQUEST,preserved_existing_running=True,job_id='1103563',new_sbatch_submissions=0,new_job=prior)),flush=True)
    raise SystemExit(0)
assert prior['JobState']=='PENDING' and prior['AllocTRES'] in ('(null)','') and not prior['NodeList']
queue=q(['squeue','-h','-u',str(os.getuid()),'-o','%i|%100j|%T']);assert queue['rc']==0
assert not any(len(x.split('|'))>=2 and x.split('|')[1].strip()==LABEL for x in queue['stdout'].splitlines())
history=q(['sacct','-X','-n','-P','-u',str(os.getuid()),'-S','2026-11-12','--format=JobIDRaw,JobName%100,State']);assert history['rc']==0
assert not any(len(x.split('|'))>=2 and x.split('|')[1].strip()==LABEL for x in history['stdout'].splitlines())
W.mkdir(mode=0o700)
dump('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,old_parent='1087656',
    replace_only_job='1103563',replacement_requires_actual_CANCELLED_unallocated=True,requested_nodes=40,requested_DCU=320,
    CPU_per_node=64,time_limit='30-00:00:00',exclusive_mode='user',replay_allowed=False))
print(json.dumps(dict(stage='REPAIR_INTENT_SAVED',claim=CLAIM,request_id=REQUEST)),flush=True)
cancel_args=['scancel','--ctld','--state=PENDING','--user=aclsylqidf','--name=training-extra40-long-20261112-v1','1103563']
dump('cancel-attempt.json',dict(argv=cancel_args,job='1103563',original_parent_not_targeted=True))
cancel=q(cancel_args);dump('cancel-result.json',cancel);assert cancel['rc']==0
accounting=None;until=min(deadline,time.time()+35)
while time.time()<until:
    row=q(['sacct','-X','-n','-P','-j','1103563','--format=JobIDRaw,State,ExitCode,AllocNodes,NodeList'])
    assert row['rc']==0
    exact=[r.split('|') for r in row['stdout'].splitlines() if r.split('|')[0]=='1103563']
    if exact and exact[0][1].startswith('CANCELLED'):
        accounting=exact[0];break
    running=q(['squeue','-h','-j','1103563','-o','%T|%N'])
    assert running['rc']==0
    if running['stdout'].startswith('RUNNING|'):
        dump('preserved-running.json',dict(job_id='1103563',claim=CLAIM,request_id=REQUEST))
        print(json.dumps(dict(claim=CLAIM,request_id=REQUEST,preserved_existing_running=True,job_id='1103563',new_sbatch_submissions=0)),flush=True)
        raise SystemExit(0)
    time.sleep(2)
assert accounting and accounting[3]=='0' and accounting[4] in ('','None assigned','None')
dump('original1103563-terminal.json',dict(accounting_columns=['JobIDRaw','State','ExitCode','AllocNodes','NodeList'],row=accounting))
old_after_cancel=job('1087656');assert old_after_cancel['JobState']=='RUNNING' and old_after_cancel['NodeList']==old['NodeList']
script=b'#!/bin/bash\nexec /bin/sleep infinity\n'
with (W/'hold.sh').open('xb') as f:f.write(script)
args=['sbatch','--parsable','--partition=hx1hdnormal01','--account=aclsylqidf','--qos=user_aclsylqidf',
    '--nodes=40','--ntasks=320','--ntasks-per-node=8','--cpus-per-task=8','--gres=dcu:8','--mem=240000M',
    '--exclusive=user','--time=30-00:00:00','--no-requeue','--export=NONE','--job-name='+LABEL,
    '--comment=extra40-v2-'+REQUEST,'--chdir='+str(R),'--output='+str(W/'hold-%j.out'),'--error='+str(W/'hold-%j.err'),str(W/'hold.sh')]
dump('submission-attempt.json',dict(claim=CLAIM,request_id=REQUEST,argv=args,script_SHA256=hashlib.sha256(script).hexdigest(),
    previous1103563_actual_terminal=accounting,replay_allowed=False))
submitted=q(args,40);dump('sbatch-result.json',submitted)
result=dict(claim=CLAIM,request_id=REQUEST,sbatch_rc=submitted['rc'],job_id=None,old_parent='1087656',
    original_parent_unchanged=True,previous1103563_terminal=accounting,new_sbatch_submissions=1,requested_nodes=40,
    requested_DCU=320,requested_CPU_per_node=64,time_limit='30-00:00:00',exclusive_mode='user',replay_allowed=False,
    observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
if submitted['rc']==0:
    new_id=submitted['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',new_id)
    dump('submitted-job.json',dict(job_id=new_id,claim=CLAIM,request_id=REQUEST));result['job_id']=new_id
    new=job(new_id);assert new['JobName']==LABEL and new['Comment']=='extra40-v2-'+REQUEST
    result['new_job']={k:new.get(k) for k in ['JobId','JobName','JobState','Reason','NumNodes','NumCPUs','ReqTRES','AllocTRES','TimeLimit','StartTime','EndTime','NodeList','OverSubscribe']}
    if new['JobState']=='RUNNING':
        rows=q(['scontrol','show','hostnames',new['NodeList']]);assert rows['rc']==0
        nodes=rows['stdout'].split();assert len(nodes)==len(set(nodes))==40 and not set(nodes)&old_set
        assert 'gres/dcu=320' in new['AllocTRES']
        result.update(allocation_confirmed=True,new_nodes=nodes,confirmed_total_nodes=80,confirmed_total_DCU=640)
final_old=job('1087656');assert final_old['JobState']=='RUNNING' and final_old['NodeList']==old['NodeList']
dump('outcome.json',result);print(json.dumps(result),flush=True)
if submitted['rc']!=0:raise SystemExit(1)
