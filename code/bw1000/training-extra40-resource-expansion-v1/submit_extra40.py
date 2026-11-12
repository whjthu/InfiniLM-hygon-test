"""Submit one extra40 long reservation; preserve original1087656 and never replay."""
from pathlib import Path
import datetime,hashlib,json,os,re,socket,subprocess,time

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/training-extra40-resource-expansion-v1'
LABEL='training-extra40-long-20261112-v1'
assert re.fullmatch('[0-9a-f]{40}',CLAIM) and re.fullmatch('[0-9a-f]{32}',REQUEST)
epoch=time.time();anchor=time.monotonic();deadline=epoch+180
def query(args,timeout=20):
    assert time.time()<deadline and time.monotonic()-anchor<180
    r=subprocess.run(args,capture_output=True,text=True,timeout=min(timeout,max(1,deadline-time.time())))
    assert len(r.stdout.encode())<=1024**2 and len(r.stderr.encode())<=65536
    return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
def fields(raw):return {x.split('=',1)[0]:x.split('=',1)[1] for x in raw.split() if '=' in x}
def dump(n,v):
    with (W/n).open('x') as f:json.dump(v,f,indent=2)

assert not W.exists()
before=query(['scontrol','show','job','-o','1087656']);assert before['rc']==0
old=fields(before['stdout'])
assert old['JobState']=='RUNNING' and old['NumNodes']=='40' and 'gres/dcu=320' in old['AllocTRES']
assert old['Partition']=='hx1hdnormal01' and old['Account']=='aclsylqidf'
old_nodes=query(['scontrol','show','hostnames',old['NodeList']]);assert old_nodes['rc']==0
old_set=set(old_nodes['stdout'].split());assert len(old_set)==40
queue=query(['squeue','-h','-u',str(os.getuid()),'-o','%i|%100j|%T']);assert queue['rc']==0
assert not any(len(x.split('|'))>=2 and x.split('|')[1].strip()==LABEL for x in queue['stdout'].splitlines())
history=query(['sacct','-X','-n','-P','-u',str(os.getuid()),'-S','2026-11-12','--format=JobIDRaw,JobName%100,State']);assert history['rc']==0
assert not any(len(x.split('|'))>=2 and x.split('|')[1].strip()==LABEL for x in history['stdout'].splitlines())
W.mkdir(mode=0o700)
owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
intent=dict(claim=CLAIM,request_id=REQUEST,owner=owner,epoch=epoch,deadline=deadline,label=LABEL,
    original_parent='1087656',original_nodes=sorted(old_set),requested_extra_nodes=40,requested_extra_cards=320,
    requested_time_limit='30-00:00:00',replay_allowed=False,new_service=False,new_poller=False)
dump('intent.json',intent)
script=b'#!/bin/bash\nexec /bin/sleep infinity\n'
with (W/'hold.sh').open('xb') as f:f.write(script)
args=['sbatch','--parsable','--partition=hx1hdnormal01','--account=aclsylqidf','--qos=user_aclsylqidf',
    '--nodes=40','--ntasks=40','--ntasks-per-node=1','--cpus-per-task=64','--gres=dcu:8','--mem=240000M',
    '--exclusive','--time=30-00:00:00','--no-requeue','--export=NONE','--job-name='+LABEL,
    '--comment=extra40-'+REQUEST,'--chdir='+str(R),'--output='+str(W/'hold-%j.out'),'--error='+str(W/'hold-%j.err'),str(W/'hold.sh')]
dump('submission-attempt.json',dict(request_id=REQUEST,claim=CLAIM,submission_attempted=True,argv=args,script_SHA256=hashlib.sha256(script).hexdigest(),replay_allowed=False))
print(json.dumps(dict(stage='SUBMISSION_INTENT_SAVED',request_id=REQUEST,claim=CLAIM,owner=owner)),flush=True)
submitted=query(args,timeout=40)
dump('sbatch-result.json',submitted)
result=dict(claim=CLAIM,request_id=REQUEST,submitted_command_once=True,sbatch_rc=submitted['rc'],
    original_parent_unchanged=True,original_parent_before={k:old[k] for k in ['JobId','JobState','NumNodes','TimeLimit','NodeList','AllocTRES']},
    replay_allowed=False,extra_nodes_requested=40,extra_cards_requested=320,job_id=None,
    old_jobs_cancelled=0,GPU_initialized=False,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
if submitted['rc']==0:
    job=submitted['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job)
    result['job_id']=job
    dump('submitted-job.json',dict(job_id=job,request_id=REQUEST,claim=CLAIM))
    current=query(['scontrol','show','job','-o',job]);result['new_job_query']=current
    if current['rc']==0:
        new=fields(current['stdout']);assert new['JobName']==LABEL and new['Comment']=='extra40-'+REQUEST
        result['new_job']={k:new.get(k) for k in ['JobId','JobName','JobState','Reason','NumNodes','NumCPUs','ReqTRES','AllocTRES','TresPerNode','TimeLimit','StartTime','EndTime','NodeList','Partition','Account','QOS','Comment']}
        if new['JobState']=='RUNNING':
            expanded=query(['scontrol','show','hostnames',new['NodeList']]);assert expanded['rc']==0
            actual=expanded['stdout'].split();assert len(actual)==len(set(actual))==40 and not set(actual)&old_set
            assert 'gres/dcu=320' in new['AllocTRES']
            result.update(new_expanded_nodes=actual,total_unique_nodes=80,total_Slurm_allocated_DCU=640,allocation_confirmed=True)
after=query(['scontrol','show','job','-o','1087656']);assert after['rc']==0
old_after=fields(after['stdout']);assert old_after['JobState']=='RUNNING' and old_after['NumNodes']=='40' and old_after['NodeList']==old['NodeList']
result['original_parent_after']={k:old_after[k] for k in ['JobId','JobState','NumNodes','TimeLimit','NodeList','AllocTRES']}
dump('outcome.json',result);print(json.dumps(result),flush=True)
if submitted['rc']!=0:raise SystemExit(1)
