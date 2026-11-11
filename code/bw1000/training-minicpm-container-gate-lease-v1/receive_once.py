"""One owned CPU Slurm verification; preserve the earlier login-container timeout."""
import base64,hashlib,json,math,os,signal,socket,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/training-minicpm-container-gate-lease-v1'
label='training-minicpm-container-gate-lease-v1';node='f11r4n09';epoch=time.time();mono=time.monotonic();deadline=epoch+240
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
def remaining(reserve=0):
    value=min(deadline-time.time(),240-(time.monotonic()-mono))-reserve
    assert value>0;return value
def query(args):return subprocess.check_output(args,text=True,timeout=min(12,remaining(5))).strip()
def accounting():return [x.split('|') for x in query(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+label+'|' in x]
assert not W.exists() and not accounting()
parent=query(['squeue','-h','-j','1087656','-o','%T|%D|%N']).split('|');assert parent[:2]==['RUNNING','40']
assert node in query(['scontrol','show','hostnames',parent[2]]).splitlines()
jobinfo=query(['scontrol','show','job','1087656','--oneliner']);jobfields={x.split('=',1)[0]:x.split('=',1)[1] for x in jobinfo.split() if '=' in x};resources={k:jobfields.get(k) for k in ('NumCPUs','NumNodes','CPUs/Task','MinCPUsNode','AllocTRES','TimeLimit','JobState')}
W.mkdir(mode=0o700);(W/'source').mkdir()
for name,row in FILES.items():
    assert name in ('scope.py','snapshot.py','run_gate.py','slurm_accounting.py')
    raw=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256']
    with (W/'source'/name).open('xb') as f:f.write(raw)
def save(name,v):
    with (W/name).open('x') as f:json.dump(v,f,indent=2)
save('intent.json',dict(claim=CLAIM,epoch=epoch,deadline=deadline,host=socket.gethostname(),node=node,
                       parent=parent,actual_parent_resources=resources,files={n:x['SHA256'] for n,x in FILES.items()},GPU_cards=8))
command=['/usr/local/python3.12/bin/python3','-B','-u',str(W/'source/run_gate.py')]
args=['srun','--jobid=1087656','--job-name='+label,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,
      '--cpus-per-task=2','--mem=4096M','--gres=dcu:8','--immediate=5','--time=3',*command]
forced=False
with (W/'CPU.log').open('xb') as log:
    environment=os.environ.copy();environment.update(LEASE_GATE_EPOCH=str(epoch),LEASE_GATE_DEADLINE=str(epoch+120),PYTHONDONTWRITEBYTECODE='1');p=subprocess.Popen(args,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()
    save('owned-srun.json',dict(PID=p.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname()))
    try:p.wait(timeout=min(140,remaining(100)))
    except subprocess.TimeoutExpired:
        forced=True;os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=min(5,remaining(25)))
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
import sys
sys.path.insert(0,str(W/'source'))
from slurm_accounting import wait_terminal,is_terminal
initial_rows=accounting();assert len(initial_rows)<=1
if forced and initial_rows and not is_terminal(initial_rows[0]):
    assert initial_rows[0][0].startswith('1087656.') and initial_rows[0][0].split('.')[-1].isdigit()
    query(['scancel',initial_rows[0][0]])
terminal_observation=wait_terminal(lambda:(accounting() or [None])[0],time.monotonic()+min(30,remaining(70)),interval=2,max_observations=15)
rows=[terminal_observation['row']] if terminal_observation['terminal_visible'] else []
post=None
if len(rows)==1:
    name=label+'-post';postargs=['srun','--jobid=1087656','--job-name='+name,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,'--cpus-per-task=2','--mem=4096M','--gres=none','--immediate=5','--time=1','/usr/local/python3.12/bin/python3','-B',str(W/'source/snapshot.py'),'pre']
    with (W/'post.log').open('xb') as log:
        child=subprocess.Popen(postargs,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();save('post-owner.json',dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname()))
        post_forced=False
        try:child.wait(timeout=min(35,remaining(25)))
        except subprocess.TimeoutExpired:
            post_forced=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=min(5,remaining(15)))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
    def postrow():
        values=[x.split('|') for x in query(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+name+'|' in x]
        assert len(values)<=1
        return values[0] if values else None
    actual=wait_terminal(postrow,time.monotonic()+min(12,remaining(5)),interval=2,max_observations=6)
    raw=(W/'post.log').read_bytes();objects=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
    snapshot=objects[0] if len(objects)==1 else None
    post=dict(returncode=child.returncode,forced_cleanup=post_forced,terminal=actual,log_SHA256=hashlib.sha256(raw).hexdigest(),snapshot=snapshot)
    post['passed']=bool(child.returncode==0 and not post_forced and actual['terminal_visible'] and actual['row'][2:4]==['COMPLETED','0:0'] and snapshot and snapshot['kfd_holder_count']==0 and snapshot['full_fd_inventory_performed'] and snapshot['master_port_available'])
out=dict(claim=CLAIM,epoch=epoch,deadline=deadline,returncode=p.returncode,forced_cleanup=forced,
         accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'],accounting=rows,
         log_SHA256=hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest(),terminal_observation=terminal_observation,post=post,wall_seconds=time.monotonic()-mono,GPU_cards=8)
for name in ('CPU-result.json',):
    path=W/name
    if path.exists():
        raw=path.read_bytes();out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out['passed']=bool(p.returncode==0 and not forced and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
                   and out.get('CPU-result.json',{}).get('value',{}).get('passed')
                   and rows[0][5]==node and out['CPU-result.json']['value']['owner']['step']==rows[0][0] and post and post['passed']
                   and post['snapshot']['pci']==out['CPU-result.json']['value']['before']['pci']
                   and post['snapshot']['boot_id']==out['CPU-result.json']['value']['before']['boot_id'])
if not out['passed']:out['error_tail']=(W/'CPU.log').read_text(errors='replace')[-1800:]
save('outcome.json',out);print(json.dumps(out))
