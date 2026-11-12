"""One finite CPU Slurm step for the pinned paired operand algebra helper; never a GPU launcher."""
import base64,hashlib,json,math,os,shlex,signal,socket,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='training-sdxl-paired-attribution-helper-CPU-v1';W=R/'results'/label;node='f11r4n11';parent='1087656';epoch=time.time();anchor=time.monotonic();deadline=epoch+300
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def left(reserve=0):
 value=min(deadline-time.time(),300-(time.monotonic()-anchor))-reserve
 if not math.isfinite(value) or value<=0:raise TimeoutError('original CPU scope exhausted')
 return value
def query(argv):return subprocess.check_output(argv,text=True,timeout=min(12,left(5))).strip()
def accounting():return [row.split('|') for row in query(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+label+'|' in row]
def save(name,value):
 with (W/name).open('x') as stream:json.dump(value,stream,indent=2)
assert not W.exists() and not accounting(),'namespace or original label exists; reconcile without replay'
allocation=query(['squeue','-h','-j',parent,'-o','%T|%D|%N']).split('|');assert allocation[:2]==['RUNNING','40']
nodes=query(['scontrol','show','hostnames',allocation[2]]).splitlines();assert len(nodes)==len(set(nodes))==40 and node in nodes
for line in query(['squeue','--steps','-h','-j',parent,'-o','%i|%N']).splitlines():
 handle,hosts=line.split('|',1)
 if handle not in (parent+'.batch',parent+'.extern') and hosts not in ('','(null)'):assert node not in query(['scontrol','show','hostnames',hosts]).splitlines(),'CPU node has a live step'
job=query(['scontrol','show','job',parent,'--oneliner']);fields={x.split('=',1)[0]:x.split('=',1)[1] for x in job.split() if '=' in x}
W.mkdir(mode=0o700);(W/'source').mkdir();(W/'evidence').mkdir()
for name,row in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(row['b64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'];path=W/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(raw)
fm=json.loads((W/'source/FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==337 and fm['executed_source_subset_files']==6 and len(fm['files'])==6
for name,pin in fm['files'].items():
 p=W/'source/framework'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,work_deadline=epoch+220,host=socket.gethostname(),node=node,parent=parent,actual_nodes=nodes,actual_parent_resources={k:fields.get(k) for k in ('NumCPUs','NumNodes','AllocTRES','TimeLimit','JobState')},source_files={name:row['SHA256'] for name,row in FILES.items()},GPU_cards=0))
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',request_id=REQUEST,claim=CLAIM,epoch=epoch,deadline=deadline)),flush=True)
binds=[str(W/'source')+':/input-stage:ro',str(W/'source/framework')+':/input-framework:ro',str(W/'evidence')+':/evidence:rw',str(R/'results/training-t06-sdxl-runtime-v1')+':/training-runtime:ro','/opt/hyhal:/opt/hyhal:ro']
shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; export PYTHONPATH=/training-runtime/site:/input-framework; export INPUT_CPU_EPOCH='+str(epoch)+' INPUT_CPU_DEADLINE='+str(epoch+220)+'; LEFT=$(python3 -B -c '+shlex.quote('import time;print(max(0,int('+str(epoch+220)+'-time.time())))')+'); test "$LEFT" -gt 20; exec /usr/bin/timeout --kill-after=5s "$LEFT" python3 -B -u /input-stage/run_checks.py'
command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/']
for bind in binds:command+=['--bind',bind]
command +=[str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
argv=['srun','--jobid='+parent,'--job-name='+label,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,'--cpus-per-task=4','--mem=8192M','--gres=none','--immediate=5','--time=4',*command]
forced=False;error=None;child=None
try:
 with (W/'CPU.log').open('xb') as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();save('owned-srun.json',dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname()))
  child.wait(timeout=min(250,left(40)))
except BaseException as failure:
 forced=True;error=dict(type=type(failure).__name__,message_redacted=True)
finally:
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=min(5,left(30)))
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=min(5,left(20)))
import sys
sys.path.insert(0,str(W/'source'));from slurm_accounting import wait_terminal,is_terminal
observed=wait_terminal(lambda:(accounting() or [None])[0],time.monotonic()+min(18,left(10)),interval=.5,max_observations=36)
rows=accounting()
if len(rows)==1 and not is_terminal(rows[0]):
 query(['scancel',rows[0][0]]);forced=True
 observed=wait_terminal(lambda:(accounting() or [None])[0],time.monotonic()+min(8,left(2)),interval=.5,max_observations=16);rows=accounting()
out=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,returncode=child.returncode if child else None,forced_cleanup=forced,error=error,accounting_columns=columns,accounting=rows,terminal=observed,log_SHA256=hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest(),wall_seconds=time.monotonic()-anchor,GPU_cards=0,owned_srun_reaped=bool(child and child.poll() is not None))
p=W/'evidence/CPU-result.json'
if p.exists():
 raw=p.read_bytes();out['CPU-result.json']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out['passed']=bool(child and child.returncode==0 and not forced and observed['terminal_visible'] and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]==node and out.get('CPU-result.json',{}).get('value',{}).get('passed') and out['CPU-result.json']['value']['owner']['step']==rows[0][0])
if not out['passed']:out['private_log_retained']=True
save('outcome.json',out);print(json.dumps(out))