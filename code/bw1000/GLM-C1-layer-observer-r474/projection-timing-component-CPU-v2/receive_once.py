"""One finite CPU Slurm step checking actual target Z-Image native model meta structure; no GPU."""
import base64,hashlib,json,math,os,shlex,signal,socket,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='GLM474-projection-timing-component-CPU-v2';W=R/'results'/label;parent='1103606';epoch=time.time();anchor=time.monotonic();deadline=epoch+360
EXPECTED_NODES=['f13r1n05']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def left(reserve=0):
 value=min(deadline-time.time(),360-(time.monotonic()-anchor))-reserve
 if not math.isfinite(value) or value<=0:raise TimeoutError('original CPU scope exhausted')
 return value
def query(argv):return subprocess.check_output(argv,text=True,timeout=min(12,left(5))).strip()
def accounting():return [row.split('|') for row in query(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+label+'|' in row]
def save(name,value):
 with (W/name).open('x') as stream:json.dump(value,stream,indent=2)
assert not W.exists() and not accounting(),'namespace or original label exists; reconcile without replay'
allocation=query(['squeue','-h','-j',parent,'-o','%T|%D|%N']).split('|');assert allocation[:2]==['RUNNING','40']
nodes=query(['scontrol','show','hostnames',allocation[2]]).splitlines();assert len(nodes)==len(set(nodes))==40 and set(EXPECTED_NODES)<=set(nodes)
nodes=EXPECTED_NODES
for line in query(['squeue','--steps','-h','-j',parent,'-o','%i|%N']).splitlines():
 handle,hosts=line.split('|',1)
 if handle not in (parent+'.batch',parent+'.extern') and hosts not in ('','(null)'):assert not set(nodes).intersection(query(['scontrol','show','hostnames',hosts]).splitlines()),'CPU node has a live step'
job=query(['scontrol','show','job',parent,'--oneliner']);fields={x.split('=',1)[0]:x.split('=',1)[1] for x in job.split() if '=' in x}
W.mkdir(mode=0o700);(W/'src').mkdir();(W/'evidence').mkdir()
for name,row in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(row['b64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'];path=W/'src'/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(raw)
mf=json.loads((W/'src/SOURCE_FILES.json').read_bytes());assert set(mf['files'])==set(FILES)-{'SOURCE_FILES.json'}
for name,pin in mf['files'].items():assert pin['SHA256']==FILES[name]['SHA256'] and pin['bytes']==FILES[name]['bytes']
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,work_deadline=epoch+265,host=socket.gethostname(),parent=parent,node=nodes[0],actual_nodes=nodes,actual_parent_resources={k:fields.get(k) for k in ('NumCPUs','NumNodes','AllocTRES','TimeLimit','JobState')},source_files={name:row['SHA256'] for name,row in FILES.items()},GPU_cards=0))
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',request_id=REQUEST,claim=CLAIM,epoch=epoch,deadline=deadline)),flush=True)
scope='/refactor/results/'+label+'/src'
argv=['srun','--jobid='+parent,'--job-name='+label,'--exact','--nodes=1','--ntasks=1',
      '--nodelist='+nodes[0],'--cpus-per-task=4','--mem=8192M','--gres=none','--immediate=5','--time=5',
      '/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm',
      '--no-home','--pwd','/',
      '--bind',str(R)+':/refactor:ro','--bind',str(W)+':/refactor/results/'+label+':rw',
      '--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),
      '/bin/bash','--noprofile','--norc','-c',
      'source "$1/ENV.sh"; exec /usr/bin/python3 -B -u "$1/build_once.py"','build',scope]
forced=False;error=None;child=None
try:
 with (W/'CPU.log').open('xb') as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();save('owned-srun.json',dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname()))
  child.wait(timeout=min(300,left(40)))
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
sys.path.insert(0,str(W/'src'));from slurm_accounting import wait_terminal,is_terminal
observed=wait_terminal(lambda:(accounting() or [None])[0],time.monotonic()+min(18,left(10)),interval=.5,max_observations=36)
rows=accounting()
if len(rows)==1 and not is_terminal(rows[0]):
 query(['scancel',rows[0][0]]);forced=True
 observed=wait_terminal(lambda:(accounting() or [None])[0],time.monotonic()+min(8,left(2)),interval=.5,max_observations=16);rows=accounting()
out=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,returncode=child.returncode if child else None,forced_cleanup=forced,error=error,accounting_columns=columns,accounting=rows,terminal=observed,log_SHA256=hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest(),wall_seconds=time.monotonic()-anchor,GPU_cards=0,owned_srun_reaped=bool(child and child.poll() is not None))
p=W/'build-result.json'
if p.is_file():
 raw=p.read_bytes();out['build']=json.loads(raw);out['build_SHA256']=hashlib.sha256(raw).hexdigest()
out['passed']=bool(child and child.returncode==0 and not forced and observed['terminal_visible']
 and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]==nodes[0]
 and out.get('build',{}).get('passed') and out['build']['step']==rows[0][0])
save('outcome.json',out)
print(json.dumps(out),flush=True)
