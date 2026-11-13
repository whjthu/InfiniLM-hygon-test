"""Finite original SDK execution with owned processes and actual terminal accounting."""
import hashlib,json,math,os,re,shlex,signal,socket,subprocess,time
from pathlib import Path
from scope import ROOT,LABEL,NODE,PARENT,WHOLE_SECONDS,GPU_SECONDS,TERM_SECONDS,KILL_SECONDS,EpochClock,gpu_minutes
from slurm_accounting import wait_terminal,is_terminal
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');COLUMNS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];clock=EpochClock(epoch);runs=[];gpu_started=False
def save(n,v):
 with (ROOT/n).open('x') as f:json.dump(v,f,indent=2)
def query(args):return subprocess.check_output(args,text=True,timeout=min(12,clock.left(WHOLE_SECONDS,20))).strip()
def accounting(name):return [x.split('|') for x in query(['sacct','-n','-P','-j',PARENT,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+name+'|' in x]
def one_row(name):
 rows=accounting(name)
 if len(rows)>1:raise RuntimeError('duplicate original Slurm label')
 return rows[0] if rows else None
def run(suffix,gpu,limit,command,*,resource_node=NODE,cpu_count=None,memory=None):
 global gpu_started
 name=LABEL+suffix
 if accounting(name):raise RuntimeError('original label already exists; reconcile instead of replay')
 if gpu:
  minutes=gpu_minutes(epoch,time.time());budget=clock.left(GPU_SECONDS,2)
 else:
  budget=min(limit,clock.left(WHOLE_SECONDS,80));minutes=max(1,int(math.ceil(budget/60)))
 args=['srun','--jobid='+PARENT,'--job-name='+name,'--exact','--nodes=1','--ntasks=1','--nodelist='+resource_node,'--cpus-per-task='+str(cpu_count if cpu_count is not None else (8 if gpu else 2)),'--mem='+str(memory if memory is not None else ('16384M' if gpu else '8192M')),'--gres='+('dcu:1' if gpu else 'none'),'--immediate=5','--time='+str(minutes),*command]
 forced=False;failure=None;child=None
 with (ROOT/(name+'.log')).open('xb') as log:
  try:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   if gpu:gpu_started=True
   stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()
   save(name+'-owner.json',dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()))
   child.wait(timeout=min(budget,clock.left(WHOLE_SECONDS,180)))
  except BaseException as error:
   failure=dict(type=type(error).__name__,message=str(error)[:300]);forced=True
  finally:
   if child is not None and child.poll() is None:
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=min(TERM_SECONDS,clock.left(WHOLE_SECONDS,100)))
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     child.wait(timeout=min(KILL_SECONDS,clock.left(WHOLE_SECONDS,80)))
    row=one_row(name)
    if row and re.fullmatch(PARENT+r'\.\d+',row[0]) and not is_terminal(row):query(['scancel',row[0]])
 observed=wait_terminal(lambda:one_row(name),time.monotonic()+min(40,clock.left(WHOLE_SECONDS,40)),interval=2,max_observations=20)
 row=observed['row'];raw=(ROOT/(name+'.log')).read_bytes()
 value=dict(name=name,GPU_cards=1 if gpu else 0,returncode=child.returncode if child else None,forced_cleanup=forced,error=failure,accounting_columns=COLUMNS,terminal=observed,log_SHA256=hashlib.sha256(raw).hexdigest(),error_tail=raw.decode(errors='replace')[-1800:] if not child or child.returncode else None)
 value['passed']=bool(child and child.returncode==0 and not forced and observed['terminal_visible'] and row[2:4]==['COMPLETED','0:0'] and row[5]==resource_node)
 save(name+'-receipt.json',value);runs.append(value);return value

def container():
 reuse=json.loads((ROOT/'src/REUSED_SOURCE.json').read_bytes());qualified=Path(reuse['root']);site=Path(reuse['dependency_root'])/'dependencies/site';runtime=R/'results/training-t06-sdxl-runtime-v1'
 shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; export PYTHONPATH=/zimage-runtime:/training-runtime/site:/run-scope/src; unset BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT; exec python3 -B -u /run-scope/src/runtime_exec.py "$1"'
 command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/']
 for host,target in [(ROOT,'/run-scope:rw'),(qualified/'framework','/frozen-framework:ro'),(qualified,'/qualified-preflight:ro'),(site,'/zimage-runtime:ro'),(runtime,'/training-runtime:ro'),(Path('/opt/hyhal'),'/opt/hyhal:ro')]:command+=['--bind',str(host)+':'+target]
 return command+[str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'sdk-scope','device_entry.py']

def snap(stage):
 started=time.monotonic();receipt=run('-'+stage,False,50,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'src/snapshot.py')]);rtt=time.monotonic()-started
 if not receipt['passed']:raise RuntimeError('snapshot step did not terminate cleanly')
 raw=(ROOT/(LABEL+'-'+stage+'.log')).read_bytes();row,=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
 assert row['node']==NODE and row['step']==receipt['terminal']['row'][0] and row['kfd_holder_count']==0 and len(row['pci'])==8
 row['controller_monotonic_roundtrip_seconds']=rtt
 if stage=='pre':assert rtt<40
 save('before.json' if stage=='pre' else 'after.json',row);return row

owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),host=socket.gethostname(),epoch=epoch)
save('controller-ready.json',owner);before=None;after=None;gpu=None
try:
 subprocess.run(['/usr/local/python3.12/bin/python3','-B',str(ROOT/'src/verify_inputs.py')],
                check=True,timeout=clock.left(100,5))
 before=snap('pre')
 gpu=run('',True,480,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'src/host_device_gate.py'),'--',*container()])
 after=snap('post');assert before['boot_id']==after['boot_id'] and before['pci']==after['pci']
 diagnostic=None
 p=ROOT/'diagnostic/result.json'
 if p.exists():
  raw=p.read_bytes();assert len(raw)<512*1024;diagnostic=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 passed=bool(gpu['passed'] and diagnostic and diagnostic['value']['passed'])
 if passed:
  v=diagnostic['value'];assert v['counters']==dict(forward_attempted=12,forward_returned=12,backward_attempted=12,backward_synchronized=12)
  assert v['input_audits']['completed']==12 and v['input_audits']['buffers_checked']==48 and len(v['cases'])==3 and all(x['complete'] for x in v['cases'])
  a=json.loads((ROOT/'assigned.json').read_bytes());assert a['node']==NODE and a['step']==gpu['terminal']['row'][0] and a['pci'] in before['pci'] and a['visible_cards']==1
 result=dict(completed=True,SDK_numerical_passed=passed,diagnostic=diagnostic,actual_GPU_step=gpu['terminal']['row'][0] if gpu['terminal']['row'] else None,GPU_started=gpu_started,post_clean=True)
except BaseException as error:
 if gpu_started and before is not None and after is None:
  try:after=snap('failure-post')
  except BaseException:pass
 result=dict(completed=False,SDK_numerical_passed=False,error_type=type(error).__name__,GPU_started=gpu_started,post_clean=bool(after is not None and after['pci']==before['pci'] and after['boot_id']==before['boot_id']))
result.update(claim_commit=intent['claim_commit'],request_id=intent['request_id'],epoch=epoch,deadline=intent['deadline'],owner=owner,runs=runs,accounting_columns=COLUMNS,framework_commit='bcb2ad9369ac9f12325fe4152b516f3435173de5',actual_native_model_loads=0,actual_native_optimizer_updates=0,model_numerical_qualified=False,continuous25_qualified=False,formal_performance=False,quality_effect=False,Model_GO=False)
save('controller-result.json',result)
if not result['completed']:raise SystemExit(1)
