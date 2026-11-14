"""Finite per-job controller; no SSH service, timer service or run replay."""
import hashlib,json,math,os,re,shlex,signal,socket,subprocess,time
from pathlib import Path
from scope import ROOT,LABEL,NODE,PARENT,PORT,PREFLIGHT_SECONDS,NATIVE_SECONDS,INSIDE_SECONDS,GPU_SECONDS,WHOLE_SECONDS,POST_SECONDS,TERM_SECONDS,KILL_SECONDS,EpochClock,gpu_minutes
from slurm_accounting import wait_terminal,is_terminal
from verify_package import verify
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
COLUMNS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
intent=json.loads((ROOT/'intent.json').read_bytes());epoch=intent['epoch'];clock=EpochClock(epoch);runs=[];gpu_requested=False

def save(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)

def query(args):return subprocess.check_output(args,text=True,timeout=min(12,clock.left(WHOLE_SECONDS,20))).strip()

def accounting(name):
 return [x.split('|') for x in query(['sacct','-n','-P','-j',PARENT,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+name+'|' in x]

def one_row(name):
 rows=accounting(name)
 if len(rows)>1:raise RuntimeError('duplicate original Slurm label')
 return rows[0] if rows else None

def run(suffix,gpu,limit,command):
 global gpu_requested
 name=LABEL+suffix
 if accounting(name):raise RuntimeError('original label already exists; reconcile instead of replay')
 if gpu:
  minutes=gpu_minutes(epoch,time.time());budget=clock.left(GPU_SECONDS,2)
 else:
  budget=min(limit,clock.left(WHOLE_SECONDS,80));minutes=max(1,int(math.ceil(budget/60)))
 args=['srun','--jobid='+PARENT,'--job-name='+name,'--exact','--nodes=1','--ntasks=1','--nodelist='+NODE,'--cpus-per-task='+('64' if gpu else '2'),'--mem='+('224000M' if gpu else '4096M'),'--gres='+('dcu:8' if gpu else 'none'),'--immediate=5','--time='+str(minutes),*command]
 forced=False;failure=None;child=None
 with (ROOT/(name+'.log')).open('xb') as log:
  try:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   if gpu:gpu_requested=True
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
 row=observed['row'];
 raw=(ROOT/(name+'.log')).read_bytes()
 value=dict(name=name,GPU_cards=8 if gpu else 0,returncode=child.returncode if child else None,forced_cleanup=forced,error=failure,accounting_columns=COLUMNS,terminal=observed,log_SHA256=hashlib.sha256(raw).hexdigest(),error_tail=raw.decode(errors='replace')[-1800:] if not child or child.returncode else None)
 value['passed']=bool(child and child.returncode==0 and not forced and observed['terminal_visible'] and row[2:4]==['COMPLETED','0:0'] and row[5]==NODE)
 save(name+'-receipt.json',value);runs.append(value);return value

def snap(stage):
 value=run('-'+stage,False,60 if stage=='pre' else POST_SECONDS,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'src/snapshot.py'),stage])
 if not value['passed']:raise RuntimeError(stage+' snapshot step did not pass actual terminal accounting')
 raw=(ROOT/(LABEL+'-'+stage+'.log')).read_bytes();objects=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
 if len(objects)!=1:raise ValueError('snapshot schema differs')
 observed=objects[0];assert observed['node']==NODE and observed['step']==value['terminal']['row'][0] and len(observed['pci'])==8 and observed['kfd_holder_count']==0 and observed['master_port_available']
 if stage=='post':assert observed['owned_processes_remaining']==[]
 save('before.json' if stage=='pre' else 'after.json',observed);return observed

def main():
 def expired(*_):raise TimeoutError('original whole scope deadline reached')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,clock.left(WHOLE_SECONDS))
 stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
 save('controller-ready.json',dict(PID=os.getpid(),PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname(),epoch=epoch,deadline=epoch+WHOLE_SECONDS,CPU_controller_alive=True,GPU_launched=False))
 error=None;before=None;after=None
 try:
  signal.setitimer(signal.ITIMER_REAL,clock.left(PREFLIGHT_SECONDS))
  parent=query(['squeue','-h','-j',PARENT,'-o','%T|%D|%N']).split('|');assert parent[:2]==['RUNNING','40']
  assert NODE in query(['scontrol','show','hostnames',parent[2]]).splitlines()
  for line in query(['squeue','--steps','-h','-j',PARENT,'-o','%i|%N']).splitlines():
   handle,hosts=line.split('|',1)
   if handle not in (PARENT+'.batch',PARENT+'.extern') and hosts not in ('','(null)'):assert NODE not in query(['scontrol','show','hostnames',hosts]).splitlines()
  save('source-and-asset-preflight.json',verify())
  before=snap('pre');clock.left(PREFLIGHT_SECONDS)
  signal.setitimer(signal.ITIMER_REAL,clock.left(WHOLE_SECONDS))
  from verify_package import FRAMEWORK, STAGE, MODEL, INITIAL, ASSETS, SWIFT, PROCESSOR, VISION
  framework=str(FRAMEWORK)
  binds=[f"{p}:{p}:{mode}" for p,mode in [
    (ROOT,"rw"),(FRAMEWORK,"ro"),(STAGE,"ro"),(MODEL,"ro"),(INITIAL.parent,"ro"),(ASSETS,"ro")]]
  binds += [f"{VISION}:/vision:ro",f"{SWIFT}:/swift-runtime:ro",f"{PROCESSOR}:/minicpm-transformers:ro","/opt/hyhal:/opt/hyhal:ro"]
  shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; export PYTHONPATH='+shlex.quote('/vision:/minicpm-transformers:/swift-runtime/site:'+framework+':'+str(ROOT/'src'))+'; LEFT=$(python3 -B -c '+shlex.quote('import time;print(max(0,int('+str(epoch+INSIDE_SECONDS)+'-time.time())))')+'); test "$LEFT" -gt 60; exec /usr/bin/timeout --kill-after=10s "$LEFT" python3 -B -u '+shlex.quote(str(ROOT/'src/native.py'))
  command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/']
  for binding in binds:command+=['--bind',binding]
  command +=[str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
  gpu=run('',True,0,command)
  if not gpu['terminal']['terminal_visible']:raise RuntimeError('original GPU terminal remains unknown; keep reservation')
 except BaseException as failure:error=dict(type=type(failure).__name__,message=str(failure)[:350])
 finally:
  signal.setitimer(signal.ITIMER_REAL,clock.left(WHOLE_SECONDS))
  if gpu_requested:
   try:
    gpu_row=one_row(LABEL)
    if not is_terminal(gpu_row):raise RuntimeError('no actual GPU terminal; cannot post/release')
    after=snap('post');assert before and after['pci']==before['pci'] and after['boot_id']==before['boot_id']
   except BaseException as failure:error=dict(type=type(failure).__name__,message=str(failure)[:350],stage='post',previous=error)
 out=dict(claim_commit=intent['claim_commit'],epoch=epoch,deadline=epoch+WHOLE_SECONDS,wall_seconds=WHOLE_SECONDS-clock.left(WHOLE_SECONDS),GPU_step_requested=gpu_requested,native_intent_present=(ROOT/'native-intent.json').exists(),error=error,runs=runs,before=before,after=after,independent_model_qualified=False,formal_performance=False)
 for name in ('assigned.json','native-outcome.json','actual-versions.json','torch-provider.json','source-and-asset-preflight.json'):
  path=ROOT/name
  if path.exists():
   raw=path.read_bytes();out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 out['keeper_passed']=bool(error is None and before and after and len(runs)==3 and all(x['passed'] for x in runs) and out.get('native-outcome.json',{}).get('value',{}).get('normal_native_exit'))
 save('outcome.json',out);print(json.dumps(dict(keeper_passed=out['keeper_passed'],native_intent_present=(ROOT/'native-intent.json').exists(),error=error,runs=[dict(name=x['name'],terminal=x['terminal'],passed=x['passed']) for x in runs])))
if __name__=='__main__':main()
