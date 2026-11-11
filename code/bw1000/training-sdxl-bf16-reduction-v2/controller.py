"""Finite owned SDXL policy experiment with bounded original accounting."""
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
 args=['srun','--jobid='+PARENT,'--job-name='+name,'--exact','--nodes=1','--ntasks=1','--nodelist='+resource_node,'--cpus-per-task='+str(cpu_count if cpu_count is not None else (8 if gpu else 2)),'--mem='+str(memory if memory is not None else ('64000M' if gpu else '8192M')),'--gres='+('dcu:1' if gpu else 'none'),'--immediate=5','--time='+str(minutes),*command]
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

def container(script,gpu):
 model=R/'results/training-t06-sdxl-unet-login-stage-v1/model';reference=R/'results/training-t06-sdxl-first-reference-r2';deps=R/'results/training-t06-sdxl-runtime-v1'
 shell='set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; export PYTHONPATH=/training-runtime/site:/t06-run/src; unset BASH_ENV ENV PYTHONSTARTUP PYTHONINSPECT; '
 if not gpu:shell+='export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
 return ['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec',*(['--rocm'] if gpu else []),'--no-home','--pwd','/','--bind',str(ROOT)+':/t06-run:rw','--bind',str(model)+':/model:ro','--bind',str(reference)+':/reference:ro','--bind',str(deps)+':/training-runtime:ro','--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell+script]
def snap(stage):
 started=time.monotonic();receipt=run('-'+stage,False,60,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'src/snapshot.py')]);rtt=time.monotonic()-started
 if not receipt['passed']:raise RuntimeError('snapshot terminal not complete')
 raw=(ROOT/(LABEL+'-'+stage+'.log')).read_bytes();row,=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
 assert row['node']==NODE and row['step']==receipt['terminal']['row'][0] and row['kfd_holder_count']==0 and len(row['pci'])==8
 row['controller_monotonic_roundtrip_seconds']=rtt
 if stage=='pre':assert rtt<40
 save('before.json' if stage=='pre' else 'after.json',row);return row
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),host=socket.gethostname(),epoch=epoch)
save('controller-ready.json',owner);result=None;before=None;after=None
try:
 mf=json.loads((ROOT/'src/SOURCE_FILES.json').read_bytes())
 for n,pin in mf['files'].items():
  p=ROOT/'src'/n;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
 for n,pin in json.loads((ROOT/'src/FRAMEWORK_SUBSET.json').read_bytes())['files'].items():
  p=ROOT/'src/source'/n;target=ROOT/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];target.write_bytes(raw)
 reuse=json.loads((ROOT/'src/REUSED_CPU_POLICY.json').read_bytes());old=Path(reuse['original_root']);runtime_raw=(old/'runtime-preflight.json').read_bytes();policy_raw=(old/'preflight-policy/policy.json').read_bytes();assert hashlib.sha256(runtime_raw).hexdigest()==reuse['runtime_preflight_SHA256'] and hashlib.sha256(policy_raw).hexdigest()==reuse['policy_SHA256']
 row=one_row(reuse['original_CPU_label']);assert row and row[0]==reuse['original_CPU_step'] and row[2:4]==['COMPLETED','0:0'];observed=json.loads(policy_raw);assert not observed['GPU_initialized'];save('reused-original-policy.json',dict(step=row[0],runtime_SHA256=reuse['runtime_preflight_SHA256'],policy_SHA256=reuse['policy_SHA256'],reexecuted=False))
 if not observed['original_policy']:
  result=dict(completed=True,numerical_passed=None,disposition='ALREADY_DISABLED_NO_GPU',GPU_cards_executed=0,policy=observed)
 else:
  before=snap('pre');gpu=run('',True,780,container('exec python3 -B -u /t06-run/src/device_entry.py',True));after=snap('post');assert before['pci']==after['pci'] and before['boot_id']==after['boot_id']
  compare=None;summary=None
  if gpu['passed']:
   assigned=json.loads((ROOT/'assigned.json').read_bytes());assert assigned['step']==gpu['terminal']['row'][0] and assigned['node']==NODE and assigned['visible_cards']==1 and not assigned['assigned_PCI_pending'] and assigned['pci'] in before['pci']
   decision=json.loads((ROOT/'candidate/policy.json').read_bytes());assert decision['executed_update'] and decision['observed_policy'] is False and decision['restored_policy'] is True
   command='exec python3 -B -u /t06-run/source/tools/hygon_sdxl_compare_first_step.py --reference /reference/reference.pt --policy /reference/policy.json --observed /t06-run/candidate/replay/observed.pt --receipt /t06-run/candidate/replay/summary.json --output /t06-run/comparison.json'
   compare=run('-compare',False,120,container(command,False),cpu_count=4,memory='8192M');observations=compare['terminal']['row'];assert observations and observations[2].split()[0] in ('COMPLETED','FAILED') and compare['returncode'] in (0,1) and not compare['forced_cleanup']
   raw=(ROOT/'comparison.json').read_bytes();full=json.loads(raw);summary=dict(passed=full['passed'],checks=full.get('checks'),metrics={k:{n:v for n,v in values.items() if n not in ('per_tensor','worst_relative_l2')} for k,values in full.get('metrics',{}).items()},observed_SHA256=full['observed_sha256'],reference_SHA256=full['reference_sha256'],policy_SHA256=full['policy_sha256'],comparison_SHA256=hashlib.sha256(raw).hexdigest())
  result=dict(completed=True,numerical_passed=summary['passed'] if summary else False,GPU_cards_executed=1,actual_GPU_step=gpu['terminal']['row'][0] if gpu['terminal']['row'] else None,GPU_receipt=gpu,compare_receipt=compare,comparison=summary,post_clean=True,policy=json.loads((ROOT/'candidate/policy.json').read_bytes()) if (ROOT/'candidate/policy.json').exists() else None)
except BaseException as error:
 if gpu_started and before is not None and after is None:
  try:after=snap('failure-post')
  except BaseException:pass
 result=dict(completed=False,error_type=type(error).__name__,message=str(error)[-300:],GPU_started=gpu_started,post_clean=after is not None,numerical_passed=False)
result.update(claim_commit=intent['claim_commit'],request_id=intent['request_id'],epoch=epoch,deadline=intent['deadline'],owner=owner,runs=runs,accounting_columns=COLUMNS,framework_commit='e56e2f5b647830bf87697ae031b3b260e0748f8e',formal_performance=False,quality_effect=False,continuous25_qualified=False)
save('controller-result.json',result)
if not result['completed']:raise SystemExit(1)
