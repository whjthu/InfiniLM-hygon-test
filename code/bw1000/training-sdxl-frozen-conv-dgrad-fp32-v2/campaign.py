"""Finite owned SDXL backward diagnostic with bounded original accounting."""
import hashlib,json,math,os,re,shlex,signal,socket,subprocess,sys,time
from pathlib import Path
from scope import ROOT,LABEL,NODE,PARENT,WHOLE_SECONDS,GPU_SECONDS,TERM_SECONDS,KILL_SECONDS,EpochClock,gpu_minutes
from slurm_accounting import wait_terminal,is_terminal
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');COLUMNS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];
epoch, request = float(sys.argv[1]), sys.argv[2]
clock=EpochClock(epoch)
claim=json.loads((ROOT/'CLAIM.json').read_bytes())
assert request==claim['request'] and re.fullmatch('[0-9a-f]{32}',request)
assert 0 <= time.time()-epoch < 30
intent=dict(request=request,epoch=epoch,deadline=epoch+WHOLE_SECONDS)
with (ROOT/'intent.json').open('x',encoding='utf-8') as f:
    json.dump(intent,f)
    f.flush()
    os.fsync(f.fileno())
runs=[]
gpu_started=False
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
 started=time.monotonic();receipt=run('-'+stage,False,60,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'snapshot.py')]);rtt=time.monotonic()-started
 if not receipt['passed']:raise RuntimeError('snapshot terminal not complete')
 raw=(ROOT/(LABEL+'-'+stage+'.log')).read_bytes();row,=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
 assert row['node']==NODE and row['step']==receipt['terminal']['row'][0] and row['kfd_holder_count']==0 and len(row['pci'])==8
 row['controller_monotonic_roundtrip_seconds']=rtt
 if stage=='pre':assert rtt<40
 save('before.json' if stage=='pre' else 'after.json',row);return row
result=None
before=None
after=None
gpu=None
try:
    args=['scontrol','show','job',PARENT,'-o']
    live=query(args)
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)',live))
    assert fields['JobState']=='RUNNING' and fields['UserId'].endswith('('+str(os.getuid())+')')
    assert NODE in query(['scontrol','show','hostnames',fields['NodeList']]).splitlines()
    for line in query(['scontrol','show','step',PARENT,'-o']).splitlines():
        step = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)',line))
        handle = step['StepId']
        if handle in (PARENT+'.batch',PARENT+'.extern'):
            continue
        assert re.fullmatch(re.escape(PARENT)+r'\.\d+',handle) and step['NodeList'] not in ('','(null)')
        assert NODE not in query(['scontrol','show','hostnames',step['NodeList']]).splitlines()

    for suffix in ('-pre','-CPU-contract','','-post','-failure-post','-compare'):
        assert not accounting(LABEL+suffix), 'original label already exists'
    fixture=claim["original_robot_cpu_fixture"]
    prior=R/"results"/fixture["directory"]
    prior_terminal=(prior/"terminal.json").read_bytes()
    assert hashlib.sha256(prior_terminal).hexdigest()==fixture["terminal_SHA256"]
    terminal=json.loads(prior_terminal)
    assert terminal["request"]==fixture["request"] and all(row["GPU_cards"]==0 for row in terminal["runs"])
    [pinned]=[run for run in terminal["runs"] if run["name"].endswith("-CPU-contract")]
    assert pinned["terminal"]["row"]==fixture["cpu"] and pinned["passed"]
    old_log=(prior/(pinned["name"]+".log")).read_bytes()
    assert hashlib.sha256(old_log).hexdigest()==fixture["cpu_log_SHA256"]==pinned["log_SHA256"]
    assert any(json.loads(line)=={"cpu_fixture_shapes":3,"gpu_initialized":False}
               for line in old_log.splitlines() if line.startswith(b'{"cpu_fixture_shapes"'))
    assert hashlib.sha256((ROOT/"intervention_replay.py").read_bytes()).digest() == hashlib.sha256(
        (prior/"intervention_replay.py").read_bytes()).digest()
    before=snap('pre')
    inner='exec python3 -B -u /t06-run/run_platform.py hip /reference/reference.pt /reference/policy.json /model'
    gpu=run('',True,780,['/usr/local/python3.12/bin/python3','-B',str(ROOT/'host_device_gate.py'),
                        '--',*container(inner,True)])
    after=snap('post')
    assert before['pci']==after['pci'] and before['boot_id']==after['boot_id']
    result=dict(gpu_passed=gpu['passed'],clean_post=True,receipt=gpu,
                step=gpu['terminal']['row'][0] if gpu['terminal']['row'] else None)
    if gpu['passed']:
        checked=run('-compare',False,135,container('exec python3 -B -u /t06-run/compare_update_CPU.py',False),
                    cpu_count=4,memory='8192M')
        comparison=ROOT/'comparison-summary.json'
        if checked['passed']:
            deadline=time.monotonic()+min(80,max(0,clock.left(WHOLE_SECONDS,20)))
            while not comparison.is_file() and time.monotonic()<deadline:
                time.sleep(min(2,max(0,deadline-time.monotonic())))
        result['comparison_complete']=bool(checked['passed'] and comparison.is_file())
    else:
        result['comparison_complete']=False
except BaseException as exc:
    if gpu_started and before is not None and after is None:
        try:
            after=snap('failure-post')
        except BaseException:
            pass
    result=dict(gpu_passed=False,clean_post=after is not None,error_class=type(exc).__name__)
result.update(request=intent['request'],epoch=epoch,runs=runs,before=before,after=after)
save('terminal.json',result)
print(json.dumps(dict(request=intent['request'],gpu_passed=result['gpu_passed'],
                      clean_post=result['clean_post'],step=result.get('step'))),flush=True)
raise SystemExit(int(not (result['gpu_passed'] and result.get('comparison_complete'))))
