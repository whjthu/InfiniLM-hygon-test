"""One source-pinned CPU Slurm step on the owned T05 second node."""
import base64,datetime,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
epoch=time.time();origin=time.monotonic();deadline=epoch+180
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ROOT=R/'results/training-vlm-native-dropout-CPU-v1';label='training-vlm-native-dropout-CPU-v1';node='f11r4n10'
def query(*args):return subprocess.check_output(args,text=True,timeout=min(12,max(1,deadline-time.time()-5))).strip()
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw']
def accounting():return [x.split('|') for x in query('sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw').splitlines() if '|'+label+'|' in x]
assert not ROOT.exists() and not accounting()
parent=query('squeue','-h','-j','1087656','-o','%T|%D|%N').split('|')
assert parent[:2]==['RUNNING','40'] and node in query('scontrol','show','hostnames',parent[-1]).splitlines()
for line in query('squeue','--steps','-h','-j','1087656','-o','%i|%N').splitlines():
    handle,names=line.split('|',1)
    if handle not in ('1087656.batch','1087656.extern') and names not in ('','(null)'):
        assert node not in query('scontrol','show','hostnames',names).splitlines()
ROOT.mkdir(mode=0o700)
for name,row in FILES.items():
    path=Path(name);assert not path.is_absolute() and '..' not in path.parts
    raw=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256']
    target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(raw)
with (ROOT/'intent.json').open('x') as f:json.dump({'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'node':node,'GPU_allocation':0,'files':{n:x['SHA256'] for n,x in FILES.items()}},f)
assert deadline-time.time()>30
shell=('set -euo pipefail; source /opt/dtk/env.sh; '
       'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
       'CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
       'export CPU_EPOCH='+shlex.quote(str(epoch))+' CPU_DEADLINE='+shlex.quote(str(deadline))+'; '
       'exec python3 -B -u /replay/inside_CPU.py')
args=['srun','--jobid=1087656','--job-name='+label,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,
      '--cpus-per-task=2','--mem=8192M','--gres=none','--immediate=5','--time=00:02:30',
      '/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/',
      '--bind',str(ROOT)+':/replay:rw','--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),
      '/bin/bash','--noprofile','--norc','-c',shell]
with (ROOT/'CPU.log').open('xb') as log:
    process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    values=(Path('/proc')/str(process.pid)/'stat').read_text().split(') ',1)[1].split()
    with (ROOT/'owner.json').open('x') as f:json.dump({'PID':process.pid,'PGID':int(values[2]),'startticks':int(values[19]),'claim_commit':CLAIM},f)
    try:rc=process.wait(timeout=max(1,180-(time.monotonic()-origin)-12))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,15)
        try:rc=process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,9);rc=process.wait(timeout=5)
rows=[]
for _ in range(12):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'):break
    time.sleep(.5)
log=(ROOT/'CPU.log').read_bytes()
out={'claim_commit':CLAIM,'returncode':rc,'epoch':epoch,'deadline':deadline,'wall_seconds':time.monotonic()-origin,
     'accounting_columns':columns,'accounting':rows,'log_SHA256':hashlib.sha256(log).hexdigest(),'error_tail':log.decode(errors='replace')[-1500:] if rc else None,'GPU_allocation':0}
if (ROOT/'CPU-result.json').is_file():
    raw=(ROOT/'CPU-result.json').read_bytes();out['CPU-result.json']={'SHA256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
with (ROOT/'outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
