"""One bounded CPU-only corrected Swift help and normal runtime probe; no model."""
import base64,hashlib,io,json,math,os,signal,socket,subprocess,tarfile,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/training-vlm-canonical-swift-entry-CPU-v2'
label='training-vlm-canonical-swift-entry-CPU-v2';node='f11r4n09';epoch=time.time();mono=time.monotonic();deadline=epoch+240
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
W.mkdir(mode=0o700);(W/'source').mkdir();(W/'evidence').mkdir()
for name,row in FILES.items():
    part=Path(name);assert not part.is_absolute() and '..' not in part.parts
    raw=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256']
    path=W/'source'/part;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw)
V=R/'results/training-vlm-canonical-swift-entry-CPU-v1'
fm_raw=(V/'source/FRAMEWORK_FILES.json').read_bytes();assert hashlib.sha256(fm_raw).hexdigest()=='99b59b345e42b5d48a977434624e8d03f53721dc178009def84fbab7c00fbd5c'
fm=json.loads(fm_raw);assert fm['framework_commit']=='0345a8c9ba2966b0fb7301edfbbe86d9cdd608b4' and len(fm['files'])==284
for name,pin in fm['files'].items():
 p=V/'source'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']

def save(name,v):
    with (W/name).open('x') as f:json.dump(v,f,indent=2)
save('intent.json',dict(claim=CLAIM,epoch=epoch,deadline=deadline,host=socket.gethostname(),node=node,
                       parent=parent,actual_parent_resources=resources,files={n:x['SHA256'] for n,x in FILES.items()},GPU_cards=0))
shell=('set -euo pipefail; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
       'CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; '
       'export PYTHONPATH=/shared-training:/minicpm-transformers:/swift-runtime/site; '
       'export RECORDER_CPU_EPOCH='+str(epoch)+' RECORDER_CPU_DEADLINE='+str(epoch+150)+' FRAMEWORK_EPOCH='+str(epoch)+' FRAMEWORK_DEADLINE='+str(epoch+90)+'; '
       'exec python3 -B -u /stage-source/run_CPU.py')
container=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/',
           '--bind',str(W/'source')+':/stage-source:ro','--bind',str(V/'source/framework')+':/shared-training:ro',
           '--bind',str(W/'evidence')+':/evidence:rw',
           '--bind',str(R/'results/training-vlm-swift-offline-runtime-v1')+':/swift-runtime:ro',
           '--bind',str(R/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site')+':/minicpm-transformers:ro',
           '--bind',str(R/'results/training-minicpm-original-model-v1/MiniCPM-V-4.6')+':'+str(R/'results/training-minicpm-original-model-v1/MiniCPM-V-4.6')+':ro','--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
args=['srun','--jobid=1087656','--job-name='+label,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,
      '--cpus-per-task=4','--mem=8192M','--gres=none','--immediate=5','--time=3',*container]
forced=False
with (W/'CPU.log').open('xb') as log:
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()
    save('owned-srun.json',dict(PID=p.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname()))
    try:p.wait(timeout=min(200,remaining(35)))
    except subprocess.TimeoutExpired:
        forced=True;os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=min(5,remaining(25)))
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
rows=[]
for _ in range(10):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in terminal:break
    time.sleep(min(.5,remaining(10)))
if len(rows)==1 and rows[0][2] not in terminal:
    query(['scancel',rows[0][0]])
out=dict(claim=CLAIM,epoch=epoch,deadline=deadline,returncode=p.returncode,forced_cleanup=forced,
         accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'],accounting=rows,
         log_SHA256=hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest(),wall_seconds=time.monotonic()-mono,GPU_cards=0)
for name in ('CPU-result.json','owned-node.json'):
    path=W/'evidence'/name
    if path.exists():
        raw=path.read_bytes();out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out['passed']=bool(p.returncode==0 and not forced and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
                   and out.get('CPU-result.json',{}).get('value',{}).get('passed')
                   and rows[0][5]==node and out['owned-node.json']['value']['step']==rows[0][0]
                   and out['CPU-result.json']['value']['framework_commit']=='0345a8c9ba2966b0fb7301edfbbe86d9cdd608b4')
if not out['passed']:out['error_tail']=(W/'CPU.log').read_text(errors='replace')[-1800:]
save('outcome.json',out);print(json.dumps(out))
