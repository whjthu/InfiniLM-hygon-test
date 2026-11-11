"""One finite source-pinned HIP step, with independent CPU admission and post snapshot."""
import base64,hashlib,json,math,os,re,shlex,signal,socket,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/O01-GLM-empty-stream-drain-cost-v1'
node='f12r3n17';label='O01-GLM-empty-stream-drain-cost-v1';epoch=time.time();origin=time.monotonic();deadline=epoch+360
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
gpu_started=False
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw'];runs=[]
def remaining(reserve=0):
 value=min(deadline-time.time(),360-(time.monotonic()-origin))-reserve
 if not math.isfinite(value) or value<=0:raise TimeoutError('single original epoch exhausted')
 return value
def query(*args):return subprocess.check_output(args,text=True,timeout=min(12,remaining(3))).strip()
def accounting(name):return [x.split('|') for x in query('sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw').splitlines() if '|'+name+'|' in x]
def save(name,v):
 with (W/name).open('x') as f:json.dump(v,f,indent=2)
assert not W.exists() and not any(accounting(label+s) for s in ('','-pre','-post'))
parent=query('squeue','-h','-j','1087656','-o','%T|%D|%N').split('|');assert parent[:2]==['RUNNING','40']
assert node in query('scontrol','show','hostnames',parent[2]).splitlines()
for row in query('squeue','--steps','-h','-j','1087656','-o','%i|%N').splitlines():
 handle,hosts=row.split('|',1)
 if handle not in ('1087656.batch','1087656.extern') and hosts not in ('','(null)'):assert node not in query('scontrol','show','hostnames',hosts).splitlines()
W.mkdir(mode=0o700);(W/'src').mkdir(mode=0o700)
for name,item in FILES.items():
 assert Path(name).name==name
 raw=base64.b64decode(item['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['SHA256']
 with (W/'src'/name).open('xb') as f:f.write(raw)
save('intent.json',{'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'host':socket.gethostname(),'boot_id_SHA256':hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),'controller_PID':os.getpid(),'controller_startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),'node':node,'parent_observation':parent,'files':{n:x['SHA256'] for n,x in FILES.items()}})
def run(suffix,gpu,limit,reserve,command):
 global gpu_started
 name=label+suffix;budget=min(limit,remaining(reserve));assert budget>10
 args=['srun','--jobid=1087656','--job-name='+name,'--exact','--nodes=1','--ntasks=1','--nodelist='+node,'--cpus-per-task='+('8' if gpu else '2'),'--mem='+('16000M' if gpu else '4096M'),'--gres='+('dcu:1' if gpu else 'none'),'--immediate=5','--time='+str(max(1,int(math.ceil(budget/60)))),*command]
 logpath=W/(name+'.log');forced=False
 with logpath.open('xb') as log:
  p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  if gpu:gpu_started=True
  stat=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split();owner={'PID':p.pid,'PGID':int(stat[2]),'startticks':int(stat[19]),'host':socket.gethostname()};save(name+'-owner.json',owner)
  try:p.wait(timeout=min(budget,remaining(reserve)))
  except subprocess.TimeoutExpired:
   forced=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=min(5,remaining(reserve-5)))
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
   observed=accounting(name)
   if len(observed)==1 and re.fullmatch(r'1087656\.\d+',observed[0][0]) and observed[0][2] not in terminal:
    query('scancel',observed[0][0])
 rows=[]
 for _ in range(10):
  rows=accounting(name)
  if len(rows)==1 and rows[0][2] in terminal:break
  if remaining(5)<1:break
  time.sleep(.5)
 raw=logpath.read_bytes();objects=[json.loads(x) for x in raw.splitlines() if x.startswith(b'{')]
 value={'name':name,'GPU_cards':int(gpu),'returncode':p.returncode,'forced_cleanup':forced,'owner':owner,'accounting_columns':columns,'accounting':rows,'log_SHA256':hashlib.sha256(raw).hexdigest(),'objects':objects,'error_tail':raw.decode(errors='replace')[-1800:] if p.returncode else None};runs.append(value);save(name+'-receipt.json',value);return value

def snap(suffix,reserve):
 v=run(suffix,False,55,reserve,['/usr/local/python3.12/bin/python3','-B',str(W/'src/snapshot.py')]+([str(W/'assigned.json')] if suffix=='-post' and (W/'assigned.json').exists() else []));assert v['returncode']==0 and not v['forced_cleanup'] and len(v['accounting'])==1 and v['accounting'][0][2:4]==['COMPLETED','0:0']
 a,=v['objects'];assert a['node']==node and a['step']==v['accounting'][0][0] and a['kfd_holder_count']==0 and len(a['pci'])==8
 return a
error=None;before=None;after=None;assigned=None
try:
 before=snap('-pre',210)
 shell=('set -euo pipefail; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/probe/src; export DIAGNOSTIC_EPOCH='+shlex.quote(str(epoch))+' DIAGNOSTIC_DEADLINE='+shlex.quote(str(deadline))+' DIAGNOSTIC_CLAIM='+shlex.quote(CLAIM)+'; exec python3 -B -u /probe/src/check_GPU.py')
 command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/','--bind',str(W)+':/probe:rw','--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
 gpu=run('',True,145,90,command)
 if (W/'assigned.json').exists():
  assigned=json.loads((W/'assigned.json').read_bytes());assert assigned['PCI'] in before['pci'] and assigned['visible_cards']==1 and assigned['step']==gpu['accounting'][0][0]
except BaseException as e:error={'type':type(e).__name__,'message':str(e)[:350]}
finally:
 if gpu_started:
  try:
   after=snap('-post',12);assert after['boot_id']==before['boot_id'] and after['pci']==before['pci']
  except BaseException as e:error={'type':type(e).__name__,'message':str(e)[:350],'stage':'post','previous':error}
out={'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'wall_seconds':time.monotonic()-origin,'node':node,'runs':runs,'error':error,'assigned':assigned,'before':before,'after':after,'model_loaded':False,'formal_performance':False}
for name in ('result.json','failure.json','progress.json'):
 p=W/name
 if p.exists():
  raw=p.read_bytes();out[name]={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
out['keeper_passed']=bool(error is None and before and after and assigned and len(runs)==3 and all(x['returncode']==0 and not x['forced_cleanup'] and len(x['accounting'])==1 and x['accounting'][0][2:4]==['COMPLETED','0:0'] for x in runs) and 'result.json' in out)
save('outcome.json',out)
print(json.dumps({'claim_commit':CLAIM,'keeper_passed':out['keeper_passed'],'error':error,'wall_seconds':out['wall_seconds'],'runs':[{k:v for k,v in x.items() if k not in ('objects','owner')} for x in runs],'result':{k:v for k,v in out.get('result.json',{}).get('value',{}).items() if k!='cases'},'failure':out.get('failure.json')}))
