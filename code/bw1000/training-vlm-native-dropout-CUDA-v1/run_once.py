"""One finite, fresh-admitted, single-A100 native dropout test and cleanup record."""
import datetime,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;SRC=ROOT/'src';epoch=time.time();origin=time.monotonic();deadline=epoch+180
installed=json.loads((ROOT/'installed.json').read_bytes());claim=installed['claim_commit']
for name,pin in installed['files'].items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==pin,name
expected=json.loads((SRC/'GPU_BINDING.json').read_bytes())['UUID_by_index']
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=max(1,min(12,deadline-time.time()-3))).strip().splitlines()
def inventory():
    rows=[[x.strip() for x in line.split(',')] for line in query(['--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]
    assert len(rows)==8 and {x[0]:x[1] for x in rows}==expected
    apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
    return {'observed_epoch':time.time(),'cards':rows,'compute_apps':apps}
first=inventory();busy={x.split(',')[0].strip() for x in first['compute_apps']}
eligible=[x for x in first['cards'] if x[1] not in busy and int(x[5])<=16 and int(x[6])==0]
if not eligible:
    save('prelaunch.json',{'state':'PRELAUNCH_NO_IDLE_A100_CARD','claim_commit':claim,'GPU_launched':False,'admission':first})
    print(json.dumps({'state':'PRELAUNCH_NO_IDLE_A100_CARD','GPU_launched':False}));raise SystemExit(0)
selected=eligible[0];time.sleep(1);second=inventory()
assert not any(x.split(',')[0].strip()==selected[1] for x in second['compute_apps'])
fresh=next(x for x in second['cards'] if x[0]==selected[0]);assert fresh[1]==selected[1] and int(fresh[5])<=16 and int(fresh[6])==0
save('intent.json',{'claim_commit':claim,'epoch':epoch,'deadline':deadline,'host':socket.gethostname(),'boot_id_SHA256':hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),'controller_PID':os.getpid(),'controller_startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),'selected':selected,'first_admission':first,'second_admission':second,'GPU_cards':1})
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=selected[0],HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(SRC),EXPECTED_PLATFORM='cuda',EXPECTED_UUID=selected[1],DIAGNOSTIC_EPOCH=str(epoch),DIAGNOSTIC_DEADLINE=str(deadline),DIAGNOSTIC_CLAIM=claim)
child=None;timed_out=False;forced=False;failure=None
try:
    with (ROOT/'GPU.log').open('xb') as log:
        child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(SRC/'check_GPU.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        values=(Path('/proc')/str(child.pid)/'stat').read_text().split(') ',1)[1].split()
        save('child-owner.json',{'PID':child.pid,'PGID':int(values[2]),'startticks':int(values[19]),'host':socket.gethostname(),'claim_commit':claim})
        try:child.wait(timeout=max(1,deadline-time.time()-25))
        except subprocess.TimeoutExpired:timed_out=True
except BaseException as error:failure=type(error).__name__
finally:
    if child is not None and child.poll() is None:
        forced=True;os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
post=inventory()
for _ in range(6):
    if child is None or not any(int(x.split(',')[1])==child.pid for x in post['compute_apps']):break
    time.sleep(.3);post=inventory()
out={'claim_commit':claim,'epoch':epoch,'deadline':deadline,'native_returncode':child.returncode if child else None,'timed_out':timed_out,'forced_cleanup':forced,'controller_error':failure,'wall_seconds':time.monotonic()-origin,'selected':selected,'post':post,'GPU_log_SHA256':hashlib.sha256((ROOT/'GPU.log').read_bytes()).hexdigest() if (ROOT/'GPU.log').is_file() else None,'model_loaded':False,'formal_performance':False}
for name in ('result.json','failure.json','progress.json'):
    p=ROOT/name
    if p.is_file():
        raw=p.read_bytes();out[name]={'SHA256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
out['keeper_passed']=bool(child and child.returncode==0 and not timed_out and not forced and failure is None and not any(int(x.split(',')[1])==child.pid for x in post['compute_apps']))
save('outcome.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ('result.json','progress.json')}))
