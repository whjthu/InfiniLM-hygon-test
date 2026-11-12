"""Own four serial fresh CUDA optimizer workers,without loading a model."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
claim=json.loads((S/'CLAIM.json').read_bytes());UUID=claim['GPU_UUID'];PCI=claim['GPU_PCI']
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE']);anchor=time.monotonic();remaining=deadline-time.time()
assert 0<remaining<=540
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original CUDA optimizer epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def proc(pid):
 f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
def idle():
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 selected=[[v.strip() for v in row.split(',')] for row in cards if UUID in row]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 value=dict(cards=selected,compute_apps=[row for row in apps if UUID in row])
 assert len(selected)==1 and selected[0][1]==PCI and int(selected[0][2])<=16 and not value['compute_apps']
 return value
def forbid(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):
  raise RuntimeError('pretrained weights or dataset bodies forbidden')
 if event=='socket.connect':raise RuntimeError('network forbidden in optimizer-only experiment')
owner=proc(os.getpid());dump('GPU-owner.json',owner);phases=[];stage='source'
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==367
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 # The same audit restriction is installed in each actual torch worker through
 # a small runpy wrapper;the supervisor itself never imports torch/CUDA.
 worker="import os,sys,runpy\ndef audit(event,args):\n if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/mnt/data/nvidia-baseline-20261010/inputs/'):raise RuntimeError('pretrainedweights/dataset forbidden')\n if event=='socket.connect':raise RuntimeError('network forbidden')\nsys.addaudithook(audit)\np=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name='__main__')"
 for foreach in ('default','off'):
  for arm in ('control','resume'):
   stage=foreach+'-'+arm;before=idle();assert time.time()<deadline
   args=[sys.executable,'-B','-u','-c',worker,str(F/'tools/hygon_bf16_cuda_fresh_resume_probe.py'),'--binding',str(S/'BINDING.json'),'--output',str(W/'probe'),'--arm',arm,'--foreach',foreach,'--epoch',str(epoch),'--deadline',str(deadline)]
   with (W/(stage+'.private.log')).open('xb') as log:
    child=subprocess.Popen(args,cwd=F,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    owned=proc(child.pid);assert owned['PGID']==owner['PGID'];dump(stage+'-owner.json',owned)
    rc=child.wait(timeout=min(deadline-time.time(),remaining-(time.monotonic()-anchor)))
   case='default' if foreach=='default' else 'foreach-off';rp=W/'probe'/case/(arm+'-result.json')
   value=json.loads(rp.read_bytes()) if rp.is_file() else None
   row=dict(foreach=foreach,arm=arm,owner=owned,returncode=rc,result=value,result_SHA256=sha(rp) if rp.is_file() else None,before_GPU=before)
   phases.append(row);dump(stage+'-returned.json',row)
   assert rc==0 and value and value['passed'];dump(stage+'-after-GPU.json',idle())
 assert len({v['owner']['PID'] for v in phases})==4
 assert sum(v['result']['actual_optimizer_updates'] for v in phases)==60
 finalpost=idle();dump('final-after-GPU.json',finalpost)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=367,
  phases=phases,actual_GPU_worker_processes=4,actual_optimizer_steps=60,fresh_process_restore=True,
  GPU_cards=1,wall_seconds=time.monotonic()-anchor,private_operands_exported=False,native_model_loads=0,native_model_updates=0,
  original_native_resume_FAIL_preserved=True,native_model_resume_acceptance=False,formal_performance=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc()[-9000:],completed_phases=phases))
 result=dict(passed=False,stage=stage,error_type=type(error).__name__,owner=owner,epoch=epoch,deadline=deadline,phases=phases,wall_seconds=time.monotonic()-anchor,GPU_cards=1,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],stage=stage,error_type=result.get('error_type'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
