"""Own two serial fresh native Python workers under the same original epoch."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
claim=json.loads((S/'CLAIM.json').read_bytes());GPU_UUID=claim['GPU_UUID'];GPU_PCI=claim['GPU_PCI']
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE']);anchor=time.monotonic();remaining=deadline-time.time()
assert 0<remaining<=3500
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original native25 shared epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def check():
 if min(deadline-time.time(),remaining-(time.monotonic()-anchor))<=0:raise TimeoutError('original shared epoch')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def proc(pid):
 f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
owner=proc(os.getpid());dump('GPU-owner.json',owner);phases=[];stage='source';comparison=None
def idle():
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 selected=[[v.strip() for v in line.split(',')] for line in cards if GPU_UUID in line]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 value=dict(cards=selected,compute_apps=[line for line in apps if GPU_UUID in line])
 assert len(selected)==1 and selected[0][1]==GPU_PCI and int(selected[0][2])<=16 and not value['compute_apps']
 return value
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==365
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 for mode in ('continuous','resume20'):
  check();stage=mode;postbefore=idle();args=[os.sys.executable,'-B','-u',str(F/'tools/hygon_zimage_native_record.py'),
    '--binding',str(S/'BINDING.json'),'--output',str(W/mode),'--mode',mode,'--epoch',str(epoch),'--deadline',str(deadline)]
  if mode=='resume20':args+=['--checkpoint',str(W/'continuous/checkpoint-20/state.private.pt')]
  with (W/(mode+'.private.log')).open('xb') as log:
   child=subprocess.Popen(args,cwd=F,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
   owned=proc(child.pid);assert owned['PGID']==owner['PGID']
   dump(mode+'-owner.json',owned)
   try:rc=child.wait(timeout=min(deadline-time.time(),remaining-(time.monotonic()-anchor)))
   except BaseException:
    # Outer supervisor owns this same process group and performs TERM/KILL.
    raise
  rp=W/mode/'result.json';value=json.loads(rp.read_bytes()) if rp.is_file() else None
  row=dict(mode=mode,owner=owned,returncode=rc,result=value,result_SHA256=sha(rp) if rp.is_file() else None,before_GPU=postbefore)
  phases.append(row);dump(mode+'-returned.json',row)
  assert rc==0 and value and value['passed']
  after=idle();dump(mode+'-after-GPU.json',after)
 assert phases[0]['owner']['PID']!=phases[1]['owner']['PID']
 stage='independent-private-checkpoint-comparison'
 # Only the already-owned private checkpoints are loaded;no model is created.
 import torch
 sys.path.insert(0,str(F/'generation/src'))
 from gencuda import zimage_training_resume_checkpoint as resume
 from gencuda import hy_resume_checkpoint as common
 a=resume.load_checkpoint(W/'continuous/checkpoint-25/state.private.pt');b=resume.load_checkpoint(W/'resume20/checkpoint-25/state.private.pt')
 assert a['metadata']==b['metadata'] and a['metadata']['global_step']==25
 names=set(a['state']['trainable']);assert names==set(b['state']['trainable']) and len(names)==476
 maxabs=max(float((a['state']['trainable'][n].float()-b['state']['trainable'][n].float()).abs().max()) for n in names)
 metricsA={v['global_step']:v for v in phases[0]['result']['step_metrics']};metricsB={v['global_step']:v for v in phases[1]['result']['step_metrics']}
 lossdiff=max(abs(metricsA[i]['loss']-metricsB[i]['loss']) for i in range(21,26))
 comparison=dict(final_global_step=25,compared_steps=list(range(21,26)),trainable_tensors=476,
  parameter_max_abs=maxabs,loss_max_abs=lossdiff,original_resume_atol=1e-6,
  complete_state_equal=common.tree_sha256(a['state'])==common.tree_sha256(b['state']),
  scheduler_equal=common.tree_sha256(a['scheduler'])==common.tree_sha256(b['scheduler']),RNG_equal=common.tree_sha256(a['rng'])==common.tree_sha256(b['rng']),
  integrity_equal=a['integrity_sha256']==b['integrity_sha256'],independent_processes=True)
 comparison['passed']=bool(maxabs<=1e-6 and lossdiff<=1e-6 and comparison['complete_state_equal'] and comparison['scheduler_equal'] and comparison['RNG_equal'])
 dump('resume-comparison.json',comparison);assert comparison['passed']
 assert not torch.cuda.is_initialized();finalpost=idle();dump('final-after-GPU.json',finalpost)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=365,
  phases=phases,continuous_updates=25,resumed_updates=5,actual_GPU_worker_processes=2,
  first_update_capture_complete=phases[0]['result']['first_update_capture_complete'],resume_comparison=comparison,
  original_single_card_recipe=True,GPU_cards=1,wall_seconds=time.monotonic()-anchor,
  formal_performance=False,cross_platform_numerical_qualified=False,effect_qualified=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc()[-10000:],completed_phases=phases))
 result=dict(passed=False,stage=stage,error_type=type(error).__name__,owner=owner,epoch=epoch,deadline=deadline,
  phases=phases,resume_comparison=comparison,wall_seconds=time.monotonic()-anchor,GPU_cards=1,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],stage=stage,error_type=result.get('error_type'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
