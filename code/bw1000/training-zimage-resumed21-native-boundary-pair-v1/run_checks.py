"""One CPU precheck and two native resumed21 arms under one owned epoch."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
claim=json.loads((S/'CLAIM.json').read_bytes());binding=json.loads((S/'BINDING.json').read_bytes())
UUID=claim['GPU_UUID'];PCI=claim['GPU_PCI'];checkpoint=Path(binding['diagnostic_checkpoint20']['path'])
epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE']);anchor=time.monotonic();remaining=deadline-time.time()
assert 0<remaining<=1700
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original resumed21 pair epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
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
 assert len(selected)==1 and selected[0][1]==PCI and int(selected[0][2])<=16 and not value['compute_apps'];return value
owner=proc(os.getpid());dump('GPU-owner.json',owner);phases=[];stage='source';CPU=None;comparison=None
def launch(label,args,env,limit):
 with (W/(label+'.private.log')).open('xb') as log:
  p=subprocess.Popen(args,cwd=F,env=env,stdout=log,stderr=subprocess.STDOUT);owned=proc(p.pid);assert owned['PGID']==owner['PGID'];dump(label+'-owner.json',owned)
  rc=p.wait(timeout=min(limit,deadline-time.time(),remaining-(time.monotonic()-anchor)))
 return rc,owned
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==369
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 assert checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size==binding['diagnostic_checkpoint20']['bytes'] and sha(checkpoint)==binding['diagnostic_checkpoint20']['SHA256']
 stage='CPU-recorder-precheck';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
 rc,owned=launch('CPU-recorder',[sys.executable,'-B','-u',str(S/'check_recorder_CPU.py')],env,180)
 CPU=json.loads((W/'CPU-recorder-result.json').read_bytes());dump('CPU-recorder-returned.json',dict(returncode=rc,owner=owned,result=CPU))
 assert rc==0 and CPU['passed'] and CPU['tests']==3 and CPU['skipped']==0 and not CPU['GPU_initialized']
 for arm in ('cold','warm-restored'):
  stage=arm;before=idle()
  args=[sys.executable,'-B','-u',str(F/'tools/hygon_zimage_native_record.py'),'--binding',str(S/'BINDING.json'),'--output',str(W/arm),
   '--mode','resume20','--checkpoint',str(checkpoint),'--first-resumed-update-only','--epoch',str(epoch),'--deadline',str(deadline)]
  if arm=='warm-restored':args.append('--warmup-before-restored-update')
  rc,owned=launch(arm,args,os.environ.copy(),deadline-time.time())
  rp=W/arm/'result.json';value=json.loads(rp.read_bytes()) if rp.is_file() else None
  row=dict(arm=arm,owner=owned,returncode=rc,result=value,result_SHA256=sha(rp) if rp.is_file() else None,before_GPU=before)
  phases.append(row);dump(arm+'-returned.json',row)
  assert rc==0 and value and value['passed'] and value['completed_global_steps']==[21] and value['diagnostic_update_only']
  assert value['actual_optimizer_updates']==(1 if arm=='cold' else 2)
  if arm=='warm-restored':assert value['diagnostic_warmup']['restored20_after_warmup']
  dump(arm+'-after-GPU.json',idle())
 assert len({v['owner']['PID'] for v in phases})==2
 stage='actual-private-tensor-comparison'
 import torch
 sys.path.insert(0,str(F/'generation/src'));from gencuda import hy_resume_checkpoint as common
 assert not torch.cuda.is_initialized();groups=[]
 def compare(a,b):
  count=0;bad=0;maximum=0.
  def visit(x,y):
   nonlocal count,bad,maximum
   if isinstance(x,torch.Tensor):
    count+=1
    if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:bad+=1;return
    equal=torch.equal(x,y);bad+=int(not equal)
    if x.numel():maximum=max(maximum,float((x.float()-y.float()).abs().max()))
   elif isinstance(x,dict):
    if not isinstance(y,dict) or set(x)!=set(y):bad+=1;return
    for k in x:visit(x[k],y[k])
   elif isinstance(x,(list,tuple)):
    if type(x) is not type(y) or len(x)!=len(y):bad+=1;return
    for l,r in zip(x,y):visit(l,r)
   elif type(x) is not type(y) or x!=y:bad+=1
  visit(a,b)
  return dict(tensor_leaves=count,mismatched_leaves=bad,max_abs=maximum,tree_equal=common.tree_sha256(a)==common.tree_sha256(b))
 roots=[W/arm/'resumed-update-private' for arm in ('cold','warm-restored')]
 manifests=[json.loads((p/'progress.json').read_bytes()) for p in roots]
 required=['initial.pt','initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','preclip.pt','postclip.pt','updated.pt','optimizer.pt','clip_norm.pt','optimizer_groups.pt']
 for m in manifests:
  assert m['phase']=='RESUMED21_RETURNED' and m['planned_global_steps']==m['completed_global_steps']==[21] and m['trainable_tensors']==476 and m['trainable_elements']==9922560
  assert {v['name'] for v in m['files']}==set(required)
 pins=[{v['name']:v for v in m['files']} for m in manifests]
 for n in required:
  tensors=[]
  for root,manifest in zip(roots,pins):
   p=root/n;pin=manifest[n];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];tensors.append(torch.load(p,map_location='cpu',weights_only=True))
  summary=compare(*tensors);groups.append(dict(name=n,**summary));del tensors
 cps=[torch.load(W/arm/'checkpoint-21/state.private.pt',map_location='cpu',weights_only=False) for arm in ('cold','warm-restored')]
 assert cps[0]['metadata']==cps[1]['metadata'] and cps[0]['metadata']['global_step']==21
 complete=compare(cps[0],cps[1]);del cps
 comparison=dict(passed=True,groups=groups,complete_checkpoint21=complete,all_groups_equal=all(v['tree_equal'] for v in groups),
  first_different_group=next((v['name'] for v in groups if not v['tree_equal']),None),
  actual_global_step=21,actual_optimizer_calls=3,physical_GPU_cards=1,diagnostic_only=True,original_native_resume_acceptance=False)
 dump('comparison.json',comparison);finalpost=idle();dump('final-after-GPU.json',finalpost)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=369,CPU_precheck=CPU,
  phases=phases,actual_GPU_worker_processes=2,actual_optimizer_calls=3,recorded_updates=2,extra_warmup_updates=1,comparison=comparison,
  GPU_cards=1,wall_seconds=time.monotonic()-anchor,original_native_resume_FAIL_preserved=True,original_native_resume_acceptance=False,
  formal_performance=False,private_values_exported=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc()[-10000:],completed_phases=phases))
 result=dict(passed=False,stage=stage,error_type=type(error).__name__,owner=owner,epoch=epoch,deadline=deadline,CPU_precheck=CPU,phases=phases,
  comparison=comparison,wall_seconds=time.monotonic()-anchor,GPU_cards=1,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],stage=stage,error_type=result.get('error_type'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
