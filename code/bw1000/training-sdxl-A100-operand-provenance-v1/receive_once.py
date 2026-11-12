from pathlib import Path
import base64,hashlib,importlib.metadata,json,math,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-sdxl-original-operand-provenance-v1');epoch=time.time();anchor=time.monotonic();deadline=epoch+900;assert not W.exists(),'original namespace exists;reconcile without replay';W.mkdir(mode=0o700);(W/'source').mkdir();available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(available[:8]))
def left(offset=900,reserve=0):
 value=min(epoch+offset-time.time(),offset-(time.monotonic()-anchor))-reserve
 if not math.isfinite(value) or value<=0:raise TimeoutError('original source epoch exhausted')
 return value
def save(n,x):
 with (W/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nvml():
 g=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=min(10,left(reserve=5)));p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=min(10,left(reserve=5)));return dict(GPUs=[z.split(', ') for z in g.splitlines()],processes=[z.split(', ') for z in p.splitlines()])
for n,pin in FILES.items():
 part=Path(n);assert not part.is_absolute() and '..' not in part.parts;raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=W/'source'/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
claim=json.loads((W/'source/CLAIM.json').read_bytes());deps=json.loads((W/'source/DEPENDENCIES.json').read_bytes());uuid=claim['GPU_UUID'];selected=claim['physical_GPU'];save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,CPU_deadline=epoch+120,GPU_deadline=epoch+660,compare_deadline=epoch+780,host=socket.gethostname(),source_files={n:v['SHA256'] for n,v in FILES.items()},GPU_UUID=uuid,physical_GPU=selected,planned_CPU_tests=14,planned_optimizer_updates=1,planned_comparisons=1,raw_operand_cross_host_transfer_bytes=0));print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline)),flush=True)
phases={};originals={};forced=False;error=None;before=None;post=None;cpu=None;gpu=None;comparison=None;child=None
base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(W/'private-pycache'),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',SOURCE_EPOCH=str(epoch),SOURCE_CPU_DEADLINE=str(epoch+120),SOURCE_GPU_DEADLINE=str(epoch+660),SOURCE_COMPARE_DEADLINE=str(epoch+780),HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
def phase(name,script,offset,gpu_on=False):
 global child,forced
 env=base.copy();env['CUDA_VISIBLE_DEVICES']=uuid if gpu_on else '-1';budget=left(offset,5);assert budget>5
 with (W/(name+'.log')).open('xb') as log:
  child=subprocess.Popen(['/usr/bin/timeout','--kill-after=5s',str(budget),'/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(W/'source'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rest=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();originals[name]=dict(PID=child.pid,PGID=int(rest[2]),startticks=int(rest[19]),host=socket.gethostname());save(name+'-owned.json',originals[name]);child.wait(timeout=left(offset,1));phases[name]=dict(returncode=child.returncode,reaped=True,log_SHA256=sha(W/(name+'.log')));assert child.returncode==0,name+' phase failed'
try:
 for n,v in deps['versions'].items():
  # Child env-vlm versions are checked in the actual GPU consumer; thisHOST is not that env.
  assert isinstance(v,str) and v
 assert sha(Path(deps['original_trainer']))==deps['original_trainer_SHA256'];assert Path(deps['reference']).stat().st_size==74257307 and Path(deps['weights'],'unet/diffusion_pytorch_model.safetensors').stat().st_size==10270077736
 phase('CPU','run_cpu_checks.py',120);cpu=json.loads((W/'CPU-result.json').read_bytes());assert cpu['passed']
 before=nvml();save('before.json',before);rows=[z for z in before['GPUs'] if int(z[0])==selected];assert len(rows)==1 and rows[0][4]==uuid and int(rows[0][3])==0 and not any(z[1]==uuid for z in before['processes'])
 phase('GPU','run_GPU_once.py',660,True);gpu=json.loads((W/'GPU-result.json').read_bytes());assert gpu['passed'] and gpu['actual_optimizer_updates']==1
 phase('compare','compare_source.py',780);comparison=json.loads((W/'comparison-result.json').read_bytes());assert comparison['evidence_processed']
except BaseException as failure:error=dict(type=type(failure).__name__)
finally:
 if child is not None and child.poll() is None:
  forced=True
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=min(5,left(reserve=35)))
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=min(5,left(reserve=25)))
 try:post=nvml();save('post.json',post)
 except BaseException as e:post=dict(observation_failed=True,error_type=type(e).__name__)
for n,p in [('CPU',W/'CPU-result.json'),('GPU',W/'GPU-result.json'),('compare',W/'comparison-result.json')]:
 if p.is_file():
  value=json.loads(p.read_bytes());phases.setdefault(n,{});phases[n].update(result=dict(bytes=p.stat().st_size,SHA256=sha(p),value=value))
postclean=bool(post and 'GPUs' in post and not any(z[1]==uuid for z in post['processes']) and all(int(z[3])==0 for z in post['GPUs'] if z[4]==uuid));out=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,phases=phases,owned_processes=originals,forced_cleanup=forced,error=error,before=before,post=post,post_clean=postclean,wall_seconds=time.monotonic()-anchor,passed=bool(error is None and not forced and postclean and comparison and comparison['evidence_processed']),source_numerical_passed=comparison['numerical_passed'] if comparison else False,Model_GO=False,formal_performance=False,raw_operand_cross_host_transfer_bytes=0);save('outcome.json',out);print(json.dumps(out,allow_nan=False))
