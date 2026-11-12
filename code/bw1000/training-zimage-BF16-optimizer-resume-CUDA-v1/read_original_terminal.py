"""Independently consume originalCUDAoptimizer terminal and process ownership."""
from pathlib import Path
import hashlib,json,os,signal,time,datetime,subprocess
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-BF16-optimizer-resume-CUDA-v1')
signal.alarm(60);anchor=time.monotonic();artifacts={};failure=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(n):
 p=W/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size<2*1024**2
 raw=p.read_bytes();v=json.loads(raw);artifacts[n]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=v);return v
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('GPU-result.json');c=json.loads((W/'src/CLAIM.json').read_bytes())
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+600 and r['deadline']==i['epoch']+540
 assert o['passed'] and o['returncode']==0 and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['GPU_result']==r and o['GPU_result_SHA256']==artifacts['GPU-result.json']['SHA256']
 assert r['passed'] and r['actual_GPU_worker_processes']==4 and r['actual_optimizer_steps']==60 and r['fresh_process_restore']
 assert [(v['foreach'],v['arm']) for v in r['phases']]==[('default','control'),('default','resume'),('off','control'),('off','resume')]
 assert len({p['owner']['PID'] for p in r['phases']})==4
 owners=[o['owner'],o['child'],r['owner']]
 for p in r['phases']:
  case='default' if p['foreach']=='default' else 'foreach-off';name='probe/'+case+'/'+p['arm']+'-result.json';v=small(name)
  assert v==p['result'] and artifacts[name]['SHA256']==p['result_SHA256'] and p['returncode']==0 and v['passed']
  assert v['actual_operand_tensors']==476 and v['actual_operand_elements']==9922560 and v['native_model_updates']==0 and not v['model_loaded']
  assert v['epoch']==i['epoch'] and v['deadline']==r['deadline'] and v['strict_free_floor_bytes']==16*1024**3
  if p['arm']=='control':
   assert v['actual_optimizer_updates']==25 and set(v['checkpoints'])=={str(n) for n in range(20,26)}
   for n,pin in v['checkpoints'].items():
    cp=W/'probe'/case/('state-%s.private.pt'%n);assert not cp.is_symlink() and cp.stat().st_size==pin['bytes']
   # Stored private checkpoint bytes were consumed and SHA-checked by the
   # actualfreshworker;this reader doesnot repeat tensor-body hashing.
  else:
   assert v['actual_optimizer_updates']==5 and v['state_before_first_resumed_update_equal']
   assert [x['global_step'] for x in v['updates']]==list(range(21,26))
   assert all(x['complete_state_equal'] and x['parameter_max_abs']==0 and all(x['component_equal'].values()) for x in v['updates'])
  owners.append(p['owner'])
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==367 and fm['framework_commit']==r['framework_commit']
 for n,pin in fm['files'].items():
  f=W/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
 assert r['native_model_loads']==r['native_model_updates']==0 and r['original_native_resume_FAIL_preserved'] and not r['native_model_resume_acceptance']
 assert len(o['GPU_affinity'])==2 and o['peak_group_RSS_bytes']<=4*1024**3 and o['wall_seconds']<600
 for owner in owners:
  now=proc(owner['PID']);assert now is None or now['startticks']!=owner['startticks'] or now['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   now=proc(int(n));assert not(now and now['PGID']==o['child']['PGID'] and now['state']!='Z')
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 selected=[[v.strip() for v in row.split(',')] for row in cards if c['GPU_UUID'] in row]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 assert len(selected)==1 and selected[0][1]==c['GPU_PCI'] and int(selected[0][2])<=16 and not any(c['GPU_UUID'] in row for row in apps)
 post=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cards=selected,owned_compute_apps=[])
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:]);post=None
print(json.dumps(dict(passed=failure is None,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,post=post,
 framework_files=367,new_CPU_or_GPU_runs=0,reader_private_tensor_bytes_read=0,inside_seconds=time.monotonic()-anchor,Model_GO=False)))
if failure is not None:raise SystemExit(1)
