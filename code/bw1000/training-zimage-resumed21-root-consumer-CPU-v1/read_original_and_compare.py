"""Independent source/actualnativeboundary/terminal/ownership consumption."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-resumed21-native-boundary-pair-v1')
signal.alarm(60);anchor=time.monotonic();artifacts={};failure=None;post=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(n):
 p=W/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size<2*1024**2
 b=p.read_bytes();v=json.loads(b);artifacts[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v);return v
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
READ_BYTES=0
def recover_comparison(original):
 import io,sys,torch
 sys.path.insert(0,str(FRAMEWORK_ROOT/'generation/src'))
 from gencuda.checkpoint_compare import compare_trees
 assert not torch.cuda.is_initialized()
 def tensor_file(path,pin):
  global READ_BYTES
  assert not path.is_symlink() and 0<path.stat().st_size==pin['bytes']<=256*1024**2
  raw=path.read_bytes();READ_BYTES+=len(raw);assert hashlib.sha256(raw).hexdigest()==pin['SHA256']
  return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
 roots=[W/arm/'resumed-update-private' for arm in ('cold','warm-restored')]
 manifests=[json.loads((p/'progress.json').read_bytes()) for p in roots];pins=[{v['name']:v for v in m['files']} for m in manifests]
 names=['initial.pt','initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','preclip.pt','postclip.pt','updated.pt','optimizer.pt','clip_norm.pt','optimizer_groups.pt'];groups=[]
 for n in names:
  a=tensor_file(roots[0]/n,pins[0][n]);b=tensor_file(roots[1]/n,pins[1][n]);groups.append(dict(name=n,**compare_trees(a,b)));del a,b
 states=[]
 for arm,phase in zip(('cold','warm-restored'),original['phases']):
  pin=phase['result']['checkpoints'][0];path=W/arm/'checkpoint-21/state.private.pt'
  assert not path.is_symlink() and path.stat().st_size==pin['bytes']<=256*1024**2
  raw=path.read_bytes();global READ_BYTES;READ_BYTES+=len(raw);assert hashlib.sha256(raw).hexdigest()==pin['SHA256']
  # Our own SHA-bound original native checkpoint includes NumPy RNG values.
  states.append(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False))
 assert states[0]['metadata']==states[1]['metadata'] and states[0]['metadata']['global_step']==21
 complete=compare_trees(*states);del states
 result=dict(passed=True,groups=groups,complete_checkpoint21=complete,all_groups_equal=all(v['tree_equal'] for v in groups),first_different_group=next((v['name'] for v in groups if not v['tree_equal']),None),actual_global_step=21,actual_optimizer_calls=3,
  original_GPU_overall_passed=False,original_GPU_comparator_failure_preserved=True,new_GPU_runs=0)
 with (RECOVERY_ROOT/'recovered-comparison.json').open('x') as f:json.dump(result,f,indent=2)
 assert not torch.cuda.is_initialized()
 return result

try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('GPU-result.json');c=json.loads((W/'src/CLAIM.json').read_bytes())
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+1800 and r['deadline']==i['epoch']+1700
 assert o['passed'] is False and o['returncode']==1 and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['GPU_result']==r and o['GPU_result_SHA256']==artifacts['GPU-result.json']['SHA256']
 cpu=small('CPU-recorder-result.json');assert cpu==r['CPU_precheck'] and cpu['passed'] and cpu['tests']==3 and not cpu['errors'] and not cpu['failures'] and not cpu['skipped'] and not cpu['GPU_initialized']
 assert len(r['phases'])==2 and sum(p['result']['actual_optimizer_updates'] for p in r['phases'])==3
 assert [p['arm'] for p in r['phases']]==['cold','warm-restored'] and len({p['owner']['PID'] for p in r['phases']})==2
 owners=[o['owner'],o['child'],r['owner'],small('CPU-recorder-owner.json')]
 required={'initial.pt','initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','preclip.pt','postclip.pt','updated.pt','optimizer.pt','clip_norm.pt','optimizer_groups.pt'}
 binding=json.loads((W/'src/BINDING.json').read_bytes())
 for p in r['phases']:
  arm=p['arm'];v=small(arm+'/result.json');m=small(arm+'/resumed-update-private/progress.json');rb=small(arm+'/resumed-update-private/binding.json')
  assert v==p['result'] and artifacts[arm+'/result.json']['SHA256']==p['result_SHA256'] and p['returncode']==0 and v['passed']
  assert v['completed_global_steps']==[21] and v['actual_optimizer_updates']==(1 if arm=='cold' else 2) and v['diagnostic_update_only'] and not v['original_full_recipe']
  assert v['first_update_capture_complete'] and v['frozen_parameters_unchanged'] and v['initial_tensors']==476 and v['initial_elements']==9922560
  assert len(v['checkpoints'])==1 and v['checkpoints'][0]['global_step']==21
  cp=v['checkpoints'][0];assert (W/arm/'checkpoint-21/state.private.pt').stat().st_size==cp['bytes']
  assert m['phase']=='RESUMED21_RETURNED' and m['planned_global_steps']==m['completed_global_steps']==[21] and m['completed_recorded_updates']==1 and not m['failed']
  assert m['trainable_tensors']==476 and m['trainable_elements']==9922560 and len(m['files'])==12 and {x['name'] for x in m['files']}==required
  assert rb['actual_global_step']==21 and rb['checkpoint20_SHA256']==binding['diagnostic_checkpoint20']['SHA256']
  for x in m['files']:
   f=W/arm/'resumed-update-private'/x['name'];assert not f.is_symlink() and f.stat().st_size==x['bytes']
  assert m['artifact_bytes']==sum(x['bytes'] for x in m['files'])==v['first_update_capture_bytes'] and m['artifact_bytes']<=2*1024**3
  assert sha(W/arm/'frozen-before.json')==sha(W/arm/'frozen-after.json')
  if arm=='warm-restored':assert v['diagnostic_warmup']['actual_optimizer_calls']==1 and v['diagnostic_warmup']['restored20_after_warmup']
  else:assert v['diagnostic_warmup'] is None
  owners.append(p['owner'])
 comp=recover_comparison(r);assert comp['passed'] and comp['actual_global_step']==21 and comp['actual_optimizer_calls']==3
 assert {x['name'] for x in comp['groups']}==required and comp['all_groups_equal']==all(x['tree_equal'] for x in comp['groups'])
 for x in comp['groups']:
  if x['name'] in ('initial.pt','preclip.pt','postclip.pt','updated.pt'):assert x['tensor_leaves']==476
 assert r['passed'] is False and r['stage']=='actual-private-tensor-comparison' and r['error_type']=='ValueError' and r['comparison'] is None
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==369 and fm['framework_commit']==r['framework_commit']
 for n,pin in fm['files'].items():
  f=W/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
 assert len(o['GPU_affinity'])==4 and o['peak_group_RSS_bytes']<=64*1024**3 and o['wall_seconds']<1800
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
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:])
print(json.dumps(dict(passed=failure is None,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,post=post,
 framework_files=369,new_GPU_runs=0,original_overall_passed=False,recovered_comparison=comp if failure is None else None,reader_private_tensor_bytes_read=READ_BYTES,inside_seconds=time.monotonic()-anchor,Model_GO=False)))
if failure is not None:raise SystemExit(1)
