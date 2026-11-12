"""Independent CPU experiment terminal/source/ownership consistency reader."""
from pathlib import Path
import hashlib,json,os,signal,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-BF16-optimizer-resume-CPU-v1')
signal.alarm(60);started=time.monotonic();artifacts={};failure=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(name):
 p=W/name;assert not p.is_symlink() and p.is_file() and p.stat().st_size<2*1024**2
 raw=p.read_bytes();artifacts[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw));return artifacts[name]['value']
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('CPU-result.json')
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+300 and r['deadline']==i['epoch']+260
 assert o['passed'] and o['returncode']==0 and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['CPU_result']==r and o['CPU_result_SHA256']==artifacts['CPU-result.json']['SHA256']
 p=small('probe/result.json');assert p==r['probe'] and sha(W/'probe/result.json')==r['probe_result_SHA256']
 assert p['passed'] and p['actual_optimizer_steps']==r['actual_CPU_optimizer_steps']==60 and p['actual_input_tensors']==476 and p['actual_input_elements']==9922560
 assert len(p['cases'])==2 and [v['foreach'] for v in p['cases']]==[None,True]
 for case in p['cases']:
  assert case['passed'] and case['state_before_first_resumed_update_equal'] and case['complete_final_state_equal']
  assert case['control_updates']==25 and case['restored_updates']==5 and [v['global_step'] for v in case['updates']]==list(range(21,26))
  assert all(v['parameters_equal'] and v['parameter_max_abs']==0 for v in case['updates'])
 assert p['native_full_model_loaded'] is False and p['native_model_updates']==0 and p['GPU_initialized'] is False and not p['model_resume_acceptance'] and not p['CUDA_optimizer_or_backward_cause_qualified']
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==366 and fm['framework_commit']==r['framework_commit']
 for n,pin in fm['files'].items():
  f=W/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
 binding=json.loads((W/'src/BINDING.json').read_bytes());mp=Path(binding['reference_root'])/binding['capture_manifest']['relative_path']
 assert sha(mp)==binding['capture_manifest']['SHA256'];pins={v['name']:v for v in json.loads(mp.read_bytes())['files']}
 assert p['operand_pins']=={n:pins[n] for n in ('initial.pt','postclip.pt')}
 assert r['CPU_only'] and r['native_model_training_updates']==0 and not r['GPU_initialized'] and not r['pretrained_weight_or_dataset_bodies_read'] and not r['private_operands_exported']
 assert len(o['CPU_affinity'])==2 and o['peak_group_RSS_bytes']<=4*1024**3 and o['wall_seconds']<300
 for owner in (o['owner'],o['child'],r['owner']):
  current=proc(owner['PID']);assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   row=proc(int(n));assert not(row and row['PGID']==o['child']['PGID'] and row['state']!='Z')
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:])
print(json.dumps(dict(passed=failure is None,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,
 framework_files=366,new_CPU_or_GPU_runs=0,reader_private_operand_bytes_read=0,inside_seconds=time.monotonic()-started,original_native_resume_FAIL_preserved=True,Model_GO=False)))
if failure is not None:raise SystemExit(1)
