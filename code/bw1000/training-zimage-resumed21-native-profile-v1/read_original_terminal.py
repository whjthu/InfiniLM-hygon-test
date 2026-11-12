"""Consume profile or qualification failure without assuming success-only fields."""
from pathlib import Path
from collections import Counter,defaultdict
import datetime,hashlib,json,os,signal,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-resumed21-native-profile-v1')
signal.alarm(75);anchor=time.monotonic();artifacts={};failure=None;profile_consumption=None;post=None;experiment_passed=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(n):
 p=W/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size<3*1024**2
 b=p.read_bytes();v=json.loads(b);artifacts[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v);return v
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
def trace(n,summary):
 p=W/n;assert not p.is_symlink() and p.stat().st_size==summary['raw_event_bytes']<=64*1024**2
 digest=hashlib.sha256();events=0;CUDA=Counter();durations=defaultdict(float);CPU=Counter();backward={}
 with p.open('rb') as f:
  for raw in f:
   assert len(raw)<=128*1024;digest.update(raw);row=json.loads(raw);events+=1
   if row['device_type'].endswith('CUDA'):
    CUDA[row['name']]+=1;durations[row['name']]+=row['duration_us']
   else:
    CPU[row['name']]+=1
    if 'backward' in row['name'].lower() and row['kernel_links']:
     item=backward.setdefault(row['name'],dict(calls=0,linked_device_us=0.,kernels=Counter()))
     item['calls']+=1
     for k in row['kernel_links']:item['linked_device_us']+=k['duration_us'];item['kernels'][k['name']]+=1
 assert events==summary['events'] and sum(CUDA.values())==summary['CUDA_device_events']>0 and digest.hexdigest()==summary['raw_event_SHA256']
 for item in summary['top_CUDA_events']:
  assert CUDA[item['name']]==item['calls'] and abs(durations[item['name']]-item['duration_us'])<=1e-6
 return dict(raw_event_bytes=p.stat().st_size,raw_event_SHA256=digest.hexdigest(),events=events,CUDA_device_events=sum(CUDA.values()),phase_counts={k:CPU[k] for k in ('native.forward','native.backward','native.clip_grad','native.optimizer','native.synchronize')},
  backward_operators=[dict(name=k,calls=v['calls'],linked_device_us=v['linked_device_us'],kernels=dict(v['kernels'])) for k,v in sorted(backward.items(),key=lambda kv:-kv[1]['linked_device_us'])[:100]])
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('GPU-result.json');c=json.loads((W/'src/CLAIM.json').read_bytes());experiment_passed=r['passed']
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+1800 and r['deadline']==i['epoch']+1700
 assert o['GPU_result']==r and o['GPU_result_SHA256']==artifacts['GPU-result.json']['SHA256'] and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['passed']==r['passed'] and o['returncode']==(0 if r['passed'] else 1)
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==374 and fm['framework_commit']==c['framework_commit']
 for n,pin in fm['files'].items():
  p=W/'framework'/n;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 owners=[o['owner'],o['child'],r['owner']]
 for phase in r['phases']:owners.append(phase['owner'])
 if r['passed']:
  assert r['framework_commit']==fm['framework_commit'] and r['actual_GPU_worker_processes']==2 and r['actual_native_model_loads']==1 and r['actual_native_optimizer_updates']==2
  assert [p['label'] for p in r['phases']]==['profiler-qualification','native-profile'] and all(p['returncode']==0 for p in r['phases'])
  q=small('profiler-qualification/profile-summary.json');assert q==r['profiler_qualification'] and q['passed'] and q['native_models_loaded']==0
  qualification=trace('profiler-qualification/events.private.jsonl',q)
  native=small('native-profile/result.json');p=small('native-profile/operator-profile-private/profile-summary.json')
  assert native==r['native'] and artifacts['native-profile/result.json']['SHA256']==r['native_result_SHA256'] and native['passed']
  assert native['completed_global_steps']==[21] and native['actual_optimizer_updates']==2 and native['diagnostic_warmup']['restored20_after_warmup']
  assert not native['first_update_capture_complete'] and native['frozen_parameters_unchanged'] and native['initial_tensors']==476 and native['initial_elements']==9922560
  assert p==native['profile_summary']==r['profile'] and p['passed'] and p['attention_source_SHA256']=='1b55a91812b4e82b6b0ba9c7f46036231acca3234a0b6a105ed210954508e2e4'
  actual=trace('native-profile/operator-profile-private/events.private.jsonl',p);assert all(v==1 for v in actual['phase_counts'].values())
  assert p['routes'] and sum(row['calls'] for row in p['routes'])>0 and not p['formal_performance']
  assert sha(W/'native-profile/frozen-before.json')==sha(W/'native-profile/frozen-after.json')
  profile_consumption=dict(qualification=qualification,native=actual,routes=p['routes'],configured_native_implementation=p['configured_native_implementation'],top_CUDA_events=p['top_CUDA_events'],top_CPU_ops=p['top_CPU_ops'])
 else:
  # Failurefields are consumed as failuredata;never require a missing successfield.
  fail=small('failure-private.json');assert fail['stage']==r['stage'] and fail['error_type']==r['error_type']
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
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-7000:])
print(json.dumps(dict(passed=failure is None,experiment_passed=experiment_passed,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,profile_consumption=profile_consumption,post=post,
 framework_files=374,new_GPU_runs=0,raw_tensor_bytes_read=0,private_profile_metadata_read=True,inside_seconds=time.monotonic()-anchor,Model_GO=False)))
if failure is not None:raise SystemExit(1)
