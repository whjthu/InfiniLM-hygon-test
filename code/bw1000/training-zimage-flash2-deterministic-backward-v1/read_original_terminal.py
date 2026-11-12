"""Independent policy chain terminal/contracts/counters/process consumption."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-flash2-deterministic-backward-v1')
signal.alarm(75);anchor=time.monotonic();artifacts={};failure=None;post=None;experiment_passed=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def small(n):
 p=W/n;assert not p.is_symlink() and p.is_file() and p.stat().st_size<3*1024**2;b=p.read_bytes();v=json.loads(b);artifacts[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v);return v
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('GPU-result.json');c=json.loads((W/'src/CLAIM.json').read_bytes());b=json.loads((W/'src/BINDING.json').read_bytes());experiment_passed=r['passed']
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+3600 and r['deadline']==i['epoch']+3500
 assert o['GPU_result']==r and o['GPU_result_SHA256']==artifacts['GPU-result.json']['SHA256'] and o['owned_group_reaped']
 assert o['passed']==r['passed'] and o['returncode']==(0 if r['passed'] else 1)
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==379 and fm['framework_commit']==c['framework_commit']
 for n,pin in fm['files'].items():
  f=W/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
 owners=[o['owner'],o['child'],r['owner']]+[p['owner'] for p in r['phases']]
 if r['passed']:
  assert r['framework_commit']==fm['framework_commit'] and o['stop_reason'] is None
  cpu=small('CPU-policy-checks.json');assert cpu==r['CPU_checks'] and cpu['passed'] and cpu['tests']==2 and not cpu['failures'] and not cpu['errors'] and not cpu['skipped'] and not cpu['GPU_initialized']
  sdk=small('SDK-policy-qualification/result.json');assert sdk==r['SDK_qualification'] and sdk['passed'] and sdk['actual_forward_calls']==sdk['actual_backward_calls']==12
  assert [x['shape'] for x in sdk['cases']]==[[1,n,30,128] for n in (32,256,288)] and all(x['passed'] and x['forward_default_vs_deterministic_equal'] and x['three_deterministic_output_and_gradients_equal'] for x in sdk['cases'])
  pair=small('policy-causal-pair.json');assert pair==r['pair'] and pair['passed']==(all(x['tree_equal'] for x in pair['groups']) and pair['complete_checkpoint21']['tree_equal'])
  assert r['candidate_causal_passed']==pair['passed']
  names=['diagnostic-cold','diagnostic-warm']+(['candidate-continuous25','candidate-resume20'] if pair['passed'] else [])
  assert [p['label'] for p in r['phases']]==['CPU-policy-checks','SDK-policy-qualification']+names and all(p['returncode']==0 for p in r['phases'])
  assert r['actual_native_model_loads']==len(names) and r['actual_GPU_worker_processes']==len(names)+1 and r['actual_native_optimizer_calls']==(33 if pair['passed'] else 3)
  frozen=[]
  for name in names:
   v=small(name+'/result.json');returned=small(name+'-native-result.json');contract=small(name+'/contract.json')
   assert v==returned['result'] and returned['result_SHA256']==artifacts[name+'/result.json']['SHA256'] and v['passed'] and v['frozen_parameters_unchanged']
   assert v['initial_tensors']==476 and v['initial_elements']==9922560 and not v['original_full_recipe']
   policy=v['attention_backward_policy'];assert policy['forced_calls']>0 and policy['policy']==sdk['policy'] and policy['policy']['deterministic_backward']
   assert contract['identity']['config']==b['config_canonical_SHA256'] and contract['identity']['runtime']==canonical(dict(versions=b['versions'],platform=b['platform'],GPU_name=b['GPU_name'],attention_backward_policy=policy['policy']))
   isparent=name.startswith('diagnostic-');assert v['diagnostic_parent_checkpoint']==isparent
   expected_steps=([21] if isparent else list(range(1,26)) if name=='candidate-continuous25' else list(range(21,26)))
   assert v['completed_global_steps']==expected_steps and v['actual_optimizer_updates']==len(expected_steps)+(1 if name=='diagnostic-warm' else 0)
   if isparent:
    parent=small(name+'/diagnostic-parent-contract.json');assert parent['identity']['config']==b['diagnostic_parent_config_SHA256'] and parent!=contract
    m=small(name+'/resumed-update-private/progress.json');assert m['phase']=='RESUMED21_RETURNED' and m['completed_global_steps']==[21] and m['trainable_tensors']==476 and m['trainable_elements']==9922560 and len(m['files'])==12
   if name=='diagnostic-warm':assert v['diagnostic_warmup']['restored20_after_warmup']
   for cp in v['checkpoints']:
    f=W/name/('checkpoint-%d/state.private.pt'%cp['global_step']);assert not f.is_symlink() and f.stat().st_size==cp['bytes']
   before=sha(W/name/'frozen-before.json');assert before==sha(W/name/'frozen-after.json');frozen.append(before)
  assert len(set(frozen))==1
  if pair['passed']:
   full=small('candidate-full25-resume-comparison.json');assert full==r['full25_resume_comparison'] and full['actual_continuous_updates']==25 and full['actual_resumed_updates']==5 and full['original_resume_atol']==1e-6
   assert full['passed']==(full['complete_checkpoint25']['tree_equal'] and full['loss_max_abs']<=1e-6) and r['candidate_full25_resume_passed']==full['passed']
  else:assert r['full25_resume_comparison'] is None and not r['candidate_full25_resume_passed']
 else:
  f=small('failure-private.json');assert f['stage']==r['stage'] and f['error_type']==r['error_type']
 assert len(o['GPU_affinity'])==4 and o['peak_group_RSS_bytes']<=64*1024**3 and o['wall_seconds']<3600
 for owner in owners:
  now=proc(owner['PID']);assert now is None or now['startticks']!=owner['startticks'] or now['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   now=proc(int(n));assert not(now and now['PGID']==o['child']['PGID'] and now['state']!='Z')
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines();cards=[[v.strip() for v in row.split(',')] for row in rows if c['GPU_UUID'] in row]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 assert len(cards)==1 and cards[0][1]==c['GPU_PCI'] and int(cards[0][2])<=16 and not any(c['GPU_UUID'] in row for row in apps)
 post=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cards=cards,owned_compute_apps=[])
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-7000:])
print(json.dumps(dict(passed=failure is None,experiment_passed=experiment_passed,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,post=post,framework_files=379,
 new_CPU_tests=0,new_GPU_runs=0,raw_tensor_bytes_read=0,inside_seconds=time.monotonic()-anchor,Model_GO=False)))
if failure is not None:raise SystemExit(1)
