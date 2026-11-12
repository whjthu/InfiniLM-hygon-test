"""Independent actual CPU-consumer terminal/source/ownership verification."""
from pathlib import Path
import hashlib,json,os,signal,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-resumed21-root-consumer-CPU-v1')
signal.alarm(60);anchor=time.monotonic();artifacts={};failure=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def small(n):
 p=W/n;assert not p.is_symlink() and p.stat().st_size<2*1024**2;b=p.read_bytes();v=json.loads(b);artifacts[n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),value=v);return v
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('CPU-result.json');v=small('ORIGINAL-CONSUME.json');comp=small('recovered-comparison.json')
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+300 and r['deadline']==i['epoch']+260
 assert o['passed'] and o['returncode']==0 and o['owned_group_reaped'] and o['stop_reason'] is None and o['CPU_result']==r and o['CPU_result_SHA256']==artifacts['CPU-result.json']['SHA256']
 assert r['CPU_tests']==dict(tests=3,failures=0,errors=0,skipped=0,passed=True)
 assert v['passed'] and v['original_overall_passed'] is False and sha(W/'ORIGINAL-CONSUME.json')==r['original_consume_SHA256']
 assert comp==r['recovered_comparison']==v['recovered_comparison'] and comp['original_GPU_overall_passed'] is False and comp['actual_optimizer_calls']==3
 assert r['new_model_loads']==r['new_GPU_runs']==0 and not r['GPU_initialized'] and not r['private_values_exported']
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==371 and fm['framework_commit']==r['framework_commit']
 for n,pin in fm['files'].items():
  f=W/'framework'/n;assert not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['SHA256'] and f.stat().st_mode&0o777==pin['mode']
 assert len(o['CPU_affinity'])==2 and o['peak_group_RSS_bytes']<=4*1024**3 and o['wall_seconds']<300
 for owner in (o['owner'],o['child'],r['owner']):
  now=proc(owner['PID']);assert now is None or now['startticks']!=owner['startticks'] or now['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   now=proc(int(n));assert not(now and now['PGID']==o['child']['PGID'] and now['state']!='Z')
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:])
print(json.dumps(dict(passed=failure is None,failure=failure,claim=CLAIM,request_id=REQUEST,artifacts=artifacts,framework_files=371,new_GPU_runs=0,reader_private_tensor_bytes_read=0,inside_seconds=time.monotonic()-anchor,Model_GO=False)))
if failure is not None:raise SystemExit(1)
