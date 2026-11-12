"""Independent bounded original CPU fixture terminal,source and owner reader."""
from pathlib import Path
import datetime,hashlib,json,os,signal,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-first-update-recorder-CPU-v1')
signal.alarm(60);start=time.monotonic();artifacts={};failure=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 for n in ('intent.json','outcome.json','CPU-owner.json','CPU-child-owner.json','CPU-result.json','CPU-tests.private.log','CPU.private.log','failure-private.json'):
  p=W/n
  if not p.is_file():continue
  assert not p.is_symlink() and p.stat().st_size<1024**2
  artifacts[n]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  if n.endswith('.json'):artifacts[n]['value']=json.loads(p.read_bytes())
  elif n=='CPU-tests.private.log':artifacts[n]['text']=p.read_text()
 i=artifacts['intent.json']['value'];o=artifacts['outcome.json']['value'];r=artifacts['CPU-result.json']['value']
 assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+300 and r['deadline']==i['epoch']+260
 assert o['returncode']==1 and o['passed'] is False and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['CPU_result']==r and artifacts['CPU-result.json']['SHA256']==o['CPU_result_SHA256']
 assert o['wall_seconds']<300 and o['peak_group_RSS_bytes']<4*1024**3 and len(o['CPU_affinity'])==2
 assert r['passed'] is False and r['real_CPU_tests']==3 and r['failures']==r['skipped']==0 and r['errors']==1 and r['framework_files']==362
 assert not r['weight_or_dataset_bodies_read'] and not r['GPU_initialized'] and r['GPU_cards']==0 and not r['GPU_GO'] and not r['Model_GO']
 assert artifacts['CPU-tests.private.log']['SHA256']==r['tests_log_SHA256'] and 'Ran 3 tests' in artifacts['CPU-tests.private.log']['text']
 text=artifacts['CPU-tests.private.log']['text']
 assert 'test_full25_exact_loss_parameters_optimizer_and_rng (test_zimage_first_update_capture.RealCPUCapture.test_full25_exact_loss_parameters_optimizer_and_rng) ... ok' in text
 assert 'test_prepared_input_mutation_preserves_failure_denominator (test_zimage_first_update_capture.RealCPUCapture.test_prepared_input_mutation_preserves_failure_denominator) ... ok' in text
 assert 'first-update artifact exceeds its byte budget' in text and 'RuntimeError: [enforce fail' in text
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert fm['framework_commit']==r['framework_commit'] and len(fm['files'])==362
 for n,pin in fm['files'].items():
  p=W/'framework'/n;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 binding=json.loads((W/'src/BINDING.json').read_bytes());assert sha(Path(binding['original_loss_path']))==binding['original_loss_SHA256']==r['native_loss_SHA256']
 for owner in (o['owner'],o['child'],r['owner']):
  current=proc(owner['PID']);assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   row=proc(int(n));assert not(row and row['PGID']==o['child']['PGID'] and row['state']!='Z')
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-6000:])
print(json.dumps(dict(passed=failure is None,failure=failure,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,
 artifacts=artifacts,original_run_passed=False,original_returncode=1,passed_tests_retained=2,failed_tests_retained=1,original_failure_preserved=True,original_source_files=len(EXPECTED_FILES),framework_files=362,new_CPU_or_GPU_invocations=0,reader_weight_or_dataset_bytes=0,inside_seconds=time.monotonic()-start,GPU_GO=False,Model_GO=False)))
if failure is not None:raise SystemExit(1)
