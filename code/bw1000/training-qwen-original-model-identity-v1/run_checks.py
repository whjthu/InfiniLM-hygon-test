"""Pin the original four Qwen weight shards with the shared streaming inspector."""
from pathlib import Path
import datetime,hashlib,io,json,os,resource,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];stage=W/'src';framework=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic()
assert 0<deadline-time.time()<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('model identity original epoch exhausted')))
signal.setitimer(signal.ITIMER_REAL,deadline-time.time());sys.path.insert(0,str(framework))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source';events=[]
def check_deadline():
 if not epoch<=time.time()<deadline or time.monotonic()-start>=260:raise TimeoutError('original hash deadline exhausted')
def progress(value):
 event=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**value);events.append(event)
 p=W/'hash-progress.tmp';p.write_text(json.dumps(event));p.replace(W/'hash-progress.json')
 with (W/'hash-progress.private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():
  p=framework/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='shared-CPU-fixtures';log=io.StringIO()
 suite=unittest.defaultTestLoader.discover(str(framework/'tests'),pattern='test_indexed_model_identity.py')
 unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.txt').write_text(log.getvalue());assert unit.wasSuccessful() and unit.testsRun==6
 from tools.hygon_indexed_model_identity import inspect_indexed_model
 observed=json.loads((stage/'ORIGINAL_MODEL_METADATA.json').read_bytes());config=observed['files']['config'];p=Path(config['path'])
 assert p.stat().st_size==config['bytes'] and sha(p)==config['SHA256']
 index=observed['files']['index'];root=Path(index['path']).parent
 expected={s['name']:{k:s[k] for k in ('bytes','mtime_ns')} for s in observed['shards']}
 assert len(expected)==4 and observed['tensors']==750
 phase='complete-original-four-shard-stream';hash_started=time.monotonic()
 identity=inspect_indexed_model(root,Path(index['path']).name,index['SHA256'],expected,check_deadline=check_deadline,progress=progress)
 hash_wall=time.monotonic()-hash_started
 assert identity['tensors']==750 and identity['full_file_bytes_read']==sum(s['bytes'] for s in observed['shards'])
 identity.update(model_root=str(root),config=config,case='T01.sft',observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  epoch=epoch,deadline=deadline,initial_values_qualified=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False)
 dump('MODEL_IDENTITY.json',identity)
 assert 'torch' not in sys.modules
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=6,
  original_shards=4,tensors=750,full_file_bytes_read=identity['full_file_bytes_read'],tensor_payload_bytes=identity['tensor_payload_bytes'],serialized_tensor_elements=identity['serialized_tensor_elements'],
  MODEL_IDENTITY_SHA256=sha(W/'MODEL_IDENTITY.json'),hash_progress_SHA256=sha(W/'hash-progress.private.jsonl'),progress_events=len(events),
  hash_wall_seconds=hash_wall,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  weight_bytes_read_for_hash=True,tensor_values_decoded=False,model_loaded=False,Torch_imported=False,GPU_cards=0,GPU_GO=False,Model_GO=False,
  qualification='Complete original file identities and safetensors/index structural closure only; not external canonical provenance,loaded values,forward,numerics or performance.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 dump('hash-failure.json',dict(phase=phase,error_type=type(error).__name__,last_progress=events[-1] if events else None,planned_shards=4))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
