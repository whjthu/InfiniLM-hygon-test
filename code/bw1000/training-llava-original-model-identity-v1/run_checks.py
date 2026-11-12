"""One complete original OneVision indexed-checkpoint read; no tensor decode."""
from pathlib import Path
import datetime,hashlib,json,os,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic()
initial=deadline-time.time();assert 0<initial<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original identity epoch exhausted')))
signal.setitimer(signal.ITIMER_REAL,initial);sys.path.insert(0,str(F))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def check():
 if min(deadline-time.time(),initial-(time.monotonic()-start))<=0:raise TimeoutError('original hash epoch exhausted')
events=[];phase='source'
def progress(v):
 event=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**v);events.append(event)
 p=W/'hash-progress.tmp';p.write_text(json.dumps(event));p.replace(W/'hash-progress.json')
 with (W/'hash-progress.private.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
dump('CPU-owner.json',owner())
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():assert (F/name).stat().st_size==pin['bytes'] and sha(F/name)==pin['SHA256']
 original=json.loads((S/'ORIGINAL_MODEL_METADATA.json').read_bytes())
 config=original['files']['config'];index=original['files']['index'];root=Path(index['path']).parent
 for pin in (config,index):
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 from tools.hygon_indexed_model_identity import inspect_indexed_model
 expected={s['name']:{k:s[k] for k in ('bytes','mtime_ns','inode','device')} for s in original['shards']}
 assert len(expected)==4 and original['tensors']==698
 phase='complete-original-four-shard-stream';hash_start=time.monotonic()
 identity=inspect_indexed_model(root,Path(index['path']).name,index['SHA256'],expected,check_deadline=check,progress=progress)
 hash_wall=time.monotonic()-hash_start
 assert identity['tensors']==698 and identity['full_file_bytes_read']==17054506776 and identity['tensor_payload_bytes']==17054429248
 identity.update(case='T04.sft',model_root=str(root),config=config,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  epoch=epoch,deadline=deadline,original_metadata_SHA256=sha(S/'ORIGINAL_MODEL_METADATA.json'),
  native_loaded_schema_correspondence_qualified=False,official_release_provenance_qualified=False,historical_training_file_bytes_qualified=False,
  initial_values_qualified=False,actual_weight_loading_qualified=False,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False)
 dump('MODEL_IDENTITY.json',identity);check();assert 'torch' not in sys.modules
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),original_shards=4,tensors=698,
  full_file_bytes_read=identity['full_file_bytes_read'],tensor_payload_bytes=identity['tensor_payload_bytes'],serialized_tensor_elements=identity['serialized_tensor_elements'],
  MODEL_IDENTITY_SHA256=sha(W/'MODEL_IDENTITY.json'),hash_progress_SHA256=sha(W/'hash-progress.private.jsonl'),progress_events=len(events),
  hash_wall_seconds=hash_wall,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  weight_bytes_read_for_hash=True,tensor_values_decoded=False,weight_values_exported=False,model_loaded=False,Torch_imported=False,GPU_cards=0,GPU_GO=False,Model_GO=False,
  new_stdlib_fixture_runs=0,native_meta_replayed=False,
  qualification='Current original four full-file SHA,index698/header range closure only; no tensor decode,loading,initialtrainables,forward,update,numerics,resume,performance or official release provenance.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 dump('hash-failure.json',dict(phase=phase,error_type=type(error).__name__,last_progress=events[-1] if events else None,planned_shards=4))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result);print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
