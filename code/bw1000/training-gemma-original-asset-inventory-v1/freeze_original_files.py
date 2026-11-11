"""Freeze exact original Gemma file identities without loading a model or device."""
from pathlib import Path
import datetime,hashlib,json,os,re,signal,struct,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-original-file-pins-v1');epoch=time.time();anchor=time.monotonic();deadline=epoch+240
assert not W.exists() and len(CLAIM)==40 and len(REQUEST)==32
W.mkdir(mode=0o700);(W/'source.py').write_bytes(SOURCE);(W/'SOURCE_MODEL_PLAN.json').write_bytes(PLAN_BYTES)
plan=json.loads(PLAN_BYTES);root=Path(plan['model_root']);assert root==Path('/mnt/data/nvidia-baseline-20261010/inputs/models/gemma-3-4b-it') and len(plan['files'])==16
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def write(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
write('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,source_SHA256=hashlib.sha256(SOURCE).hexdigest(),plan_SHA256=hashlib.sha256(PLAN_BYTES).hexdigest(),owner=owner,GPU_cards=0))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original Gemma identity budget exhausted')));signal.setitimer(signal.ITIMER_REAL,max(.01,epoch+230-time.time()))
rows=[];hashed=0;sidecar_bytes=0;header_read=0;stage='exact_original_files';result=None
try:
 for pin in plan['files']:
  if min(deadline-time.time(),240-(time.monotonic()-anchor))<10:raise TimeoutError('Gemma identity closing reserve reached')
  name=pin['name'];assert Path(name).name==name;p=root/name;before=p.stat();assert p.is_file() and not p.is_symlink() and before.st_size==pin['bytes']
  companion=root/(name+'.source-sha256');assert companion.is_file() and not companion.is_symlink() and companion.stat().st_size==65;expected_raw=companion.read_bytes();sidecar_bytes+=len(expected_raw);expected=expected_raw.decode().strip();assert re.fullmatch('[0-9a-f]{64}',expected)
  h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    block=f.read(4*1024**2)
    if not block:break
    h.update(block);hashed+=len(block)
  after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and h.hexdigest()==expected
  if pin.get('SHA256'):assert pin['SHA256']==h.hexdigest()
  row=dict(name=name,bytes=before.st_size,SHA256=h.hexdigest(),source_companion_SHA256=hashlib.sha256(expected_raw).hexdigest(),matches_source_companion=True,stable_during_hash=True)
  if name.endswith('.safetensors'):
   with p.open('rb') as f:prefix=f.read(8);length=struct.unpack('<Q',prefix)[0];assert length==pin['header_bytes'];raw=f.read(length);assert len(raw)==length
   header_read+=8+length;header=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==pin['header_SHA256'];keys={k:x for k,x in header.items() if k!='__metadata__'};assert len(keys)==pin['tensor_count'] and all(x['dtype']=='BF16' for x in keys.values());row.update(header_SHA256=pin['header_SHA256'],header_bytes=length,tensor_count=len(keys),dtype='BF16')
  rows.append(row)
 index=json.loads((root/'model.safetensors.index.json').read_bytes());assert len(index['weight_map'])==883 and set(index['weight_map'].values())=={x['name'] for x in rows if x['name'].endswith('.safetensors')}
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,model_root=str(root),files=rows,planned_files=16,completed_files=len(rows),total_file_bytes=hashed,weight_file_bytes=sum(x['bytes'] for x in rows if x['name'].endswith('.safetensors')),weight_tensor_count=883,weight_dtype='BF16',header_extra_read_bytes=header_read,source_companion_bytes_read=sidecar_bytes,scope='Exact original source file bytes match their recorded source-sha256 companions; no copy or model execution.',historical_performance_weight_binding_independently_proven=False,raw_tensor_export_bytes=0,GPU_cards=0,torch_imported=False,original_model_loaded=False,target_files_installed=False,Model_GO=False,wall_seconds=time.monotonic()-anchor)
except BaseException as error:
 result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,owner=owner,stage=stage,error_type=type(error).__name__,message=str(error)[:400],planned_files=16,completed_files=len(rows),completed_rows=rows,hash_input_bytes=hashed,GPU_cards=0,Model_GO=False,wall_seconds=time.monotonic()-anchor)
finally:write('result.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
