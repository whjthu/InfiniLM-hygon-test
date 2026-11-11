"""Receive a bounded original processor archive; never execute a model or device."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,tarfile,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-gemma-processor-assets-v1';epoch=time.time();start=time.monotonic();assert not W.exists();W.mkdir(mode=0o700);model=W/'models/gemma-3-4b-it';model.mkdir(parents=True)
assert shutil.disk_usage(W).free>=100*1024**2;plan=PAYLOAD['manifest'];assert len(plan['files'])==14 and sum(x['bytes'] for x in plan['files'].values())==39352681
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('processor staging budget exhausted')));signal.alarm(170)
with (W/'intent.json').open('x') as f:json.dump(dict(claim=PAYLOAD['claim'],request_id=PAYLOAD['request'],epoch=epoch,deadline=epoch+180,archive_SHA256=PAYLOAD['archive_SHA256'],archive_bytes=PAYLOAD['archive_bytes'],GPU_cards=0),f)
class Reader:
 def __init__(self):self.sha=hashlib.sha256();self.bytes=0
 def read(self,size=-1):
  assert 0<size<=4*1024**2;raw=sys.stdin.buffer.read(size);self.sha.update(raw);self.bytes+=len(raw);assert self.bytes<=40500000;return raw
stream=Reader();done=[];result=None
try:
 parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D'],text=True,timeout=12).strip();assert parent=='RUNNING|40'
 with tarfile.open(fileobj=stream,mode='r|') as archive:
  for member in archive:
   name=member.name;assert member.isfile() and name in plan['files'] and name not in done and Path(name).name==name and not name.endswith('.safetensors');pin=plan['files'][name];assert member.size==pin['bytes'];h=hashlib.sha256();count=0;src=archive.extractfile(member)
   with (model/name).open('xb') as target:
    while True:
     chunk=src.read(1024**2)
     if not chunk:break
     count+=len(chunk);h.update(chunk);target.write(chunk)
   assert count==pin['bytes'] and h.hexdigest()==pin['SHA256'];done.append(name)
 while stream.read(1024**2):pass
 assert set(done)==set(plan['files']) and stream.bytes==PAYLOAD['archive_bytes'] and stream.sha.hexdigest()==PAYLOAD['archive_SHA256']
 (W/'MODEL_PROCESSOR_FILES.json').write_text(json.dumps(plan,indent=2));result=dict(passed=True,claim=PAYLOAD['claim'],request_id=PAYLOAD['request'],observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epoch,deadline=epoch+180,model_directory=str(model),installed_files=[dict(name=n,**plan['files'][n]) for n in sorted(done)],file_bytes=39352681,archive_bytes=stream.bytes,archive_SHA256=stream.sha.hexdigest(),full_model_weight_files_present=False,GPU_cards=0,model_loaded=False,wall_seconds=time.monotonic()-start)
except BaseException as error:result=dict(passed=False,claim=PAYLOAD['claim'],request_id=PAYLOAD['request'],error_type=type(error).__name__,completed_files=done,planned_files=14,GPU_cards=0,model_loaded=False)
with (W/'outcome.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
