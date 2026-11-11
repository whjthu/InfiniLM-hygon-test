"""Stream only the fourteen verified original Gemma processor assets."""
from pathlib import Path
import hashlib,json,os,signal,sys,tarfile,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-processor-archive-v1');epoch=time.time();start=time.monotonic();assert not W.exists();W.mkdir(mode=0o700)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('processor archive budget exhausted')));signal.alarm(170)
plan=json.loads(PLAN);root=Path(plan['original_model_root']);assert len(plan['files'])==14 and sum(x['bytes'] for x in plan['files'].values())==39352681
with (W/'intent.json').open('x') as f:json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+180,plan_SHA256=hashlib.sha256(PLAN).hexdigest()),f)
class Writer:
 def __init__(self):self.sha=hashlib.sha256();self.bytes=0
 def write(self,data):self.sha.update(data);self.bytes+=len(data);return sys.stdout.buffer.write(data)
 def flush(self):sys.stdout.buffer.flush()
writer=Writer();done=[]
try:
 with tarfile.open(fileobj=writer,mode='w|') as archive:
  for name,pin in sorted(plan['files'].items()):
   p=root/name;assert Path(name).name==name and not name.endswith('.safetensors') and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'];header=tarfile.TarInfo(name);header.size=pin['bytes'];header.mode=0o644;header.mtime=0
   with p.open('rb') as f:archive.addfile(header,f)
   done.append(name)
 writer.flush();result=dict(passed=True,claim=CLAIM,request_id=REQUEST,files=done,file_bytes=39352681,archive_bytes=writer.bytes,archive_SHA256=writer.sha.hexdigest(),weight_files=0,raw_activations=0,GPU_cards=0,wall_seconds=time.monotonic()-start)
except BaseException as error:result=dict(passed=False,claim=CLAIM,request_id=REQUEST,completed_files=done,error_type=type(error).__name__,archive_bytes=writer.bytes,GPU_cards=0)
with (W/'outcome.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),file=sys.stderr)
if not result['passed']:raise SystemExit(1)
