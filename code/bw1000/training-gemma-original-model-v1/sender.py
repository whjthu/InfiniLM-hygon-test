"""Send only the two original Gemma shards; full SHA is checked during this stream."""
from pathlib import Path
import datetime,hashlib,json,math,os,signal,socket,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-original-model-v1');assert not W.exists();W.mkdir(mode=0o700);start=time.monotonic();remaining=DEADLINE-time.time();assert math.isfinite(remaining) and 0<remaining<=43200
owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def save(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def expired(*_):raise TimeoutError('original Gemma transfer epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.1,remaining-10))
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,files=PLAN['weights'],GPU_cards=0))
done={};active=None;copied=0;checkpoint=0;result=None
try:
 for name,pin in sorted(PLAN['weights'].items()):
  active=name;p=Path(PLAN['source_root'])/name;before=p.stat();expected=PLAN['preflight_source_files'][name];assert not p.is_symlink() and before.st_size==pin['bytes'] and dict(bytes=before.st_size,mtime_ns=before.st_mtime_ns,inode=before.st_ino,device=before.st_dev)==expected
  sys.stdout.buffer.write((json.dumps(dict(name=name,**pin))+'\n').encode());sys.stdout.buffer.flush();h=hashlib.sha256();file_copied=0
  with p.open('rb') as f:
   while file_copied<pin['bytes']:
    raw=f.read(min(1024**2,pin['bytes']-file_copied));assert raw;h.update(raw);sys.stdout.buffer.write(raw);file_copied+=len(raw);copied+=len(raw)
    if copied-checkpoint>=8*1024**2:
     checkpoint=copied;tmp=W/'progress.tmp';tmp.write_text(json.dumps(dict(claim=CLAIM,request_id=REQUEST,owner=owner,epoch=EPOCH,deadline=DEADLINE,active_file=name,active_file_bytes=file_copied,total_weight_bytes_read=copied,active_prefix_SHA256=h.hexdigest(),completed_files=list(done),wall_seconds=time.monotonic()-start)));tmp.replace(W/'progress.json')
   assert not f.read(1)
  sys.stdout.buffer.flush();after=p.stat();assert h.hexdigest()==pin['SHA256'] and (before.st_size,before.st_ino,before.st_mtime_ns,before.st_dev)==(after.st_size,after.st_ino,after.st_mtime_ns,after.st_dev);done[name]=pin
 result=dict(passed=True,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,files=done,weight_bytes=copied,wall_seconds=time.monotonic()-start,GPU_cards=0,model_loaded=False)
except BaseException as error:result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,error_type=type(error).__name__,active_file=active,completed_files=done,total_weight_bytes_read=copied,wall_seconds=time.monotonic()-start,GPU_cards=0,model_loaded=False)
save('outcome.json',result);print(json.dumps(result),file=sys.stderr)
if not result['passed']:raise SystemExit(1)
