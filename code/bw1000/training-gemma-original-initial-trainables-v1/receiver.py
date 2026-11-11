"""Receive the original update-zero Gemma trainables into a new immutable directory."""
from pathlib import Path
import datetime,hashlib,json,math,os,shutil,signal,socket,subprocess,sys,time
W=Path(PLAN['target_root']);assert not W.exists();assert shutil.disk_usage(W.parent).free>sum(p['bytes'] for p in PLAN['files'].values())+1024**3
remaining=DEADLINE-time.time();assert math.isfinite(remaining) and 0<remaining<=1800
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D'],text=True,timeout=12).strip();assert parent=='RUNNING|40'
W.mkdir(mode=0o700);model=W/'model-trainables';model.mkdir(mode=0o700);start=time.monotonic();owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
def save(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def expired(*_):raise TimeoutError('original Gemma transfer epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.1,remaining-10));save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,files=PLAN['files'],GPU_cards=0))
done={};active=None;checkpoint=0;copied=0;result=None
try:
 for name,pin in sorted(PLAN['weights'].items()):
  active=name;header=sys.stdin.buffer.readline(1024);assert header.endswith(b'\n') and json.loads(header)==dict(name=name,**pin);partial=model/(name+'.partial')
  with partial.open('xb'):pass
  before_completed=copied
  def progress(written,digest):
   global checkpoint
   total=before_completed+written
   if total-checkpoint<8*1024**2:return
   checkpoint=total;tmp=W/'progress.tmp';tmp.write_text(json.dumps(dict(claim=CLAIM,request_id=REQUEST,owner=owner,epoch=EPOCH,deadline=DEADLINE,active_file=name,active_file_bytes=written,total_weight_bytes_written=total,active_prefix_SHA256=digest,completed_files=list(done),wall_seconds=time.monotonic()-start)));tmp.replace(W/'progress.json')
  copied_file=append_verified(partial,FileSlice(sys.stdin.buffer,pin['bytes']),0,pin['bytes'],hashlib.sha256(b'').hexdigest(),pin['SHA256'],progress)
  assert copied_file['bytes']==pin['bytes'];os.link(partial,model/name);partial.unlink();copied+=pin['bytes'];done[name]=pin
 assert not sys.stdin.buffer.read(1);assert sorted(p.name for p in model.iterdir())==sorted(PLAN['files']) and done==PLAN['files']
 result=dict(passed=True,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,state='ORIGINAL_GEMMA_INITIAL_TRAINABLE_FILE_STREAM_SHA_VERIFIED',files=done,weight_bytes=copied,total_file_bytes=sum(p['bytes'] for p in done.values()),metadata_locally_reused_bytes=0,model_directory=str(model),wall_seconds=time.monotonic()-start,GPU_cards=0,model_loaded=False,Model_GO=False)
except BaseException as error:result=dict(passed=False,claim=CLAIM,request_id=REQUEST,epoch=EPOCH,deadline=DEADLINE,owner=owner,error_type=type(error).__name__,active_file=active,active_partial_bytes=(model/(active+'.partial')).stat().st_size if active and (model/(active+'.partial')).exists() else None,completed_files=done,wall_seconds=time.monotonic()-start,GPU_cards=0,model_loaded=False,Model_GO=False)
save('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
