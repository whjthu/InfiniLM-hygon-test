"""One independent bounded full target-file read, without decoding a checkpoint."""
from pathlib import Path
import datetime,hashlib,json,math,os,resource,signal,socket,subprocess,sys,time
epoch=time.time();anchor=time.monotonic();deadline=epoch+300;work_deadline=epoch+270
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results');W=R/'training-vggt-target-full-SHA-CPU-v1'
assert not W.exists();W.mkdir(mode=0o700)
fields=Path('/proc/self/stat').read_text().split(') ',1)[1].split();owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=int(fields[2]),startticks=int(fields[19]))
def save(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def remaining():
 value=min(work_deadline-time.time(),270-(time.monotonic()-anchor))
 if value<=0:raise TimeoutError('original target fullSHA epoch exhausted')
 return value
def expired(*_):raise TimeoutError('original target fullSHA epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining())
affinity=sorted(os.sched_getaffinity(0))[:2];assert len(affinity)==2;os.sched_setaffinity(0,affinity)
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
save('intent.json',dict(claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,work_deadline=work_deadline,owner=owner,original_transfer_request=ORIGINAL['request_id'],GPU_cards=0,replay_allowed=False))
print(json.dumps(dict(stage='REMOTE_HASH_INTENT',request_id=REQUEST,owner=owner,epoch=epoch,deadline=deadline)),flush=True)
phase='original_transfer_receipts';read_bytes=0;h=hashlib.sha256();checkpoints=0;result=None
try:
 original=R/'training-vggt-original-public-model-v1'
 raw=(original/'outcome.json').read_bytes();assert len(raw)<=65536;receiver=json.loads(raw)
 intent_raw=(original/'intent.json').read_bytes();assert len(intent_raw)<=65536;intent=json.loads(intent_raw)
 assert receiver==ORIGINAL['receiver_result'] and receiver['passed'] and receiver['model_loaded'] is False and receiver['GPU_cards']==0
 assert intent['claim']==receiver['claim']==ORIGINAL['claim'] and intent['request_id']==receiver['request_id']==ORIGINAL['request_id'] and intent['epoch']==receiver['epoch']==ORIGINAL['epoch'] and intent['deadline']==receiver['deadline']==ORIGINAL['deadline']
 assert receiver['files']==intent['files']=={'model.pt':{'bytes':5026874952,'SHA256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}}
 parent=subprocess.run(['squeue','-h','-j','1087656','-o','%T|%D'],capture_output=True,text=True,timeout=min(12,remaining()));assert parent.returncode==0 and parent.stdout.strip()=='RUNNING|40'
 model=original/'model/model.pt';assert not model.is_symlink() and model.resolve()==Path(receiver['model_directory'])/'model.pt' and sorted(p.name for p in model.parent.iterdir())==['model.pt']
 def stat():
  s=model.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode)
 before=stat();assert before['bytes']==5026874952
 phase='independent_complete_target_SHA';hash_start=time.monotonic()
 with model.open('rb') as stream:
  while True:
   remaining();chunk=stream.read(8*1024**2)
   if not chunk:break
   h.update(chunk);read_bytes+=len(chunk)
   if read_bytes//(256*1024**2)>checkpoints:
    checkpoints=read_bytes//(256*1024**2);p=W/'progress.tmp';p.write_text(json.dumps(dict(request_id=REQUEST,planned_bytes=5026874952,completed_bytes=read_bytes,epoch=epoch,deadline=deadline,failed=False)));p.replace(W/'progress.json')
 hash_seconds=time.monotonic()-hash_start;after=stat()
 assert read_bytes==5026874952 and before==after and h.hexdigest()=='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
 assert 'torch' not in sys.modules and 'pickle' not in sys.modules
 result=dict(passed=True,claim_commit=CLAIM,request_id=REQUEST,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epoch,deadline=deadline,work_deadline=work_deadline,owner=owner,original_transfer_claim=ORIGINAL['claim'],original_transfer_request=ORIGINAL['request_id'],original_receiver_outcome_SHA256=hashlib.sha256(raw).hexdigest(),original_receiver_intent_SHA256=hashlib.sha256(intent_raw).hexdigest(),original_receiver_matches_local_terminal=True,parent_state='RUNNING',parent_nodes=40,model_path=str(model),file_bytes_read=read_bytes,full_file_SHA256=h.hexdigest(),before_stat=before,after_stat=after,CPU_affinity=sorted(os.sched_getaffinity(0)),address_space_limit_bytes=512*1024**2,hash_seconds=hash_seconds,wall_seconds=time.monotonic()-anchor,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,progress_checkpoints=checkpoints,byte_stream_only=True,Torch_imported=False,pickle_imported=False,checkpoint_values_decoded=False,model_loaded=False,CO3D_input_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False,qualification='Independent complete target public-file SHA only; no checkpoint decoding,state-dict schema,model loading,data,numerics,training,resume,performance or task effects.')
except BaseException as e:
 result=dict(passed=False,claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,work_deadline=work_deadline,owner=owner,error_type=type(e).__name__,phase=phase,file_bytes_read=read_bytes,planned_bytes=5026874952,partial_prefix_SHA256=h.hexdigest(),wall_seconds=time.monotonic()-anchor,GPU_cards=0,GPU_GO=False,Model_GO=False)
save('outcome.json',result);print(json.dumps(dict(stage='REMOTE_HASH_TERMINAL',request_id=REQUEST,result=result)),flush=True)
raise SystemExit(0 if result['passed'] else 1)
