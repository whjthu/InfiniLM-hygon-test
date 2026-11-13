"""One finite owned SSH relay for the 14 SHA-identified public Qwen files."""
from pathlib import Path
import ctypes,ctypes.wintypes,datetime,hashlib,json,math,os,shlex,subprocess,sys,threading,time,uuid
B=Path(__file__).resolve().parents[2];D=B/'training-qwen-original-public-model-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
def winowner(pid):
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.wintypes.DWORD,ctypes.wintypes.BOOL,ctypes.wintypes.DWORD];k.OpenProcess.restype=ctypes.wintypes.HANDLE;k.GetProcessTimes.argtypes=[ctypes.wintypes.HANDLE]+[ctypes.POINTER(ctypes.wintypes.FILETIME)]*4;k.CloseHandle.argtypes=[ctypes.wintypes.HANDLE]
 handle=k.OpenProcess(0x1000,False,pid)
 if not handle:raise OSError(ctypes.get_last_error(),'Cannot identify owned process')
 values=[ctypes.wintypes.FILETIME() for _ in range(4)]
 try:
  if not k.GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):raise OSError(ctypes.get_last_error(),'Cannot read owned process start')
  return dict(PID=pid,startcreationticks=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime)
 finally:k.CloseHandle(handle)
def dump(path,value):
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
def main():
 epoch=time.time();anchor=time.monotonic();deadline=epoch+43200;request=uuid.uuid4().hex;claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
 def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
 task=json.loads(blob('coordination/tasks/training-qwen-original-public-model-v1.json'));assert task['transfer_GO'] and not task['GPU_GO'];files={}
 for name,pin in task['source_SHA256'].items():
  raw=blob(task['source']+'/'+name);assert hashlib.sha256(raw).hexdigest()==pin;files[name]=raw
 assert Path(__file__).read_bytes()==files['transfer_once.py'];rate_ns={};exec(compile(files['rate_limit.py'],'published_public_model_rate','exec'),rate_ns);plan=json.loads(files['MODEL_FILES.json']);owner=winowner(os.getpid());shared='PLAN='+repr(plan)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEPOCH='+repr(epoch)+'\nDEADLINE='+repr(deadline)+'\n'
 sender_code=shared+files['sender.py'].decode();receiver_code=shared+files['resume_stream.py'].decode()+'\n'+files['framed_stream.py'].decode()+'\n'+files['receiver.py'].decode()
 dump(D/'transfer-intent.json',dict(claim=claim,request_id=request,epoch=epoch,deadline=deadline,owner=owner,source_code_SHA256=hashlib.sha256(sender_code.encode()).hexdigest(),receiver_code_SHA256=hashlib.sha256(receiver_code.encode()).hexdigest(),weight_bytes=17545907231,raw_activations=0,replay_allowed=False))
 pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns);receiver=ns['ssh'];receiver[-1]=receiver[-1].removesuffix(' -')+' -c '+shlex.quote(receiver_code)
 A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');sender=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -c '+shlex.quote(sender_code)];assert len(subprocess.list2cmdline(receiver))<30000 and len(subprocess.list2cmdline(sender))<30000
 copied=0;checkpoint=0;error=None;orig=dest=None
 with (D/'sender.stderr.private.txt').open('xb') as se,(D/'receiver.stderr.private.txt').open('xb') as re,(D/'receiver.stdout.private.json').open('xb') as ro:
  dest=subprocess.Popen(receiver,stdin=subprocess.PIPE,stdout=ro,stderr=re);orig=subprocess.Popen(sender,stdout=subprocess.PIPE,stderr=se);dump(D/'owned-local-transports.json',dict(driver=owner,sender=winowner(orig.pid),receiver=winowner(dest.pid),epoch=epoch,deadline=deadline,request_id=request))
  def stop_owned():
   for p in (orig,dest):
    if p.poll() is None:
     try:p.terminate()
     except OSError:pass
  remaining=min(deadline-time.time(),43200-(time.monotonic()-anchor));assert remaining>0;timer=threading.Timer(remaining,stop_owned);timer.daemon=True;timer.start()
  try:
   while True:
    raw=orig.stdout.read(1024**2)
    if not raw:break
    dest.stdin.write(raw);copied+=len(raw);assert copied<=17545907231+16384
    remaining=min(deadline-time.time(),43200-(time.monotonic()-anchor));delay=rate_ns['delay_after_chunk'](copied,time.monotonic()-anchor,plan['relay_rate_bytes_per_second'],remaining);until=time.monotonic()+delay
    while time.monotonic()<until:time.sleep(min(.25,max(0.,until-time.monotonic())))
    if copied-checkpoint>=8*1024**2:
     checkpoint=copied;value=dict(claim=claim,request_id=request,epoch=epoch,deadline=deadline,pipe_bytes_relayed=copied,wall_seconds=time.monotonic()-anchor,driver=owner,sender=winowner(orig.pid),receiver=winowner(dest.pid),sender_live=orig.poll() is None,receiver_live=dest.poll() is None);tmp=D/'transfer-progress.tmp';tmp.write_bytes(json.dumps(value).encode());tmp.replace(D/'transfer-progress.json');print(json.dumps({k:value[k] for k in ['pipe_bytes_relayed','wall_seconds','sender_live','receiver_live']}),flush=True)
   dest.stdin.close()
   for child in (orig,dest):
    remaining=min(deadline-time.time(),43200-(time.monotonic()-anchor))-15
    if remaining<=0:raise TimeoutError('original transfer closing reserve exhausted')
    child.wait(timeout=remaining)
  except BaseException as failure:
   error=type(failure).__name__;stop_owned()
   for p in (orig,dest):
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  finally:timer.cancel()
 raw=(D/'receiver.stdout.private.json').read_bytes();result=dict(claim=claim,request_id=request,epoch=epoch,deadline=deadline,owner=owner,source_rc=orig.returncode,destination_rc=dest.returncode,error_type=error,pipe_bytes_relayed=copied,wall_seconds=time.monotonic()-anchor,stdout_SHA256=hashlib.sha256(raw).hexdigest(),GPU_cards=0,Model_GO=False)
 if dest.returncode==0:
  try:result['receiver_result']=json.loads(raw)
  except ValueError:result['receipt_parse_failed']=True
 result['passed']=bool(orig.returncode==dest.returncode==0 and error is None and result.get('receiver_result',{}).get('passed')
     and result['receiver_result']['request_id']==request and result['receiver_result']['claim']==claim
     and result['receiver_result']['files']==plan['files'] and result['receiver_result']['total_file_bytes']==17545907231)
 dump(D/'transfer-outcome.json',result);print(json.dumps({k:v for k,v in result.items() if k!='receiver_result'}))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
