from pathlib import Path
import base64,hashlib,json,math,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-reference-values-CPU-v1')
epoch=time.time();mono=time.monotonic();deadline=epoch+720
assert not W.exists();W.mkdir(mode=0o700);(W/'source').mkdir();(W/'evidence').mkdir()
def save(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def left(reserve=0):
 n=min(deadline-time.time(),720-(time.monotonic()-mono))-reserve;assert n>0;return n
for name,row in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256']
 p=W/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
plan=json.loads((W/'source/PLAN.json').read_bytes());capture=Path(plan['capture_root'])
for r in plan['reference_receipts']:assert sha(capture/('rank-%02d'%r['rank'])/'receipt.json')==r['SHA256']
affinity=sorted(os.sched_getaffinity(0))[:4];assert len(affinity)==4
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,host=socket.gethostname(),PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_pins={n:x['SHA256'] for n,x in FILES.items()},CPU_affinity=affinity,GPU_cards=0,raw_tensor_transfer_bytes=0))
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',request_id=REQUEST,claim=CLAIM,epoch=epoch,deadline=deadline)),flush=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',VALUES_EPOCH=str(epoch),VALUES_DEADLINE=str(epoch+600))
forced=False
with (W/'CPU.log').open('xb') as log:
 p=subprocess.Popen(['/usr/bin/python3','-B','-u',str(W/'source/audit_values.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,affinity))
 st=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split();owned=dict(PID=p.pid,PGID=int(st[2]),startticks=int(st[19]),host=socket.gethostname());save('owned-child.json',owned)
 try:p.wait(timeout=min(660,left(35)))
 except subprocess.TimeoutExpired:
  forced=True;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
child_alive=False
try:
 st=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split();child_alive=int(st[19])==owned['startticks']
except FileNotFoundError:pass
out=dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,host=socket.gethostname(),returncode=p.returncode,forced_cleanup=forced,owned_child=owned,owned_child_alive=child_alive,log_SHA256=sha(W/'CPU.log'),wall_seconds=time.monotonic()-mono,GPU_cards=0,raw_tensor_transfer_bytes=0)
for name in ('CPU-result.json','progress.json'):
 path=W/'evidence'/name
 if path.exists():
  raw=path.read_bytes();out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out['passed']=bool(p.returncode==0 and not forced and not child_alive and out.get('CPU-result.json',{}).get('value',{}).get('passed'))
if not out['passed']:out['error_tail']=(W/'CPU.log').read_text(errors='replace')[-1400:]
save('outcome.json',out);print(json.dumps(out),flush=True)
