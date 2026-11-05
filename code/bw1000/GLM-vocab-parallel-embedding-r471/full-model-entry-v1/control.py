"""Full Model lifecycle barriers using original876 scalar framing/socketpairs.
No SDK, Runtime, constructor, collective or tensor is imported/executed here.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,secrets,socket,subprocess,sys,time,traceback,re
from ready_socket_parent import send_json,receive_json,canonical
D=Path(__file__).resolve().parent;R=Path('/refactor')if str(D).startswith('/refactor/')else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');LABEL='GLM-vocab-parallel-embedding-r471-full-Model-entry-v1';OUT=R/'results'/LABEL;NODES=['a02r3n15','a02r3n16'];IPS=['10.1.2.55','10.1.2.56'];PORT=43844;H=lambda b:hashlib.sha256(b).hexdigest()
CHILDREN=[];CURRENT_PHASE='created';PHASES=['created','loaded','case0','case1','quiescent','closed']
class Run:
 def stop_requested(self):return any(p.poll()not in [None,0]for p in CHILDREN)
RUN=Run()
def set_phase(phase):
 global CURRENT_PHASE
 assert phase in PHASES;CURRENT_PHASE=phase

def save(p,q):
 t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(q,indent=2)+'\n');os.replace(str(t),str(p))
def limit(epoch):
 cap=1800 if CURRENT_PHASE=='created'else 10800 if CURRENT_PHASE=='loaded'else 13800
 remaining=epoch+cap-time.time()
 if remaining<=0:raise TimeoutError('Original fullModel epoch phase cap:'+CURRENT_PHASE)
 return time.monotonic()+remaining

def send(s,v,e):send_json(s,v,RUN,lambda:False,limit(e))
def recv(s,e):return receive_json(s,RUN,lambda:False,limit(e))
def identity(epoch):return dict(protocol='r470-fixed16-pipe-scalar-v1',label=LABEL,epoch=epoch,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H((D/'SOURCE_MANIFEST.json').read_bytes()),CPU_only=False,scope='globalTP16-full78layer-checkpoint-entry-notbenchmark')
def validate(records,node,phase,ident):
 assert type(records)is list and len(records)==8
 for dev,r in enumerate(records):
  assert set(r)=={'identity','phase','rank','node','device','PID','startticks'}and r['identity']==ident and r['phase']==phase and type(r['rank'])is int and r['rank']==node*8+dev and r['node']==NODES[node]and r['device']==dev and type(r['PID'])is int and r['PID']>0 and type(r['startticks'])is str and r['startticks'].isdigit()
 return records

def child(node,device,fd):
 from owner import main_owned
 return main_owned(node,device,fd)

def exchange(conn,node,records,phase,ident,epoch):
 validate(records,node,phase,ident)
 if node==1:
  nonce=secrets.token_hex(32);send(conn,dict(kind='challenge',identity=ident,phase=phase,nonce=nonce),epoch);other=recv(conn,epoch);assert set(other)=={'kind','identity','phase','nonce','records'}and other['kind']=='records'and other['identity']==ident and other['phase']==phase and other['nonce']==nonce;allrecords=validate(other['records'],0,phase,ident)+records;digest=H(canonical(allrecords));send(conn,dict(kind='all-ready',identity=ident,phase=phase,nonce=nonce,records=allrecords,digest=digest),epoch);assert recv(conn,epoch)==dict(kind='ACK',identity=ident,phase=phase,nonce=nonce,digest=digest)
 else:
  challenge=recv(conn,epoch);assert set(challenge)=={'kind','identity','phase','nonce'}and challenge['kind']=='challenge'and challenge['identity']==ident and challenge['phase']==phase and re.fullmatch('[0-9a-f]{64}',challenge['nonce']);nonce=challenge['nonce'];send(conn,dict(kind='records',identity=ident,phase=phase,nonce=nonce,records=records),epoch);v=recv(conn,epoch);assert set(v)=={'kind','identity','phase','nonce','records','digest'}and v['kind']=='all-ready'and v['identity']==ident and v['phase']==phase and v['nonce']==nonce;assert v['records'][:8]==records;validate(v['records'][8:],1,phase,ident);allrecords=v['records'];digest=H(canonical(allrecords));assert v['digest']==digest;send(conn,dict(kind='ACK',identity=ident,phase=phase,nonce=nonce,digest=digest),epoch)
 return digest

def worker():
 from shared import verify
 verify(os.environ['GLM470_GPU_MF'])
 node=int(os.environ['SLURM_PROCID']);assert node in(0,1)and socket.gethostname().split('.')[0]==NODES[node];epoch=float(os.environ['GLM470_EPOCH']);ident=identity(epoch);q=dict(node=NODES[node],node_index=node,actual_GPU=ident['step'],gpu_epoch=epoch,common_CPUmask=sorted(os.sched_getaffinity(0)),identity=ident,PID=os.getpid(),UID=os.getuid(),startticks=Path('/proc/self/stat').read_text().split()[21],rounds=[],owners=[],passed=False,completed=False);assert len(q['common_CPUmask'])>=64;children=[];socks=[];conn=None;listener=None
 local=Path('/local')/('r470-leader-%d-%d.jsonl'%(node,os.getpid()));trace=local.open('xb',buffering=0)
 def mark(p,**kw):trace.write((json.dumps(dict(phase=p,epoch=time.time(),monotonic=time.monotonic(),**kw))+'\n').encode())
 try:
  assert (D/'ready_socket_parent.py').is_file()
  if node==1:listener=socket.socket();listener.set_inheritable(False);listener.bind((IPS[1],PORT));listener.listen(1);mark('listener-live')
  for dev in range(8):
   parent,childsock=socket.socketpair();childsock.set_inheritable(True)
   with(OUT/('owner-%02d.log'%(node*8+dev))).open('xb')as f:p=subprocess.Popen([sys.executable,'-B',str(D/'control.py'),'child','--node',str(node),'--device',str(dev),'--fd',str(childsock.fileno())],pass_fds=(childsock.fileno(),),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   childsock.close();children.append(p);CHILDREN.append(p);socks.append(parent);q['owners'].append(dict(rank=node*8+dev,global_rank=node*8+dev,PID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]));save(OUT/('worker-%d.json'%node),q);mark('child-spawn',rank=node*8+dev,PID=p.pid)
  for phase in PHASES:
   set_phase(phase)
   records=[]
   for dev,s in enumerate(socks):
    mark('local-lastmissing',round=phase,rank=node*8+dev);r=recv(s,epoch);assert r['PID']==q['owners'][dev]['PID']and r['startticks']==q['owners'][dev]['startticks'];records.append(r)
   validate(records,node,phase,ident);mark('local8-ready',round=phase)
   if conn is None:
    if node==1:
     listener.settimeout(max(.001,epoch+13800-time.time()));conn,peer=listener.accept();assert peer[0]==IPS[0];listener.close();listener=None
    else:
     # Retry only a definitely refused pre-protocol connect. No accepted
     # connection, sent frame, child work or collective is ever replayed.
     attempts=0
     while True:
      limit(epoch);attempts+=1;assert attempts<=1500;candidate=socket.socket();candidate.set_inheritable(False);candidate.bind((IPS[0],0));candidate.settimeout(min(15,max(.001,epoch+13800-time.time())))
      try:candidate.connect((IPS[1],PORT));conn=candidate;mark('connect-established',attempts=attempts);break
      except ConnectionRefusedError:candidate.close();mark('listener-not-yet-live',attempts=attempts);time.sleep(min(.1,max(0,epoch+13800-time.time())))
      except BaseException:candidate.close();raise
    from owner import sockets
    q['owned_socket_inodes']=sockets();save(OUT/('worker-%d.json'%node),q);mark('peer-connected')
   begin=time.monotonic();digest=exchange(conn,node,records,phase,ident,epoch);mark('global16-ready',round=phase,digest=digest);q['rounds'].append(dict(phase=phase,digest=digest,exchange_seconds=time.monotonic()-begin))
   for s in socks:send(s,dict(kind='GO',identity=ident,phase=phase,digest=digest),epoch)
   mark('local8-GO',round=phase)
  if node==0:conn.close();conn=None
  else:
   conn.settimeout(min(5,max(.001,epoch+13800-time.time())));assert conn.recv(1)==b'';conn.close();conn=None
  for p in children:assert p.wait(timeout=max(.001,epoch+14100-time.time()))==0
  q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  for s in socks:s.close()
  if conn:conn.close()
  if listener:listener.close()
  for p in children:
   if p.poll()is None:p.terminate();q['forced_TERM']=True
  for p in children:
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2);q['forced_KILL']=True
  mark('leader-finally',passed=q['passed']);trace.close();save(OUT/('worker-%d.json'%node),dict(q,completed=True,trace_path=str(local)))
  # Copy completed local evidence after the control protocol, never a GO gate.
  (OUT/('leader-%d-trace.jsonl'%node)).write_bytes(local.read_bytes())
  for dev,p in enumerate(children):
   src=Path('/local')/('r470-pipe-owner-%02d-%d.jsonl'%(node*8+dev,p.pid))
   if src.is_file():(OUT/('owner-%02d-trace.jsonl'%(node*8+dev))).write_bytes(src.read_bytes())
   for kind in ['native-phase','owner-state']:
    src=Path('/local')/('r470-%s-%02d-%d.%s'%(kind,node*8+dev,p.pid,'jsonl'if kind=='native-phase'else'json'))
    if src.is_file():(OUT/('%s-%02d.%s'%(kind,node*8+dev,'jsonl'if kind=='native-phase'else'json'))).write_bytes(src.read_bytes())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','child']);p.add_argument('--node',type=int);p.add_argument('--device',type=int);p.add_argument('--fd',type=int);p.add_argument('--publication');a=p.parse_args()
 if a.mode=='worker':worker()
 else:child(a.node,a.device,a.fd)
