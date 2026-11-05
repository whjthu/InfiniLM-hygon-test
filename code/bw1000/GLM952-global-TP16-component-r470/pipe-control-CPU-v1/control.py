"""Fixed two-round16owner control; qualified scalar framing, local socketpairs.
No SDK, Runtime, constructor, collective or tensor is imported/executed here.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,secrets,socket,subprocess,sys,time,traceback,re
from ready_socket_parent import send_json,receive_json,canonical
D=Path(__file__).resolve().parent;R=Path('/refactor')if str(D).startswith('/refactor/')else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');LABEL='GLM-r470-pipe-control-CPU-v1';OUT=R/'results'/LABEL;NODES=['a02r3n15','a02r3n16'];IPS=['10.1.2.55','10.1.2.56'];PORT=43844;H=lambda b:hashlib.sha256(b).hexdigest()
class Run:
 def stop_requested(self):return False
RUN=Run()
def save(p,q):
 t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(q,indent=2)+'\n');os.replace(str(t),str(p))
def limit(epoch):
 remaining=epoch+150-time.time()
 if remaining<=0:raise TimeoutError('Original150 epoch exhausted')
 return time.monotonic()+remaining
def send(s,v,e):send_json(s,v,RUN,lambda:False,limit(e))
def recv(s,e):return receive_json(s,RUN,lambda:False,limit(e))
def identity(epoch):return dict(protocol='r470-fixed16-pipe-scalar-v1',label=LABEL,epoch=epoch,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H((D/'SOURCE_MANIFEST.json').read_bytes()),CPU_only=True)
def validate(records,node,phase,ident):
 assert type(records)is list and len(records)==8
 for dev,r in enumerate(records):
  assert set(r)=={'identity','phase','rank','node','device','PID','startticks'}and r['identity']==ident and r['phase']==phase and type(r['rank'])is int and r['rank']==node*8+dev and r['node']==NODES[node]and r['device']==dev and type(r['PID'])is int and r['PID']>0 and type(r['startticks'])is str and r['startticks'].isdigit()
 return records

def child(node,device,fd):
 epoch=float(os.environ['GLM470_PIPE_EPOCH']);ident=identity(epoch);rank=node*8+device;path=Path('/local')/('r470-pipe-owner-%02d-%d.jsonl'%(rank,os.getpid()));log=path.open('xb',buffering=0)
 def mark(phase,**kw):log.write((json.dumps(dict(phase=phase,rank=rank,PID=os.getpid(),epoch=time.time(),monotonic=time.monotonic(),**kw))+'\n').encode())
 sock=socket.socket(fileno=fd);record=dict(identity=ident,phase='created',rank=rank,node=NODES[node],device=device,PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21]);q=dict(rank=rank,PID=os.getpid(),startticks=record['startticks'],identity=ident,CPU_only=True,SDK=False,GPU=False,passed=False,completed=False,trace_path=str(path))
 try:
  for phase in ['created','proved']:
   record['phase']=phase;mark('barrier-enter',round=phase,lastmissing='leader-global-GO');send(sock,record,epoch);go=recv(sock,epoch);assert set(go)=={'kind','identity','phase','digest'}and go['kind']=='GO'and go['identity']==ident and go['phase']==phase and re.fullmatch('[0-9a-f]{64}',go['digest']);mark('barrier-exit',round=phase,digest=go['digest'])
   if phase=='created':mark('CAPI-enter-recording-leaf',actual_CAPI=False)
   else:mark('normal-close-recording-leaf',actual_Runtime=False)
  q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:q['completed']=True;sock.close();log.close();save(OUT/('owner-%02d.json'%rank),q)

def exchange(conn,node,records,phase,ident,epoch):
 validate(records,node,phase,ident)
 if node==1:
  nonce=secrets.token_hex(32);send(conn,dict(kind='challenge',identity=ident,phase=phase,nonce=nonce),epoch);other=recv(conn,epoch);assert set(other)=={'kind','identity','phase','nonce','records'}and other['kind']=='records'and other['identity']==ident and other['phase']==phase and other['nonce']==nonce;allrecords=validate(other['records'],0,phase,ident)+records;digest=H(canonical(allrecords));send(conn,dict(kind='all-ready',identity=ident,phase=phase,nonce=nonce,records=allrecords,digest=digest),epoch);assert recv(conn,epoch)==dict(kind='ACK',identity=ident,phase=phase,nonce=nonce,digest=digest)
 else:
  challenge=recv(conn,epoch);assert set(challenge)=={'kind','identity','phase','nonce'}and challenge['kind']=='challenge'and challenge['identity']==ident and challenge['phase']==phase and re.fullmatch('[0-9a-f]{64}',challenge['nonce']);nonce=challenge['nonce'];send(conn,dict(kind='records',identity=ident,phase=phase,nonce=nonce,records=records),epoch);v=recv(conn,epoch);assert set(v)=={'kind','identity','phase','nonce','records','digest'}and v['kind']=='all-ready'and v['identity']==ident and v['phase']==phase and v['nonce']==nonce;assert v['records'][:8]==records;validate(v['records'][8:],1,phase,ident);allrecords=v['records'];digest=H(canonical(allrecords));assert v['digest']==digest;send(conn,dict(kind='ACK',identity=ident,phase=phase,nonce=nonce,digest=digest),epoch)
 return digest

def worker():
 node=int(os.environ['SLURM_PROCID']);assert node in(0,1)and socket.gethostname().split('.')[0]==NODES[node];epoch=float(os.environ['GLM470_PIPE_EPOCH']);ident=identity(epoch);q=dict(node=NODES[node],identity=ident,PID=os.getpid(),rounds=[],owners=[],passed=False,completed=False);children=[];socks=[];conn=None;listener=None
 local=Path('/local')/('r470-leader-%d-%d.jsonl'%(node,os.getpid()));trace=local.open('xb',buffering=0)
 def mark(p,**kw):trace.write((json.dumps(dict(phase=p,epoch=time.time(),monotonic=time.monotonic(),**kw))+'\n').encode())
 try:
  assert (D/'ready_socket_parent.py').is_file()
  if node==1:listener=socket.socket();listener.set_inheritable(False);listener.bind((IPS[1],PORT));listener.listen(1);mark('listener-live')
  for dev in range(8):
   parent,childsock=socket.socketpair();childsock.set_inheritable(True)
   with(OUT/('owner-%02d.log'%(node*8+dev))).open('xb')as f:p=subprocess.Popen([sys.executable,'-B',str(D/'control.py'),'child','--node',str(node),'--device',str(dev),'--fd',str(childsock.fileno())],pass_fds=(childsock.fileno(),),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   childsock.close();children.append(p);socks.append(parent);q['owners'].append(dict(rank=node*8+dev,PID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]));mark('child-spawn',rank=node*8+dev,PID=p.pid)
  for phase in ['created','proved']:
   records=[]
   for dev,s in enumerate(socks):
    mark('local-lastmissing',round=phase,rank=node*8+dev);r=recv(s,epoch);assert r['PID']==q['owners'][dev]['PID']and r['startticks']==q['owners'][dev]['startticks'];records.append(r)
   validate(records,node,phase,ident);mark('local8-ready',round=phase)
   if conn is None:
    if node==1:
     listener.settimeout(max(.001,epoch+150-time.time()));conn,peer=listener.accept();assert peer[0]==IPS[0];listener.close();listener=None
    else:
     # Retry only a definitely refused pre-protocol connect. No accepted
     # connection, sent frame, child work or collective is ever replayed.
     attempts=0
     while True:
      limit(epoch);attempts+=1;assert attempts<=1500;candidate=socket.socket();candidate.set_inheritable(False);candidate.bind((IPS[0],0));candidate.settimeout(min(15,max(.001,epoch+150-time.time())))
      try:candidate.connect((IPS[1],PORT));conn=candidate;mark('connect-established',attempts=attempts);break
      except ConnectionRefusedError:candidate.close();mark('listener-not-yet-live',attempts=attempts);time.sleep(min(.1,max(0,epoch+150-time.time())))
      except BaseException:candidate.close();raise
    mark('peer-connected')
   begin=time.monotonic();digest=exchange(conn,node,records,phase,ident,epoch);mark('global16-ready',round=phase,digest=digest);q['rounds'].append(dict(phase=phase,digest=digest,exchange_seconds=time.monotonic()-begin))
   for s in socks:send(s,dict(kind='GO',identity=ident,phase=phase,digest=digest),epoch)
   mark('local8-GO',round=phase)
  if node==0:conn.close();conn=None
  else:
   conn.settimeout(min(5,max(.001,epoch+150-time.time())));assert conn.recv(1)==b'';conn.close();conn=None
  for p in children:assert p.wait(timeout=max(.001,epoch+155-time.time()))==0
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

def launch(publication):
 assert re.fullmatch('[0-9a-f]{40}',publication);claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H((D/'SOURCE_MANIFEST.json').read_bytes())==claim['source_manifest_SHA256'];assert all(H((D/n).read_bytes())==h for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items());assert not any(p.exists()for p in [OUT,R/'results'/(LABEL+'-intent.json'),R/'results'/(LABEL+'-launch.json')]);z=subprocess.run(['sacct','-j','910371','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert z.returncode==0 and not any(LABEL in v for v in z.stdout.splitlines());epoch=time.time();args=['srun','--jobid=910371','--overlap','--exact','--immediate=10','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c1','--mem=1024M','--gres=none','-w',','.join(NODES),'-t','00:03:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'env','GLM470_PIPE_EPOCH='+str(epoch),'WAVE_INSIDE_R2323=GLM952-global-TP16-component-r470/pipe-control-CPU-v1/inside.sh','bash',str(D/'outer.sh'),'probe']
 with(R/'results'/(LABEL+'-intent.json')).open('x')as f:json.dump(dict(publication=publication,epoch=epoch,argv=args,claim_SHA256=H((D/'CPU_RESOURCE_CLAIM.json').read_bytes()),HOST=socket.gethostname(),GPU=False),f)
 OUT.mkdir(exist_ok=False)
 with(R/'logs'/(LABEL+'-srun.log')).open('xb')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],epoch=epoch,HOST=socket.gethostname(),GPU=False);save(R/'results'/(LABEL+'-launch.json'),q);print(json.dumps(q))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','worker','child']);p.add_argument('--node',type=int);p.add_argument('--device',type=int);p.add_argument('--fd',type=int);p.add_argument('--publication');a=p.parse_args()
 if a.mode=='launch':launch(a.publication)
 elif a.mode=='worker':worker()
 else:child(a.node,a.device,a.fd)
