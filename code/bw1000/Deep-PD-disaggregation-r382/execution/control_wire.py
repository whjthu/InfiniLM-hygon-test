"""Early held sockets and one authenticated benchmark control connection per role."""
from pathlib import Path
import socket,time,json,hashlib,hmac,os
from contract import ROLE_IPS,CONTROL_PORTS,STATE_PORT,check_live,run_identity

def atom(path,value):
 p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

class EarlyRoleSockets:
 def __init__(self,role,run,nonce,source_SHA,root,epoch,wire):
  self.role,self.run,self.nonce,self.source_SHA=role,run,nonce,source_SHA
  self.root,self.epoch,self.wire=Path(root),epoch,wire;self.listener=self.state_listener=self.control=None
  try:
   self.listener=self.bind(ROLE_IPS[role],CONTROL_PORTS[role])
   if role=='D':self.state_listener=self.bind(ROLE_IPS[role],STATE_PORT)
   marker=self.root/'control'/('Deep382-'+run+'-'+role+'-listening.json');assert not marker.exists()
   atom(marker,dict(**run_identity(run,nonce),role=role,source_manifest_SHA256=source_SHA,control_IP=ROLE_IPS[role],control_port=CONTROL_PORTS[role],state_listener_held=role=='D',before_Torch_Native_weights=True,actual_step=os.environ.get('SLURM_JOB_ID','CPU')+'.'+os.environ.get('SLURM_STEP_ID','RECORDING'),PID=os.getpid()))
  except BaseException:self.close();raise
 @staticmethod
 def bind(address,port):
  sock=socket.socket()
  try:sock.bind((address,port));sock.listen(1);return sock
  except BaseException:sock.close();raise
 def authenticate_once(self,cancel_file):
  check_live(self.epoch,cancel_file);self.listener.settimeout(min(180,max(1,self.epoch+3540-time.time())))
  self.control,self.peer=self.listener.accept();self.listener.close();self.listener=None
  deadline=time.monotonic()+min(180,max(1,self.epoch+3540-time.time()))
  self.wire.send_json(self.control,dict(kind='hello',**run_identity(self.run,self.nonce),role=self.role,source_manifest_SHA256=self.source_SHA),deadline)
  auth=self.wire.recv_json(self.control,deadline)
  assert set(auth)=={'kind','run','nonce','source_manifest_SHA256'}and auth['kind']=='authenticate'and auth['run']==self.run and auth['source_manifest_SHA256']==self.source_SHA and hmac.compare_digest(auth['nonce'],self.nonce)
  self.send(dict(kind='authenticated',role=self.role,before_weights=True));return self.control
 def deadline(self):return time.monotonic()+max(.001,self.epoch+3600-time.time())
 def send(self,value):self.wire.send_json(self.control,value,self.deadline())
 def receive(self):return self.wire.recv_json(self.control,self.deadline())
 def close(self):
  for name in ['control','listener','state_listener']:
   obj=getattr(self,name,None)
   if obj is not None:obj.close();setattr(self,name,None)

def connect_role_once(role,run,nonce,source_SHA,root,epoch,cancel_file,wire):
 marker=Path(root)/'control'/('Deep382-'+run+'-'+role+'-listening.json');limit=min(epoch+180,epoch+3540)
 while True:
  check_live(epoch,cancel_file)
  assert time.time()<limit,'Bounded early-listener visibility expired'
  try:q=json.loads(marker.read_bytes())
  except FileNotFoundError:time.sleep(.1);continue
  assert q['run']==run and q['role']==role and q['source_manifest_SHA256']==source_SHA and q['session_nonce_SHA256']==run_identity(run,nonce)['session_nonce_SHA256']
  break
 sock=socket.create_connection((ROLE_IPS[role],CONTROL_PORTS[role]),timeout=15)
 try:
  deadline=time.monotonic()+max(.001,epoch+3600-time.time());hello=wire.recv_json(sock,deadline)
  assert hello==dict(kind='hello',**run_identity(run,nonce),role=role,source_manifest_SHA256=source_SHA)
  wire.send_json(sock,dict(kind='authenticate',run=run,nonce=nonce,source_manifest_SHA256=source_SHA),deadline)
  assert wire.recv_json(sock,deadline)==dict(kind='authenticated',role=role,before_weights=True)
  return sock
 except BaseException:sock.close();raise
