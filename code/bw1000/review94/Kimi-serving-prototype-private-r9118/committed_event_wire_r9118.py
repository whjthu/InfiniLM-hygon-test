"""Fixed JSON token-event frames for NEXT private native protocol only; default OFF.
No Python source, tensor payload, file paths, eval, or per-token filesystem writes.
Caller invokes in actual accepted-token callback before features/commit/replay; all
send/backpressure time stays inside measured native decode wall.
"""
import socket,struct,json,time,re
MAX_HEADER=65536
class CommittedEventSender:
 def __init__(self,host=None,port=None,nonce=None,timeout=2.):
  self.enabled=bool(host and port and nonce);self.sock=None;self.seq=0;self.failed=False;self.errors=[]
  if self.enabled:
   self.sock=socket.create_connection((host,int(port)),timeout=timeout);self.sock.settimeout(timeout);self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);self._send({'type':'hello','nonce':nonce,'schema':1})
 def _send(self,obj):
  raw=json.dumps(obj,allow_nan=False,separators=(',',':')).encode();assert 0<len(raw)<=MAX_HEADER;self.sock.sendall(struct.pack('!II',len(raw),0)+raw)
 def event(self,request_id,token_ids,phase,terminal=False):
  if not self.enabled or self.failed:return not self.failed
  if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request_id)or phase not in ('prefill','AR','verified'):raise ValueError('invalid committed event identity/phase')
  ids=list(token_ids)
  if not 1<=len(ids)<=16 or any(isinstance(t,bool)or not isinstance(t,int)or not 0<=t<163840 for t in ids):raise ValueError('invalid committed token frame')
  return self.send_event({'type':'tokens','request_id':request_id,'token_ids':ids,'phase':phase,'terminal':bool(terminal),'emitter_perf_counter':time.perf_counter(),'emitter_wall_utc':time.time()})
 def finished(self,request_id,reason):
  if reason not in ('stop','length','error','cancelled'):raise ValueError('invalid finish reason')
  return self.send_event({'type':'finished','request_id':request_id,'reason':reason,'transaction_complete':True})
 def send_event(self,obj):
  if not self.enabled or self.failed:return not self.failed
  try:self._send(dict(obj,seq=self.seq));self.seq+=1;return True
  except OSError as e:self.failed=True;self.errors.append(repr(e));self.close();return False
 def close(self):
  if self.sock is not None:self.sock.close();self.sock=None

def recv_frame(sock):
 def exact(n):
  b=bytearray()
  while len(b)<n:
   part=sock.recv(n-len(b))
   if not part:raise EOFError('event socket closed')
   b.extend(part)
  return bytes(b)
 n,body=struct.unpack('!II',exact(8))
 if not 0<n<=MAX_HEADER or body!=0:raise ValueError('bounded JSON metadata frames only')
 obj=json.loads(exact(n))
 if not isinstance(obj,dict):raise ValueError('object event required')
 return obj
