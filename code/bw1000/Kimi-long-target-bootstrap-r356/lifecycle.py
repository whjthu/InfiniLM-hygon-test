"""Original absolute budget and the single delayed draft connection."""
from pathlib import Path
import hashlib,json,math,os,re,socket,time

NATIVE_SHA='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
WORKER_SHA='49fa1d269f18f78820c7589496011300a7648fe340933e3c7a0e9105958b93c2'

class Settings:
 def __init__(self,env,root=Path('/refactor')):
  if env.get('KIMI_LONG_PAIR_R346')!='1' or env.get('INFINILM_KIMI_LONG_FEATURES_R305')!='1':raise RuntimeError('Process creation long pair opt-in required')
  self.root=Path(root);self.run=env['KIMI_LONG_RUN_R347'];self.nonce=env['KIMI_LONG_SESSION_R347']
  if not re.fullmatch('[a-z0-9][a-z0-9-]{7,39}',self.run)or len(self.nonce)<32:raise ValueError('Bound session/run required')
  if env['GLOBAL_RUN_R2354']!=self.run:raise ValueError('Run identities differ')
  self.start=float(env['KIMI_LONG_GPU_START_UNIX_R347']);self.deadline=float(env['KIMI_LONG_NATIVE_DEADLINE_UNIX_R347'])
  if not math.isfinite(self.start)or self.start<=0 or self.deadline!=self.start+28800:raise ValueError('Original Native deadline must equal GPU start plus28800')
  self.cancel=Path(env['KIMI_LONG_CANCEL_FILE_R347']);self.node=int(env['SLURM_PROCID']);self.head=env['KIMI_ENGINE_HEAD']
  if self.cancel.parent!=self.root/'control' or self.run not in self.cancel.name or self.node not in range(4):raise ValueError('Explicit cancellation/node binding')
  self.port=int(env['GLOBAL_PORT_R2354']);self.capture_port=int(env['KIMI_LONG_CAPTURE_PORT_R347']);self.draft_port=int(env['KIMI_LONG_DRAFT_PORT_R347']);self.draft_host=env['KIMI_LONG_DRAFT_HOST_R347']
  if [self.port,self.capture_port,self.draft_port]!=[32947,45947,46947]:raise ValueError('Frozen ports required')
  self.identity=dict(schema=1,run=self.run,nonce_SHA256=hashlib.sha256(self.nonce.encode()).hexdigest(),original_GPU_start_unix=self.start,target_Native_SHA256=NATIVE_SHA,worker_manifest_SHA256=WORKER_SHA,requested_by_node=0)
  self.start_file=self.root/'control'/('Kimi-long-draft-start-r347-'+self.run+'.json')
  self.listening_file=self.root/'control'/('Kimi-long-draft-listening-r347-'+self.run+'.json')

class Gate:
 def __init__(self,settings,clock=time.monotonic,wall=time.time):
  self.settings=settings;self.clock=clock;self.deadline=clock()+max(0.,settings.deadline-wall());self.counts={};self.signal_cancel=False
 def cancelled(self):return self.signal_cancel or self.settings.cancel.exists()
 def check(self,label):
  self.counts[label]=self.counts.get(label,0)+1
  if self.clock()>=self.deadline:raise TimeoutError('Original Native deadline before '+label)
  if self.cancelled():raise RuntimeError('Original controller cancellation before '+label)
 def remaining(self):return max(0.,self.deadline-self.clock())

class GuardedFunction:
 def __init__(self,raw,gate,name):object.__setattr__(self,'raw',raw);object.__setattr__(self,'gate',gate);object.__setattr__(self,'name',name)
 def __getattr__(self,name):return getattr(self.raw,name)
 def __setattr__(self,name,value):setattr(self.raw,name,value)
 def __call__(self,*args,**kwargs):
  if self.name not in ['modern_kimi_engine_destroy_r2475','modern_kimi_engine_error_r2475']:self.gate.check('Native:'+self.name)
  return self.raw(*args,**kwargs)

class GuardedLibrary:
 def __init__(self,raw,gate):self.raw=raw;self.gate=gate;self.functions={}
 def __getattr__(self,name):
  if name not in self.functions:self.functions[name]=GuardedFunction(getattr(self.raw,name),self.gate,name)
  return self.functions[name]

def guarded_safe_open(function,gate):
 def call(*args,**kwargs):gate.check('safe_open');return function(*args,**kwargs)
 return call

def publish_once(path,value):
 with Path(path).open('x',encoding='utf8',newline='\n')as f:
  f.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def connect_draft(settings,gate,control,draft_client,save,*,clock=time.monotonic,sleep=time.sleep,connect=socket.create_connection):
 """Called by immutable337 after both AR arms; a single authenticated socket."""
 connection=client=None;status=dict(node=settings.node,ok=False)
 try:
  gate.check('delayed-draft-start')
  if settings.node==0:
   publish_once(settings.start_file,settings.identity);save({'draft_start_marker_written':True})
   while True:
    gate.check('wait-draft-listening')
    if settings.listening_file.exists():
     raw=settings.listening_file.read_bytes()
     if raw.endswith(b'\n'):
      value=json.loads(raw);extra={'actual_draft_step','listener_ready'}
      if set(value)!=set(settings.identity)|extra or any(value[k]!=v or type(value[k])is not type(v)for k,v in settings.identity.items()):raise ValueError('Draft listener identity mismatch')
      if value['listener_ready']is not True or not isinstance(value['actual_draft_step'],str)or not re.fullmatch(r'910371[.]\d+',value['actual_draft_step']):raise ValueError('Actual sole draft listener required')
      break
    sleep(min(.1,gate.remaining()))
   gate.check('draft-connect-once');timeout=min(7200.,gate.remaining())
   if timeout<=0:raise TimeoutError('No budget for draft connection')
   connection=connect((settings.draft_host,settings.draft_port),timeout=timeout);connection.settimeout(timeout)
   gate.check('draft-hello');client=draft_client(connection,settings.nonce);save({'draft_listener_step':value['actual_draft_step'],'draft_connected_once':True})
  status['ok']=True
 except BaseException as e:
  status['error_type']=type(e).__name__
  if client is not None:client.abort()
  elif connection is not None:connection.close()
  # Every leader participates in this bounded admission before publishing.
 rows=control.collect('r346-delayed-draft-connected',status)
 if len(rows)!=4 or {v['node']for v in rows}!=set(range(4))or not all(v.get('ok')is True for v in rows):
  if client is not None:client.abort()
  raise RuntimeError('Delayed sole draft connection failed; no retry')
 return client
