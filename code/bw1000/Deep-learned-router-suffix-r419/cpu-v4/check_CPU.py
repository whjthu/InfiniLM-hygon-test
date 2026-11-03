"""Actual owner runner/configure/proof/stats with fake device leaves; no Native/weights/GPU."""
from pathlib import Path
import json,ctypes as C,threading,concurrent.futures,ast
import numpy as np
import interface as api,driver_core

def main():
 fixture=json.loads((Path(__file__).parent/'FIXTURE_METADATA.json').read_bytes());expected={'layers.16.ffn.gate.weight','layers.16.ffn.gate.bias'}
 for name in [f'experts.{i}'for i in range(256)]+['shared_experts']:
  for w in ['w1','w2','w3']:
   for field in ['weight','scale']:expected.add(f'layers.16.ffn.{name}.{w}.{field}')
 assert expected==set(fixture['keys'])and len(expected)==1544 and fixture['payload_read']is False
 keys=sorted(expected);completed=[]
 class Fn:
  def __init__(self,fn):self.fn=fn
  def __call__(self,*args):return self.fn(*args)
 def owner(rank,fail=None):
  trace=[];attempts=0;done=0;tid=threading.get_ident()
  class Lib:
   def __init__(self):
    self.mode=0;self.loaded=set();self.projected=False;self.fields={};self.forward_index=-1
    for name in ['error','id_size','id','create','keys','load','finish','forward','project','same_logits','proof_info','proof_read','stats','dtype','destroy','shutdown']:setattr(self,'deep_router_'+name+'_r419',Fn(getattr(self,name)))
   def error(self):return (b'destroy injected'if trace and trace[-1]=='destroy'else b'load injected')if fail=='load_and_destroy'else b'injected primary error'
   def id_size(self):return 128
   def id(self,p):C.memset(p,rank,128);return 0
   def create(self,device,cfg,uid):trace.append('create');assert device==rank;return None if fail=='create'else tid
   def keys(self,ctx):assert ctx==tid;return json.dumps(keys).encode()
   def load(self,ctx,name,p,ndim,shape,dtype):assert ctx==tid and name.decode()in expected;self.loaded.add(name.decode());return 2 if fail in ['load','load_and_destroy']else 0
   def finish(self,ctx):trace.append('finish');assert len(self.loaded)==1544;return 2 if fail=='finish'else 0
   def project(self,ctx,x,producer):trace.append('project');self.projected=True;return 2 if fail=='project'else 0
   def populate(self):
    self.fields={0:(np.full((1,4096),0x3f80,np.uint16),101),1:(np.full((1,6),.25,np.float32),100),2:(np.arange(6,dtype=np.int64)[None,:],102),3:(np.ones((1,256),np.float32),100),4:(np.ones((1,256),np.float32),100),5:(np.ones((1,256),np.float32),100)}
   def same_logits(self,ctx,mode):trace.append(('diag',mode));assert self.projected;self.mode=mode;self.populate();return 2 if fail=='diag_ON'and mode==1 else 0
   def forward(self,ctx,tokens,x,producer,mode):self.forward_index+=1;trace.append(('forward',self.forward_index));assert tokens==1;self.mode=mode;self.populate();return 2 if fail=='forward2'and self.forward_index==2 else 0
   def proof_info(self,ctx,k,shape,size,dt):
    a,d=self.fields[k];shape[0]=a.ndim
    for i,n in enumerate(a.shape):shape[i+1]=n
    C.cast(size,C.POINTER(api.Z))[0]=a.nbytes;C.cast(dt,C.POINTER(api.I))[0]=d;return 0
   def proof_read(self,ctx,k,out,size):a,d=self.fields[k];assert size==a.nbytes;C.memmove(out,int(a.ctypes.data),size);return 0
   def stats(self,ctx,t,a):t[0]=.001;t[1]=.0011;a[0]=1;a[1]=self.mode;a[2]=1-self.mode;a[3]=tid;return 0
   def dtype(self,i):return [100,101,102][i]
   def destroy(self,ctx):trace.append('destroy');assert ctx==tid;return 2 if fail in ['destroy','load_and_destroy']else 0
   def shutdown(self):trace.append('shutdown');return 2 if fail=='shutdown'else 0
  lib=Lib();f=api.configure(lib)
  def check(rc):
   if rc:raise RuntimeError(f('error')().decode())
  def create(r):
   uid=C.create_string_buffer(f('id_size')());check(f('id')(uid));ctx=f('create')(r,b'{}',uid)
   if not ctx:raise RuntimeError('injected primary error')
   return ctx
  def initialize(ctx):
   names=json.loads(f('keys')(ctx));assert names==keys
   for name in names:
    spec=fixture['parameters'][name];shape=(api.Z*len(spec['shape']))(*spec['shape']);dtype={'I8':0,'F32':1,'BF16':2}[spec['dtype']];check(f('load')(ctx,name.encode(),api.P(1),len(shape),shape,dtype))
   check(f('finish')(ctx))
  def read(ctx):
   data={k:api.proof(f,check,ctx,k)for k in [0,1,2]};assert data[0]['shape']==(1,4096)and data[0]['dtype']==101
   ids=data[2]['raw'].view(np.int64);weights=data[1]['raw'].view(np.float32);assert len(set(ids.tolist()))==6 and np.isfinite(weights).all()and np.isclose(weights.sum(),1.5)
   return data
  def qualify(ctx):
   check(f('project')(ctx,api.P(1),api.P(2)));pack=[]
   for mode in [0,1]:
    check(f('same_logits')(ctx,mode));pack.append({k:api.proof(f,check,ctx,k)for k in [1,2,3,4,5]});s=api.stats(f,check,ctx);assert s['calls']==1 and s['hits']==mode and s['fallbacks']==1-mode and s['owner_thread_hash']==tid
   assert np.array_equal(pack[0][3]['raw'],pack[1][3]['raw'])and np.array_equal(pack[0][1]['raw'],pack[1][1]['raw'])
  def trial(ctx,item):
   nonlocal attempts,done
   attempts+=1;check(f('forward')(ctx,1,api.P(1),api.P(2),item['mode']));s=api.stats(f,check,ctx);assert s['calls']==1 and s['hits']==item['mode'] and s['fallbacks']==1-item['mode'];read(ctx);done+=1;return dict(index=item['index'],mode=item['mode'],stats=s)
  def destroy(ctx):check(f('destroy')(ctx))
  def shutdown():check(f('shutdown')())
  try:rows=driver_core.run_owner(rank,create,initialize,qualify,trial,destroy,shutdown,lambda:trace.append('abort'))
  except RuntimeError as exc:
   assert fail and 'shutdown'in trace and all(not isinstance(x,tuple)for x in trace[trace.index('shutdown')+1:])
   if fail=='forward2':assert attempts==3 and done==2
   if fail in ['project','diag_ON']:assert attempts==done==0
   if fail!='create':assert 'destroy'in trace
   if fail=='load_and_destroy':assert str(exc)=='load injected'and 'destroy injected'in str(getattr(exc,'cleanup_errors',[]))
   return
  assert fail is None and len(rows)==attempts==done==10 and trace[-2:]==['destroy','shutdown'];completed.append(rank)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
 assert sorted(completed)==list(range(8))
 for fail in ['create','load','finish','project','diag_ON','forward2','destroy','shutdown','load_and_destroy']:owner(0,fail)
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 return dict(passed=True,CPU_only=True,Native_dlopen=False,GPU=False,actual_shared_owner_runner_and_CABI_configure=True,owner_records=8,planned_timed_query_records=80,same_logit_qualification_before_formal=True,all1544_metadata_keys_match_checkpoint_header=True,payload_opened=False,nine_error_paths_close=True,primary_error_retained_with_cleanup_errors=True,nonzero_forward_not_counted_completed=True,SDK_device_leaves_fake=True,scope='Actual glue/proof decoder/metadata plan with fake GPU leaves, not device, mathematics or fullMoE qualification')

if __name__=='__main__':print(json.dumps(main()))
