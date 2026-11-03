"""Recording CPU leaves only. These are not Runtime, checkpoint or Native qualification."""
import ctypes,json,hashlib,threading,time
from types import SimpleNamespace as NS
from native_codec import Codec,ABI

class Function:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*args):return self.fn(*args)

class FakeCAPI:
 def __init__(self,leaf,wire):
  self.leaf,self.wire=leaf,wire;self.generation=1;self.session=None;self.sid=0;self.last_epoch=0;self.poisoned=False;self.calls=[]
  for name in ['abi','generation','begin_export','descriptor','read','begin_import','write','validate','commit','abort','poison']:
   setattr(self,'deep_pd_state_'+name+'_r382',Function(lambda *a,n=name:self.call(n,*a)))
 def check(self):
  assert threading.get_ident()==self.leaf.owner
  if self.poisoned:raise RuntimeError('CPUrecording Native sticky poison')
 def call(self,name,*args):
  self.calls.append(name)
  if name=='abi':return ABI.encode()
  if name=='generation':return 0 if self.poisoned else self.generation
  if name=='poison':
   if args[1]!=self.generation:return 1
   self.poisoned=True;self.session=None;return 0
  try:
   self.check()
   if name=='begin_export':
    _,past,epoch,sid=args;assert past==self.leaf.past==self.leaf.current['past'] and epoch>self.last_epoch and self.session is None
    self.sid+=1;self.last_epoch=epoch;values={};shards=[]
    for rank in range(8):
     blocks=[]
     for i in range(273):
      data=bytes([(rank+i)%256])* (8 if i%7==0 else 0);values[(rank,i)]=data
      blocks.append(dict(block=i,key='recording_block_%03d'%i,kind='CPU_RECORDING_NOT_NATIVE_STATE',dtype='uint8',shape=[len(data)],storage_shape=[len(data)],livepages=[0]if data else[],nbytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
     shards.append(dict(rank=rank,blocks=blocks))
    desc=dict(abi=ABI,past=past,rows=1,tp=8,epoch=epoch,source_generation=self.generation,shards=shards,payload_nbytes=sum(map(len,values.values())))
    self.session=dict(importing=False,descriptor=desc,values=values);ctypes.cast(sid,ctypes.POINTER(ctypes.c_uint64))[0]=self.sid;return 0
   if name=='begin_import':
    _,raw,epoch,generation,sid=args;assert generation==self.generation and self.leaf.past==0 and epoch>self.last_epoch and self.session is None
    desc=self.wire.decode(raw);assert desc['epoch']==epoch;self.sid+=1;self.last_epoch=epoch
    values={(r['rank'],b['block']):bytearray()for r in desc['shards']for b in r['blocks']}
    self.session=dict(importing=True,descriptor=desc,values=values);self.leaf.shared['events'].append(dict(role=self.leaf.role,event='begin_import_current_generation',epoch=epoch,generation=generation,tick=time.monotonic()));ctypes.cast(sid,ctypes.POINTER(ctypes.c_uint64))[0]=self.sid;return 0
   assert self.session and args[1]==self.sid
   s=self.session
   if name=='descriptor':return self.wire.canonical(s['descriptor'])
   if name in ['read','write']:
    _,sid,rank,block,offset,buffer,count=args;key=(rank,block)
    if name=='read':ctypes.memmove(buffer,s['values'][key][offset:offset+count],count)
    else:assert offset==len(s['values'][key]);s['values'][key].extend(ctypes.string_at(buffer,count))
    return 0
   if name=='validate':
    assert s['importing']
    assert all(hashlib.sha256(s['values'][(rank['rank'],b['block'])]).hexdigest()==b['sha256']for rank in s['descriptor']['shards']for b in rank['blocks']);s['validated']=True;return 0
   if name=='commit':
    assert s['importing']and s['validated'];self.leaf.past=s['descriptor']['past'];self.session=None;return 0
   if name=='abort':
    if s['importing']:self.poisoned=True
    else:self.leaf.shared['events'].append(dict(role=self.leaf.role,event='export_snapshot_release',epoch=self.leaf.current['epoch'],tick=time.monotonic()))
    self.session=None;return 0
   raise AssertionError(name)
  except Exception as e:self.leaf.last_error=repr(e);return 1

class Leaf:
 actual_Native=False
 def __init__(self,wire,role,shared):
  self.owner=threading.get_ident();self.past=0;self.last_error=None;self.prompt_ids=[7]*8489;self.trace=[];self.resets=0
  self.role,self.shared,self.current=role,shared,None
  self.reasons=NS(EOS_TOKEN='EOS_TOKEN',LENGTH='LENGTH',CANCELED='CANCELED')
  self.owner_observation=dict(owners=[dict(device=i,tid=-i-1,CPU_RECORDING=True)for i in range(8)])
  self.receipt=dict(actual_Native=False,actual_model=False,actual_prepared_encoder=False,prompt_and_load_leaf='CPUrecording only, no officialID/Model claim',recorded_owners=8)
  self.lib=FakeCAPI(self,wire);self.codec=Codec(self.lib,1,self.check,wire,'a'*64,'b'*64)
 def check(self,code):
  if code:raise RuntimeError('CPUrecording CAPI rc: '+str(self.last_error))
 def reset(self):
  self.lib.check();assert self.lib.session is None;self.past=0;self.lib.generation+=1;self.resets+=1
  if self.current:self.shared['events'].append(dict(role=self.role,event='reset',epoch=self.current['epoch'],tick=time.monotonic()))
 def stage_started(self,operation,meta):
  self.current=meta;self.shared['events'].append(dict(role=self.role,event='stage_enter',operation=operation,epoch=meta['epoch'],group=meta['group'],tick=time.monotonic()))
  if self.role=='P'and operation=='prefill'and meta['phase']=='capacity'and meta['mode']=='PD'and meta.get('ordinal_in_group')==1:
   self.shared['gates'][meta['group']].set()
 def finite(self,logits):assert logits==['CPU_RECORDING_FINITE']
 def forward(self,ids,positions,rows,past,total,offsets):
  self.lib.check();assert rows==[0]and past==[self.past]and positions==list(range(self.past,self.past+len(ids)))and total==[self.past+len(ids)]and offsets==[0,len(ids)]
  self.trace.append(dict(ids=list(ids),positions=list(positions),past=list(past),total=list(total),offsets=list(offsets)))
  if self.role=='D'and past[0]==self.current['past']+1 and self.current['mode']=='PD'and self.current['phase']=='capacity'and self.current['ordinal_in_group']==0:
   self.shared['events'].append(dict(role='D',event='D1_lasttoken_wait_P2_prefill',group=self.current['group'],tick=time.monotonic()))
   assert self.shared['gates'][self.current['group']].wait(3),'Pipeline serialized: D1 awaited P2 but P2 never began'
   self.shared['pipeline_proofs'].append(dict(group=self.current['group'],P2_entered_before_D1_lasttoken=True))
  token=100 if total[0]<=self.current['past'] else 101 if past[0]==self.current['past'] else 1
  if self.shared.get('quality_fail')=='warm_cap'and self.current['phase']=='warm'and self.current['mode']=='PD'and self.role=='D':token=101
  self.past=total[0]
  return 0,['CPU_RECORDING_FINITE'],[token]
 def prompt_for(self,meta):
  if self.shared.get('fixture_inputs'):
   value=self.shared['fixture_inputs'][meta['fixture']];ids=value['ids']
   assert len(ids)==meta['past']and value['prompt_IDs_SHA256']==meta['prompt_IDs_SHA256']
   self.shared.setdefault('actual_fixture_consumed',[]).append(dict(role=self.role,epoch=meta['epoch'],fixture=meta['fixture'],prompt_tokens=len(ids),prompt_IDs_SHA256=meta['prompt_IDs_SHA256']))
   return ids
  return [7]*meta['past']
 def quality(self,ids,meta):
  wrong=self.shared.get('quality_fail')=='warm_wrong'and self.current['phase']=='warm'and self.current['mode']=='PD'and self.role=='D'
  return dict(task_correct=not wrong,CPU_recording_quality_leaf=True,not_actual_answer=True)
 def begin_export(self,*args):return self.codec.begin_export(*args)
 def prepare_import(self,*args):return self.codec.prepare_import(*args)
 def close_python_provider(self):self.provider_closed=True

def loader_factory(holders,role,shared):
 def loader(**kw):
  from support import load
  wire=load(__import__('pathlib').Path(__file__).resolve().parent)['state_wire'];leaf=Leaf(wire,role,shared);holders.append(leaf)
  # Only provider/constructor/parameter payload leaves are recorded. Actual
  # bootstrap, role loop, splitstages, codec ctypes, wire and client execute.
  provider=dict(CPU_recording_prepared_encode=True);ns=dict(recording_leaf=leaf,report={})
  try:
   kw['serve'](ns,provider);return ns,dict(CPU_recording_provider_load=True,actual_parameter_payload_open=False),dict(CPU_recording=True)
  finally:kw['on_namespace_exit'](dict(report=dict(engine_worker_shutdown_completed=True,CPU_recording_destroy=True)))
 return loader
