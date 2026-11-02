"""ctypes consumer of the private complete-state ABI. No independent Native qualification."""
import ctypes,hashlib,json

ABI='deep-pd-r382-c1-tp8-raw-pages-v1'
def normalized(descriptor,wire):
 assert descriptor['abi']==ABI and descriptor['tp']==8 and descriptor['rows']==1 and 0<descriptor['past']<32768
 result=[];indices=[]
 assert len(descriptor['shards'])==8
 for rank,shard in enumerate(descriptor['shards']):
  assert shard['rank']==rank and len(shard['blocks'])==273
  for index,b in enumerate(shard['blocks']):
   assert b['block']==index and type(b['nbytes'])is int and b['nbytes']>=0
   layout={k:v for k,v in b.items()if k!='sha256'}
   result.append(dict(key=b['key'],shard=rank,nbytes=b['nbytes'],layout_SHA256=wire.sha(wire.canonical(layout))))
   indices.append((rank,index))
 assert wire.validate_blocks(result)==descriptor['payload_nbytes']
 return result,indices

class Codec:
 def __init__(self,lib,ctx,check,wire,model_SHA,ABI_SHA):
  self.lib,self.ctx,self.check,self.wire,self.model_SHA,self.ABI_SHA=lib,ctx,check,wire,model_SHA,ABI_SHA
  ptr=ctypes.c_void_p;u64=ctypes.c_uint64;u32=ctypes.c_uint32
  specs={'abi':([],ctypes.c_char_p),'generation':([ptr],u64),'begin_export':([ptr,u64,u64,ctypes.POINTER(u64)],ctypes.c_int),
   'descriptor':([ptr,u64],ctypes.c_char_p),'read':([ptr,u64,u32,u32,u64,ptr,u64],ctypes.c_int),
   'begin_import':([ptr,ctypes.c_char_p,u64,u64,ctypes.POINTER(u64)],ctypes.c_int),
   'write':([ptr,u64,u32,u32,u64,ptr,u64],ctypes.c_int),'validate':([ptr,u64],ctypes.c_int),'commit':([ptr,u64],ctypes.c_int),'abort':([ptr,u64],ctypes.c_int),'poison':([ptr,u64],ctypes.c_int)}
  self.fn={}
  for name,(args,result)in specs.items():
   f=getattr(lib,'deep_pd_state_'+name+'_r382');f.argtypes=args;f.restype=result;self.fn[name]=f
  assert self.fn['abi']()==ABI.encode()
 def identity(self,request,past,first):
  return dict(model_SHA256=self.model_SHA,ABI_SHA256=self.ABI_SHA,prompt_IDs_SHA256=request['prompt_IDs_SHA256'],request_nonce=request['request_nonce'],epoch=request['epoch'],past=past,TP=8,next_token=first,output_budget=request['output_budget'])
 def describe(self,sid):
  value=self.fn['descriptor'](self.ctx,sid)
  if value is None:self.check(1)
  assert len(value)<=1<<20;return self.wire.decode(value)
 def begin_export(self,request,past,first):
  sid=ctypes.c_uint64();self.check(self.fn['begin_export'](self.ctx,past,request['epoch'],ctypes.byref(sid)))
  try:return Export(self,sid.value,request,past,first)
  except BaseException:self.check(self.fn['abort'](self.ctx,sid.value));raise
 def prepare_import(self,request,handoff):return Sink(self,request,handoff)

class Export:
 def __init__(self,parent,sid,request,past,first):
  self.parent,self.sid,self.closed=parent,sid,False;descriptor=parent.describe(sid)
  assert descriptor['epoch']==request['epoch']and descriptor['past']==past
  self.blocks,self.indices=normalized(descriptor,parent.wire);self.identity=parent.identity(request,past,first)
  self.handoff=dict(past=past,first_token=first,descriptor=descriptor)
  # Fail in source/CPU geometry closure if the whole authenticated control
  # metadata cannot fit its existing bounded frame; never discover via reload.
  assert len(parent.wire.canonical(dict(kind='prefill_ready',request_nonce=request['request_nonce'],epoch=request['epoch'],handoff=self.handoff,report={})))<parent.wire.MAX_HEADER-16384
 def read_block(self,index,offset,count):
  assert not self.closed;rank,block=self.indices[index];buf=ctypes.create_string_buffer(count)
  self.parent.check(self.parent.fn['read'](self.parent.ctx,self.sid,rank,block,offset,buf,count));return buf.raw
 def close(self):
  if not self.closed:self.parent.check(self.parent.fn['abort'](self.parent.ctx,self.sid));self.closed=True
 def abort(self):self.close()

class Sink:
 def __init__(self,parent,request,handoff):
  self.parent,self.closed,self.group_published,self.poisoned=parent,False,False,False
  self.identity=parent.identity(request,handoff['past'],handoff['first_token']);sid=ctypes.c_uint64()
  raw=parent.wire.canonical(handoff['descriptor']);assert len(raw)<=1<<20
  generation=parent.fn['generation'](parent.ctx);assert generation>0,'Invalid/failed target Native generation';self.target_generation=generation
  parent.check(parent.fn['begin_import'](parent.ctx,raw,request['epoch'],generation,ctypes.byref(sid)));self.sid=sid.value
  try:
   descriptor=parent.describe(self.sid);assert parent.wire.canonical(descriptor)==raw
   self.blocks,self.indices=normalized(descriptor,parent.wire);self.target_generation=generation
  except BaseException:self.abort();raise
 def prepare(self,identity,blocks):
  assert not self.closed and self.parent.wire.canonical(identity)==self.parent.wire.canonical(self.identity)and self.parent.wire.canonical(blocks)==self.parent.wire.canonical(self.blocks)
 def write(self,index,offset,part):
  assert not self.closed;rank,block=self.indices[index];buffer=ctypes.create_string_buffer(part)
  self.parent.check(self.parent.fn['write'](self.parent.ctx,self.sid,rank,block,offset,buffer,len(part)))
 def verify(self):
  assert not self.closed;self.parent.check(self.parent.fn['validate'](self.parent.ctx,self.sid));self.validated=True
 def publish(self):
  assert self.validated and not self.closed
  self.parent.check(self.parent.fn['commit'](self.parent.ctx,self.sid));self.closed=True;self.group_published=True
 def abort(self):
  if not self.closed:
   # A commit/owner failure can poison Native and make abort return an error;
   # retain that first failure and let the original bootstrap destroy ctx.
   self.parent.fn['abort'](self.parent.ctx,self.sid);self.closed=True
  self.poison_code=self.parent.fn['poison'](self.parent.ctx,self.target_generation)
  self.poisoned=self.poison_code==0;self.group_published=False
 def evidence(self):return dict(ABI=ABI,shards=8,blocks=len(self.blocks),payload_bytes=sum(b['nbytes']for b in self.blocks),all8_validate_then_commit=self.group_published,poisoned=self.poisoned,poison_returncode=getattr(self,'poison_code',None),target_generation_before_import=self.target_generation)
