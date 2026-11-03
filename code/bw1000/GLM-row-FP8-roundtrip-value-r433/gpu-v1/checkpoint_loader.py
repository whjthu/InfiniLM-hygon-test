"""Only layer38 AttentionPrepare+Output keys. Shared actual/recording loop."""
import json,hashlib
KEYS=sorted([n+'.'+s for n in ['q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj','o_proj']for s in ['weight','weight_scale_inv']]+['q_a_layernorm.weight','kv_a_layernorm.weight'])
PREFIX='model.layers.38.self_attn.'
def load(lib,ctx,keys,reader,check,expired):
 assert len(keys)==len(set(keys))and sorted(keys)==KEYS
 count=0;payload=0
 try:
  for key in keys:
   expired();held,descriptor,nbytes=reader.tensor(PREFIX+key)
   d=dict(name=key,shape=descriptor['shape'],dtype=descriptor['dtype'],pointer=descriptor['pointer'])
   assert d['dtype']in [0,1,2]and d['pointer']and all(n>0 for n in d['shape'])
   check(lib.glm_attention_load_r398(ctx,json.dumps([d]).encode()));count+=1;payload+=nbytes
   del held
  check(lib.glm_attention_finish_r398(ctx))
 finally:reader.close()
 return dict(keys=count,payload_referenced_bytes=payload,checkpoint_load_passes=1,keys_SHA256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest())

class Checkpoint:
 def __init__(self,model,torch):
  from pathlib import Path
  from safetensors import safe_open
  self.model,self.t,self.safe=Path(model),torch,safe_open
  self.index=json.loads((self.model/'model.safetensors.index.json').read_bytes())['weight_map'];self.handles={}
  assert all(PREFIX+k in self.index for k in KEYS)
 def tensor(self,key):
  shard=self.index[key]
  if shard not in self.handles:self.handles[shard]=self.safe(str(self.model/shard),framework='pt',device='cpu');self.handles[shard].__enter__()
  x=self.handles[shard].get_tensor(key).contiguous();dtype={self.t.float32:0,self.t.bfloat16:1,self.t.float8_e4m3fn:2,self.t.float8_e5m2:2,self.t.uint8:2}[x.dtype]
  return x,dict(shape=list(x.shape),dtype=dtype,pointer=x.data_ptr()),x.nbytes
 def close(self):
  for h in self.handles.values():h.__exit__(None,None,None)
  self.handles.clear()

def check_recording():
 class Reader:
  def __init__(self):self.read=[];self.closed=False
  def tensor(self,key):self.read.append(key);return object(),dict(shape=[1],dtype=1,pointer=1234),2
  def close(self):self.closed=True
 class Lib:
  def __init__(self,fail=False):self.calls=[];self.fail=fail
  def glm_attention_load_r398(self,ctx,data):self.calls.append(('load',json.loads(data)));return int(self.fail)
  def glm_attention_finish_r398(self,ctx):self.calls.append(('finish',));return 0
 def ck(rc):
  if rc:raise RuntimeError('recorded load failure')
 for fail in [False,True]:
  r=Reader();lib=Lib(fail)
  try:out=load(lib,123,KEYS,r,ck,lambda:None)
  except RuntimeError:assert fail
  else:assert not fail and out['keys']==12 and lib.calls[-1]==('finish',)
  assert r.closed
 r=Reader()
 try:load(Lib(),123,KEYS+[KEYS[0]],r,ck,lambda:None)
 except AssertionError:assert not r.read
 else:raise AssertionError('duplicate keys accepted')
 return dict(passed=True,actual_loader_body_recorded=True,keys=12,one_pass=True,load_error_closes_reader=True,duplicate_keys_reject_before_payload=True,checkpoint_payload_read=False,Native=False,GPU=0)
