"""Default-OFF private KV_A/KV_B prefix reuse; original attention.forward is unchanged."""
import copy,threading,weakref,time
import torch
from long_context_r323 import LongCommittedFeatureContext
from long_rpc import DraftSession

def version(tensor):
 try:return tensor._version
 except RuntimeError:return None

def full_query(model,head,projected,anchor,K):
 class RecordHead:
  def __init__(self,base):self.base=base;self.last=None
  def __call__(self,x):self.last=self.base(x);return self.last
 from query_long_projected_r323 import query_long_projected
 observed=RecordHead(head)
 values,debug=query_long_projected(model,observed,[projected],[anchor],[K],enabled=True,return_debug=True,profile=False)
 debug['logits']=observed.last.view(1,K,-1)
 return values,debug

class PrefixLinear(torch.nn.Module):
 def __init__(self,base,context,layer,kind):
  super().__init__();self.base=base;self.context=weakref.ref(context);self.layer=layer;self.kind=kind;self.training=base.training
 def forward(self,x):
  owner=self.context()
  if owner is None:raise RuntimeError('Owned prefix context released')
  return owner.linear(self.layer,self.kind,self.base,x)

def private_view(model,context):
 view=copy.copy(model);view._modules=dict(model._modules);layers=[]
 for i,original in enumerate(model.layers):
  layer=copy.copy(original);layer._modules=dict(original._modules)
  attn=copy.copy(original.self_attn);attn._modules=dict(original.self_attn._modules)
  attn.kv_a_proj_with_mqa=PrefixLinear(original.self_attn.kv_a_proj_with_mqa,context,i,'A')
  attn.kv_b_proj=PrefixLinear(original.self_attn.kv_b_proj,context,i,'B')
  layer.self_attn=attn;layers.append(layer)
 view.layers=torch.nn.ModuleList(layers)
 return view

class CachedContext(LongCommittedFeatureContext):
 def __init__(self,model,weight_tag,*,enabled=False,cache_enabled=False):
  super().__init__(model,weight_tag,enabled=enabled)
  if type(cache_enabled)is not bool:raise ValueError('Explicit Boolean cache opt-in required')
  self.original=model;self.cache_enabled=cache_enabled;self.cache={};self.cached_length=0
  self.pending=None;self.active=None;self.last_debug=None;self.last_stats=None;self.publish_count=0;self.aborted=False
  self.signature=self._weight_signature();self.model=private_view(model,self)
 def _weight_signature(self):
  items=[id(self.original),self.guard.weight_tag]
  for i,layer in enumerate(self.original.layers):
   a=layer.self_attn;items.append((i,id(a),a.num_heads,a.qk_head_dim,a.v_head_dim,a.kv_lora_rank,id(a.rotary_emb),a.softmax_scale))
   for name,module in [('A',a.kv_a_proj_with_mqa),('N',a.kv_a_layernorm),('B',a.kv_b_proj)]:
    items.append((i,name,id(module),type(module).__name__,bool(module.training),getattr(module,'variance_epsilon',None)))
    for key,p in module.named_parameters():items.append((i,name,key,id(p),p.data_ptr(),tuple(p.shape),str(p.dtype),str(p.device),version(p)))
  return tuple(items)
 def drop(self):
  self._owner();self.cache={};self.cached_length=0;self.pending=None;self.active=None
 def abort(self):
  self._owner();self.drop();self.aborted=True
 def _check(self,epoch,request):
  self._owner()
  try:
   super()._check(epoch,request)
   if self.cache_enabled and self._weight_signature()!=self.signature:raise RuntimeError('KV weight identity/version changed; new Context required')
   if self.aborted:raise RuntimeError('Aborted cached context')
  except BaseException:
   self.abort();raise
 def begin(self,*args):
  self._owner()
  try:
   super().begin(*args);self.drop();self.aborted=False;self.last_debug=None
   if self.cache_enabled and self._weight_signature()!=self.signature:raise RuntimeError('KV weights changed across epoch; new Context required')
  except BaseException:self.abort();raise
 def linear(self,layer,kind,base,x):
  self._owner()
  if not self.cache_enabled:return base(x)
  a=self.active
  if a is None:raise RuntimeError('Cached Linear called outside owned query')
  if (layer,kind)in a['values']:raise RuntimeError('Repeated per-layer Linear in one query')
  L=a['length'];old=self.cache.get((layer,kind))if a['incremental']else None
  assert x.ndim==3 and x.shape[0]==1 and x.shape[1]==L+8
  prefix=self.cached_length if old is not None else 0
  tick=time.perf_counter();computed=base(x[:,prefix:]);compute_wall=time.perf_counter()-tick
  tick=time.perf_counter();out=torch.cat([old,computed],dim=1)if old is not None else computed;cat_wall=time.perf_counter()-tick
  assert out.shape[1]==L+8
  cached=old if old is not None and prefix==L else out[:,:L]
  a['values'][(layer,kind)]=cached;a['finite_inputs'].append(computed)
  a['stats'].append(dict(layer=layer,kind=kind,input_rows=L+8,computed_rows=int(computed.shape[1]),cached_prefix_rows=prefix,output_rows=int(out.shape[1]),compute_host_wall_s=compute_wall,cat_host_wall_s=cat_wall,cat_output_bytes=out.numel()*out.element_size()if old is not None else 0))
  return out
 def query(self,epoch,request,head,anchor,K=7,*,return_debug=False):
  plan=self.query_plan(epoch,request,anchor,K)
  if plan['mode']=='ar_tail':self.drop();self.last_debug=None;self.last_stats=dict(mode='ar_tail',linear_calls=[]);return dict(plan,values=None)
  if self.pending is not None:raise RuntimeError('Previous query pending final codec fence')
  L=self.length;delta=L-self.cached_length
  incremental=bool(self.cache)and 0<=delta<=8
  mode='hit'if incremental else('full_refresh_gap'if self.cache else'cold_full')
  self.active=dict(length=L,incremental=incremental,values={},finite_inputs=[],stats=[])
  try:
   values,debug=full_query(self.model,head,self.projected,anchor,K);self.last_debug=debug
   if self.cache_enabled:
    a=self.active;assert len(a['values'])==2*len(self.model.layers)
    finite=torch.stack([torch.isfinite(t).all()for t in a['finite_inputs']]).all()
    if not bool(finite):raise ValueError('Nonfinite pending context projections')
    self._check(epoch,request)
    self.pending=dict(epoch=epoch,request=request,length=L,values=a['values'],signature=self.signature)
    self.last_stats=dict(mode=mode,delta=delta,linear_calls=a['stats'],pending_bytes=sum(t.numel()*t.element_size()for t in a['values'].values()),cache_committed_before_fence=False)
   else:self.last_stats=dict(mode='OFF_original',linear_calls=[])
   return dict(plan,values=(values,debug)if return_debug else values)
  except BaseException:self.abort();raise
  finally:self.active=None
 def publish_after_fence(self):
  self._owner()
  if self.pending is None:return
  p=self.pending;self._check(p['epoch'],p['request'])
  assert p['length']==self.length and p['signature']==self.signature
  self.cache=p['values'];self.cached_length=p['length'];self.pending=None;self.publish_count+=1
  if self.last_stats is not None:self.last_stats['cache_committed_before_fence']=False;self.last_stats['published_after_original_fence']=True
 def retire(self,*args):
  self._owner()
  try:return super().retire(*args)
  finally:self.drop();self.last_debug=None
 def close(self):
  self._owner()
  try:return super().close()
  finally:self.drop();self.last_debug=None

class PrivateSession(DraftSession):
 def dispatch(self,meta,body):
  self._owner()
  try:return super().dispatch(meta,body)
  except BaseException:self.context.abort();raise

def committing_codec(base,context):
 """Call original codec/4GiB guard first; only then publish all5 pending layers."""
 class Codec(base):
  def fence(self):
   try:super().fence();context.publish_after_fence()
   except BaseException:context.abort();raise
 return Codec
