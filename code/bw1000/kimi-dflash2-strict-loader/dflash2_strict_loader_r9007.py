"""Pinned BF16 loader. A mapped byte storage backs CPU weights without copies."""
import dataclasses,hashlib,json,math,pathlib,threading
import torch
import torch.nn.functional as F
from dflash2_incremental_worker_r9003 import IncrementalDFlash2

REVISION='e77935fb4804e17eb55085bffd045eae1d779769'
LFS_SHA256='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2'
FILE_BYTES=7385765464
HEADER_SHA256='d250fe199d420a34c0bba811511d6e0ee79f11c95ad8f9482cd014ce9180fb24'
CONFIG_SHA256='aac0900105b9099b99a452c728ff9c4685889d264f7bed0101e57788df8f09c8'

def creator_key(key):
 return {'fc.weight':'context_proj.weight','hidden_norm.weight':'context_norm.weight','norm.weight':'final_norm.weight'}.get(key,key)

def unique_json(raw):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('duplicate safetensors key '+k)
   d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs)

@dataclasses.dataclass(frozen=True)
class _Spec:
 file_bytes:int
 sha256:str
 header_sha256:str
 header:dict

def _validate(path,spec):
 path=pathlib.Path(path)
 if '.partial'in path.name or path.suffix!='.safetensors':raise ValueError('partial/non-safetensors path refused')
 stat=path.stat()
 if stat.st_size!=spec.file_bytes:raise ValueError('incomplete/extra file length')
 with path.open('rb')as f:
  prefix=f.read(8)
  if len(prefix)!=8:raise ValueError('missing header length')
  length=int.from_bytes(prefix,'little')
  if not 2<=length<=1<<20 or 8+length>stat.st_size:raise ValueError('invalid bounded header length')
  raw=f.read(length)
  if hashlib.sha256(raw).hexdigest()!=spec.header_sha256:raise ValueError('pinned header SHA mismatch')
  header=unique_json(raw)
  tensors={k:v for k,v in header.items()if k!='__metadata__'};expected={k:v for k,v in spec.header.items()if k!='__metadata__'}
  if set(tensors)!=set(expected)or len(tensors)!=87:raise ValueError('strict 87-key mismatch')
  mapped=[creator_key(k)for k in tensors]
  if len(set(mapped))!=len(mapped):raise ValueError('creator-key collision')
  spans=[]
  for key,t in tensors.items():
   if t['dtype']!='BF16'or t['shape']!=expected[key]['shape']or t!=expected[key]:raise ValueError('key shape/dtype/offset mismatch '+key)
   if not all(type(x)is int and x>0 for x in t['shape']):raise ValueError('invalid dimensions')
   a,b=t['data_offsets']
   if type(a)is not int or type(b)is not int or a<0 or b-a!=2*math.prod(t['shape']):raise ValueError('invalid BF16 span')
   spans.append((a,b))
  spans.sort();end=0
  for a,b in spans:
   if a!=end:raise ValueError('overlap/hole in tensor payload')
   end=b
  if 8+length+end!=stat.st_size:raise ValueError('payload endpoint mismatch')
  f.seek(0);h=hashlib.sha256()
  while True:
   block=f.read(8<<20)
   if not block:break
   h.update(block)
  if h.hexdigest()!=spec.sha256:raise ValueError('complete LFS SHA mismatch')
  after=path.stat()
  if(stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('file changed during validation')
 return path,header,8+length,h.hexdigest()

class ExternalTargetHead:
 """Borrowed weight reference; matching shape never certifies value compatibility."""
 def __init__(self,weight,*,values_compatibility=None,proof=None):
  if weight.ndim!=2 or weight.dtype!=torch.bfloat16:raise ValueError('external BF16 [vocab,H] head required')
  self.weight=weight;self.values_compatibility=values_compatibility;self.proof=proof;self.owner=threading.get_ident();self.closed=False
 def __call__(self,x):
  if self.closed or self.owner!=threading.get_ident():raise RuntimeError('external head closed/foreign owner')
  return F.linear(x,self.weight)
 def close(self):self.weight=None;self.closed=True

class LoadedDraft:
 def __init__(self,worker,storage,receipt,head):self.worker=worker;self._storage=storage;self.receipt=receipt;self._borrowed_head=head
 def close(self):
  self.worker.close();self._storage=None;self._borrowed_head=None
  # ExternalTargetHead remains caller-owned; tensors retained by a caller keep
  # their PyTorch mapped storage alive. No unsafe manual mmap.close is used.

def _load_verified(path,spec,factory,head,*,device='cpu',enabled=False,eos_ids=()):
 if not enabled:raise RuntimeError('strict incremental loader defaultOFF')
 if not isinstance(head,ExternalTargetHead)or head.closed:raise ValueError('explicit live external target head missing')
 path,header,payload,actual_sha=_validate(path,spec)
 old_dtype=torch.get_default_dtype()
 try:
  torch.set_default_dtype(torch.bfloat16)
  with torch.device('meta'):model=factory()
 finally:torch.set_default_dtype(old_dtype)
 expected={creator_key(k):v for k,v in header.items()if k!='__metadata__'};parameters=dict(model.named_parameters())
 if set(parameters)!=set(expected):raise ValueError('creator model parameter-key mismatch')
 for key,param in parameters.items():
  if not param.is_meta or param.dtype!=torch.bfloat16 or list(param.shape)!=expected[key]['shape']:raise ValueError('creator model meta/BF16/shape mismatch '+key)
 if tuple(head.weight.shape)!=(model.embed_tokens.num_embeddings,model.embed_tokens.embedding_dim):raise ValueError('head shape incompatible; values remain unqualified')
 if head.weight.device!=torch.device(device):raise ValueError('external head must already be on destination device')
 storage=torch.from_file(str(path),shared=False,size=spec.file_bytes,dtype=torch.uint8)
 for serving,t in header.items():
  if serving=='__metadata__':continue
  key=creator_key(serving);a,b=t['data_offsets'];view=storage.narrow(0,payload+a,b-a).view(torch.bfloat16).reshape(t['shape'])
  value=view if torch.device(device).type=='cpu'else view.to(device)
  parent,_,leaf=key.rpartition('.');module=model.get_submodule(parent)if parent else model;setattr(module,leaf,torch.nn.Parameter(value,requires_grad=False))
 # Nonpersistent RoPE buffers are absent from the 87-tensor checkpoint.
 # Recreate through the unchanged canonical builder, never materialize random
 # model parameters or construct a second complete state_dict.
 with torch.device(device):
  for layer in model.layers:layer.self_attn._init_rope()
 if any(x.is_meta for x in model.buffers()):raise ValueError('unmaterialized runtime buffer')
 model.eval();worker=IncrementalDFlash2(model,head,enabled=True,eos_ids=eos_ids)
 receipt={'revision':REVISION,'full_SHA256':actual_sha,'BF16_tensors_loaded':len(expected),'CPU_weight_storage':'single private file mapping; no allocated full state_dict copy','GPU_transfer':'one tensor at a time from file mapping'if torch.device(device).type!='cpu'else None,'external_head_borrowed':True,'external_head_values_compatibility':head.values_compatibility,'external_head_proof':head.proof,'trained_acceptance_qualified':False}
 return LoadedDraft(worker,storage,receipt,head)

def load_pinned_worker(path,external_head,*,enabled=False,device='cpu',source_dir=None):
 """Production entry: pins cannot be overridden by a fixture specification."""
 source=pathlib.Path(source_dir)if source_dir else pathlib.Path(__file__).parent
 raw=(source/'actual-safetensors-header.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==HEADER_SHA256
 config_bytes=(source/'config.json').read_bytes();assert hashlib.sha256(config_bytes).hexdigest()==CONFIG_SHA256
 from dflash2_creator_factory_r9007 import make_factory
 cfg=json.loads(config_bytes);factory=make_factory(cfg,source)
 return _load_verified(path,_Spec(FILE_BYTES,LFS_SHA256,HEADER_SHA256,unique_json(raw)),factory,external_head,device=device,enabled=enabled,eos_ids=(cfg['eos_token_id'],))
