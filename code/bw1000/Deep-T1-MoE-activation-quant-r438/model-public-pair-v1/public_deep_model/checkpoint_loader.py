"""Qualified Deep raw loader and CAPI declarations, with an explicit lifetime owner."""
import ctypes,hashlib,json,time
from pathlib import Path

NATIVE_SHA='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9'
CONFIG_SHA='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
INDEX_SHA='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'

class Binding:
 def __init__(self,lib):self.lib=lib
 def check(self,code):
  if code:raise RuntimeError(self.lib.modern_native_engine_error().decode())

def declare(lib):
 lib.modern_native_engine_error.argtypes=[];lib.modern_native_engine_error.restype=ctypes.c_char_p
 lib.modern_native_engine_create.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_size_t];lib.modern_native_engine_create.restype=ctypes.c_void_p
 lib.modern_native_engine_keys.argtypes=[ctypes.c_void_p];lib.modern_native_engine_keys.restype=ctypes.c_char_p
 for name in ['finish','reset','destroy','rank_logits_equal']:
  f=getattr(lib,'modern_native_engine_'+name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_int
 lib.modern_native_engine_load.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.modern_native_engine_load.restype=ctypes.c_int
 lib.modern_native_engine_forward.argtypes=[ctypes.c_void_p]+[ctypes.c_size_t]*2+[ctypes.c_void_p]*8;lib.modern_native_engine_forward.restype=ctypes.c_int
 for name,count in [('deep_hc_canonical_policy_r7130',1),('deep_hc_canonical_get_r7130',0),('modern_deep_layout_get_r7052',0),('deep_moe_tail_get_r7211',0),('modern_deep_c4_tile_get_r5920',0),('modern_deep_c4_tile_policy_r5920',1),('modern_deep_allselected_set_r813',1)]:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int]*count;f.restype=ctypes.c_int
 for name,count in [('deep_output_tile_calls_r4808',1),('modern_deep_c4_tile_calls_r5920',2),('modern_deep_FP32_calls_r3597',1),('deep_hc_canonical_calls_r7130',2),('deep_moe_tail_calls_r7211',2),('modern_deep_layout_calls_r7052',2)]:
  f=getattr(lib,name);f.argtypes=[ctypes.c_int]*count;f.restype=ctypes.c_uint64
 return Binding(lib)

def bind_library(path,expected):
 assert expected==NATIVE_SHA and hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
 return declare(ctypes.CDLL(str(path)))

def native_expected_kinds(cfg):
 n=int(cfg['num_hidden_layers']);ratios=cfg['compress_ratios'];mtp=int(cfg['num_nextn_predict_layers']);assert n==43 and mtp==1 and len(ratios)==n+mtp
 result=[]
 for layer in range(n):
  ratio=int(ratios[layer]);assert ratio in [0,4,128];result.append({0:0,4:2,128:1}[ratio])
 assert result[:2]==[0,0]and result.count(0)==2 and result.count(1)==20 and result.count(2)==21
 return result

def read_model_metadata(model):
 model=Path(model);raw=(model/'config.json').read_bytes();index=(model/'model.safetensors.index.json').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==CONFIG_SHA and hashlib.sha256(index).hexdigest()==INDEX_SHA
 cfg=json.loads(raw);native_expected_kinds(cfg)
 return cfg,json.loads(index)['weight_map']

def load_checkpoint(binding,ctx,*,model,torch,safe_open,mapping,report,deadline_check,progress=print):
 """Every actual key, same1024/512MiB flush and strong CPU borrow until load returns."""
 lib,check=binding.lib,binding.check;model=Path(model);keys=set(json.loads(lib.modern_native_engine_keys(ctx)))
 assert len(keys)==67569 and keys<=set(mapping);grouped={}
 for key in keys:grouped.setdefault(mapping[key],[]).append(key)
 pending=[];pending_bytes=loaded=0;start=time.perf_counter()
 def flush():
  nonlocal pending_bytes,loaded
  if not pending:return
  deadline_check('Native raw load',minimum=120)
  descriptors=[{'name':name,'pointer':value.data_ptr(),'shape':list(value.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.int8:2,torch.int64:3}[value.dtype]}for name,value in pending]
  check(lib.modern_native_engine_load(ctx,json.dumps(descriptors).encode()))
  loaded+=len(pending);pending.clear();pending_bytes=0
  progress('NATIVE_ENGINE_LOAD',loaded,len(keys),time.perf_counter()-start,flush=True)
 for filename,names in sorted(grouped.items()):
  deadline_check('weight file open',minimum=120)
  with safe_open(model/filename,framework='pt',device='cpu')as reader:
   for name in sorted(names):
    value=reader.get_tensor(name).contiguous();pending.append((name,value));pending_bytes+=value.numel()*value.element_size()
    if len(pending)>=1024 or pending_bytes>=512*1024*1024:flush()
   flush()
 assert loaded==len(keys);check(lib.modern_native_engine_finish(ctx))
 report.update(loaded_key_count=loaded,loading_seconds=time.perf_counter()-start,raw_loader_dtype_codes=dict(F32=0,BF16=1,I8=2,I64=3))
 return loaded
