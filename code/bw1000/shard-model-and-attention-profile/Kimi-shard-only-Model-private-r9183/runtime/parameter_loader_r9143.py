"""Same old safe_open/get_tensor/contiguous/A_log trim/2GiB synchronous flush. No clone."""
import json,time,collections
from shard_head_contract_r9183 import expected_from_CPU

def load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress):
 mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping);groups=collections.defaultdict(list)
 for name in keys:groups[mapping[name]].append(name)
 head_source=None;pending=[];pending_bytes=loaded=flushes=0;start=time.perf_counter()
 def flush():
  nonlocal pending_bytes,loaded,flushes
  if not pending:return
  descriptors=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending];check(lib.modern_kimi_engine_load_r2475(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0;flushes+=1
  if flushes%16==0:progress({'loaded_keys':loaded,'loading_elapsed_seconds':time.perf_counter()-start})
 for filename,names in sorted(groups.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   for name in sorted(names):
    w=shard.get_tensor(name)
    if name.endswith('.self_attn.A_log'):w=w[:text['linear_attn_config']['num_heads']]
    w=w.contiguous()
    if name=='language_model.lm_head.weight':head_source=expected_from_CPU(lib,w,torch)
    pending.append((name,w));pending_bytes+=w.numel()*w.element_size()
    if len(pending)>=16384 or pending_bytes>=2*1024**3:flush()
   flush()
 check(lib.modern_kimi_engine_finish_r2475(ctx));assert loaded==len(keys)
 return {'head_CPU_source_registration':head_source,'loaded_keys':loaded,'loader_flushes':flushes,'loading_seconds':time.perf_counter()-start,'loader':'original bounded no_clone shardwise path, only functionized'}
