"""Same old safe_open/get_tensor/contiguous/A_log trim/2GiB synchronous flush. No clone."""
import json,time,collections,ctypes
from shard_head_contract_r9183 import expected_from_CPU

def load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress):
 mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping);groups=collections.defaultdict(list)
 for name in keys:groups[mapping[name]].append(name)
 phases=collections.defaultdict(lambda:{'seconds':0.0,'calls':0})
 def record(label,begin):
  phases[label]['seconds']+=time.perf_counter()-begin;phases[label]['calls']+=1
 copy_stats=lib.modern_cpu_copy_stats_r2515;copy_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];copy_stats.restype=ctypes.c_int;flush_profiles=[]
 def copy_snapshot():
  values=(ctypes.c_uint64*5)();assert copy_stats(values,5)==0;return list(values)
 head_source=None;pending=[];pending_bytes=loaded=flushes=0;start=time.perf_counter()
 def flush():
  nonlocal pending_bytes,loaded,flushes
  if not pending:return
  begin=time.perf_counter()
  descriptors=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending];payload=json.dumps(descriptors).encode();record('descriptor_json',begin);before=copy_snapshot();begin=time.perf_counter();check(lib.modern_kimi_engine_load_r2475(ctx,payload));native_wall=time.perf_counter()-begin;record('native_load_inclusive',begin);after=copy_snapshot();assert all(b>=a for a,b in zip(before,after));flush_profiles.append({'flush':flushes,'keys':len(pending),'CPU_logical_source_bytes':pending_bytes,'native_load_host_inclusive_s':native_wall,'r2515_before':before,'r2515_after':after,'r2515_delta':[b-a for a,b in zip(before,after)],'scope':'one nodeprocess aggregate per original native_load; includes pagefault/prepare/8ownerload+originalsync, not DMA time'});loaded+=len(pending);pending.clear();pending_bytes=0;flushes+=1
  if flushes%16==0:progress({'loaded_keys':loaded,'loading_elapsed_seconds':time.perf_counter()-start})
 for filename,names in sorted(groups.items()):
  begin=time.perf_counter()
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   record('safe_open_enter',begin)
   for name in sorted(names):
    begin=time.perf_counter();w=shard.get_tensor(name);record('get_tensor_inclusive',begin)
    if name.endswith('.self_attn.A_log'):w=w[:text['linear_attn_config']['num_heads']]
    begin=time.perf_counter();w=w.contiguous();record('contiguous_inclusive',begin)
    if name=='language_model.lm_head.weight':
     begin=time.perf_counter();head_source=expected_from_CPU(lib,w,torch);record('head_source_registration',begin)
    pending.append((name,w));pending_bytes+=w.numel()*w.element_size()
    if len(pending)>=16384 or pending_bytes>=2*1024**3:flush()
   flush()
 begin=time.perf_counter();check(lib.modern_kimi_engine_finish_r2475(ctx));record('native_finish_inclusive',begin);assert loaded==len(keys)
 return {'native_load_flush_profiles':flush_profiles,'r2515_labels':['calls','bulk_blocks','bulk_bytes','scalar_leaves','scalar_bytes'],'extra_empty_load_calls':0,'host_loader_phase_profile':dict(phases),'phase_scope':'Host inclusive wall; mmap page faults can occur in native_load; not pure I/O/H2D or GPU kernel timing; no live runtime modified','head_CPU_source_registration':head_source,'loaded_keys':loaded,'loader_flushes':flushes,'loading_seconds':time.perf_counter()-start,'loader':'original bounded no_clone shardwise path, only functionized'}
