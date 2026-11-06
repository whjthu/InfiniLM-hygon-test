"""Same raw fullcheckpoint source encoding and perfile flush, typed newcontext only."""
from pathlib import Path
import collections,hashlib,json,time

def load_checkpoint(handle,*,model_path,torch,safe_open,report,deadline,memory,mark,progress=print):
 # Adapt the qualified r444 loader's rawtensor dtype, grouping, contiguous source,
 # per2048/perfile flush and completion-returning source lifetime. No presharding.
 model=Path(model_path);keys=set(handle.keys());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];assert keys and keys<=set(mapping)
 layers=sorted({int(k.split('.')[2])for k in keys if k.startswith('model.layers.')});assert layers==list(range(78));assert{'model.embed_tokens.weight','model.norm.weight','lm_head.weight'}<=keys
 grouped=collections.defaultdict(list)
 for key in keys:grouped[mapping[key]].append(key)
 pending=[];loaded=0;source_bytes=0;begin=time.monotonic();report['checkpoint_keys_SHA256']=hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest();report['checkpoint_key_count']=len(keys);report['actual_layers']=layers
 def flush():
  nonlocal loaded
  if not pending:return
  deadline('typed load_batch');records=[dict(name=name,pointer=w.data_ptr(),shape=list(w.shape),dtype={torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype],owner=w)for name,w in pending]
  # The typed caller holds allCPU owners through original Native copy+sync.
  handle.load_batch(records);loaded+=len(pending);pending.clear();del records;memory('after_load_flush');report.update(loaded_parameters=loaded,loading_seconds=time.monotonic()-begin);mark('load.progress',loaded=loaded,total=len(keys));progress('GLM470_FULL_LOAD_PROGRESS',handle.rank,loaded,len(keys),round(time.monotonic()-begin,3),flush=True)
 for filename,names in sorted(grouped.items()):
  deadline('safe_open original shard')
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   for name in sorted(names):
    w=shard.get_tensor(name).contiguous()
    if w.dtype==torch.float8_e4m3fn:w=w.view(torch.uint8)
    assert w.dtype in[torch.float32,torch.bfloat16,torch.uint8],(name,str(w.dtype));source_bytes+=w.numel()*w.element_size();pending.append((name,w))
    if len(pending)>=2048:flush()
   flush()
 assert not pending and loaded==len(keys);deadline('typed finish');mark('finish.enter');handle.finish();memory('after_finish');geo=handle.geometry();assert geo['weights_loaded']and geo['Model_READY']and geo['loaded_parameters']==len(keys);report.update(loaded_parameters=loaded,source_parameter_bytes=source_bytes,loading_seconds=time.monotonic()-begin,all78layers_loaded=True,postload_geometry=geo);mark('finish.return');return report
