"""Default-OFF CPU source package. Hold this context through native load+drain.
No CPU clone, no forced contiguous copy, no tensor .numpy/tobytes/full SHA scan.
Only header/stat checks are current; prior tensor SHA bindings are historical proof.
"""
import pathlib,json,hashlib,contextlib,os
@contextlib.contextmanager
def mapped_layer_weights(model,prefix,expected_header_sha256,rss_budget_bytes=8<<30):
 import torch
 from safetensors import safe_open
 model=pathlib.Path(model);mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];keys=sorted(k for k in mapping if k.startswith(prefix));files=sorted(set(mapping[k]for k in keys));held=[];descriptors=[];owners=contextlib.ExitStack();stats={};opened={}
 def rss():
  return int(next(v.split()[1]for v in pathlib.Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')))*1024
 try:
  for name in files:
   p=model/name;st=p.stat();identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
   with p.open('rb')as f:size=int.from_bytes(f.read(8),'little');raw=f.read(size)
   assert hashlib.sha256(raw).hexdigest()==expected_header_sha256[name];header=json.loads(raw);stats[name]={'identity_before':identity,'header_SHA256':expected_header_sha256[name],'bytes':st.st_size,'payload_offset':8+size};opened[name]=(owners.enter_context(safe_open(p,framework='pt',device='cpu')),header)
  for key in keys:
   f,header=opened[mapping[key]];v=f.get_tensor(key);h=header[key];assert v.device.type=='cpu'and v.is_contiguous()and list(v.shape)==h['shape'];assert v.dtype in (torch.float32,torch.bfloat16,torch.uint8);held.append(v);descriptors.append({'name':key[len(prefix):],'pointer':v.data_ptr(),'shape':list(v.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[v.dtype]})
   if rss()>rss_budget_bytes:raise MemoryError('lazy mmap RSS budget exceeded; stop before GPU launch')
  proof={'tensor_count':len(held),'logical_weight_bytes':sum(v.numel()*v.element_size()for v in held),'current_full_tensor_SHA_verified':False,'CPU_clones':0,'full_tensor_numpy_tobytes':False,'files':stats,'CPU_buffers_ownership':'held tensors and safe_open contexts remain alive until context exit AFTER native owner drained all copies','source_readonly_contract':'native uses these only as source; caller must not mutate Torch views'}
  yield held,descriptors,proof
  for name in files:
   st=(model/name).stat();assert tuple(stats[name]['identity_before'])==(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns),'checkpoint changed during load'
 finally:
  descriptors.clear();held.clear();opened.clear();owners.close()
