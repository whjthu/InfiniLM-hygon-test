from pathlib import Path
import collections,hashlib,json
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native25-HIP-diagnostic-v3');out=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rank in range(8):
 for name in ('runtime-providers.json','runtime-providers-final.json'):
  p=W/'capture'/('rank-%02d'%rank)/name;assert p.stat().st_size<1024**2;v=json.loads(p.read_bytes());bindings=v['bindings'];groups={}
  for b in bindings:
   assert b['source']['SHA256']=='395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6';key=json.dumps(b['callables'],sort_keys=True)
   if key not in groups:groups[key]=dict(callables=b['callables'],count=0,example=b['name'])
   groups[key]['count']+=1
  out.append(dict(rank=rank,name=name,bytes=p.stat().st_size,SHA256=sha(p),model_source_SHA256=bindings[0]['source']['SHA256'],bindings=len(bindings),provider_groups=list(groups.values()),loaded_modules=v['already_loaded_modules']))
source=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-hygon-processor-r2/cpu/runtime-site/transformers/models/qwen3_5/modeling_qwen3_5.py');raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6'
lines=raw.decode().splitlines();selected=[dict(line=i+1,text=x) for i,x in enumerate(lines) if any(k in x for k in ('is_fast_path_available','is_flash_linear_attention_available','self.chunk_gated_delta_rule =','self.recurrent_gated_delta_rule =','self.causal_conv1d_fn =','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','from fla','from causal_conv1d'))]
print(json.dumps(dict(platform='Hygon_original143',records=out,source=dict(path=str(source),SHA256=hashlib.sha256(raw).hexdigest(),lines=selected),GPU_cards=0,torch_imported=False,raw_tensor_transfer_bytes=0,qualification='Original saved bound-callable metadata and actual source bytes; no new import or kernel execution trace')))
