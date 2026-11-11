from pathlib import Path
import hashlib,json
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1');rows=[]
for rank in range(8):
 p=W/'capture'/('rank-%02d'%rank)/'runtime-providers.json';raw=p.read_bytes();assert len(raw)<1024**2;rows.append(dict(rank=rank,SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),providers=json.loads(raw)))
source=Path('/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py');raw=source.read_bytes();lines=raw.decode().splitlines();selected=[dict(line=i+1,text=x) for i,x in enumerate(lines) if any(k in x for k in ('is_fast_path_available','is_flash_linear_attention_available','self.chunk_gated_delta_rule =','self.recurrent_gated_delta_rule =','self.causal_conv1d_fn =','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','from fla','from causal_conv1d'))]
print(json.dumps(dict(platform='A100_original_common_mask',records=rows,source=dict(path=str(source),SHA256=hashlib.sha256(raw).hexdigest(),lines=selected),GPU_cards=0,torch_imported=False,raw_tensor_transfer_bytes=0,qualification='Original saved loaded-module metadata plus current source bytes; A100 original callback did not record per-instance callable bindings or kernel trace')))
