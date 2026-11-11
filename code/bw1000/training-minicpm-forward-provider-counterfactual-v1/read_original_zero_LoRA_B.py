import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import hashlib,json,time
start=time.monotonic();p=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1/capture/rank-00/initial.pt');assert p.stat().st_size==43466515;assert hashlib.sha256(p.read_bytes()).hexdigest()=='d3824dc5a788a0238861ae86dec0374dac7b5bf7e6e945e163c32877d6b86d83'
import torch
torch.set_num_threads(1);v=torch.load(p,map_location='cpu',weights_only=True,mmap=True);params=v['parameters'];assert v['global_step']==0 and len(params)==372
rows={}
for label,part in [('A','.lora_A.'),('B','.lora_B.')]:
 values=[x for n,x in params.items() if part in n];assert len(values)==186
 rows[label]=dict(tensors=len(values),elements=sum(x.numel() for x in values),nonzero_elements=sum(int(torch.count_nonzero(x)) for x in values),finite=all(bool(torch.isfinite(x).all()) for x in values),dtype=sorted({str(x.dtype) for x in values}))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,original_initial_file_SHA256='d3824dc5a788a0238861ae86dec0374dac7b5bf7e6e945e163c32877d6b86d83',groups=rows,LoRA_B_all_zero=rows['B']['nonzero_elements']==0,GPU_initialized=False,GPU_cards=0,raw_tensor_transfer_bytes=0,wall_seconds=time.monotonic()-start,qualification='Exact original initial-value group statistics, not model acceptance. Zero B removes a direct nonzero LoRA branch contribution at this forward; RNG or observation effects remain unqualified.')))
