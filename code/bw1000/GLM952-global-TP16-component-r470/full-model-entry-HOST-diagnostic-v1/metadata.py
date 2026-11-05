from pathlib import Path
import json,time,sys
print(json.dumps(dict(phase='Python_entry',at=time.time())),flush=True)
P=Path('/refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1');sys.path.insert(0,str(P))
import shared
original=shared.H
def measured(p):
 print(json.dumps(dict(phase='hash_before',file=Path(p).name,at=time.time())),flush=True)
 h=original(p)
 print(json.dumps(dict(phase='hash_after',file=Path(p).name,at=time.time())),flush=True)
 return h
shared.H=measured
print(json.dumps(dict(phase='verify_before',at=time.time())),flush=True)
m,cpu,inp,plan=shared.verify('d859262194500adb03931e9417bff0258f23350c036d90620cec82466a903361')
print(json.dumps(dict(phase='verify_after_torch_before',at=time.time())),flush=True)
import torch
print(json.dumps(dict(phase='torch_after_safetensors_before',at=time.time(),CUDA_initialized=torch.cuda.is_initialized())),flush=True)
assert not torch.cuda.is_initialized()
import safetensors
print(json.dumps(dict(phase='safetensors_after_tokenizers_before',at=time.time())),flush=True)
import tokenizers
print(json.dumps(dict(phase='tokenizers_after',at=time.time())),flush=True)
