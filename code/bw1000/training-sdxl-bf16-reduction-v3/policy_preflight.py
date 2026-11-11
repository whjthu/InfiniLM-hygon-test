from pathlib import Path
import hashlib,importlib.metadata as metadata,json,runpy,sys
W=Path('/t06-run');deps=json.loads((W/'src/DEPENDENCIES.json').read_bytes());versions={k:metadata.version(k) for k in deps['versions']};assert versions==deps['versions']
import torch
assert not torch.cuda.is_initialized() and torch.version.hip==deps['HIP'] and torch.cuda.device_count()==0
providers={}
for module in (torch,torch.backends.cuda,torch.optim.adamw if hasattr(torch.optim,'adamw') else __import__('torch.optim.adamw',fromlist=['AdamW'])):
 p=Path(module.__file__);assert p.is_file() and p.stat().st_size<2*1024**2;providers[module.__name__]=dict(path=str(p),bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
(W/'runtime-preflight.json').write_text(json.dumps(dict(versions=versions,HIP=torch.version.hip,GPU_initialized=torch.cuda.is_initialized(),providers=providers),indent=2)+'\n')
sys.path.insert(0,str(W/'source/tools'));source=str(W/'source/tools/hygon_sdxl_precision_probe.py');sys.argv=[source,'--reference','/reference/reference.pt','--policy','/reference/policy.json','--output','/t06-run/preflight-policy','--policy-only'];runpy.run_path(source,run_name='__main__');assert not torch.cuda.is_initialized()
