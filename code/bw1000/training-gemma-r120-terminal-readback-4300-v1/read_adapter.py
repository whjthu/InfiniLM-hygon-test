
import json,os,sys
from hashlib import sha256
from pathlib import Path
import torch
from safetensors.torch import load_file
from vlm.runtime.first_update_audit import fingerprint
root=Path(sys.argv[1])
assert os.environ.get('SLURM_JOB_ID')=='1103606' and os.environ.get('SLURM_STEP_ID','').isdecimal()
assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
assert {p.name for p in root.iterdir()}=={'adapter_config.json','adapter_model.safetensors','additional_config.json'}
weights=root/'adapter_model.safetensors'
def stamp(p):
 s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_ctime_ns,s.st_mtime_ns
before=stamp(weights)
assert before[2]==119280680 and not weights.is_symlink()
values=load_file(str(weights),device='cpu')
fp=fingerprint(values)
assert fp['SHA256']=='5663eaa9445a1c047af864e1d9375e152937fb4334ccab0237c10ceebdbc654e'
assert fp['tensors']==476 and fp['elements']==29802496 and fp['finite'] is True
h=sha256()
with weights.open('rb') as f:
 while data:=f.read(4<<20):h.update(data)
assert stamp(weights)==before
extra=(root/'additional_config.json').read_bytes();cfg=(root/'adapter_config.json').read_bytes()
assert len(extra)==67 and len(cfg)==1178
other=json.loads(extra); config=json.loads(cfg)
assert isinstance(other,dict) and config['r']==16 and config['lora_alpha']==32
report=dict(actual_step='1103606.'+os.environ['SLURM_STEP_ID'],request=sys.argv[2],actual_GPU=0,
 loader='safetensors.torch.load_file',values_verified=fp['elements'],tensors_verified=fp['tensors'],finite=True,
 named_value_SHA256=fp['SHA256'],adapter_bytes=before[2],adapter_file_SHA256=h.hexdigest(),
 adapter_config_SHA256=sha256(cfg).hexdigest(),additional_config_SHA256=sha256(extra).hexdigest(),
 exact_rank_value_match=True,full_model_loaded=False,actual_optimizer_updates=0)
with (Path(sys.argv[3])/'terminal.json').open('x') as o:json.dump(report,o,sort_keys=True)
