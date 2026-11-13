"""Import the native HunyuanVideo SFT dependencies inside the masked Hygon runtime."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

output = Path('/evidence/CPU-result.json')
binding = json.loads(Path('/input-stage/BINDING.json').read_bytes())
owner = dict(step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
             node=socket.gethostname().split('.')[0])
report = dict(passed=False, owner=owner, modules={}, versions={}, GPU_cards=0,
              model_constructed=False, weights_read=False, cache_read=False)
phase = 'initial'
try:
    for mask in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):
        assert os.environ[mask]=='-1'
    phase = 'verify_native_files'
    root = Path('/hunyuan')
    for name,pin in binding['files'].items():
        path = root/name
        raw = path.read_bytes()
        assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    report['native_files_verified'] = len(binding['files'])
    for package in ('torch','diffusers','peft','accelerate','einops','loguru','safetensors','transformers'):
        try:
            report['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][package] = None
    phase = 'torch'
    import torch
    report['torch_hip'] = torch.version.hip
    assert torch.version.hip and not torch.cuda.is_initialized()
    modules = {
        'hyvideo.commons.parallel_states': 'initialize_parallel_state',
        'hyvideo.models.transformers.hunyuanvideo_1_5_transformer': 'HunyuanVideo_1_5_DiffusionTransformer',
        'hyvideo.optim.muon': 'get_muon_optimizer',
        'peft': 'LoraConfig',
    }
    for module,name in modules.items():
        phase = module
        loaded = importlib.import_module(module)
        assert callable(getattr(loaded,name))
        path = Path(loaded.__file__)
        if module.startswith('hyvideo.'):
            rel = path.relative_to(root).as_posix()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==binding['files'][rel]['SHA256']
        report['modules'][module] = dict(symbol=name,path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    phase = 'device_initialization_check'
    assert not torch.cuda.is_initialized()
    report['passed'] = True
except Exception as error:
    report['failure'] = dict(phase=phase,type=type(error).__name__,
                             missing_module=getattr(error,'name',None),
                             detail=str(error).splitlines()[-1][:240] if str(error) else '')
finally:
    with output.open('x') as f:
        json.dump(report,f,indent=2)
if not report['passed']:
    raise SystemExit(1)
