"""Import the native HunyuanVideo SFT dependencies inside the masked Hygon runtime."""
from typing import Any
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

def _apply_gradient_checkpointing(transformer: Any) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    block_types = tuple(
        {
            type(block)
            for block in list(transformer.double_blocks) + list(transformer.single_blocks)
            if block is not None
        }
    )
    if not block_types:
        raise RuntimeError("HunyuanVideo transformer exposes no checkpointable blocks")

    def wrapper(module: Any) -> Any:
        return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    apply_activation_checkpointing(
        transformer,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, block_types),
    )

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
    phase = 'original_configs'
    config_path = Path('/input-stage/TRANSFORMER_CONFIG.json')
    launch_path = Path('/input-stage/ORIGINAL_LAUNCH.json')
    for path in (config_path, launch_path):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==binding['configs'][path.name]
    recipe = json.loads(launch_path.read_bytes())['config']
    config = json.loads(config_path.read_bytes())
    cls = getattr(importlib.import_module('hyvideo.models.transformers.hunyuanvideo_1_5_transformer'),
                  'HunyuanVideo_1_5_DiffusionTransformer')
    from peft import LoraConfig
    from hyvideo.optim.muon import get_muon_optimizer

    phase = 'meta_constructor'
    with torch.device('meta'):
        model = cls.from_config(config).to(dtype=torch.bfloat16)
    report['model_constructed'] = True
    report['construction_device'] = 'meta'
    assert all(p.is_meta and p.dtype==torch.bfloat16 for p in model.parameters())
    base = {name: list(p.shape) for name,p in model.named_parameters()}
    targets = set(recipe['lora_target_modules'])
    matching = {n for n,_ in model.named_modules() if n.rsplit('.',1)[-1] in targets}
    assert matching and {n.rsplit('.',1)[-1] for n in matching}==targets
    report['base_tensors'] = len(base)
    report['base_parameters'] = sum(p.numel() for p in model.parameters())

    phase = 'native_LoRA'
    model.set_attn_mode('torch')
    with torch.device('meta'):
        model.add_adapter(LoraConfig(r=recipe['lora_rank'], lora_alpha=recipe['lora_alpha'],
            target_modules=recipe['lora_target_modules'], lora_dropout=recipe['lora_dropout'],
            bias='none', task_type='FEATURE_EXTRACTION'), adapter_name='default')
    expected = {n+'.lora_'+suffix+'.default.weight' for n in matching for suffix in ('A','B')}
    params = dict(model.named_parameters())
    assert {n for n,p in params.items() if p.requires_grad}==expected
    for name in matching:
        base_shape = base[name+'.weight']
        assert list(params[name+'.lora_A.default.weight'].shape)==[recipe['lora_rank'],base_shape[1]]
        assert list(params[name+'.lora_B.default.weight'].shape)==[base_shape[0],recipe['lora_rank']]
    trainable = {id(params[n]):dict(name=n,shape=list(params[n].shape),dtype=str(params[n].dtype)) for n in expected}
    assert all(p.is_meta for p in params.values())

    phase = 'activation_checkpointing'
    assert recipe['gradient_checkpointing'] is True
    block_count = len(model.double_blocks)+len(model.single_blocks)
    _apply_gradient_checkpointing(model)
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    wrappers = sum(isinstance(m,CheckpointWrapper) for m in model.modules())
    assert wrappers==block_count and wrappers>0
    assert {id(p) for p in model.parameters() if p.requires_grad}==set(trainable)
    model.train()

    phase = 'Muon'
    optimizer = get_muon_optimizer(model,lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    owned = [p for group in optimizer.param_groups for p in group['params']]
    assert type(optimizer).__name__=='Muon' and len(owned)==len(trainable)
    assert {id(p) for p in owned}==set(trainable)
    assert all(p.is_meta and optimizer.state[p]=={'use_muon':True} for p in owned)
    assert all(g['lr']==recipe['learning_rate'] and g['wd']==recipe['weight_decay'] for g in optimizer.param_groups)
    names = sorted(trainable.values(),key=lambda row:row['name'])
    report.update(trainable_tensors=len(owned),trainable_parameters=sum(p.numel() for p in owned),
        trainable_schema=names,lora_targets_found=len(matching),checkpoint_wrappers=wrappers,
        optimizer=dict(type=type(optimizer).__name__,groups=len(optimizer.param_groups),
                       lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'],steps=0))
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
