"""Verify original Z-Image DiT/LoRA structure without reading weight values."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def check_deadline(epoch, deadline, now=None):
    now = time.time() if now is None else now
    if not all(math.isfinite(x) for x in (epoch, deadline, now)):
        raise ValueError('finite CPU epoch required')
    if not 0 < deadline - epoch <= 220 or not epoch <= now < deadline:
        raise TimeoutError('original meta CPU epoch exhausted')


def verify_sources(root, pins):
    root = root.resolve(strict=True)
    if len(pins) != 40:
        raise ValueError('all forty original native import sources required')
    verified = {}
    for name, pin in pins.items():
        part = Path(name)
        if part.is_absolute() or '..' in part.parts or not name.startswith('diffsynth/') or not name.endswith('.py'):
            raise ValueError('invalid native source path')
        path = root / part
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('native source escapes pinned root')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != pin['bytes'] or digest != pin['SHA256']:
            raise ValueError('native source identity mismatch')
        verified[name] = digest
    return verified


def run(args):
    started = time.monotonic()
    check_deadline(args.epoch, args.deadline)
    if any(os.environ.get(n) != '-1' for n in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('explicitly hidden devices required')
    raw = args.spec.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.spec_sha256:
        raise ValueError('meta input binding mismatch')
    spec = json.loads(raw)
    verified = verify_sources(args.source_root, spec['upstream_files'])
    if args.output.exists():
        raise FileExistsError('original meta output exists; reconcile instead')
    sys.path.insert(0, str(ROOT / 'generation/src'))
    from gencuda.zimage_fusion_contract import check_metadata, validate_header
    shards = []
    preset = None
    for name, header in spec['original_headers'].items():
        validate_header(header['tensors'], file_bytes=header['bytes'], header_bytes=header['header_bytes'])
        if name.startswith('diffusion_pytorch_model-'):
            shards.append(header['tensors'])
        else:
            if preset is not None:
                raise ValueError('ambiguous original adapter metadata')
            preset = header['tensors']
    expected = check_metadata(shards, preset)
    if expected['preset_pairs'] != 240 or expected['new_rank4_elements'] != 9922560:
        raise ValueError('original parameter metadata differs')
    import torch
    if torch.cuda.is_initialized() or torch.version.hip != spec['expected_HIP']:
        raise ValueError('fresh actual HIP CPU runtime required')
    def forbid(*a, **k):
        raise RuntimeError('device initialization forbidden during meta validation')
    torch.cuda._lazy_init = forbid
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.source_root.resolve()))
    native = importlib.import_module('diffsynth.models.z_image_dit')
    training = importlib.import_module('diffsynth.diffusion.training_module')
    lora = importlib.import_module('diffsynth.utils.lora.general')
    check_deadline(args.epoch, args.deadline)
    with torch.device('meta'):
        model = native.ZImageDiT()
    base = {n: v for shard in shards for n, v in shard.items()}
    state = model.state_dict()
    if set(state) != set(base) or len(state) != 521:
        raise ValueError('full native DiT state keys differ from original')
    if not all(t.is_meta and list(t.shape) == base[n]['shape'] and t.dtype == torch.float32 for n, t in state.items()):
        raise ValueError('full native DiT shape/dtype differs')
    if sum(p.numel() for p in model.parameters()) != 6154908736:
        raise ValueError('full native parameter count differs')
    names = lora.GeneralLoRALoader(device='cpu', torch_dtype=torch.bfloat16).get_name_dict(preset)
    if set(names) != set(expected['preset_dimensions']) or len(names) != 240:
        raise ValueError('native preset target mapping differs')
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear) or list(module.weight.shape) != expected['preset_dimensions'][name]['base']:
            raise ValueError('native preset target shape differs')
    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    helper = training.DiffusionTrainingModule()
    targets = helper.parse_lora_target_modules(model, spec['target_modules'])
    model = helper.add_lora_to_model(model, target_modules=targets, lora_rank=4, upcast_dtype=torch.bfloat16)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    reference = {}
    for n, shape in expected['new_target_shapes'].items():
        reference[n + '.lora_A.default.weight'] = [4, shape[1]]
        reference[n + '.lora_B.default.weight'] = [shape[0], 4]
        module = model.get_submodule(n)
        if module.r['default'] != 4 or module.lora_alpha['default'] != 4 or module.scaling['default'] != 1 or getattr(module.lora_dropout['default'], 'p', 0) != 0:
            raise ValueError('native rank/alpha/scaling/dropout differs')
    if set(trainable) != set(reference) or len(trainable) != 476 or sum(p.numel() for p in trainable.values()) != 9922560:
        raise ValueError('native trainable scope differs')
    if not all(p.is_meta and p.dtype == torch.bfloat16 and list(p.shape) == reference[n] for n, p in trainable.items()):
        raise ValueError('native trainable shape/dtype or storage differs')
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    if len(frozen) != 521 or sum(p.numel() for p in frozen.values()) != 6154908736 or not all(p.is_meta and p.dtype == torch.bfloat16 for p in frozen.values()):
        raise ValueError('native frozen boundary differs')
    imported = {}
    for name, module in sorted(sys.modules.items()):
        if (name == 'diffsynth' or name.startswith('diffsynth.')) and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(args.source_root.resolve()).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if verified.get(relative) != digest:
                raise ValueError('unpinned native import provider')
            imported[name] = dict(path=relative, SHA256=digest)
    attention = importlib.import_module('diffsynth.core.attention.attention')
    selected = attention.ATTENTION_IMPLEMENTATION
    flash = sys.modules.get('flash_attn')
    flash_api = None
    if flash is not None:
        function = flash.flash_attn_func
        provider = Path(inspect.getsourcefile(function))
        flash_api = dict(signature=str(inspect.signature(function)), version=metadata.version('flash-attn'),
                         provider_SHA256=hashlib.sha256(provider.read_bytes()).hexdigest(), operator_called=False)
    check_deadline(args.epoch, args.deadline)
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected GPU initialization')
    value = dict(passed=True, qualification='FULL_NATIVE_HIP_CPU_META_STRUCTURE', base_tensors=521,
                 base_elements=6154908736, preset_linear_targets=240, trainable_tensors=476, trainable_elements=9922560,
                 frozen_tensors=521, frozen_elements=6154908736, native_rank=4, native_alpha=4,
                 all_parameters_meta=True, all_parameters_BF16=True, native_import_providers=imported,
                 attention_selected=selected, passive_flash_API=flash_api, production_attention_qualified=False,
                 CUDA_initialized=False, HIP=torch.version.hip, weights_or_data_bytes_read=0, GPU_cards=0,
                 actual_fusion=False, native_forward_backward_updates=0, numerical_alignment=False,
                 formal_performance=False, heldout_effect=False, Model_GO=False,
                 versions={n: metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')},
                 epoch=args.epoch, deadline=args.deadline, wall_seconds=time.monotonic() - started)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    value = run(parser.parse_args())
    print(json.dumps({k: value[k] for k in ('passed', 'qualification', 'trainable_tensors', 'trainable_elements', 'Model_GO')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
