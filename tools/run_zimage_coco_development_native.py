"""Load the audited Hygon Z-Image pipeline and render one frozen development shard."""

import argparse
import ctypes
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import sys
import time

from render_zimage_coco_development_pairs import execute_pairs, load_adapters, read_inputs, require, shard_plan, write_line


BINDING_SHA256 = '20ff9fe9e9c9c05083cb64ba42d184b0228b1c9c845d94b14862e045092d9326'
FRAMEWORK = {
    'gencuda.zimage_native_record': ('zimage_native_record.py', '20273da4fa308465ef2c1672895043c7e6b7621ee05c9ad14ee78e75129c7c71'),
    'gencuda.zimage_hip_replay_binding': ('zimage_hip_replay_binding.py', '498db55a3dd18b33e201bde702a168ed7343c1528b69e94bacc64c2750c17b1b'),
    'gencuda.hy_resume_checkpoint': ('hy_resume_checkpoint.py', 'aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7'),
    'gencuda.backends.zimage_training': ('backends/zimage_training.py', '6526be513a7b55977ede0dd176757da1530fffedaefa0080639e4191b1f49d31'),
    'gencuda.zimage_attention_policy': ('zimage_attention_policy.py', '2cb33cda080b5650f2c584d583b72b43f366f69da76e2c51c95069d6f5ce6eea'),
}
PCI = re.compile(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
BLOCK = 4 * 1024 * 1024


def checked_digest(path, expected_size, deadline):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), 'asset is missing or aliased')
    digest = sha256()
    count = 0
    with path.open('rb') as stream:
        while True:
            deadline()
            block = stream.read(BLOCK)
            if not block:
                break
            digest.update(block)
            count += len(block)
            require(count <= expected_size, 'asset exceeds frozen size')
    require(count == expected_size, 'asset differs from frozen size')
    return digest.hexdigest()


def verify_assets(binding, deadline):
    files = 0
    total = 0
    groups = ((Path(binding['source_root']), binding['source_files']),
              (Path(binding['model_root']), binding['model']['files']))
    for root, manifest in groups:
        for name, pin in manifest.items():
            member = Path(name)
            require(not member.is_absolute() and '..' not in member.parts, 'asset path escapes its root')
            require(checked_digest(root / member, pin['bytes'], deadline) == pin['SHA256'], 'asset digest differs: ' + name)
            files += 1
            total += pin['bytes']
    require(files == 671 and len(binding['source_files']) == 651, 'original source or runtime extent differs')
    return {'native_source_members': 651, 'runtime_members': 20, 'total_verified_bytes': total}


def verify_framework(root, deadline):
    root = Path(root).resolve(strict=True) / 'generation/src/gencuda'
    for name, (path, expected) in FRAMEWORK.items():
        member = root / path
        loaded = sys.modules.get(name)
        spec = importlib.util.find_spec(name) if loaded is None else None
        origin = getattr(loaded, '__file__', None) if loaded else getattr(spec, 'origin', None)
        require(origin is not None and Path(origin).resolve(strict=True) == member,
                'framework module resolves outside the verified tree: ' + name)
        size = member.stat().st_size
        require(0 < size < 1 << 20 and checked_digest(member, size, deadline) == expected,
                'framework module bytes differ: ' + name)


def result_file(path, report):
    with path.open('x', encoding='utf8') as stream:
        write_line(stream, report)


def run(args):
    plan = shard_plan(args.first, args.stop)
    now = time.time()
    require(math.isfinite(args.epoch) and math.isfinite(args.deadline) and args.epoch <= now < args.deadline
            and args.deadline - args.epoch <= 3600, 'invalid single Slurm deadline')
    local_deadline = time.monotonic() + (args.deadline - now)

    def check():
        if min(args.deadline - time.time(), local_deadline - time.monotonic()) <= 0:
            raise TimeoutError('Z-Image development budget expired')

    step = os.environ['SLURM_STEP_ID']
    require(step.isascii() and step.isdecimal() and args.parent == os.environ['SLURM_JOB_ID']
            and args.node == socket.gethostname().split('.')[0] and PCI.fullmatch(args.pci),
            'execution is outside the claimed Slurm node or step')
    output = args.output
    output.mkdir(mode=0o700)
    report = {'schema': 'training.zimage.coco.native_development_runtime.v1', 'plan': plan,
              'step': args.parent + '.' + step, 'node': args.node, 'passed': False,
              'formal_effect': False, 'actual_optimizer_updates': 0, 'phase': 'frozen-inputs'}
    result_file(output / 'invocation.json', report)

    def expired(*_):
        raise TimeoutError('Z-Image development watchdog expired')

    previous_handler = None
    try:
        previous_handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, local_deadline - time.monotonic()))
        with args.binding.open('rb') as stream:
            binding_raw = stream.read(116486)
        require(len(binding_raw) == 116485 and sha256(binding_raw).hexdigest() == BINDING_SHA256,
                'original native binding differs')
        binding = json.loads(binding_raw)
        rows, replay = read_inputs(args.prompts, args.noise_manifest, args.roster)
        check()
        report['phase'] = 'framework-and-native-assets'
        verify_framework(args.framework, check)
        from gencuda import zimage_hip_replay_binding as hip
        from gencuda import zimage_native_record as native
        from gencuda.backends import zimage_training as backend
        from gencuda.zimage_attention_policy import policy_for_config

        native.validate_binding(binding)
        require(binding['profile'] == hip.PROFILE and policy_for_config(binding['config']) is None,
                'original native-default Hygon variant required')
        report.update(verify_assets(binding, check))
        versions = {name: importlib.metadata.version(name) for name in binding['versions']}
        require(versions == binding['versions'], 'original runtime package versions differ')
        import numpy as np
        import torch

        report['phase'] = 'device-admission'
        require(torch.cuda.device_count() == 1 and torch.version.hip == binding['HIP_version'], 'unexpected Hygon visibility or runtime')
        architecture = str(torch.cuda.get_device_properties(0).gcnArchName).split(':')[0]
        require(architecture == binding['GPU_arch'], 'unexpected Hygon device architecture')
        address = ctypes.create_string_buffer(24)
        hip_runtime = ctypes.CDLL('libamdhip64.so')
        require(hip_runtime.hipDeviceGetPCIBusId(address, len(address), 0) == 0, 'physical PCI read failed')
        pci = address.value.decode('ascii').lower()
        if len(pci) == 7:
            pci = '0000:' + pci
        require(pci == args.pci, 'actual physical PCI differs from admission')
        free, total = torch.cuda.mem_get_info()
        require(total == binding['GPU_total_bytes'] and free >= binding['initial_minimum_free_bytes'], 'physical memory admission differs')
        torch.set_num_threads(2)
        torch.manual_seed(20260911)
        random.seed(20260911)
        np.random.seed(20260911)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        report.update(pci=pci, versions=versions, free_before_bytes=free, phase='native-pipeline-and-fused-base')
        source_root = Path(binding['source_root'])
        sys.path.insert(0, str(source_root))
        load_started = time.monotonic()
        factory = backend._training_module_class(source_root)
        model = factory(model_paths=json.dumps(binding['model_paths']), tokenizer_path=binding['tokenizer_path'],
                        lora_base_model='dit', lora_target_modules=binding['config']['lora_target_modules'],
                        lora_rank=4, preset_lora_path=binding['preset_path'], preset_lora_model='dit',
                        use_gradient_checkpointing=True, device='cuda', task='sft')
        torch.cuda.synchronize()
        report['load_seconds'] = time.monotonic() - load_started

        def tensor_sha(tensor):
            flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            digest = sha256()
            for offset in range(0, flat.numel(), BLOCK):
                check()
                digest.update(flat[offset:offset + BLOCK].cpu().numpy().tobytes())
            return digest.hexdigest()

        report['fused_base'] = hip.verify_fused_base(torch, model, binding, tensor_sha, check)
        report['phase'] = 'original-and-trained-adapters'
        named, states = load_adapters(model, torch, np, binding, args.checkpoint, deadline=check)
        model.eval()
        report['phase'] = 'paired-native-rendering'
        report['generation'] = execute_pairs(pipe=model.pipe, torch=torch, replay=replay, rows=rows,
                                             named=named, states=states, output=output / 'pairs',
                                             first=args.first, stop=args.stop, deadline=check)
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['passed'] = report['generation']['complete_pairs'] == plan['planned_pairs']
    except Exception as error:
        report['error_type'] = type(error).__name__
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        result_file(output / 'result.json', report)
    return int(not report['passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for path_arg in ('binding', 'checkpoint', 'prompts', 'noise-manifest', 'roster', 'output', 'framework'):
        parser.add_argument('--' + path_arg, type=Path, required=True)
    for text_arg in ('node', 'parent', 'pci'):
        parser.add_argument('--' + text_arg, required=True)
    for index in ('first', 'stop'):
        parser.add_argument('--' + index, type=int, required=True)
    for clock in ('epoch', 'deadline'):
        parser.add_argument('--' + clock, type=float, required=True)
    raise SystemExit(run(parser.parse_args()))
