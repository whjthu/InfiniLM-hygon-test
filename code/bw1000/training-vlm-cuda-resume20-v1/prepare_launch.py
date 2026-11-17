"""Prepare one common-entry command inside a NEW identity-owned A100 supervisor.

This module does not spawn a process. The caller reuses qualified owned_processes
cleanup/deadline code and holds this new PID live until native exit and postcheck.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


D = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def owner(request):
    return dict(PID=os.getpid(), uid=os.getuid(), host=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                scope_request_id=request)


def prepare(case, *, framework, scope, canonical_reference, request_id, epoch, native_deadline, publication):
    """Call once before Popen, within the already authorized new case supervisor."""
    require(case in {'T03.sft', 'T04.sft'} and epoch <= time.time() < native_deadline <= epoch + 900,
            'new CUDA scope case/epoch differs')
    require(len(request_id) == 32 and not set(request_id) - set('0123456789abcdef')
            and len(publication) == 40 and not set(publication) - set('0123456789abcdef'), 'new CUDA scope context differs')
    source_manifest = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    for name, pin in source_manifest['files'].items():
        path = D / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
                and sha(path.read_bytes()) == pin['SHA256'], 'published CUDA source member differs')
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    require(claim['root_GO'] is True and claim['GPU_GO'] is True, 'CUDA source still requires root review/publication')
    framework, scope = Path(framework), Path(scope)
    require(framework.is_absolute() and scope.is_absolute() and not framework.is_symlink() and not scope.is_symlink()
            and scope.is_dir() and scope.stat().st_uid == os.getuid(), 'new CUDA installed scope differs')
    mf = json.loads((D / 'MERGED_FRAMEWORK_FILES.json').read_bytes())
    require(mf['base_framework_commit'] == 'a68952cb2285d6593b665147249d14ec1f8cecbc' and len(mf['files']) == 396,
            'complete common framework identity differs')
    for name, pin in mf['files'].items():
        path = framework / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
                and sha(path.read_bytes()) == pin['SHA256'], 'complete common framework source differs')
    template = json.loads((D / 'cases' / f'{case}-config-template.json').read_bytes())
    config = dict(template, root_GO=True, GPU_GO=True, epoch=epoch, deadline=native_deadline,
                  request_id=request_id, source_publication=publication,
                  output=str(scope / 'resume-witness'), supervisor=owner(request_id),
                  binding_path=str(D / 'cases' / f'{case}-native-binding.json'),
                  canonical64_reference_staged_path=str(canonical_reference))
    config_path = scope / 'cuda-resume-runtime-config.json'
    raw = (json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(raw) <= 65536, 'runtime CUDA config size differs')
    with config_path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    original = json.loads((D / 'cases' / f'{case}-original-launch.json').read_bytes())
    model_files = json.loads((D / 'cases' / f'{case}-model-files.json').read_bytes())
    dataset = original['command'][original['command'].index('--dataset') + 1]
    port = 29911 if case == 'T03.sft' else 29913
    command = [sys.executable, str(framework / 'tools/hygon_training_recipe.py'), case,
               '--a100-launch', str(D / 'cases' / f'{case}-original-launch.json'),
               '--swift', '/home/whj/nvidia-baseline/env-vlm/bin/swift',
               '--models', str(Path(model_files['model_root']).parent), '--dataset', dataset,
               '--plugins', '/home/whj/nvidia-baseline', '--output', str(scope / 'train'),
               '--master-port', str(port), '--diagnostic-plugin', str(framework / 'vlm/plugins/cuda_resume_training.py'),
               '--diagnostic-callback', 'native_cuda_resume_witness', '--resume-from-checkpoint', config['checkpoint'],
               '--run', '--receipt', str(scope / 'common-entry-receipt.jsonl')]
    environment = os.environ.copy()
    environment.update(PATH=original['environment']['PATH'], PYTHONPATH=str(framework) + os.pathsep + str(D),
                       NATIVE_CUDA_RESUME_GO='1', NATIVE_RESUME_CONFIG=str(config_path),
                       NATIVE_RESUME_CONFIG_SHA256=sha(raw), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    return dict(command=command, environment=environment, config_path=str(config_path), config_SHA256=sha(raw),
                request_id=request_id, epoch=epoch, native_deadline=native_deadline,
                supervisor=config['supervisor'], complete_framework_files=396,
                ownership_cleanup_required=True, actual_GPU_launched=False)
