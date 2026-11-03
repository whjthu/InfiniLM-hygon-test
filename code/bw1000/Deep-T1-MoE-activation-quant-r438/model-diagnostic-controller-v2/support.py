"""One diagnostic residency; no HTTP or automatic replay of the old Model run."""
from pathlib import Path
import hashlib
import json
import os
import re

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D = R / 'incoming/Deep-T1-MoE-activation-quant-r438/model-diagnostic-controller-v2'
RUN = 'capture-20261104-a'
LABEL = 'Deep-r438-fixed-prefix-diagnostic-v2'
NODE = 'a02r4n01'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n');out.flush();os.fsync(out.fileno())


def clean_environment(base):
    return {k: v for k, v in base.items()
            if k not in ['BASH_ENV', 'ENV', 'PYTHONSTARTUP', 'PYTHONINSPECT']
            and (not k.startswith(('SLURM_', 'PMI_', 'PMIX_', 'OMPI_', 'CUDA_', 'HIP_',
                                  'ROCR_', 'GPU_', 'GLM_', 'DEEP_', 'INFINILM_DEEP_'))
                 or k in ['SLURM_CONF', 'SLURM_CONF_SERVER'])}


def verify_tree(root, expected):
    root = Path(root)
    if root == Path('/refactor') or root.is_relative_to('/refactor'):
        root = R / root.relative_to('/refactor')
    root = root.resolve()
    assert sha(root / 'SOURCE_MANIFEST.json') == expected
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_bytes())
    for name, digest in manifest['files_SHA256'].items():
        item = root / name
        assert item.resolve().is_relative_to(root) and sha(item) == digest, name
    return manifest


def verify(expected, *, require_ready=False):
    m = verify_tree(D, expected)
    if not require_ready:
        return m
    assert m['execution_READY'] and m['GPU_GO'] and not m['performance_qualification']
    for name in ['binding-pin.json', 'config.json', 'ENTRY_CPU.json']:
        assert name in m['files_SHA256']
    cfg = json.loads((D / 'config.json').read_bytes())
    pin = json.loads((D / 'binding-pin.json').read_bytes())
    cpu = json.loads((D / 'ENTRY_CPU.json').read_bytes())
    assert cpu['passed'] and cpu['actual_input_processor'] and cpu['actual_Native_derivation']
    assert not cpu['Native_dlopen'] and not cpu['GPU'] and not cpu['Model_READY']
    assert cpu['binding_pin_SHA256'] == sha(D / 'binding-pin.json')
    assert cpu['entry_manifest_SHA256'] == m['entry_manifest_SHA256']
    assert pin['CPU_READY'] and pin['actual_CPU'] and pin['library_SHA256']
    assert cfg['run'] == RUN and cfg['node'] == NODE
    assert cfg['gpu_start_epoch'] is None and cfg['actual_step'] is None
    assert cfg['execute_authorized'] and cfg['Native_seconds'] == 2700
    assert cfg['CPUkeeper_seconds'] == 3000 and cfg['HTTP'] == cfg['free_generation'] == 0
    assert cfg['modes'] == [0, 1, 0, 1] and cfg['max_forwards'] == 368
    assert cfg['capture_outputs'] == [82, 83] and cfg['raw_limit'] == 134217728
    assert cfg['Native_SHA256'] == pin['library_SHA256']
    assert cfg['output_root'] == '/refactor/results/Deep438-diagnostic-v2-private/' + RUN
    entry = R / 'incoming/Deep-T1-MoE-activation-quant-r438/model-diagnostic-entry-v2'
    verify_tree(entry, m['entry_manifest_SHA256'])
    diag = R / 'incoming/Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2'
    verify_tree(diag, pin['source_manifest_SHA256'])
    for root, h in cpu['verified_source_manifests'].items():
        verify_tree(root, h)
    return m
