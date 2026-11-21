"""Inspect the fixed SDXL style18 data and deployed common trainer without launching it."""

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import stat

from verify_training80_manifest import verify_package_members


REPO = Path(__file__).resolve().parents[1]
PACKAGE = 'code/bw1000/training-sdxl-USGS-style18-portable-entry-v1'
PORTABLE_MANIFEST = dict(path=PACKAGE + '/SOURCE_FILES.json',
                         SHA256='2e2d4bf0a5bec8ee55ac341a6803ed824e3efdf6746d97bd522d540c8f8994b2')
PORTABLE_ENTRY = dict(path=PACKAGE + '/entry.py',
                      SHA256='008d0b548370f0ec3bc3c9db19b19fc6dd98d9148dace39d73a519d76610431e')
DATA_REQUEST = '6eef89b3d3ce4842beb9c4fa31f08ed59'
ORIGINAL_A100_SHA = '15bf8b928f58d54858f8553ce51da4eaf302928bfd627b5a44da49b66c32b13c'
REQUIRED_FLAGS = frozenset('--' + flag for flag in (
    'pretrained_model_name_or_path', 'train_data_dir', 'output_dir', 'resolution', 'center_crop',
    'train_batch_size', 'gradient_accumulation_steps', 'max_train_steps', 'checkpointing_steps',
    'checkpoints_total_limit', 'learning_rate', 'lr_scheduler', 'lr_warmup_steps', 'mixed_precision', 'seed',
    'dataloader_num_workers', 'rank', 'report_to', 'gencuda_metrics_path', 'gencuda_checkpoint_steps',
))


def load_portable():
    source = REPO / PORTABLE_ENTRY['path']
    verify_package_members(REPO, PORTABLE_MANIFEST, required_files=[PORTABLE_ENTRY])
    spec = importlib.util.spec_from_file_location('sdxl_fixed_style18_portable_entry', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_regular(path, ceiling):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('expected a canonical absolute deployed artifact')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= ceiling:
        raise ValueError('deployed artifact is empty, irregular or oversized')
    with path.open('rb') as stream:
        raw = stream.read(ceiling + 1)
    after = path.stat()
    if len(raw) != before.st_size or (before.st_ino, before.st_mtime_ns) != (after.st_ino, after.st_mtime_ns):
        raise ValueError('deployed artifact changed during inspection')
    return raw


def options(source):
    tree = ast.parse(source)
    return {value.value for call in ast.walk(tree) if isinstance(call, ast.Call) and
            isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
            for value in call.args if isinstance(value, ast.Constant) and isinstance(value.value, str)}


def inspect_source(portable, source_root, backend_path, expected_trainer_sha):
    result = {}
    backend_available = backend_path.exists() or backend_path.is_symlink()
    if backend_available:
        backend = read_regular(backend_path, 1024 * 1024)
        result['observed_common_backend_SHA256'] = sha256(backend).hexdigest()
    trainer_path = source_root / portable.TRAINER_RELATIVE
    if not trainer_path.exists() and not trainer_path.is_symlink():
        return dict(result, state='DEPLOYED_TRAINER_MISSING')
    trainer = read_regular(trainer_path, 1024 * 1024)
    trainer_sha = sha256(trainer).hexdigest()
    result.update(observed_deployed_trainer_SHA256=trainer_sha, observed_deployed_trainer_bytes=len(trainer),
                  observed_trainer_matches_original_A100=trainer_sha == ORIGINAL_A100_SHA)
    if expected_trainer_sha is None:
        return dict(result, state='DEPLOYED_TRAINER_PIN_PENDING')
    if trainer_sha != expected_trainer_sha:
        return dict(result, state='DEPLOYED_TRAINER_PIN_MISMATCH', expected_deployed_trainer_SHA256=expected_trainer_sha)
    result['deployed_trainer_pin_matched'] = True
    if not backend_available:
        return dict(result, state='DEPLOYED_COMMON_BACKEND_MISSING')
    if result['observed_common_backend_SHA256'] not in portable.BACKEND_SHA256:
        return dict(result, state='DEPLOYED_COMMON_BACKEND_PIN_MISMATCH')
    missing = REQUIRED_FLAGS - options(trainer)
    if missing:
        return dict(result, state='DEPLOYED_TRAINER_CLI_INCOMPLETE', missing_frozen_flags=sorted(missing))
    return dict(result, state='DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING',
                declared_common_flags_checked=len(REQUIRED_FLAGS))


def inspect(portable, train_dir, source_root, backend_path, expected_trainer_sha=None):
    if not all(path.is_absolute() for path in (train_dir, source_root, backend_path)):
        raise ValueError('expected absolute train, source and backend paths')
    if expected_trainer_sha is not None and re.fullmatch('[0-9a-f]{64}', expected_trainer_sha) is None:
        raise ValueError('expected a separate lowercase SHA256 for the deployed Hygon trainer')
    result = dict(case='T06.domain-scientific-chart-style18', original_data_request=DATA_REQUEST,
                  portable_manifest_SHA256=PORTABLE_MANIFEST['SHA256'], data_binding_SHA256=portable.BINDING_SHA256,
                  native_training_subprocesses=0, optimizer_updates=0, actual_GPU_steps=0, training_ready_added=0,
                  original_T06_numeric_failure_retained=True, native_tokenizer_VAE_verified=False)
    source = inspect_source(portable, source_root, backend_path, expected_trainer_sha)
    source_state = source.pop('state')
    result.update(source, deployed_source_state=source_state)
    if not train_dir.exists():
        return dict(result, state='TARGET_TRAIN_DIRECTORY_MISSING', train_inputs_verified=False)
    if train_dir.resolve(strict=True) != train_dir:
        raise ValueError('expected the canonical fixed train directory')
    config = portable.bind_config(train_dir)
    params = {key: config[key] for key in ('resolution', 'lora_rank', 'max_train_steps', 'train_batch_size',
                                           'warmup_steps', 'extra_checkpoint_steps')}
    if source_state == 'DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING':
        source_state = 'TRAIN_ONLY_DATA_AND_DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING'
    return dict(result, state=source_state, train_inputs_verified=True, checked_train_images=12,
                development_images_read=0, train_dir_SHA256=sha256(config['train_data_dir'].encode()).hexdigest(),
                planned_config=params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--expected-hygon-trainer-sha256')
    args = parser.parse_args()
    result = inspect(load_portable(), args.train_dir, args.source_root, args.backend, args.expected_hygon_trainer_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
