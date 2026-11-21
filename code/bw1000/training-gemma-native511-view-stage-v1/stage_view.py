"""Exclusive byte-only bridge from the qualified Gemma VIEW to the training RUN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

BASE = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/'
SOURCE = BASE + 'training-gemma-usgs-render72-native-CPU-v1/view'
TARGET = BASE + 'training-gemma-usgs-render72-exposure25-resume20-v1'
REQUEST = '92be758e47fd412993ced28dfe9648fb'
FILES = {
    'train800-swift.jsonl': {'bytes': 602794, 'SHA256': '74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc'},
    'exposure-ledger-private.json': {'bytes': 8196, 'SHA256': 'f2b661c7375b2a418b3aad8c901848807bd4cfe301496aade021b0242ce83a44'},
}
TREE = '6221bf33ebc85c180ace49273b67c709e007d4e5'
FORBIDDEN = ('intent.json', 'controller-entered.json', 'terminal-accounting.json',
             'gpu-step-binding.json', 'continuous', 'resumed', 'view', 'view-stage-attempt.json')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return {'bytes': len(raw), 'SHA256': hashlib.sha256(raw).hexdigest()}


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def read_exact(path, expected):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not path.is_symlink()
         and path.resolve(strict=True) == path and before.st_size == expected['bytes'], 'Unsafe source')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(expected['bytes'] + 1)
        after = os.fstat(stream.fileno())
    need(identity(before) == identity(opened) == identity(after) == identity(path.lstat()), 'Source changed')
    need(pin(raw) == expected, 'Source bytes/SHA differs')
    return raw


def write_exclusive(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    need(read_exact(path, pin(raw)) == raw, 'Destination differs')


def read_small_receipt(path):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not path.is_symlink()
         and path.resolve(strict=True) == path and before.st_size <= 4096, 'Unsafe framework receipt')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(4097)
        after = os.fstat(stream.fileno())
    need(len(raw) <= 4096 and identity(before) == identity(opened) == identity(after)
         == identity(path.lstat()), 'Framework receipt changed')
    return json.loads(raw)


def copy_view(source, target, installation, check):
    """Reusable IO contract; local tests explicitly use a synthetic installation record."""
    need(source.resolve(strict=True) == source and source.is_dir(), 'Unsafe view source root')
    need(target.resolve(strict=True) == target and target.is_dir(), 'Unsafe training root')
    need(installation.get('original_base_files_read') == 393
         and installation.get('installed_files_verified') == 396
         and installation.get('candidate_framework_tree') == TREE
         and installation.get('training_ready') is False, 'Wrong framework installation receipt')
    for name in FORBIDDEN:
        need(not (target / name).exists() and not (target / name).is_symlink(), 'Existing run requires reconciliation')
    # The attempt precedes reads/writes. Any subsequent failure remains in this exact namespace.
    check()
    write_exclusive(target / 'view-stage-attempt.json', json.dumps({
        'request_id': REQUEST, 'source': str(source), 'target': str(target),
        'training_READY': False, 'GPU_steps': 0}, sort_keys=True).encode() + b'\n')
    raw = {}
    for name, expected in FILES.items():
        check()
        raw[name] = read_exact(source / name, expected)
    need(raw['train800-swift.jsonl'].endswith(b'\n')
         and raw['train800-swift.jsonl'].count(b'\n') == 800, 'Incomplete 800-row view')
    (target / 'view').mkdir(mode=0o700)
    for name, body in raw.items():
        check()
        write_exclusive(target / 'view' / name, body)
    check()
    result = dict(state='GEMMA_TRAIN_VIEW_BYTE_STAGE_COMPLETE', request_id=REQUEST, files=FILES,
                  source_unique_rows=72, exposure_rows=800, source_images=12,
                  initial_weights_read_or_authorized=False, native_encodes=0,
                  observed_rank_loader=False, optimizer_updates=0, training_READY=False,
                  original_model_path_compatibility='NOT_CHECKED', actual_800_supervised_tokens=None)
    write_exclusive(target / 'view-stage-complete.json', json.dumps(result, sort_keys=True).encode() + b'\n')
    return result


def main():
    origin = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-target', action='store_true')
    args = p.parse_args()
    if not args.install_target:
        print(json.dumps(dict(request_id=REQUEST, source=SOURCE, target=TARGET, files=FILES,
                              CPU_GO=False, GPU_GO=False, training_READY=False), sort_keys=True))
        return
    def check():
        need(time.monotonic() - origin < 30, 'Original 30-second view-copy budget expired')
    need(os.name == 'posix', 'Target POSIX execution only')
    claim = json.loads(Path(__file__).with_name('CLAIM.json').read_text(encoding='utf8'))
    need(claim.get('root_GO') is True and claim.get('CPU_GO') is True
         and claim.get('request_id') == REQUEST, 'Root has not adopted this new view-stage scope')
    check()
    target = Path(TARGET)
    receipt = read_small_receipt(target / 'installation-complete.json')
    print(json.dumps(copy_view(Path(SOURCE), target, receipt, check), sort_keys=True))


if __name__ == '__main__':
    main()
