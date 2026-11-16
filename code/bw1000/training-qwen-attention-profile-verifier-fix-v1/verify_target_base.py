"""Verify the deployed Qwen attention profile base without running its launcher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import time

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
BASE_NAME = 'training-qwen-continuous-branches21-hip-v2'
LABEL = 'training-qwen-resumed-attention-profile21-hip-v1'
ORIGINAL_REQUEST = '42f48d022f19477b88d3d5ea32ce82d8'
BASE_COMMIT = '096157e7b63f7e9cd60cb53f31b10f2810ac2ddc'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'
LIMIT = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_pin(path, digest, expected_size=None, mode=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= LIMIT, f'invalid size/type: {path.name}')
        require(expected_size in (None, before.st_size), f'wrong size: {path.name}')
        if mode is not None:
            require(stat.S_IMODE(before.st_mode) == mode, f'wrong mode: {path.name}')
        data = stream.read(before.st_size + 1)
        require(identity(before) == identity(os.fstat(stream.fileno())) == identity(path.lstat()),
                f'identity changed during read: {path.name}')
    require(len(data) == before.st_size and sha256(data).hexdigest() == digest, f'wrong SHA-256: {path.name}')
    return data


def list_files(root):
    require(root.is_dir() and not root.is_symlink(), 'deployed framework root must be a directory')
    pending = [root]
    names = set()
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    require(entry.is_file(follow_symlinks=False), 'unexpected source file type')
                    names.add(path.relative_to(root).as_posix())
    return names


def verify_base(root, inputs):
    base = root / BASE_NAME
    manifest_pin = inputs['framework_manifest']
    terminal_pin = inputs['terminal']['SHA256']
    read_pin(base / 'outcome.json', terminal_pin)
    manifest = json.loads(read_pin(base / 'src/FRAMEWORK.json', manifest_pin['SHA256'], manifest_pin['bytes']))
    pins = manifest['files']
    require(manifest['commit'] == BASE_COMMIT and len(pins) == 406, 'wrong pinned base revision')
    deployed = base / 'framework'
    require(list_files(deployed) == set(pins), 'deployed source inventory differs')

    def one(row):
        name, pin = row
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and relative.parts and '..' not in relative.parts
                and relative.as_posix() == name and pin['mode'] in (0o644, 0o755), 'malformed source entry')
        return len(read_pin(deployed.joinpath(*relative.parts), pin['SHA256'], pin['bytes'], pin['mode']))

    # Limit NFS concurrency while avoiding hundreds of serial network round trips.
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked_bytes = sum(pool.map(one, pins.items()))
    return dict(base_commit=BASE_COMMIT, base_files=len(pins), base_source_bytes=checked_bytes,
                base_manifest_SHA256=manifest_pin['SHA256'], base_terminal_SHA256=terminal_pin,
                linux_modes_verified=True)


def original_status(root):
    output = subprocess.check_output(['sacct', '-j', '1087656', '-n', '-P', '--format=' + COLUMNS],
                                     text=True, timeout=12)
    labels = {LABEL + suffix for suffix in ('', '-pre', '-post', '-audit')}
    rows = [row for row in (line.split('|') for line in output.splitlines())
            if len(row) == 6 and row[1] in labels]
    scope = root / LABEL
    files = {}
    for name in ('intent.json', 'keeper-owner.json', 'outcome.json'):
        path = scope / name
        if path.is_file():
            with path.open('rb') as stream:
                raw = stream.read(65537)
            require(len(raw) <= 65536, 'oversized original receipt')
            receipt = json.loads(raw)
            files[name] = dict(SHA256=sha256(raw).hexdigest(),
                               request_matches=receipt.get('request_id', receipt.get('request')) == ORIGINAL_REQUEST)
    return dict(original_request=ORIGINAL_REQUEST, original_namespace_present=scope.exists(),
                accounting_columns=COLUMNS.split(','), original_exact_accounting=rows, original_known_receipts=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('inputs_sha256')
    args = parser.parse_args()
    inputs = json.loads(read_pin(args.inputs, args.inputs_sha256))
    result = dict(**verify_base(ROOT, inputs), **original_status(ROOT), observed_epoch=time.time(),
                  weight_data_bytes_read=0, checkpoint_data_bytes_read=0, new_Slurm_steps=0)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
