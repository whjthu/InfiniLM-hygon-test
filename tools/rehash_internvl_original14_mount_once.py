"""Rebind the exact 14 InternVL members after a mount-device identity change."""

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PRECHECK = ROOT / 'incoming/training-internvl-modelroot17-preflight-635df739726b41c29a71aceff6569076'
RUN = ROOT / 'results/training-internvl-original14-current-mount-rehash-CPU-v1'
REQUEST = 'aba88a0dd32848a097c851c8b138e222'
ASSEMBLER_SHA = 'ebff2c2df8665cde980758aa0a8bfb06e5edf65cd71d1346bde2fc432315d3d6'
INSPECT_SHA = '8406d84711e4002eec82383fb5e84911b7bb407c7e7e830bab28336fbdee79ee'
AUTHORITY_SHA = '8aea495d35a53a311b40a5d2b09379b8e76ee7d5cca68557496a6eb18dca5a49'
BYTE_COUNT = 4_922_323_586
LIMIT = 195


def hash_original_member(path, expected_size, expected_digest, original, *, stamp):
    path = Path(path)
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError('expected the exact original canonical path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb', buffering=0) as stream:
        before = stamp(os.fstat(fd))
        changed = {field for field in before if before[field] != original[field]}
        if (changed != {'st_dev'} or before != stamp(path.lstat())
                or before['st_size'] != expected_size or not stat.S_ISREG(before['st_mode'])):
            raise ValueError('original mtimes, inode, size or ownership changed beyond the checked mount difference')
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            block = stream.read(min(2 * 1024**2, remaining))
            if not block:
                raise ValueError('original member became shorter')
            digest.update(block)
            remaining -= len(block)
        if stream.read(1) or before != stamp(os.fstat(fd)) or before != stamp(path.lstat()):
            raise ValueError('original member changed while reading')
    if digest.hexdigest() != expected_digest:
        raise ValueError('original member SHA differs on the current mount')
    return dict(bytes=expected_size, SHA256=expected_digest, current_target_stat=before)


def main():
    if os.getuid() != 154459 or len(sys.argv) != 1:
        raise ValueError('expected the project owner and the fixed one-shot scope')
    deadline = time.monotonic() + LIMIT

    def timeout(*_):
        raise TimeoutError('original 195-second rehash deadline')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMIT)
    try:
        module = PRECHECK / 'assemble.py'
        raw = module.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ASSEMBLER_SHA:
            raise ValueError('accepted assembler source differs')
        sys.path.insert(0, str(PRECHECK))
        import assemble as assembly

        inspected, _ = assembly.document(PRECHECK / 'inspect-private.json', INSPECT_SHA, 65536)
        authority, _ = assembly.document(PRECHECK / 'three-original-seals.json', AUTHORITY_SHA, 65536)
        assets, _ = assembly.document(PRECHECK / 'ASSETS.json', assembly.ASSETS_SHA, 65536)
        outcome, _ = assembly.document(assembly.ORIGINAL_361, assembly.OUTCOME_SHA, assembly.OUTCOME_BYTES,
                                       extent=assembly.OUTCOME_BYTES)
        sealed = assembly.source_receipts(assets, authority)
        originals = assembly.old_rows(outcome, assets['old14'])
        inspected_rows = {row['name']:row for row in inspected['required_targeted_rehash']}
        if (inspected['state'] != 'OLD361_IDENTITY_CHANGED_NEEDS_TARGETED_REHASH'
                or len(sealed) != 3 or len(originals) != 14 or set(inspected_rows) != set(originals)
                or sum(row['bytes'] for row in assets['old14'].values()) != BYTE_COUNT
                or any(row['differing_fields'] != ['st_dev'] for row in inspected_rows.values())):
            raise ValueError('exact original 14-member rehash scope differs')
        if not all(inspected_rows[name]['source'] == row['source']
                   and inspected_rows[name]['bytes'] == row['bytes']
                   and inspected_rows[name]['SHA256'] == row['SHA256']
                   for name, row in assets['old14'].items()):
            raise ValueError('rehash list and declared original assets differ')
        if time.monotonic() >= deadline:
            raise TimeoutError('rehash preflight exhausted the original budget')
        os.umask(0o077)
        RUN.mkdir(mode=0o700)
        identity = dict(request=REQUEST, host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        authority_SHA256=AUTHORITY_SHA, inspection_SHA256=INSPECT_SHA, old361_outcome_SHA256=assembly.OUTCOME_SHA,
                        planned_members=14, planned_body_bytes=BYTE_COUNT, new3_weight_body_bytes=0, actual_GPU_steps=0)
        assembly.write_once(RUN/'intent.json', assembly.serialized(identity))
        checked = {}
        failure = None
        try:
            for name,row in sorted(assets['old14'].items()):
                checked[name] = hash_original_member(row['source'], row['bytes'], row['SHA256'],
                                                     originals[name], stamp=assembly.stamp)
                assembly.write_once(RUN/(name+'.verified.json'), assembly.serialized(checked[name]))
        except (OSError, ValueError) as exc:
            failure = type(exc).__name__
        finally:
            signal.alarm(0)
            passed = failure is None and len(checked) == 14
            result = dict(identity, state='CURRENT_MOUNT_ALL14_WHOLE_SHA_MATCHED' if passed else 'CURRENT_MOUNT_REHASH_FAILED',
                          verified_members=checked, actual_verified_members=len(checked), failure_type=failure,
                          actual_verified_body_bytes=sum(row['bytes'] for row in checked.values()), training_READY=False,
                          modelroot_prepared=False)
            body = assembly.serialized(result)
            assembly.write_once(RUN/'outcome.json', body)
        print(json.dumps({k:v for k,v in result.items() if k not in ('verified_members','host','boot_id')}
                         | {'receipt_SHA256':hashlib.sha256(body).hexdigest()}), flush=True)
        return int(not passed)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
