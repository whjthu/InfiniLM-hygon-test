"""Record a bootstrap owner before I/O and bind verified source without copying it."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import time
from typing import Callable


def linux_owner() -> dict:
    fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    return dict(PID=os.getpid(), PGID=os.getpgrp(), startticks=int(fields[19]),
                host=socket.gethostname(), boot_id_SHA256=hashlib.sha256(
                    Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest())


class OriginalDeadline:
    def __init__(self, epoch: float, deadline: float, *, wall=time.time, monotonic=time.monotonic):
        if (not all(math.isfinite(x) for x in (epoch, deadline))
                or epoch <= 0 or not 0 < deadline - epoch <= 86400):
            raise ValueError('finite original bootstrap epoch required')
        self.epoch, self.deadline = epoch, deadline
        self.wall, self.monotonic = wall, monotonic
        self.anchor = monotonic()
        self.elapsed = wall() - epoch
        self.check()

    def check(self) -> None:
        now = self.wall()
        remaining = min(self.deadline - now,
                        self.deadline - self.epoch - self.elapsed - (self.monotonic() - self.anchor))
        if not math.isfinite(now) or now < self.epoch or not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError('original bootstrap deadline exhausted')


def exclusive_json(path: Path, value: dict) -> dict:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def bind_frozen_framework(
    workspace: Path, *, request_id: str, claim_commit: str, source_manifest_SHA256: str,
    framework_root: Path, framework_manifest: bytes, framework_manifest_SHA256: str,
    epoch: float, deadline: float, wall: Callable = time.time,
    monotonic: Callable = time.monotonic, owner_reader: Callable = linux_owner,
) -> dict:
    """Create one durable owner receipt, then verify a read-only source reference.

    This helper never launches a process, initializes an accelerator or resumes an
    unknown request. A missing terminal receipt continues to mean UNKNOWN. The
    caller must bind returned source read-only and still verify real admission,
    execution accounting and cleanup separately.
    """
    if (not re.fullmatch('[0-9a-f]{32}', request_id)
            or not re.fullmatch('[0-9a-f]{40}', claim_commit)
            or not re.fullmatch('[0-9a-f]{64}', source_manifest_SHA256)
            or not re.fullmatch('[0-9a-f]{64}', framework_manifest_SHA256)):
        raise ValueError('exact request, source and manifest identities required')
    workspace, framework_root = Path(workspace), Path(framework_root)
    clock = OriginalDeadline(epoch, deadline, wall=wall, monotonic=monotonic)
    owner = owner_reader()
    if (set(owner) != {'PID', 'PGID', 'startticks', 'host', 'boot_id_SHA256'}
            or any(not isinstance(owner[k], int) or owner[k] <= 0 for k in ['PID', 'PGID', 'startticks'])
            or not isinstance(owner['host'], str) or not owner['host']
            or not re.fullmatch('[0-9a-f]{64}', owner['boot_id_SHA256'])):
        raise ValueError('actual installer identity required')
    if not workspace.is_absolute() or workspace.parent.is_symlink() or not workspace.parent.is_dir():
        raise ValueError('explicit existing workspace parent required')
    workspace.mkdir(mode=0o700)  # Existing or partial requests are never overwritten.
    bootstrap = dict(request_id=request_id, claim_commit=claim_commit,
                     source_manifest_SHA256=source_manifest_SHA256,
                     epoch=epoch, deadline=deadline, owner=owner,
                     phase='BOOTSTRAP_CREATED', execution_result='UNKNOWN')
    completed, verified_bytes = 0, 0
    try:
        bootstrap_pin = exclusive_json(workspace / 'bootstrap.json', bootstrap)
        clock.check()
        if hashlib.sha256(framework_manifest).hexdigest() != framework_manifest_SHA256:
            raise ValueError('frozen framework manifest differs')
        manifest = json.loads(framework_manifest)
        files = manifest['files']
        if not isinstance(files, dict) or not 0 < len(files) <= 10000:
            raise ValueError('bounded nonempty source manifest required')
        if not framework_root.is_absolute() or framework_root.is_symlink():
            raise ValueError('explicit frozen framework directory required')
        root = framework_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError('frozen framework root is not a directory')
        for name, pin in files.items():
            clock.check()
            part = PurePosixPath(name)
            if (part.is_absolute() or '..' in part.parts or str(part) != name
                    or '\\' in name or not part.parts):
                raise ValueError('noncanonical source path')
            size, digest, mode = pin['bytes'], pin['SHA256'], pin['mode']
            if (type(size) is not int or not 0 <= size <= 64 * 1024**2
                    or not re.fullmatch('[0-9a-f]{64}', digest)
                    or type(mode) is not int or mode not in [0o644, 0o755, 0o666]):
                raise ValueError('invalid bounded source identity')
            verified_bytes += size
            if verified_bytes > 256 * 1024**2:
                raise ValueError('source byte budget exceeded')
            path = root.joinpath(*part.parts)
            if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
                raise ValueError('symlink in frozen source')
            if not path.is_file() or path.resolve(strict=True).parent != path.parent:
                raise ValueError('frozen source path differs')
            before = path.stat()
            if before.st_size != size or before.st_mode & 0o777 != mode:
                raise ValueError('frozen source extent or mode differs')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b''):
                    clock.check()
                    h.update(block)
            after = path.stat()
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or h.hexdigest() != digest):
                raise ValueError('frozen source bytes changed')
            completed += 1
        clock.check()
        reference = dict(request_id=request_id, claim_commit=claim_commit,
                         bootstrap_pin=bootstrap_pin, framework_root=str(root),
                         framework_manifest_SHA256=framework_manifest_SHA256,
                         files_verified=completed, bytes_verified=verified_bytes,
                         framework_files_copied=0, required_bind_mode='read-only',
                         epoch=epoch, deadline=deadline, execution_result='UNKNOWN')
        exclusive_json(workspace / 'framework-reference.json', reference)
        return reference
    except BaseException as error:
        # Preserve the first owner even when verification, storage or deadlines fail.
        exclusive_json(workspace / 'bootstrap-failure.json',
                       dict(request_id=request_id, claim_commit=claim_commit, owner=owner,
                            epoch=epoch, deadline=deadline, error_type=type(error).__name__,
                            files_verified=completed, execution_result='UNKNOWN'))
        raise
