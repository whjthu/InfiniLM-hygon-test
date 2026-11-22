"""Read checkpoint identity with bounded memory."""
from hashlib import sha256
import os
import stat


def checkpoint_sha256(path):
    if path.resolve(strict=True) != path:
        raise ValueError("Expected the unaliased checkpoint")
    digest = sha256()
    path_before = path.stat(follow_symlinks=False)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 8 << 30:
            raise ValueError("Checkpoint has an unexpected file type or size")
        remaining = before.st_size
        while remaining:
            chunk = stream.read(min(remaining, 4 << 20))
            if not chunk:
                raise ValueError("Checkpoint shortened during identity read")
            remaining -= len(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(before, key) != getattr(after, key) for key in fields)
                or any(getattr(path_before, key) != getattr(current, key) for key in fields)
                or (os.name == "posix" and not os.path.samestat(before, current))):
            raise ValueError("Checkpoint changed during identity read")
    return digest.hexdigest()
