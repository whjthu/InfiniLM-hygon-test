"""Record the exact object backing the HIP calls already used for device assignment."""

import ctypes
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import stat


NAME = re.compile(r"lib(?:amdhip64|galaxyhip)\.so(?:\.[0-9]+)*(?:-[0-9a-f]+)?")
SYMBOLS = ("hipGetDeviceCount", "hipDeviceGetPCIBusId")
MAX_BYTES = 256 * 1024**2


def symbols(hip):
    return tuple(ctypes.cast(getattr(hip, name), ctypes.c_void_p).value for name in SYMBOLS)


def mapped_identity(raw, addresses):
    if len(addresses) != len(SYMBOLS) or any(type(addr) is not int or addr <= 0 for addr in addresses):
        raise ValueError("expected the exact existing HIP handle symbol addresses")
    matches = [[] for _ in addresses]
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or "x" not in fields[1]:
            continue
        low, high = (int(value, 16) for value in fields[0].split("-"))
        for index, addr in enumerate(addresses):
            if low <= addr < high:
                if len(fields) != 6:
                    raise ValueError("HIP symbol has no file-backed mapping")
                source = fields[5]
                path = PurePosixPath(source)
                if not path.is_absolute() or path.as_posix() != source or not NAME.fullmatch(path.name):
                    raise ValueError("HIP symbol mapping has no supported live provider filename")
                dev = tuple(int(part, 16) for part in fields[3].split(":"))
                inode = int(fields[4])
                if len(dev) != 2 or inode <= 0:
                    raise ValueError("HIP symbol mapping has no stable file identity")
                matches[index].append((source, dev, inode))
    if any(len(candidates) != 1 for candidates in matches) or len({items[0] for items in matches}) != 1:
        raise ValueError("both HIP symbols must share exactly one file-backed executable object")
    return matches[0][0]


def fingerprint(source, device, inode):
    path = Path(source)
    if path.resolve(strict=True) != path:
        raise ValueError("HIP provider path is non-canonical")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or (os.major(before.st_dev), os.minor(before.st_dev)) != device \
                or before.st_ino != inode or not 0 < before.st_size <= MAX_BYTES:
            raise ValueError("HIP provider file differs from the actual mapped object")
        digest = sha256()
        size = 0
        while chunk := stream.read(min(1024**2, before.st_size + 1 - size)):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    current = path.stat(follow_symlinks=False)
    if size != before.st_size or any(getattr(before, key) != getattr(info, key)
                                     for key in keys for info in (after, current)):
        raise ValueError("HIP provider changed during readback")
    return {"path": source, "bytes": size, "SHA256": digest.hexdigest()}


def hip_dso(hip):
    identity = mapped_identity(Path("/proc/self/maps").read_text(), symbols(hip))
    result = fingerprint(*identity)
    if mapped_identity(Path("/proc/self/maps").read_text(), symbols(hip)) != identity:
        raise ValueError("HIP provider mapping changed during readback")
    return result
