"""Read only exact regular files; a pin never substitutes for a fresh body check."""
import hashlib
import json
import os
from pathlib import Path
import stat

from fresh_contract import need

FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def identity(s):
    return [getattr(s, k) for k in FIELDS]


def read_exact(path, pin, check, collect=True):
    path = Path(path)
    check()
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == pin["bytes"], "file kind/size differs")
    sha = hashlib.sha256()
    chunks = [] if collect else None
    total = 0
    with path.open("rb") as stream:
        need(identity(os.fstat(stream.fileno())) == identity(before), "opened identity changed")
        while True:
            check()
            raw = stream.read(min(1024**2, before.st_size - total + 1))
            if not raw:
                break
            total += len(raw)
            need(total <= before.st_size, "file grew")
            sha.update(raw)
            if collect:
                chunks.append(raw)
        need(identity(os.fstat(stream.fileno())) == identity(before), "file changed while read")
    need(total == pin["bytes"] and sha.hexdigest() == pin["SHA256"]
         and identity(path.lstat()) == identity(before), "file SHA/identity differs")
    record = dict(bytes=total, SHA256=sha.hexdigest(), file_identity=identity(before))
    return (b"".join(chunks) if collect else None), record


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")


def pin(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
