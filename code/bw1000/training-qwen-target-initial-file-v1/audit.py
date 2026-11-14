"""Verify the original T01 relay and the published target file without deserialization."""
import hashlib
import json
import os
from pathlib import Path
import stat

REQUEST = "de75354b4d0e47c289df272d8888d5f5"
CLAIM = "5f36675457d7204afe64d983e7a1b7ce5c5b79c6"
EPOCH = 1794621128.5466852
SIZE = 174821899
SHA256 = "2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f"


def checked_read(path, size=None):
    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Expected regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if size is None:
            raw = stream.read(32769)
            if not 0 < len(raw) <= 32768 or len(raw) != opened.st_size:
                raise ValueError("Unexpected receipt size")
            digest = hashlib.sha256(raw).hexdigest()
        else:
            if opened.st_size != size:
                raise ValueError("Wrong published file size")
            raw = None
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if not identity(before) == identity(opened) == identity(after) == identity(path.lstat()):
        raise ValueError("File changed during verification")
    return digest, raw


def verify(root):
    scope = root / "training-qwen-target-initial-file-v1"
    destination = root / "training-qwen-initial-trainables-v1"
    if any(p.is_symlink() or not p.is_dir() for p in (scope, destination)):
        raise ValueError("Unexpected publication directories")
    intent_sha, intent_raw = checked_read(scope / "intent.json")
    terminal_sha, terminal_raw = checked_read(scope / "outcome.json")
    intent = json.loads(intent_raw)
    terminal = json.loads(terminal_raw)
    base = dict(claim=CLAIM, request_id=REQUEST, epoch=EPOCH, role="receiver")
    spec = dict(bytes=SIZE, offset=0, SHA256=SHA256)
    pid = intent.get("pid")
    if type(pid) is not int or pid <= 0 or intent != dict(base, passed=False, pid=pid, deadline=EPOCH + 1800):
        raise ValueError("Unexpected receiver intent")
    if terminal != dict(base, passed=True, suffix_bytes=SIZE, files={"initial.pt": spec}):
        raise ValueError("Unexpected receiver terminal")
    if {p.name for p in destination.iterdir()} != {"initial.pt"} or (scope / "staging").exists():
        raise ValueError("Incomplete publication")
    actual, _ = checked_read(destination / "initial.pt", SIZE)
    if actual != SHA256:
        raise ValueError("Whole-file SHA differs")
    return dict(passed=True, request_id=REQUEST, claim=CLAIM, bytes=SIZE, SHA256=actual,
                intent_SHA256=intent_sha, terminal_SHA256=terminal_sha, GPU_cards=0,
                checkpoint_deserialized=False)
