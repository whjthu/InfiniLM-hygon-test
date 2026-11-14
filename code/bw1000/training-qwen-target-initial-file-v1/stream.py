"""Relay the exact A100 Qwen initial trainable checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil

CHUNK = 1024 * 1024
COMMIT_FRAME = b"QWEN_INITIAL_SOURCE_VERIFIED\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def regular(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(fd, "rb" if flags == os.O_RDONLY else "r+b")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        stream.close()
        raise ValueError("Expected a regular checkpoint file")
    return stream


def blocks(stream, size):
    while size:
        chunk = stream.read(min(CHUNK, size))
        if not chunk:
            raise EOFError("Checkpoint ended before the declared extent")
        require(len(chunk) <= size, "Read exceeded the declared extent")
        size -= len(chunk)
        yield chunk


def prefix(stream, offset, expected):
    digest = hashlib.sha256()
    for chunk in blocks(stream, offset):
        digest.update(chunk)
    require(not offset or digest.hexdigest() == expected, "Retained prefix changed")
    return digest


def frame(name, spec):
    return (json.dumps(dict(name=name, **spec), sort_keys=True) + "\n").encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def send(plan, output):
    spec = plan["file"]
    path = Path(plan["source_file"])
    with regular(path, os.O_RDONLY) as source:
        before = os.fstat(source.fileno())
        require(before.st_size == spec["bytes"], "Source size differs")
        digest = prefix(source, spec["offset"], plan["prefix_sha256"])
        output.write(frame(plan["name"], spec))
        for chunk in blocks(source, plan["relay_bytes"]):
            digest.update(chunk)
            output.write(chunk)
        require(digest.hexdigest() == spec["SHA256"], "Source whole-file hash differs")
        require(identity(before) == identity(os.fstat(source.fileno())) == identity(path.lstat()),
                "Source changed during streaming")
    output.flush()


def receive(plan, stream, scope):
    root = Path(plan["target_root"])
    final = root / plan["target_directory"]
    require(not os.path.lexists(final), "Target already exists")
    require(shutil.disk_usage(root).free > plan["relay_bytes"] + CHUNK, "Insufficient target space")
    spec = plan["file"]
    require(stream.readline(1024) == frame(plan["name"], spec), "Unexpected checkpoint frame")
    staging = scope / "staging"
    staging.mkdir(mode=0o700)
    path = staging / plan["name"]
    digest = hashlib.sha256()
    with path.open("xb") as target:
        for chunk in blocks(stream, plan["relay_bytes"]):
            require(target.write(chunk) == len(chunk), "Short target write")
            digest.update(chunk)
        target.flush()
        os.fsync(target.fileno())
        require(os.fstat(target.fileno()).st_size == spec["bytes"], "Target size differs")
    require(digest.hexdigest() == spec["SHA256"], "Target hash differs")
    require(stream.readline(len(COMMIT_FRAME)) == COMMIT_FRAME, "Missing verified sender commit")
    require(not stream.read(1) and not os.path.lexists(final), "Excess bytes or existing target")
    sync_directory(staging)
    # Exclusive link prevents overwriting a concurrent publication.
    final.mkdir(mode=0o700)
    os.link(path, final / plan["name"], follow_symlinks=False)
    sync_directory(final)
    sync_directory(root)
    path.unlink()
    staging.rmdir()


def run_endpoint(plan, claim, request, epoch, role):
    import signal
    import sys
    import time

    def expired(*_):
        raise TimeoutError("Qwen initial file deadline reached")

    require(role in ("sender", "receiver"), "Unknown transfer role")
    deadline = epoch + plan["timeout_seconds"]
    remaining = deadline - time.time()
    require(0 < remaining <= plan["timeout_seconds"], "Invalid epoch budget")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    directory = Path(plan["source_scope"] if role == "sender" else plan["target_root"])
    scope = directory / "training-qwen-target-initial-file-v1"
    scope.mkdir(mode=0o700)
    result = dict(claim=claim, request_id=request, epoch=epoch, role=role, passed=False)
    with (scope / "intent.json").open("x", encoding="utf8") as out:
        json.dump(dict(result, deadline=deadline, pid=os.getpid()), out)
        out.flush()
        os.fsync(out.fileno())
    try:
        if role == "sender":
            send(plan, sys.stdout.buffer)
        else:
            receive(plan, sys.stdin.buffer, scope)
        result.update(passed=True, suffix_bytes=plan["relay_bytes"], files={plan["name"]: plan["file"]})
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    with (scope / "outcome.json").open("x", encoding="utf8") as out:
        json.dump(result, out)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(result), file=sys.stderr if role == "sender" else sys.stdout, flush=True)
    return 0 if result["passed"] else 1
