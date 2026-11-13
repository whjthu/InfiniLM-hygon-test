"""Stream only the missing suffix of the verified public Qwen checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil

CHUNK = 1024 * 1024
COMMIT_FRAME = b"QWEN_SUFFIX_SOURCE_VERIFIED\n"


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


def send(plan, output):
    root = Path(plan["source_root"])
    for name, spec in plan["remaining"].items():
        path = root / name
        with regular(path, os.O_RDONLY) as stream:
            info = os.fstat(stream.fileno())
            observed = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino, device=info.st_dev)
            require(observed == plan["source_identities"][name], "Original source file changed")
            digest = prefix(stream, spec["offset"], plan["prefix_sha256"])
            output.write(frame(name, spec))
            for chunk in blocks(stream, spec["bytes"] - spec["offset"]):
                digest.update(chunk)
                output.write(chunk)
            require(digest.hexdigest() == spec["SHA256"], "Source whole-file hash differs")
            require(identity(info) == identity(os.fstat(stream.fileno())) == identity(path.stat()),
                    "Source file changed while streaming")
        output.flush()


def terminal(path, digest, plan):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "Original transfer receipt changed")
    row = json.loads(raw)
    require(row["request_id"] == plan["original_request"] and row["claim"] == plan["original_claim"]
            and row["passed"] is False and row["error_type"] == "TimeoutError", "Wrong original terminal receipt")


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def receive(plan, stream):
    root = Path(plan["target_root"])
    staging = root / "model.incomplete"
    final = root / "model"
    terminal(root / "outcome.json", plan["original_receiver_terminal_sha256"], plan)
    require(staging.is_dir() and not staging.is_symlink() and not os.path.lexists(final), "Wrong target directory")
    require(shutil.disk_usage(root).free > plan["relay_bytes"] + CHUNK, "Insufficient target disk space")
    allowed = set(plan["prior_done"]) | {plan["active"] + ".partial"}
    require({p.name for p in staging.iterdir()} == allowed, "Target has unexpected files")
    for name, spec in plan["prior_done"].items():
        with regular(staging / name, os.O_RDONLY) as prior:
            require(os.fstat(prior.fileno()).st_size == spec["bytes"], "Retained completed file changed size")
            require(hashlib.file_digest(prior, "sha256").hexdigest() == spec["SHA256"], "Retained completed file changed")
    for name, spec in plan["remaining"].items():
        require(stream.readline(1024) == frame(name, spec), "Unexpected file or offset")
        partial = staging / (name + ".partial")
        flags = os.O_RDWR if spec["offset"] else os.O_RDWR | os.O_CREAT | os.O_EXCL
        with regular(partial, flags) as target:
            before = os.fstat(target.fileno())
            require(before.st_size == spec["offset"], "Retained extent changed")
            digest = prefix(target, spec["offset"], plan["prefix_sha256"])
            require(identity(before) == identity(os.fstat(target.fileno())), "Retained file changed during verification")
            for chunk in blocks(stream, spec["bytes"] - spec["offset"]):
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
            require(os.path.samestat(os.fstat(target.fileno()), partial.stat()), "Target file identity changed")
            require(digest.hexdigest() == spec["SHA256"], "Received whole-file hash differs")
        os.link(partial, staging / name)
        partial.unlink()
    require(stream.readline(len(COMMIT_FRAME)) == COMMIT_FRAME, "Sender completion not verified")
    require(not stream.read(1), "Excess bytes after the registered files")
    require({p.name for p in staging.iterdir()} == set(plan["files"]), "Incomplete final file set")
    require(not os.path.lexists(final), "Final model path already exists")
    sync_directory(staging)
    staging.rename(final)
    sync_directory(root)


def run_endpoint(plan, claim, request, epoch, role, source_scope=None):
    import signal
    import sys
    import time

    def expired(*_):
        raise TimeoutError("Qwen continuation deadline reached")

    require(role in ("sender", "receiver"), "Unknown transfer endpoint")
    remaining = epoch + plan["timeout_seconds"] - time.time()
    require(0 < remaining <= plan["timeout_seconds"], "Invalid continuation deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    original = Path(source_scope if role == "sender" else plan["target_root"])
    require(role != "sender" or source_scope == "/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-original-public-model-v1",
            "Wrong original sender scope")
    prior = plan["original_sender_terminal_sha256" if role == "sender" else "original_receiver_terminal_sha256"]
    terminal(original / "outcome.json", prior, plan)
    scope = original / "verified-suffix-v1"
    scope.mkdir(mode=0o700)
    result = dict(claim=claim, request_id=request, epoch=epoch, role=role, passed=False)
    with (scope / "intent.json").open("x", encoding="utf8") as out:
        json.dump(dict(result, deadline=epoch + plan["timeout_seconds"], pid=os.getpid()), out)
        out.flush()
        os.fsync(out.fileno())
    try:
        if role == "sender":
            send(plan, sys.stdout.buffer)
        else:
            receive(plan, sys.stdin.buffer)
        result.update(passed=True, suffix_bytes=plan["relay_bytes"], files=plan["files"])
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    with (scope / "outcome.json").open("x", encoding="utf8") as out:
        json.dump(result, out)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(result), file=sys.stderr if role == "sender" else sys.stdout, flush=True)
    return 0 if result["passed"] else 1
