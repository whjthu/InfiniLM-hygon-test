"""Stream only the missing suffix of the verified original HunyuanVideo Transformer."""
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil

CHUNK = 1024 * 1024
COMMIT_FRAME = b"HUNYUAN_TRANSFORMER_SUFFIX_SOURCE_VERIFIED\n"


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
        expected = plan["source_stat"]
        require(identity(before) == tuple(expected[k] for k in
                ("device", "inode", "bytes", "mtime_ns", "ctime_ns")), "Source identity differs")
        digest = prefix(source, spec["offset"], plan["prefix_sha256"])
        output.write(frame(plan["name"], spec))
        for chunk in blocks(source, plan["relay_bytes"]):
            digest.update(chunk)
            output.write(chunk)
        require(digest.hexdigest() == spec["SHA256"], "Source whole-file hash differs")
        require(identity(before) == identity(os.fstat(source.fileno())) == identity(path.lstat()),
                "Source changed during streaming")
    output.flush()


def prior_receipt(path, pin):
    with regular(path, os.O_RDONLY) as stream:
        raw = stream.read(65537)
    require(len(raw) <= 65536 and hashlib.sha256(raw).hexdigest() == pin, "Prior receipt changed")
    return json.loads(raw)


def receive(plan, stream, scope, common):
    root = Path(plan["target_root"])
    staging = root / plan["stage"]
    final = root / plan["final"]
    folder = staging / plan["folder"]
    require(all(p.is_dir() and not p.is_symlink() for p in (root, staging, folder.parent, folder)),
            "Wrong target directory")
    prior = root / "retained-suffix-v2"
    outcome = prior_receipt(prior / "outcome.json", plan["prior_outcome_sha256"])
    durable = prior_receipt(prior / "last-fsynced.json", plan["prior_durable_sha256"])
    spec = plan["file"]
    expected = dict(request_id=plan["prior_request"], claim=plan["prior_claim"], offset=spec["offset"])
    require(all(record.get(key) == value for record in (outcome, durable) for key, value in expected.items())
            and outcome["passed"] is False and outcome["error_type"] == "EOFError"
            and durable["prefix_SHA256"] == plan["prefix_sha256"], "Prior terminal differs")
    require(not os.path.lexists(final) and not os.path.lexists(folder / plan["name"]),
            "Final model or staged weight already exists")
    require(shutil.disk_usage(root).free > plan["relay_bytes"] + CHUNK, "Insufficient target disk space")
    with regular(folder / "config.json", os.O_RDONLY) as config:
        require(hashlib.file_digest(config, "sha256").hexdigest() == plan["config_sha256"], "Config hash differs")
    require(stream.readline(1024) == frame(plan["name"], spec), "Unexpected suffix frame")
    path = folder / plan["partial"]
    with regular(path, os.O_RDWR) as target:
        before = os.fstat(target.fileno())
        require(before.st_size == spec["offset"], "Retained size differs")
        digest = prefix(target, spec["offset"], plan["prefix_sha256"])
        require(identity(before) == identity(os.fstat(target.fileno())) == identity(path.lstat()), "Retained file changed")
        position = spec["offset"]
        next_receipt = position + 64 * CHUNK

        def checkpoint(name):
            target.flush()
            os.fsync(target.fileno())
            record = dict(common, offset=position, prefix_SHA256=digest.hexdigest())
            temporary = scope / (name + ".tmp")
            with temporary.open("x", encoding="utf8") as state:
                json.dump(record, state)
                state.flush()
                os.fsync(state.fileno())
            temporary.replace(scope / name)
            sync_directory(scope)

        try:
            for chunk in blocks(stream, plan["relay_bytes"]):
                require(target.write(chunk) == len(chunk), "Short target write")
                position += len(chunk)
                digest.update(chunk)
                if position >= next_receipt:
                    checkpoint("progress.json")
                    next_receipt = position + 64 * CHUNK
            target.flush()
            os.fsync(target.fileno())
            require(os.path.samestat(os.fstat(target.fileno()), path.lstat())
                    and os.fstat(target.fileno()).st_size == spec["bytes"]
                    and digest.hexdigest() == spec["SHA256"], "Target whole-file verification failed")
        except Exception:
            target.flush()
            # An unaccounted short write or external change must not advance the prefix.
            if os.path.samestat(os.fstat(target.fileno()), path.lstat()) and path.stat().st_size == position:
                checkpoint("last-fsynced.json")
            raise
        checkpoint("last-fsynced.json")
        verified_file = os.fstat(target.fileno())
    closed_file = path.lstat()
    require(os.path.samestat(verified_file, closed_file) and closed_file.st_size == spec["bytes"],
            "Verified weight replaced during close")
    verified_identity = identity(closed_file)
    require(stream.readline(len(COMMIT_FRAME)) == COMMIT_FRAME, "Missing verified sender commit")
    require(not stream.read(1) and not os.path.lexists(final), "Excess data or existing final path")
    require(identity(path.lstat()) == verified_identity, "Verified weight changed before publication")
    weight = folder / plan["name"]
    require(not os.path.lexists(weight), "Staged weight already exists")
    path.rename(weight)
    sync_directory(folder)
    sync_directory(folder.parent)
    sync_directory(staging)
    staging.rename(final)
    sync_directory(root)


def run_endpoint(plan, claim, request, epoch, role):
    import signal
    import sys
    import time

    def expired(*_):
        raise TimeoutError("Transformer suffix deadline reached")

    require(role in ("sender", "receiver"), "Unknown transfer role")
    deadline = epoch + plan["timeout_seconds"]
    remaining = deadline - time.time()
    require(0 < remaining <= plan["timeout_seconds"], "Invalid epoch budget")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    directory = Path(plan["source_scope"] if role == "sender" else plan["target_root"])
    scope = directory / "a100-verified-transformer-suffix-v1"
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
            receive(plan, sys.stdin.buffer, scope,
                    dict(claim=claim, request_id=request, deadline=deadline))
        result.update(passed=True, suffix_bytes=plan["relay_bytes"], files={plan["name"]: plan["file"]})
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    with (scope / "outcome.json").open("x", encoding="utf8") as out:
        json.dump(result, out)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(result), file=sys.stderr if role == "sender" else sys.stdout, flush=True)
    return 0 if result["passed"] else 1
