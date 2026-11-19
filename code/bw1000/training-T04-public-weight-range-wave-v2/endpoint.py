"""Remote receiver and independent read-only audit for one frozen range wave."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time


def need(valid, detail):
    if not valid:
        raise ValueError(detail)


def save(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_sha(path, length):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == length, "FILE_IDENTITY")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")
    need(all(getattr(before, key) == getattr(after, key) for key in fields),
         "FILE_CHANGED_DURING_READ")
    return digest


def read_receipt(path, maximum=8192):
    s = path.lstat()
    need(stat.S_ISREG(s.st_mode) and 0 < s.st_size <= maximum, "RECEIPT_IDENTITY")
    return json.loads(path.read_bytes())


def name_for(chunk):
    return f"shard01.offset{chunk['offset']:010d}"


def paths(root, chunk):
    base = name_for(chunk)
    return root / (base + ".part"), root / (base + ".receipt.json"), root / (base + ".incomplete")


def check_prior(plan):
    prior = plan["prior"]
    root = Path(prior["root"])
    receipt = root / "receipt.json"
    need(file_sha(receipt, prior["receipt_bytes"]) == prior["receipt_sha"], "ORIGINAL_PREFIX_RECEIPT")
    record = read_receipt(receipt)
    need(record["request"] == prior["request"] and record["range_SHA256"] == prior["range_sha"],
         "ORIGINAL_PREFIX_REQUEST")
    need(file_sha(root / "shard01.offset0.part", prior["length"]) == prior["range_sha"],
         "ORIGINAL_PREFIX_RANGE")


def transfer(root, p):
    manifest = read_receipt(root / "intent.json")
    need(all(p[key] == manifest[key] for key in ("request", "epoch", "deadline", "publication", "revision")),
         "ORIGINAL_WAVE_IDENTITY")
    chunk = p["chunk"]
    need(chunk in manifest["chunks"], "PRECOMMITTED_RANGE")
    final, receipt, temporary = paths(root, chunk)
    need(not final.exists() and not receipt.exists(), "RANGE_ALREADY_HAS_EVIDENCE")
    expected = p["source_sha"]
    remaining = chunk["length"]
    hasher = hashlib.sha256()
    with temporary.open("xb") as stream:
        while remaining:
            part = sys.stdin.buffer.read(min(remaining, 1024**2))
            need(part, "TRUNCATED_STDIN")
            stream.write(part)
            hasher.update(part)
            remaining -= len(part)
        need(not sys.stdin.buffer.read(1), "EXTRA_STDIN")
        need(hasher.hexdigest() == expected, "RECEIVED_RANGE_DIGEST")
        stream.flush()
        os.fsync(stream.fileno())
    need(file_sha(temporary, chunk["length"]) == expected, "RECEIVER_REREAD")
    need(time.time() < p["deadline"], "ORIGINAL_WAVE_DEADLINE")
    os.link(temporary, final)
    temporary.unlink()
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    result = dict(p, published_bytes=final.stat().st_size, finished_epoch=time.time())
    save(receipt, result)
    return result


def audit(root, p):
    rows = []
    if not root.is_dir():
        return dict(request=p["request"], directory_exists=False, read_only=True, ranges=[])
    manifest = read_receipt(root / "intent.json")
    need(manifest == p["expected_intent"], "READ_EXACT_ORIGINAL_WAVE")
    check_prior(manifest)
    for chunk in manifest["chunks"]:
        final, receipt, temporary = paths(root, chunk)
        report = dict(chunk, state="MISSING", independently_verified_bytes=0)
        if final.is_file() and receipt.is_file():
            r = read_receipt(receipt)
            need(all(r[key] == manifest[key] for key in ("request", "epoch", "deadline", "publication", "revision")),
                 "READ_ORIGINAL_RECEIVER")
            need(r["chunk"] == chunk and r["published_bytes"] == chunk["length"], "READ_RANGE_RECEIVER")
            actual = file_sha(final, chunk["length"])
            need(actual == r["source_sha"], "READ_RANGE_CONTENT")
            report.update(state="VERIFIED_RANGE", source_sha=actual, independently_verified_bytes=chunk["length"],
                          receiver_receipt_sha=file_sha(receipt, receipt.stat().st_size))
        elif final.exists() or receipt.exists() or temporary.exists():
            report.update(state="INCOMPLETE_OR_UNKNOWN", temporary_bytes=temporary.stat().st_size if temporary.exists() else None)
        rows.append(report)
    return dict(request=p["request"], directory_exists=True, original_prefix_verified=True, read_only=True, ranges=rows)


def main(p):
    limit = p["deadline"]
    need(time.time() < limit, "ORIGINAL_DEADLINE")
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, limit - time.time())
    root = Path(p["target"])
    mode = p["mode"]
    if mode == "prepare":
        plan = p["intent"]
        check_prior(plan)
        fs = os.statvfs(root.parent)
        need(fs.f_bavail * fs.f_frsize >= 16 * 1024**3, "TARGET_FREE_SPACE")
        root.mkdir(mode=0o700)
        save(root / "intent.json", plan)
        result = dict(request=plan["request"], original_prefix_verified=True, target_created=True)
    elif mode == "receive":
        result = transfer(root, p)
    elif mode == "audit":
        result = audit(root, p)
    else:
        raise ValueError("UNKNOWN_MODE")
    print(json.dumps(result, allow_nan=False, sort_keys=True))


main(PARAMETERS)
