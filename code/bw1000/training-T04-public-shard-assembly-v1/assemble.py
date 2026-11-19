"""Assemble one completely staged public OneVision shard on the target filesystem."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time


HERE = Path(__file__).resolve().parent
BLOCK = 1024**2


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def stat_identity(s):
    return s.st_dev, s.st_ino, s.st_uid, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def same_open_file(path_stat, descriptor_stat):
    path, descriptor = stat_identity(path_stat), stat_identity(descriptor_stat)
    if os.name == "nt":
        # Windows may report creation time on the path and change time on the handle.
        return path[:-1] == descriptor[:-1]
    return path == descriptor


def bounded_json(path, limit):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "METADATA_FILE_IDENTITY")
    with path.open("rb") as f:
        opened = os.fstat(f.fileno())
        need(same_open_file(before, opened), "METADATA_CHANGED")
        raw = f.read(limit+1)
        need(stat_identity(os.fstat(f.fileno())) == stat_identity(opened), "METADATA_CHANGED")
    need(len(raw) <= limit and stat_identity(path.lstat()) == stat_identity(before), "METADATA_CHANGED")
    return raw, json.loads(raw)


def path_in(root, value):
    path = Path(value)
    need(path.is_absolute() and path.is_relative_to(root) and
         all(not p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)),
         "EXACT_STAGING_PATH")
    need(path.resolve().is_relative_to(root.resolve()), "EXACT_STAGING_PATH")
    return path


def bind_parts(plan, shard, staging_root):
    parts = plan["parts"]
    need(0 < len(parts) <= 4096, "PART_COUNT")
    offset = 0
    data_paths = set()
    for part in parts:
        need(type(part["offset"]) is int and part["offset"] == offset and
             type(part["length"]) is int and 0 < part["length"] <= 32*BLOCK, "CONTIGUOUS_RANGE_LAYOUT")
        data = path_in(staging_root, part["data"])
        receipt = path_in(staging_root, part["receipt"])
        need(data not in data_paths, "RANGE_PATH_REUSE")
        data_paths.add(data)
        raw, proof = bounded_json(receipt, 8192)
        need(hashlib.sha256(raw).hexdigest() == part["receipt_sha"] and
             proof["request"] == part["request"] and proof["revision"] == plan["revision"], "SOURCE_RANGE_RECEIPT")
        if part["format"] == "prefix":
            need(offset == proof["offset"] == 0 and proof["bytes_saved"] == part["length"] and
                 proof["file"] == plan["file"] and proof["shard_SHA256"] == shard["lfs_sha256"] and
                 proof["range_SHA256"] == part["sha256"], "ORIGINAL_PREFIX_RECEIPT")
        else:
            need(part["format"] == "wave", "RANGE_RECEIPT_FORMAT")
            wave_path = path_in(staging_root, part["wave_intent"])
            wave_raw, wave = bounded_json(wave_path, 65536)
            chunk = dict(offset=part["offset"], length=part["length"])
            need(hashlib.sha256(wave_raw).hexdigest() == part["wave_intent_sha"] and
                 wave["request"] == proof["request"] and wave["revision"] == proof["revision"] and
                 wave["publication"] == proof["publication"] and wave["file"] == plan["file"] and
                 wave["shard_sha"] == shard["lfs_sha256"] and wave["shard_bytes"] == shard["bytes"] and
                 proof["chunk"] == chunk and chunk in wave["chunks"] and proof["source_sha"] == part["sha256"] and
                 proof["published_bytes"] == part["length"], "ORIGINAL_SUFFIX_RECEIPT")
        offset += part["length"]
    need(offset == shard["bytes"], "COMPLETE_SHARD_REQUIRED")
    return parts


def copy_parts(parts, shard, stream, left):
    complete = hashlib.sha256()
    written = 0
    for part in parts:
        path = Path(part["data"])
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and (os.name == "nt" or before.st_uid == os.getuid()) and
             before.st_size == part["length"], "DURABLE_RANGE_FILE_IDENTITY")
        range_digest = hashlib.sha256()
        remaining = part["length"]
        with path.open("rb") as data:
            opened = os.fstat(data.fileno())
            need(same_open_file(before, opened), "RANGE_CHANGED")
            while remaining:
                left()
                raw = data.read(min(BLOCK, remaining))
                need(raw and stream.write(raw) == len(raw), "RANGE_IO")
                remaining -= len(raw)
                written += len(raw)
                complete.update(raw)
                range_digest.update(raw)
            need(stat_identity(os.fstat(data.fileno())) == stat_identity(opened), "RANGE_CHANGED")
        need(stat_identity(path.lstat()) == stat_identity(before) and
             range_digest.hexdigest() == part["sha256"], "RANGE_DIGEST")
    need(written == shard["bytes"] and complete.hexdigest() == shard["lfs_sha256"], "OFFICIAL_WHOLE_SHARD_DIGEST")
    return written, complete.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args):
    binding = json.loads((HERE/"BINDING.json").read_bytes())
    root = Path(binding["target_parent"])
    started = time.monotonic()

    def left():
        if time.monotonic() >= started+args.seconds:
            raise TimeoutError("ONE_ASSEMBLY_DEADLINE")

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("ONE_ASSEMBLY_DEADLINE")))
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    raw, plan = bounded_json(Path(args.plan), 4*BLOCK)
    need(hashlib.sha256(raw).hexdigest() == args.plan_sha, "PUBLISHED_PLAN_REQUIRED")
    need(plan["revision"] == binding["revision"] and plan["file"] in binding["shards"], "OFFICIAL_SHARD_IDENTITY")
    shard = binding["shards"][plan["file"]]
    parts = bind_parts(plan, shard, root)
    target = path_in(root, plan["target"])
    need(target.parent.is_dir() and not target.exists(), "UNUSED_ASSEMBLY_DIRECTORY")
    capacity = os.statvfs(target.parent)
    need(capacity.f_bavail*capacity.f_frsize >= shard["bytes"], "ASSEMBLY_FREE_SPACE")
    target.mkdir(mode=0o700)
    temporary = target/"file.incomplete"
    final = target/plan["file"]
    with temporary.open("xb") as stream:
        written, digest = copy_parts(parts, shard, stream, left)
        stream.flush()
        os.fsync(stream.fileno())
        need(os.fstat(stream.fileno()).st_size == written, "ASSEMBLED_STAT")
    left()
    os.link(temporary, final, follow_symlinks=False)
    temporary.unlink()
    sync_dir(target)
    result = dict(file=plan["file"], bytes=written, SHA256=digest, plan_sha=args.plan_sha,
                  source_revision=plan["revision"], ranges=len(parts), verified_whole_shard=True,
                  actual_gpu_steps=0, all_model_weights_admitted=False)
    with (target/"receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(target)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("plan_sha")
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    need(1 <= args.seconds <= 7200, "BOUNDED_ASSEMBLY")
    run(args)


if __name__ == "__main__":
    main()
