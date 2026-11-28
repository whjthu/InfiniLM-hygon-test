"""Fingerprint the actual project DTK sandbox without initializing a device."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import stat


METHOD = "canonical-dtk-rootfs-v1-file-bytes-type-mode-owner-and-links"
ROOTFS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1")
MAX_FILES = 650_000
MAX_BYTES = 100 << 30


def stable(stat_result):
    return [stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ctime_ns]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fingerprint(root):
    initial = root.lstat()
    require(root.is_absolute() and root.resolve(strict=True) == root and stat.S_ISDIR(initial.st_mode),
            "The sandbox must be its canonical directory")
    listing = sha256()
    counts = dict(entries=0, dirs=0, files=0, link_count=0, nodes=0, file_bytes=0)
    paths_to_recheck = []

    def record(entry):
        listing.update((json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii"))

    def inspect(path, relative):
        before = path.lstat()
        mode = before.st_mode
        require(before.st_dev == initial.st_dev, "A child crosses the sandbox filesystem")
        fields = [relative, stat.S_IMODE(mode), before.st_uid, before.st_gid]
        counts["entries"] += 1
        require(counts["entries"] <= MAX_FILES, "Sandbox entry bound exceeded")
        if stat.S_ISDIR(mode):
            record(["d", *fields])
            counts["dirs"] += 1
            paths_to_recheck.append((path, stable(before)))
            with os.scandir(path) as entries:
                children = sorted((entry.name for entry in entries), key=os.fsencode)
            for name in children:
                inspect(path / name, f"{relative}/{name}" if relative else name)
        elif stat.S_ISREG(mode):
            counts["files"] += 1
            counts["file_bytes"] += before.st_size
            require(counts["file_bytes"] <= MAX_BYTES, "Sandbox byte bound exceeded")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(path, flags), "rb") as stream:
                opened = os.fstat(stream.fileno())
                require(stat.S_ISREG(opened.st_mode) and opened.st_dev == initial.st_dev
                        and stable(opened) == stable(before), "Sandbox file changed before hashing")
                sha = file_digest(stream, "sha256").hexdigest()
                require(stable(os.fstat(stream.fileno())) == stable(opened), "Sandbox file changed during hashing")
            record(["f", *fields, before.st_size, sha])
        elif stat.S_ISLNK(mode):
            counts["link_count"] += 1
            record(["l", *fields, os.readlink(path)])
        elif stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode):
            counts["nodes"] += 1
            record(["n", *fields, stat.S_IFMT(mode), before.st_rdev])
        else:
            raise ValueError("Unrecognized sandbox entry type")
        require(stable(path.lstat()) == stable(before), "Sandbox entry changed during the walk")

    record([METHOD])
    inspect(root, "")
    for path, original in paths_to_recheck:
        require(stable(path.lstat()) == original, "Sandbox directory changed before sealing")
    return dict(method=METHOD, host_path=root.as_posix(), image_id="sha256:" + listing.hexdigest(),
                stat=stable(initial), counts=counts, device_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    result = fingerprint(ROOTFS)
    with args.write.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
