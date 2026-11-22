"""Durable SCNet receiver for the immutable public Z-Image runtime assets."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys

import stage_internvl_weight_segment as stage


HERE = Path(__file__).resolve().parent
TARGET = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-publisher-runtime-transfer-v1")
MANIFEST_SHA = "88b9886c9c4aeee3c5468fa935539f0166bf9d1b8a37eb30cef0f55a1f7bf13b"


def assets():
    raw = (HERE / "RUNTIME_FETCH_MANIFEST.json").read_bytes()
    stage.need(sha256(raw).hexdigest() == MANIFEST_SHA, "publisher manifest bytes differ")
    return json.loads(raw)["assets"]


def storage_spec(row):
    return {"file_id": row["asset_id"], "filename": row["filename"], "segment_bytes": row["segment_bytes"],
            "bytes": row["bytes"], "SHA256": row["SHA256"]}


def staged_directory(root, row):
    return root / "segments" / f'{row["asset_id"]:02d}'


def layout_file(layout, row):
    name = PurePosixPath(row["target_relative_path"])
    stage.need(not name.is_absolute() and ".." not in name.parts and name.name == row["filename"],
               "runtime path differs")
    return layout.joinpath(*name.parts)


def provision_directory(path, *, parents=False):
    path.mkdir(mode=0o700, parents=parents, exist_ok=True)
    stage.need(path.resolve(strict=True) == path, "aliased target directory")


def inspect(root, row):
    path = staged_directory(root, row)
    if not path.exists() and not path.is_symlink():
        return {"file_id": row["asset_id"], "state": "ABSENT", "confirmed_bytes": 0,
                "whole_file_qualified_at_seal": False}
    return stage.inspect(path, storage_spec(row))


def receive(root, row, offset, digest, stream):
    provision_directory(root)
    provision_directory(root / "segments")
    path = staged_directory(root, row)
    if offset == 0 and not path.exists() and not path.is_symlink():
        stage.initialize(path, storage_spec(row))
    return stage.append(path, storage_spec(row), offset, digest, stream)


def seal(root, row):
    path = staged_directory(root, row)
    spec = storage_spec(row)
    with stage.locked(path):
        # Link only into the private assembly tree. The verified checkpoint records
        # the inode after this link, and assembly never adds a second link later.
        state = stage.load(path, spec)
        stage.need(state["confirmed_bytes"] == spec["bytes"] and not state["sealed"], "asset is incomplete or sealed")
        candidate = layout_file(root / "candidate", row)
        provision_directory(candidate.parent, parents=True)
        source = path / "payload.partial"
        identity = source.lstat()
        stage.need(stat.S_ISREG(identity.st_mode) and [identity.st_dev, identity.st_ino] == state["file"],
                   "payload inode differs before assembly")
        if not candidate.exists() and not candidate.is_symlink():
            os.link(source, candidate)
            stage.sync_dir(candidate.parent)
        stage.need(not candidate.is_symlink() and os.path.samestat(identity, candidate.lstat()),
                   "assembly path refers to a different inode")
    # A competing seal must recheck the same locked immutable staging checkpoint.
    return stage.seal(path, spec)


def publish_runtime(root, rows):
    candidate, runtime = root / "candidate", root / "runtime"
    with stage.locked(root):
        stage.need(not candidate.is_symlink() and not runtime.is_symlink()
                   and (candidate.is_dir() != runtime.is_dir()), "runtime publication tree differs")
        layout = runtime if runtime.is_dir() else candidate
        for row in rows:
            proof = inspect(root, row)
            member = layout_file(layout, row)
            source = staged_directory(root, row) / row["filename"]
            stage.need(proof["whole_file_qualified_at_seal"]
                       and member.resolve(strict=True) == member and not member.is_symlink()
                       and os.path.samestat(source.lstat(), member.lstat()), "runtime member is not the sealed asset")
        actual = {path.relative_to(layout).as_posix() for path in layout.rglob("*")
                  if path.is_file() or path.is_symlink()}
        stage.need(actual == {row["target_relative_path"] for row in rows}, "unexpected runtime members")
        if layout == candidate:
            candidate.rename(runtime)
            stage.sync_dir(root)
        return {"state": "PUBLISHED_RUNTIME", "files": len(rows), "bytes": sum(row["bytes"] for row in rows),
                "manifest_SHA256": MANIFEST_SHA}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("inspect", "append", "seal", "publish"))
    p.add_argument("--asset", type=int)
    p.add_argument("--offset", type=int)
    p.add_argument("--digest")
    p.add_argument("--budget", type=int, default=150)
    args = p.parse_args()
    stage.need(1 <= args.budget <= 180, "receiver budget differs")

    def expired(*_):
        raise TimeoutError("receiver deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget)
    rows = assets()
    if args.operation == "publish":
        value = publish_runtime(TARGET, rows)
    else:
        matches = [row for row in rows if row["asset_id"] == args.asset]
        stage.need(len(matches) == 1, "asset identity differs")
        row = matches[0]
        if args.operation == "inspect":
            value = inspect(TARGET, row)
        elif args.operation == "append":
            stage.segment_length(storage_spec(row), args.offset)
            value = receive(TARGET, row, args.offset, args.digest, sys.stdin.buffer)
        else:
            value = seal(TARGET, row)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
