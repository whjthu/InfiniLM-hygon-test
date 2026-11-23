"""Seal the already-frozen private development noise bank on experiment storage."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
import time


REQUEST = "bcfe71f64ff5457db0a75b81ef3ebf1c"
BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL = "training-zimage-t07-coco-development-noise-v1-" + REQUEST
SIZES = (
    ("development-noise.private.json", 77483, "5c4cb6303071502ebdee928dbd13df10678143dbfba1b8e38086ba832a33c4c4"),
    ("development-bf16-le.private.bin", 8388608, "e553bf8ce030ed1b74413cb0507ed3d11495f01f68072511e93684de8a5551c5"),
)
ROSTER_SHA256 = "9e1603e5626fdca032cf8b59b63a69c4e26ee410064f507b9f521df8c8234a63"


def receive(directory, wire, publication):
    deadline = time.monotonic() + 45
    directory.mkdir(mode=0o700)
    files = {}
    for name, count, expected_digest in SIZES:
        digest = sha256()
        written = 0
        with (directory / name).open("xb") as stream:
            while written < count:
                if time.monotonic() >= deadline:
                    raise TimeoutError("noise bank transfer exceeded 45 seconds")
                block = wire.read(min(count - written, 1 << 20))
                if not block:
                    raise ValueError("short noise stream")
                stream.write(block)
                digest.update(block)
                written += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != expected_digest:
            raise ValueError("noise transfer identity differs")
        files[name] = {"bytes": written, "SHA256": expected_digest}
    if wire.read(1):
        raise ValueError("unexpected trailing bytes")
    doc = {"request": REQUEST, "publication": publication,
           "source_roster_SHA256": ROSTER_SHA256, "cohort": "development", "images": 256, "files": files,
           "target_seal": True, "measured_images": 0}
    with (directory / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(doc, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(doc, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    args = parser.parse_args()
    if len(args.publication) != 40 or not set(args.publication) <= set("0123456789abcdef"):
        raise ValueError("publication must be the verified full commit")
    os.umask(0o077)
    def expired(_signum, _frame):
        raise TimeoutError("noise bank receiver budget expired")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(45)
    try:
        receive(BASE / LABEL, sys.stdin.buffer, args.publication)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
