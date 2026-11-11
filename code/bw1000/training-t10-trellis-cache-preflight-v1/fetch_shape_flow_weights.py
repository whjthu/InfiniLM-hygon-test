"""Acquire the exact public TRELLIS Shape Flow checkpoint into private storage."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REVISION = "81c44e3aefb363742e68d7eb4540bb65839d9ab3"
NAME = "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
SIZE = 2584574424
DIGEST = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
URL = f"https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/{REVISION}/ckpts/{NAME}?download=true"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.private_dir.resolve()
    root.mkdir(exist_ok=False)
    if shutil.disk_usage(root).free < SIZE + 256 * 1024**2:
        raise OSError("insufficient space for exact checkpoint")
    digest = hashlib.sha256()
    received = 0
    next_report = 256 * 1024**2
    part = root / (NAME + ".part")
    request = urllib.request.Request(URL, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=35) as response, part.open("xb") as stream:
        length = response.headers.get("Content-Length")
        if response.status != 200 or (length is not None and int(length) != SIZE):
            raise OSError("unexpected checkpoint response or content length")
        while received < SIZE:
            block = response.read(min(4 * 1024**2, SIZE - received))
            if not block:
                raise OSError("checkpoint stream ended early")
            stream.write(block)
            digest.update(block)
            received += len(block)
            if received >= next_report:
                print(json.dumps({"received_bytes": received, "expected_bytes": SIZE}), flush=True)
                next_report += 256 * 1024**2
        if response.read(1):
            raise OSError("unexpected bytes beyond checkpoint size")
        stream.flush()
        os.fsync(stream.fileno())
    if received != SIZE or digest.hexdigest() != DIGEST:
        raise OSError("checkpoint size or SHA256 differs from the A100 pin")
    part.rename(root / NAME)
    receipt = {"revision": REVISION, "file": NAME, "bytes": SIZE, "SHA256": DIGEST, "scope": "local private asset"}
    with (root / "receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
