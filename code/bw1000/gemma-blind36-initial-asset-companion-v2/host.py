"""Enter the already approved Gemma image with only the preflight entry replaced."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ORIGINAL = Path(
    "/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/"
    "training-gemma-usgs72-blind36-58925c79ef98-v1-source"
)
ORIGINAL_MANIFEST = "34380f1660e572ecbf232ee7e1ba1be672fd06c8789e2d603ca95e5c9b3a90bd"
GUEST_ENTRY = "/gemma-companion-v2/asset_preflight.py"


def compose_image(argv, companion):
    command = list(argv)
    image = next(i for i, part in enumerate(command) if part.endswith("/containers/dtk26-sandbox-r1"))
    shell = image + 5
    original_entry = str(ORIGINAL / "core/asset_preflight.py")
    if not (
        "--rocm" not in command
        and command[image + 1 : shell] == ["/bin/bash", "--noprofile", "--norc", "-c"]
        and command[shell].endswith(f"exec python3 -B -u {original_entry}")
        and command[shell].count(original_entry) == 1
    ):
        raise ValueError("Original zero-GRES container command has changed")
    command[shell] = command[shell].replace(original_entry, GUEST_ENTRY)
    command[image:image] = ["--bind", f"{companion / 'asset_preflight.py'}:{GUEST_ENTRY}:ro"]
    return command


def main():
    companion = Path(__file__).resolve().parent
    if companion.name != "gemma-blind36-initial-asset-companion-v2":
        raise ValueError("Unexpected installed companion directory")
    manifest_raw = (companion / "SOURCE_FILES.json").read_bytes()
    if len(sys.argv) != 2 or sha256(manifest_raw).hexdigest() != sys.argv[1]:
        raise ValueError("Companion manifest differs from the bound launch")
    manifest = json.loads(manifest_raw)
    if set(manifest["files"]) != {"asset_preflight.py", "host.py"}:
        raise ValueError("Unexpected companion members")
    for name, expected in manifest["files"].items():
        path = companion / name
        raw = path.read_bytes()
        if path.resolve(strict=True) != path or not path.is_file() or len(raw) != expected["bytes"]:
            raise ValueError("Companion file identity differs")
        if sha256(raw).hexdigest() != expected["SHA256"]:
            raise ValueError("Companion source digest differs")
    if sha256((ORIGINAL / "SOURCE_FILES.json").read_bytes()).hexdigest() != ORIGINAL_MANIFEST:
        raise ValueError("Original Gemma source manifest differs")
    sys.path.insert(0, str(ORIGINAL))
    from controller import container_command, verify

    if verify("READ")["SHA256"] != ORIGINAL_MANIFEST:
        raise ValueError("Original Gemma package failed verification")
    argv = compose_image(container_command(preflight=True), companion)
    os.execve(argv[0], argv, dict(os.environ))


if __name__ == "__main__":
    main()
