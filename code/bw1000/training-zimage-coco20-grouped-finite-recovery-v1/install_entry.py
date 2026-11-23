"""Stage the published recovery package once and enter its foreground controller."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys


FILES = {"CLAIM.json", "controller.py", "resume_entry.py", "install_entry.py"}
PYTHON = "/usr/local/python3.12/bin/python3"
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")


def launch():
    line = sys.stdin.buffer.readline(180000)
    if not line.endswith(b"\n") or len(line) >= 180000:
        raise ValueError("Invalid bounded recovery package")
    payload = json.loads(base64.b64decode(line.strip(), validate=True))
    publication = payload["publication"]
    if re.fullmatch("[a-f0-9]{40}", publication) is None:
        raise ValueError("Missing published revision")
    files = {name: base64.b64decode(raw, validate=True) for name, raw in payload["files"].items()}
    if set(files) != FILES | {"SOURCE_FILES.json"}:
        raise ValueError("Unexpected package member")
    inventory = json.loads(files["SOURCE_FILES.json"])["files"]
    if set(inventory) != FILES:
        raise ValueError("Unexpected source inventory")
    for name, pin in inventory.items():
        if pin != {"bytes": len(files[name]), "SHA256": sha256(files[name]).hexdigest()}:
            raise ValueError("Source member changed: " + name)
    claim = json.loads(files["CLAIM.json"])
    if claim["label"] != "training-zimage-coco20-grouped-finite-recovery-v1" \
            or claim["request"] != "6e1148b90e76412785df3fc3aacf68816":
        raise ValueError("Original recovery identity differs")
    target = RESULTS / claim["label"]
    os.umask(0o077)
    target.mkdir()
    for name, data in files.items():
        with (target/name).open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        (target/name).chmod(0o400)
    os.execv(PYTHON, [PYTHON, "-B", "-u", str(target/"controller.py"), publication])


if __name__ == "__main__":
    launch()
