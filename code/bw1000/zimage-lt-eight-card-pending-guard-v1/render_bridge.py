"""Freeze the existing standby compute bridge with one final shared-lock admission hook."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve(strict=True).parent
BASE = HERE.with_name("fleet-dcu-standby-compute-bridge-v1") / "bridge.py"
CONTROLLER = HERE.with_name("fleet-dcu-standby-v2") / "controller.py"
BASE_SHA = "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360"
CONTROLLER_SHA = "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"


def bind(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen fleet source changed: " + path.name)
    return raw


def render():
    raw = bind(BASE, BASE_SHA)
    controller = bind(CONTROLLER, CONTROLLER_SHA)
    boundary = b"        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        owners, active = live_inventory()"
    if controller.count(boundary) != 1:
        raise ValueError("Final admission is outside the expected scheduler lock")
    source = (HERE / "pending.py").read_bytes()
    before = b'    try:\n        namespace["run"](inputs)\n'
    if raw.count(before) != 1:
        raise ValueError("Frozen bridge call boundary changed")
    lines = (f'    pending_namespace = {{"__name__": "o01_pending_guard"}}\n'
             f'    exec(compile({source!r}, "o01-pending-guard.py", "exec"), pending_namespace)\n'
             '    pending_namespace["install"](namespace, inputs, directory)\n').encode()
    result = raw.replace(before, lines + before)
    compile(result, "guarded_bridge.py", "exec")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = render()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"bytes": len(raw), "sha256": sha256(raw).hexdigest()}))
