"""Stage the exact published SFT packet and start its original bounded controller."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL = "training-qwen-usgs-table-sft2-hip-v1"
NODE = "f13r1n10"
PARENT = "1103606"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def query(*args):
    return subprocess.check_output(args, timeout=10, text=True).splitlines()


def write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def record(path, payload):
    write(path, json.dumps(payload, allow_nan=False).encode("utf8"))


def main(payload):
    os.umask(0o077)
    source = {name: base64.b64decode(raw, validate=True) for name, raw in payload["files"].items()}
    claim = json.loads(source["CLAIM.json"])
    assert (claim["label"], claim["node"], claim["parent"], claim["optimizer_updates"], claim["device_count"]) == (
        LABEL, NODE, PARENT, 2, 8)
    assert claim["GPU_GO"] is True and claim["training_GO"] is True
    assert re.fullmatch("[0-9a-f]{32}", claim["request"]) and re.fullmatch("[0-9a-f]{40}", payload["publication"])
    pinned = json.loads(source["SOURCE_FILES.json"])["files"]
    assert set(source) == set(pinned) | {"SOURCE_FILES.json"}
    for name, pin in pinned.items():
        assert Path(name).name == name and len(source[name]) == pin["bytes"] and digest(source[name]) == pin["SHA256"]
    root = ROOT / LABEL
    assert not root.exists()
    expected_labels = {LABEL + "-" + phase for phase in ("pre", "gpu", "post")}
    assert not any(line.split("|")[1] in expected_labels for line in
                   query("sacct", "-nP", "-j", PARENT, "--format=JobIDRaw,JobName%100") if "|" in line)
    root.mkdir(mode=0o700)
    area = root / "src"
    area.mkdir(mode=0o700)
    for name, raw in source.items():
        write(area / name, raw)
    epoch = time.time()
    receipt = dict(epoch=epoch, deadline=epoch+1620, node=NODE, parent=PARENT, arm="continuous",
                   request_id=claim["request"], publication=payload["publication"],
                   max_GPU_invocations=1, planned_optimizer_updates=2)
    record(root / "intent.json", receipt)
    with (root / "controller.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(area / "treasury_continuous.py"), "host"],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    receipt["controller_PID"] = child.pid
    record(root / "launch-receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main(PAYLOAD)
