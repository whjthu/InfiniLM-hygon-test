"""Continue the f12r2n13 standby with its current zero-GRES controls."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1103606"
NODE = "f12r2n13"
OLD_STEP = "1103606.2190"
OLD_REQUEST = "897ffd8b4e6f465e95e84e6a855cdaeb"
OLD_LABEL = f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}"
PLANNED_OLD_END_EPOCH = 1795491344.2637608
EIGHT_DCU = "cpu=8,gres/dcu=8,mem=16G,node=1"
PROTECTED_CPU = {"1103606.1691": {
    "name": "training80-syncbusy-5af4008d8703-control", "resources": "cpu=2,mem=512M,node=1",
    "preserve_reservation": True, "replay_allowed": False}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def document(path):
    with path.open("rb") as stream:
        raw = stream.read(8193)
    require(0 < len(raw) <= 8192, "wrong old worker evidence size")
    return json.loads(raw), sha256(raw).hexdigest()


def verify_previous(rows, read, claim):
    prior = claim["previous_standby"]
    require((prior["request"], prior["step"], prior["label"], prior["worker_expected_end_epoch"])
            == (OLD_REQUEST, OLD_STEP, OLD_LABEL, PLANNED_OLD_END_EPOCH), "wrong preceding standby binding")
    evidence = rows(OLD_STEP)
    require(len(evidence) == 1, "preceding standby accounting is not unique")
    [row] = evidence
    require(row[:4] == [OLD_STEP, OLD_LABEL, "COMPLETED", "0:0"] and row[5:] == [NODE, EIGHT_DCU],
            "preceding standby has no successful terminal accounting")
    folder = RESULTS / f"training80-dcu-standby-{OLD_REQUEST}" / NODE
    expected, expected_sha = read(folder / "expected.json")
    worker, worker_sha = read(folder / "worker.json")
    terminal, terminal_sha = read(folder / "terminal.json")
    require((expected["request"], expected["parent"], expected["node"], expected["label"], expected["seconds"])
            == (OLD_REQUEST, PARENT, NODE, OLD_LABEL, 10500) and expected["pci"] == claim["pci"],
            "preceding worker PCI or scope changed")
    require((expected_sha, worker_sha) == (prior["expected_SHA256"], prior["worker_SHA256"]),
            "preceding standby original bytes changed")
    require((worker["node"], worker["step"], worker["epoch"]) == (NODE, OLD_STEP, prior["worker_start_epoch"])
            and sorted(worker["pci"]) == expected["pci"], "preceding worker device identity changed")
    require((terminal["node"], terminal["step"], terminal["state"]) == (NODE, OLD_STEP, "WORKER_RETURNED"),
            "preceding worker did not return normally")
    return dict(accounting=row, expected_SHA256=expected_sha, worker_SHA256=worker_sha, terminal_SHA256=terminal_sha)


def run(claim, sources):
    require((claim["parent"], claim["node"], claim["original_probe"], claim["GPU_cards"], claim["CPU_cores"],
             claim["memory_MiB"], claim["worker_seconds"], claim["slurm_seconds"])
            == (PARENT, NODE, "1103606.1913", 8, 8, 16384, 10500, 10800), "node or resource budget changed")
    require(claim["synthetic"] is True and claim["preemptible"] is True and claim["replay"] is False
            and claim["qualified_pilot"]["step"] == "1103606.1915", "qualified standby scope changed")
    require(claim["protected_zero_GRES_accounting_conflicts"] == PROTECTED_CPU,
            "original zero-GRES reservation changed")
    controls = claim["zero_GRES_controls"]
    require(len(controls) == 12 and all(re.fullmatch(PARENT + r"\.[0-9]+", h) and
            re.fullmatch(r"training80-standby-compute-[0-9a-f]{12}", name) for h, name in controls.items()),
            "live control scope changed")
    require(set(sources) == {"standby", "pilot", "controller", "worker"}
            and all(sha256(raw).hexdigest() == claim["dependencies_SHA256"][name] for name, raw in sources.items()),
            "published dependency bytes changed")
    request = claim["request"]
    used = {OLD_REQUEST, "56860672a8194e308af016757aaa6ece", claim["qualified_pilot"]["request"]}
    require(re.fullmatch(r"[0-9a-f]{32}", request) and request not in used, "wrong renewal request")

    pilot = {"__name__": "published_colocated_pilot"}
    exec(compile(sources["pilot"], "published-pilot.py", "exec"), pilot)
    controller = {"__name__": "published_standby_controller"}
    exec(compile(sources["controller"], "published-controller.py", "exec"), controller)
    preceding = verify_previous(pilot["rows"], document, claim)
    pilot["guard"](claim, controller)
    standby = {"__name__": "published_colocated_standby"}
    exec(compile(sources["standby"], "published-standby.py", "exec"), standby)
    os.umask(0o077)
    root = RESULTS / f"training80-dcu-standby-{request}"
    root.mkdir(mode=0o700)
    (root / "worker.py").write_bytes(sources["worker"])
    print(json.dumps(dict(co_resident_renewal_gate=True, request=request, previous=preceding, epoch=time.time()),
                     sort_keys=True), flush=True)
    standby["run"](root, claim, sources["worker"], sources["controller"], sources["pilot"])
