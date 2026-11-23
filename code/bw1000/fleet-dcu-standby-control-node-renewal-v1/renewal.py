"""Renew the qualified co-resident standby after the original worker exits."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1103606"
NODE = "f12r2n13"
OLD_STEP = "1103606.1917"
OLD_REQUEST = "56860672a8194e308af016757aaa6ece"
OLD_LABEL = f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}"
EIGHT_DCU = "cpu=8,gres/dcu=8,mem=16G,node=1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_old(rows, claim):
    previous = claim["previous_standby"]
    require((previous["request"], previous["step"], previous["label"]) == (OLD_REQUEST, OLD_STEP, OLD_LABEL),
            "unexpected previous standby identity")
    [row] = rows(OLD_STEP)
    require(row[1:4] == [OLD_LABEL, "COMPLETED", "0:0"] and row[5:] == [NODE, EIGHT_DCU],
            "the prior standby does not have exact successful terminal accounting")
    old = RESULTS / f"training80-dcu-standby-{OLD_REQUEST}" / NODE
    expected = json.loads((old / "expected.json").read_bytes())
    terminal = json.loads((old / "terminal.json").read_bytes())
    require(expected["request"] == OLD_REQUEST and expected["node"] == NODE and expected["parent"] == PARENT
            and expected["pci"] == claim["pci"] and terminal["node"] == NODE and terminal["step"] == OLD_STEP
            and terminal["state"] == "WORKER_RETURNED", "previous worker evidence differs")


def run(claim, sources):
    require((claim["parent"], claim["node"], claim["original_probe"], claim["worker_seconds"], claim["slurm_seconds"])
            == (PARENT, NODE, "1103606.1913", 10500, 10800), "scope or budget changed")
    require(claim["synthetic"] is True and claim["preemptible"] is True and claim["replay"] is False
            and claim["qualified_pilot"]["step"] == "1103606.1915", "co-resident qualification changed")
    require(claim["protected_zero_GRES_accounting_conflicts"] == {
        "1103606.1691": {"name": "training80-syncbusy-5af4008d8703-control", "preserve_reservation": True,
                         "replay_allowed": False, "resources": "cpu=2,mem=512M,node=1"}},
        "the original zero-GRES reservation must be preserved")
    controls = claim["zero_GRES_controls"]
    require(controls and all(re.fullmatch(PARENT + r"\.[0-9]+", step)
                             and re.fullmatch(r"training80-standby-compute-[0-9a-f]{12}", label)
                             for step, label in controls.items()), "unexpected live zero-GRES control identity")
    require(set(sources) == {"standby", "pilot", "controller", "worker"}
            and all(sha256(raw).hexdigest() == claim["dependencies_SHA256"][name] for name, raw in sources.items()),
            "published dependencies differ")
    request = claim["request"]
    require(re.fullmatch(r"[0-9a-f]{32}", request) and request != OLD_REQUEST, "invalid fixed renewal request")

    pilot = {"__name__": "published_co_resident_pilot"}
    exec(compile(sources["pilot"], "published-pilot.py", "exec"), pilot)
    controller = {"__name__": "published_standby_controller"}
    exec(compile(sources["controller"], "published-controller.py", "exec"), controller)
    verify_old(pilot["rows"], claim)
    pilot["guard"](claim, controller)

    standby = {"__name__": "published_co_resident_standby"}
    exec(compile(sources["standby"], "published-standby.py", "exec"), standby)
    os.umask(0o077)
    root = RESULTS / f"training80-dcu-standby-{request}"
    root.mkdir(mode=0o700)
    (root / "worker.py").write_bytes(sources["worker"])
    standby["run"](root, claim, sources["worker"], sources["controller"], sources["pilot"])
