"""Continue the f12r2n13 standby with its current zero-GRES controls."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
import subprocess


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1103606"
NODE = "f12r2n13"
OLD_STEP = "1103606.2545"
OLD_REQUEST = "949f1e83051f4e84a6349d61dbdbe141"
SUCCESSOR = "0f8e2cdecb2340dab2d26dd7410ab13a"
OLD_LABEL = f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}"
PLANNED_OLD_END_EPOCH = 1795517157.4568815
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
            and sorted(worker["pci"]) == expected["pci"] and type(worker["worker_pid"]) is int and worker["worker_pid"] > 0
            and worker["synthetic"] is True, "preceding worker device identity changed")
    require((terminal["node"], terminal["step"], terminal["state"]) == (NODE, OLD_STEP, "WORKER_RETURNED"),
            "preceding worker did not return normally")
    require(type(terminal["epoch"]) in (int, float)
            and terminal_sha == prior["terminal_SHA256"]
            and PLANNED_OLD_END_EPOCH - 1 <= terminal["epoch"] <= time.time() + 2,
            "preceding worker did not reach its original natural budget")
    return dict(accounting=row, expected_SHA256=expected_sha, worker_SHA256=worker_sha,
                terminal_SHA256=terminal_sha, worker_terminal_epoch=terminal["epoch"])


def run(claim, sources):
    require((claim["parent"], claim["node"], claim["original_probe"], claim["GPU_cards"], claim["CPU_cores"],
             claim["memory_MiB"], claim["worker_seconds"], claim["slurm_seconds"])
            == (PARENT, NODE, "1103606.1913", 8, 8, 16384, 10500, 10800), "node or resource budget changed")
    require(claim["synthetic"] is True and claim["preemptible"] is True and claim["replay"] is False
            and claim["qualified_pilot"]["step"] == "1103606.1915", "qualified standby scope changed")
    require(claim["protected_zero_GRES_accounting_conflicts"] == PROTECTED_CPU,
            "original zero-GRES reservation changed")
    controls = claim["zero_GRES_controls"]
    require(len(controls) == 13 and all(re.fullmatch(PARENT + r"\.[0-9]+", h) and
            re.fullmatch(r"training80-standby-compute-[0-9a-f]{12}", name) for h, name in controls.items()),
            "live control scope changed")
    require(set(sources) == {"pilot", "controller", "worker"}
            and all(sha256(raw).hexdigest() == claim["dependencies_SHA256"][name] for name, raw in sources.items()),
            "published dependency bytes changed")
    request = claim["request"]
    used = {OLD_REQUEST, "b4c58b1681d5485e91d2c76ac15aad02", "56860672a8194e308af016757aaa6ece", claim["qualified_pilot"]["request"]}
    require(request == SUCCESSOR and request not in used, "wrong renewal request")

    pilot = {"__name__": "published_colocated_pilot"}
    exec(compile(sources["pilot"], "published-pilot.py", "exec"), pilot)
    controller = {"__name__": "published_standby_controller"}
    exec(compile(sources["controller"], "published-controller.py", "exec"), controller)
    with (RESULTS / ".training80-dcu-standby.lock").open("ab") as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preceding = verify_previous(pilot["rows"], document, claim)
        admit(pilot, controller, claim)
        pre_step, = pilot["rows"]("1103606.2544")
        require(pre_step[1:4] == ["training80-admit-control-pilot-" + OLD_REQUEST, "COMPLETED", "0:0"]
                and pre_step[5:] == [NODE, "cpu=1,mem=256M,node=1"], "old zero-GRES probe changed")
        require(all(sha256((pilot["PROBE"] / name).read_bytes()).hexdigest() == pin
                    for name, pin in claim["probe_source_SHA256"].items()), "probe source changed")
        [past_pilot] = pilot["rows"](claim["qualified_pilot"]["step"])
        require(past_pilot[1:4] == [f'training80-dcu-standby-{claim["qualified_pilot"]["request"][:12]}-{NODE}',
                                    "COMPLETED", "0:0"] and past_pilot[5:] == [NODE, EIGHT_DCU],
                "qualified pilot terminal drift")
        os.umask(0o077)
        root = RESULTS / f"training80-dcu-standby-{request}"
        root.mkdir(mode=0o700)
        (root / "worker.py").write_bytes(sources["worker"])
        folder = root / NODE
        folder.mkdir()
        pre = pilot["probe"](request, claim)
        pilot["save"](root / "probe-candidate.json", pre)
        deadline = time.monotonic() + 5
        while True:
            observed = pilot["rows"](pre["step"])
            if len(observed) == 1 and observed[0][2:4] == ["COMPLETED", "0:0"]:
                break
            require(not observed or len(observed) == 1 and observed[0][2] == "RUNNING",
                    "fresh zero-GRES probe C0 failed")
            require(time.monotonic() < deadline, "fresh zero-GRES accounting missing")
            time.sleep(0.25)
        require(observed[0][1] == "training80-admit-control-pilot-" + request
                and observed[0][5:] == [NODE, "cpu=1,mem=256M,node=1"], "fresh zero-GRES probe identity changed")
        admit(pilot, controller, claim)
        label = f"training80-dcu-standby-{request[:12]}-{NODE}"
        seconds = claim["worker_seconds"]
        pilot["save"](folder / "expected.json", dict(node=NODE, parent=PARENT, pci=claim["pci"],
                      seconds=seconds, label=label, request=request))
        pilot["save"](root / "intent.json", dict(request=request, node=NODE, parent=PARENT, pre=pre,
                      previous=preceding, epoch=time.time(), synthetic=True, replay_allowed=False))
        argv = controller["command"](NODE, PARENT, label, root, folder, seconds)
        with (folder / "slurm.stdout.private").open("xb") as out, (folder / "slurm.stderr.private").open("xb") as err:
            child = subprocess.Popen([argv[0], "--threads=1", *argv[1:]], stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, start_new_session=True)
        result = dict(request=request, label=label, node=NODE, parent=PARENT, pre=pre, previous=preceding,
                      launcher_pid=child.pid, preserved_controls=len(controls)+1, worker_seconds=seconds)
        pilot["save"](root / "submit.json", result)
        pilot["save"](root / "submission-receipt.json", dict(request=request, node=NODE, parent=PARENT,
                      synthetic=True, submit_records=[result]))
    print(json.dumps(result, sort_keys=True), flush=True)


def admit(pilot, controller, claim):
    pilot["guard"](claim, controller)
    controls = set(claim["zero_GRES_controls"]) | set(PROTECTED_CPU)
    expand = {}
    for line in pilot["query"]("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        step, hosts = line.split("|")
        if step in {PARENT + ".batch", PARENT + ".extern"}:
            continue
        require(re.fullmatch(PARENT + r"\.[0-9]+", step) and hosts and hosts not in {"(null)", "None", "Unknown", "N/A"},
                "unresolved queued step membership")
        if hosts not in expand:
            listed = [hosts] if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", hosts) else pilot["query"](
                "scontrol", "show", "hostnames", hosts)
            require(listed and all(re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", member) for member in listed),
                    "unresolved queued host expansion")
            expand[hosts] = frozenset(listed)
        require(NODE not in expand[hosts] or step in controls, "additional target step reported by squeue")
