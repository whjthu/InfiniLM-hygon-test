"""Qualify and fill only f13r4n03 while preserving its original zero-GRES record."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1194900"
NODE = "f13r4n03"
ORIGINAL = "1194900.378"
ORIGINAL_LABEL = "training80-admit-zero189-overlap-da23271e45384f55bcf7475b6b513409"
OLD = "1194900.189"
OLD_LABEL = "training80-admit-b0c43a4cae196c1f4f0a4a9ee71c0950"
ZERO_GRES = "cpu=1,mem=256M,node=1"
EIGHT_DCU = "cpu=8,gres/dcu=8,mem=16G,node=1"
LOCK = ".training80-dcu-standby.lock"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def document(path):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    require(0 < len(raw) <= 65536, "unexpected receipt size")
    return json.loads(raw), digest(raw)


def bind(claim, sources):
    require((claim["parent"], claim["node"], claim["original_probe"]["step"], claim["original_probe"]["label"])
            == (PARENT, NODE, ORIGINAL, ORIGINAL_LABEL), "fixed node or original probe changed")
    require(claim["protected_zero_GRES_accounting_conflicts"] == {
        OLD: {"name": OLD_LABEL, "resources": ZERO_GRES, "preserve_reservation": True, "replay_allowed": False}},
        "protected old step changed")
    require(claim["zero_GRES_controls"] == {} and claim["synthetic"] is True and claim["replay_allowed"] is False
            and (claim["GPU_cards"], claim["CPU_cores"], claim["memory_MiB"], claim["maximum_GPU_nodes"])
            == (8, 8, 16384, 1), "standby resource scope changed")
    require(set(sources) == {"pilot", "controller", "worker"}
            and all(digest(raw) == claim["dependencies_SHA256"][name] for name, raw in sources.items()),
            "published source bytes changed")
    require(set(claim["stages"]) == {"pilot", "standby"}
            and len({row["request"] for row in claim["stages"].values()}) == 2,
            "request scopes overlap")
    for stage, seconds in (("pilot", 120), ("standby", 10500)):
        row = claim["stages"][stage]
        require(re.fullmatch(r"[0-9a-f]{32}", row["request"]) and row["worker_seconds"] == seconds
                and row["slurm_seconds"] == seconds + 300, "invalid fixed stage budget")
    pci = claim["pci"]
    require(pci == sorted(set(pci)) and len(pci) == 8
            and all(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", p) for p in pci),
            "invalid fixed PCI set")
    helper = {"__name__": "published_colocated_pilot"}
    exec(compile(sources["pilot"], "published-pilot.py", "exec"), helper)
    helper.update(PARENT=PARENT, NODE=NODE, ORIGINAL_PROBE=ORIGINAL)
    probe = helper["PROBE"]
    for name, pin in claim["probe_source_SHA256"].items():
        require(digest((probe / name).read_bytes()) == pin, "deployed probe dependency changed")
    controller = {"__name__": "published_standby_controller"}
    exec(compile(sources["controller"], "published-controller.py", "exec"), controller)
    [row] = helper["rows"](ORIGINAL)
    require(row[1:4] == [ORIGINAL_LABEL, "COMPLETED", "0:0"] and row[5:] == [NODE, ZERO_GRES],
            "original .378 is not an exact successful zero-GRES terminal step")
    return helper, controller


def phase_details(claim, stage):
    phase = claim["stages"][stage]
    request = phase["request"]
    label = f"training80-dcu-standby-{request[:12]}-{NODE}"
    return phase, label, RESULTS / f"training80-dcu-standby-{request}"


def live_step(helper, controller, label):
    matches = []
    for line in helper["query"]("scontrol", "show", "step", PARENT, "--oneliner"):
        row = controller["pinned_fields"](line)
        if row.get("Name") == label:
            matches.append(row)
    if not matches:
        return None
    require(len(matches) == 1, "ambiguous standby step")
    [row] = matches
    require(row["NodeList"] == NODE and row["TRES"] == EIGHT_DCU
            and re.fullmatch(PARENT + r"\.[0-9]+", row["StepId"]), "wrong standby step location or GRES")
    return row["StepId"] if row["State"] == "RUNNING" else None


def pulse(folder, step, pci):
    worker, _ = document(folder / "worker.json")
    physical = worker["pci"]
    require(worker["step"] == step and worker["node"] == NODE and worker["synthetic"] is True
            and sorted(physical) == pci, "worker PCI or identity changed")
    result = {}
    for index, address in enumerate(physical):
        row, _ = document(folder / f"card-{index}.json")
        require(row["pci"] == address and row["active"] is True and row["synthetic"] is True
                and type(row["calls"]) is int and row["calls"] > 0
                and -1 <= time.time() - row["epoch"] <= 18, "one card has no fresh actual progress")
        busy = row.get("busy")
        result[address] = {"calls": row["calls"], "epoch": row["epoch"],
                           "busy": busy if type(busy) is int and 0 <= busy <= 100 else None}
    return result


def growing(first, second, pci):
    return (sorted(first) == sorted(second) == pci
            and all(second[p]["calls"] > first[p]["calls"] and second[p]["epoch"] > first[p]["epoch"] for p in pci))


def verify(claim, stage, helper, controller):
    phase, label, root = phase_details(claim, stage)
    recorded, _ = document(root / "step.json")
    step = recorded["step"]
    require(recorded["request"] == phase["request"] and live_step(helper, controller, label) == step,
            "original recorded stage is not running")
    helper["guard"](claim, controller, own=(step, label))
    first = pulse(root / NODE, step, claim["pci"])
    time.sleep(5.6)
    second = pulse(root / NODE, step, claim["pci"])
    helper["guard"](claim, controller, own=(step, label))
    require(live_step(helper, controller, label) == step and growing(first, second, claim["pci"]),
            "original step did not advance on all eight PCI devices")
    evidence = dict(request=phase["request"], step=step, node=NODE, stage=stage,
                    first=first, second=second, growing_cards=8, epoch=time.time())
    helper["save"](root / "verified-progress.json", evidence)
    print(json.dumps(evidence, sort_keys=True), flush=True)


def require_terminal_pilot(claim, helper):
    phase, label, root = phase_details(claim, "pilot")
    progress, progress_sha = document(root / "verified-progress.json")
    binding, _ = document(root / "step.json")
    terminal, terminal_sha = document(root / NODE / "terminal.json")
    step = binding["step"]
    require(binding["request"] == progress["request"] == phase["request"] and progress["stage"] == "pilot"
            and terminal["step"] == progress["step"] == step and progress["node"] == terminal["node"] == NODE
            and terminal["state"] == "WORKER_RETURNED" and growing(progress["first"], progress["second"], claim["pci"]),
            "pilot lacks independent two-point evidence or a normal return")
    [row] = helper["rows"](step)
    require(row[1:4] == [label, "COMPLETED", "0:0"] and row[5:] == [NODE, EIGHT_DCU],
            "pilot has not terminated successfully with eight GRES")
    return dict(step=step, progress_SHA256=progress_sha, worker_terminal_SHA256=terminal_sha)


def launch(claim, stage, sources, helper, controller):
    import fcntl

    phase, label, root = phase_details(claim, stage)
    request = phase["request"]
    with (RESULTS / LOCK).open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        helper["guard"](claim, controller)
        require(not any(row[1] == label for line in helper["query"]("sacct", "-n", "-P", "-j", PARENT,
                        "-o", helper["ACCOUNTING_COLUMNS"]) if len(row := line.split("|")) == 7),
                "this unique label already exists")
        pilot = require_terminal_pilot(claim, helper) if stage == "standby" else None
        root.mkdir(mode=0o700)
        folder = root / NODE
        folder.mkdir()
        (root / "worker.py").write_bytes(sources["worker"])
        clean = helper["probe"](request, claim)
        helper["guard"](claim, controller)
        helper["save"](folder / "expected.json", dict(node=NODE, parent=PARENT, pci=claim["pci"],
                       seconds=phase["worker_seconds"], label=label, request=request))
        intent = dict(request=request, node=NODE, parent=PARENT, stage=stage, epoch=time.time(), synthetic=True,
                      replay_allowed=False, pre=clean, pilot_terminal=pilot)
        helper["save"](root / "intent.json", intent)
        args = controller["command"](NODE, PARENT, label, root, folder, phase["worker_seconds"])
        with (folder / "slurm.stdout.private").open("xb") as stdout, (folder / "slurm.stderr.private").open("xb") as stderr:
            child = subprocess.Popen([args[0], "--threads=1", *args[1:]], stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
        submission = dict(request=request, label=label, node=NODE, parent=PARENT, launcher_pid=child.pid,
                          worker_seconds=phase["worker_seconds"], synthetic=True)
        helper["save"](root / "submit.json", submission)
        helper["save"](root / "submission-receipt.json", dict(request=request, synthetic=True, submit_records=[submission]))
        limit = time.monotonic() + 14
        while time.monotonic() < limit:
            step = live_step(helper, controller, label)
            if step:
                helper["guard"](claim, controller, own=(step, label))
                binding = dict(submission, step=step, epoch=time.time())
                helper["save"](root / "step.json", binding)
                print(json.dumps(binding, sort_keys=True), flush=True)
                return
            if child.poll() is not None:
                break
            time.sleep(0.3)
        print(json.dumps(dict(submission, step=None, remote_outcome="UNKNOWN", replay_allowed=False)), flush=True)


def run(action, stage, claim, sources):
    require(action in {"launch", "verify"} and stage in {"pilot", "standby"}, "unsupported fixed action")
    os.umask(0o077)
    helper, controller = bind(claim, sources)
    if action == "launch":
        launch(claim, stage, sources, helper, controller)
    else:
        verify(claim, stage, helper, controller)
