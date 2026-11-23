"""Run one guarded eight-DCU standby pilot alongside pinned zero-GRES controls."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time


PARENT = "1103606"
NODE = "f12r2n13"
PYTHON = "/usr/local/python3.12/bin/python3"
PROBE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
             "training80-admission-standby-9cfcaff1ae2342868d72861af5392acd")
ACCOUNTING_COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
ORIGINAL_PROBE = "1103606.1913"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def query(*args):
    return subprocess.check_output(args, text=True, timeout=10).splitlines()


def rows(step):
    return [row for line in query("sacct", "-n", "-P", "-j", step, "-o", ACCOUNTING_COLUMNS)
            if (row := line.split("|"))[0] == step and len(row) == 7]


def step_resources(row):
    return row.get("TRES", row.get("AllocTRES"))


def guard(claim, namespace, *, own=None):
    fields = namespace["pinned_fields"]
    [line] = query("scontrol", "show", "job", "-o", PARENT)
    parent = fields(line)
    require(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING"
            and parent["UserId"].endswith(f"({os.getuid()})")
            and "gres/dcu=320" in parent["AllocTRES"].split(",")
            and NODE in query("scontrol", "show", "hostnames", parent["NodeList"]), "parent ownership changed")
    local = {}
    for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
        row = fields(line)
        step = row.get("StepId", "")
        if step in {PARENT + ".batch", PARENT + ".extern"}:
            continue
        require(re.fullmatch(PARENT + r"\.[0-9]+", step) and row.get("NodeList"), "unidentified numbered step")
        members = [row["NodeList"]] if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", row["NodeList"]) else query(
            "scontrol", "show", "hostnames", row["NodeList"])
        if NODE in members:
            require(members == [NODE] and step not in local, "unresolved shared node membership")
            local[step] = (row["Name"], row["State"], step_resources(row))
    expected = {step: (name, "RUNNING", "cpu=2,mem=2G,node=1")
                for step, name in claim["zero_GRES_controls"].items()}
    for step, record in claim["protected_zero_GRES_accounting_conflicts"].items():
        name, resource = record["name"], record["resources"]
        require(local.get(step) == (name, "TIMEOUT", resource), "protected CPU controller changed")
        [accounting] = rows(step)
        require(accounting[1:4] == [name, "RUNNING", "0:0"]
                and accounting[5:] == [NODE, resource], "protected CPU accounting changed")
        expected[step] = (name, "TIMEOUT", resource)
    if own:
        step, label = own
        expected[step] = (label, "RUNNING", "cpu=8,gres/dcu=8,mem=16G,node=1")
    require(local == expected, "shared node control or GPU step changed")


def probe(request, claim):
    result = subprocess.run(["srun", "--jobid=" + PARENT, "--job-name=training80-admit-control-pilot-" + request,
                             "--exact", "--overlap", "--immediate=5", "--nodes=1", "--ntasks=1",
                             "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=256M", "--gres=none",
                             "--time=00:01:00", "--kill-on-bad-exit=0", "--nodelist=" + NODE,
                             PYTHON, "-E", "-s", "-B", "-u", str(PROBE / "collect_training80_dispatch_snapshot.py"),
                             "node", request, PARENT, "[]"], capture_output=True, check=True, timeout=15)
    require(not result.stderr, "zero-GRES probe returned warnings")
    row = json.loads(result.stdout)
    require(row["request_id"] == request and row["parent"] == PARENT and row["node"] == NODE
            and row["observer_uid"] == os.getuid() and re.fullmatch(PARENT + r"\.[0-9]+", row["step"])
            and sorted(row["PCIs"]) == claim["pci"] and row["ports_checked"] == []
            and row["kfd_process_count"] == row["own_device_fd_process_count"] == 0,
            "independent PCI or device holder check failed")
    return dict(step=row["step"], payload_SHA256=sha256(result.stdout).hexdigest(), device_checks_passed=True)


def save(path, evidence):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def run(root, claim, worker, controller):
    import fcntl

    request = root.name.removeprefix("training80-dcu-standby-")
    require(re.fullmatch(r"[a-f0-9]{32}", request), "invalid pilot request identity")
    require(sha256(worker).hexdigest() == claim["worker_SHA256"]
            and sha256(controller).hexdigest() == claim["controller_SHA256"]
            and all(sha256((PROBE / file).read_bytes()).hexdigest() == pin
                    for file, pin in claim["probe_source_SHA256"].items()), "pilot dependency identity drift")
    namespace = {"__name__": "standby_v2_qualified_controller"}
    exec(compile(controller, "qualified-controller.py", "exec"), namespace)
    require(any(row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE and row[6] == "cpu=1,mem=256M,node=1"
                for row in rows(ORIGINAL_PROBE)), "original zero-GRES probe has no matching terminal accounting")
    label = f"training80-dcu-standby-{request[:12]}-{NODE}"
    require(not any(row[1] == label for row in [line.split("|") for line in query(
        "sacct", "-n", "-P", "-j", PARENT, "-o", ACCOUNTING_COLUMNS)] if len(row) == 7), "pilot label already used")
    seconds = 120
    with (root.parent / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(claim, namespace)
        folder = root / NODE
        folder.mkdir(parents=False)
        pre = probe(request, claim)
        guard(claim, namespace)
        save(folder / "expected.json", dict(node=NODE, parent=PARENT, pci=claim["pci"], seconds=seconds,
                                           label=label, request=request))
        save(root / "intent.json", dict(request=request, node=NODE, parent=PARENT, pre=pre,
                                        epoch=time.time(), synthetic=True, replay_allowed=False))
        argv = namespace["command"](NODE, PARENT, label, root, folder, seconds)
        with (folder / "slurm.stdout.private").open("xb") as out, (folder / "slurm.stderr.private").open("xb") as err:
            process = subprocess.Popen([argv[0], "--threads=1", *argv[1:]], stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
        save(root / "submit.json", dict(request=request, label=label, launcher_pid=process.pid,
                                        worker_seconds=seconds, synthetic=True))
    print(json.dumps(dict(request=request, label=label, node=NODE, pre=pre, launcher_pid=process.pid)), flush=True)
