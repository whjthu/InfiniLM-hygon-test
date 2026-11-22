"""Stage the published package and run exactly one bounded recovery diagnostic."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-zimage-COCO20-recovery-boundaries-hip-v2"
ROOT = PROJECT / "results" / LABEL
PARENT = "1103606"
NODE = "f13r2n07"
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%100,State%24,ExitCode,ElapsedRaw,NodeList%60,AllocTRES%150"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def query(*argv):
    return subprocess.check_output(argv, timeout=8, text=True).splitlines()


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def accounting(label):
    rows = [row.split("|") for row in query("sacct", "-nP", "-j", PARENT, "--format", COLUMNS)]
    found = [row for row in rows if len(row) == 7 and row[1] == label]
    require(len(found) <= 1, "A claimed Slurm label is duplicated")
    return found[0] if found else None


def current_placement():
    [line] = query("scontrol", "show", "job", PARENT, "-o")
    job = fields(line)
    require(job["JobId"] == PARENT and job["JobState"] == "RUNNING"
            and "gres/dcu=320" in job["AllocTRES"].split(",") and job["NumNodes"] == "40"
            and job["UserId"].endswith(f"({os.getuid()})")
            and NODE in query("scontrol", "show", "hostnames", job["NodeList"]), "Parent or node differs")
    steps = query("scontrol", "show", "step", PARENT, "-o")
    require(bool(steps), "Live step inventory unavailable")
    for row in steps:
        step = fields(row)
        if step["StepId"] in {PARENT + ".batch", PARENT + ".extern"}:
            continue
        require(re.fullmatch(PARENT + r"\.\d+", step["StepId"])
                and NODE not in query("scontrol", "show", "hostnames", step["NodeList"]),
                "An active numbered step controls the claimed node")


def bootstrap(publication, manifest_sha):
    from slurm_accounting import wait_terminal

    claim = json.loads((ROOT / "CLAIM.json").read_bytes())
    try:
        current_placement()
        phases = ("admission", "pre", "continuous", "between", "resume20", "post")
        require(not any(accounting(f"{LABEL}-{phase}") for phase in phases), "A claimed phase already exists")
        short_label = LABEL + "-admission"
        command = [
            "srun", f"--jobid={PARENT}", f"--job-name={short_label}", "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", f"--nodelist={NODE}", "--cpus-per-task=2", "--mem=4096M",
            "--gres=none", "--time=1", PYTHON, "-I", "-S", "-B", str(ROOT / "node_inventory.py"), PARENT, NODE,
        ]
        probe = subprocess.run(command, capture_output=True, text=True, timeout=70)
        visibility = wait_terminal(lambda: accounting(short_label), time.monotonic() + 18,
                                   interval=1, max_observations=12)
        save("admission-accounting.json", visibility)
        row = visibility["row"]
        require(probe.returncode == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                and "gres/" not in row[6], "Independent admission is incomplete")
        [value] = [json.loads(line) for line in probe.stdout.splitlines() if line.startswith("{")]
        require(value["step"] == row[0] and value["node"] == NODE
                and len(value["PCIs"]) == len(set(value["PCIs"])) == 8
                and value["kfd_process_count"] == value["own_device_fd_process_count"] == 0,
                "Target admission found device holders or a changed device identity")
        current_placement()
        epoch = time.time()
        save("intent.json", {"request": claim["request"], "epoch": epoch, "deadline": epoch + claim["whole_seconds"],
                             "manifest_SHA256": manifest_sha, "publication": publication,
                             "single_parent_admission": value})
    except Exception as error:
        save("bootstrap-failure.json", {"request": claim["request"], "error_type": type(error).__name__,
                                       "message": str(error)[:180]})
        raise
    os.execv(PYTHON, [PYTHON, "-B", "-u", str(ROOT / "controller.py")])


def install(publication, manifest_sha):
    require(re.fullmatch(r"[a-f0-9]{40}", publication), "Publication is invalid")
    raw = sys.stdin.buffer.read(900001)
    require(0 < len(raw) <= 900000, "Bad published envelope extent")
    supplied = {key: base64.b64decode(value, validate=True) for key, value in json.loads(raw).items()}
    manifest_raw = supplied.pop("SOURCE_FILES.json")
    require(sha256(manifest_raw).hexdigest() == manifest_sha, "Manifest pin differs")
    members = json.loads(manifest_raw)["files"]
    require(set(members) == set(supplied), "Published source set differs")
    for name, pin in members.items():
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in {".", ".."}
                and len(supplied[name]) == pin["bytes"] and sha256(supplied[name]).hexdigest() == pin["SHA256"],
                "Published package member differs")
    claim = json.loads(supplied["CLAIM.json"])
    require((claim["label"], claim["parent"], claim["node"], claim["gpu"], claim["actual_optimizer_updates_planned"])
            == (LABEL, PARENT, NODE, 1, 30) and claim["GPU_GO"] is True and claim["replay_allowed"] is False
            and claim["whole_seconds"] == 1740 and claim["worker_seconds"] == 1630
            and re.fullmatch(r"[a-f0-9]{32}", claim["request"]), "Frozen recovery scope differs")
    previous = PROJECT / "results" / "training-zimage-COCO20-recovery-boundaries-hip-v1"
    require(sha256((previous / "outcome.json").read_bytes()).hexdigest() == claim["previous_outcome_SHA256"],
            "Previous original controller terminal differs")
    worker_raw = (previous / "continuous/worker-result.json").read_bytes()
    prior = json.loads(worker_raw)
    require(sha256(worker_raw).hexdigest() == claim["previous_worker_SHA256"]
            and prior["step"] == "1103606.1384" and prior["actual_optimizer_updates"] == 0
            and not prior["optimizer_intended_steps"] and not prior["optimizer_returned_steps"],
            "Earlier failed worker did not prove zero optimizer updates")
    ledger = json.loads(supplied["SURVIVOR_NODE_MAP.json"])
    require(NODE in ledger["groups"]["T07"]["nodes"], "Claimed node does not belong to T07")
    supplied["SOURCE_FILES.json"] = manifest_raw
    current_placement()
    ROOT.mkdir(mode=0o700)
    for name, content in supplied.items():
        with (ROOT / name).open("xb") as handle:
            handle.write(content)
        (ROOT / name).chmod(0o400)
    with (ROOT / "bootstrap.log").open("xb") as log:
        child = subprocess.Popen([PYTHON, "-B", "-u", str(ROOT / "deploy.py"), "bootstrap", publication, manifest_sha],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f"/proc/{child.pid}/stat").read_text()
    receipt = {"request": claim["request"], "controller_pid": child.pid,
               "start_ticks": int(stat.rsplit(")", 1)[1].split()[19]), "manifest_SHA256": manifest_sha,
               "origin_login_identity_SHA256": sha256(socket.gethostname().encode()).hexdigest()}
    save("controller-start.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    mode, commit, manifest = sys.argv[1:]
    if mode == "install":
        install(commit, manifest)
    elif mode == "bootstrap":
        bootstrap(commit, manifest)
    else:
        raise SystemExit("Unknown recovery controller role")
