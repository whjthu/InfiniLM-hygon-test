"""Refill a published set of unoccupied held nodes through the qualified standby bridge."""

from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
SOURCE = ROOT / "training80-n03-reclaim-bridge-a5362d962be24077a8a025544464e409"
PROBE = SOURCE / "collect_training80_dispatch_snapshot.py"
BRIDGE = ROOT / "training80-standby-compute-879c5fe1eb35448f926b5596849ce16e/bridge.py"
PINS = {PROBE: "3e1b069f93219a165cc19431a54b47bc6044cf54c557e941f5bde797f3404078",
        SOURCE / "standby_controller.py": "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba",
        SOURCE / "standby_worker.py": "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059",
        BRIDGE: "2e796e5a136bb7fcac5892b49a34f0fd81fabd1f6f8774945725a0b199e58360"}
PYTHON = "/usr/local/python3.12/bin/python3"
PARENTS = ("1103606", "1194900")
EXCLUDED = {"f12r2n12", "f12r2n13", "f12r3n03", "f13r4n03", "f13r4n06", "f13r4n09"}
PORTS = [43843, 43844]
H64 = re.compile(r"[a-f0-9]{64}")
NODE_ID = re.compile(r"f\d{2}r\d+n\d{2}")
PCI = re.compile(r"[a-f0-9]{4}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]")


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def wire(obj):
    return (json.dumps(obj, sort_keys=True, allow_nan=False) + "\n").encode()


def save(path, obj):
    raw = wire(obj)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return digest(raw)


def query(*args):
    result = subprocess.run(args, capture_output=True, check=True, timeout=9)
    require(not result.stderr and len(result.stdout) <= 2 << 20, "Slurm query went out of bounds")
    return result.stdout.decode().splitlines()


def fields(row):
    return dict(term.split("=", 1) for term in row.split() if "=" in term)


def current(parent):
    [job] = map(fields, query("scontrol", "show", "job", parent, "--oneliner"))
    require(job["JobId"] == parent and job["JobState"] == "RUNNING"
            and job["UserId"].endswith(f"({os.getuid()})")
            and "gres/dcu=320" in job["AllocTRES"].split(","), "Allocation identity changed")
    expansion = {}

    def nodes(expression):
        if expression not in expansion:
            names = [expression] if NODE_ID.fullmatch(expression) else query("scontrol", "show", "hostnames", expression)
            require(names and len(names) == len(set(names)), "Numbered placement cannot be expanded")
            expansion[expression] = set(names)
        return expansion[expression]

    owned = nodes(job["NodeList"])
    require(len(owned) == 40, "Allocation extent changed")
    active = set()
    rows = query("scontrol", "show", "step", parent, "--oneliner")
    for line in rows:
        row = fields(line)
        if re.fullmatch(parent + r"\.\d+", row["StepId"]):
            active.update(nodes(row["NodeList"]))
    for line in query("squeue", "--steps", "-h", "-j", parent, "-o", "%i|%N"):
        handle, expression = line.split("|", 1)
        if re.fullmatch(parent + r"\.\d+", handle):
            active.update(nodes(expression))
    require(active <= owned, "Numbered step outside allocation")
    return owned, active


def probe_group(parent, targets, probe_id, workspace):
    owned, live = current(parent)
    require(set(targets) <= owned, "Candidate parent changed")
    eligible = sorted(set(targets) - live)
    if not eligible:
        return {}, {"admitted": 0, "active_veto": sorted(live.intersection(targets))}
    name = "training80-admit-" + probe_id
    count = str(len(eligible))
    args = ["srun", "--threads=1", "--jobid=" + parent, "--job-name=" + name, "--exact", "--immediate=5",
            "--nodes=" + count, "--ntasks=" + count, "--ntasks-per-node=1", "--nodelist=" + ",".join(eligible),
            "--cpus-per-task=1", "--mem=256M", "--gres=none", "--time=00:01:00", "--kill-on-bad-exit=0",
            PYTHON, "-B", str(PROBE), "node", probe_id, parent, json.dumps(PORTS)]
    result = subprocess.run(args, capture_output=True, timeout=26)
    require(result.returncode == 0, "Zero-GRES on target group did not exit normally")
    require(len(result.stdout) <= 2 << 20, "Physical receipt too large")
    rows = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        node = row["node"]
        require(node in eligible and node not in rows and row["request_id"] == probe_id
                and row["parent"] == parent and row["observer_uid"] == os.getuid()
                and row["ports_checked"] == PORTS and re.fullmatch(parent + r"\.\d+", row["step"]),
                "Unbound physical node row")
        rows[node] = row
    require(set(rows) == set(eligible), "Incomplete zero-GRES node report")
    step, = {row["step"] for row in rows.values()}
    terminal = None
    until = time.monotonic() + 10
    while time.monotonic() < until:
        matches = [line.split("|") for line in query("sacct", "-nP", "-j", step, "-o",
                   "JobIDRaw,JobName%128,State,ExitCode,NodeList,AllocTRES%120") if line.split("|")[0] == step]
        if len(matches) == 1 and matches[0][1:4] == [name, "COMPLETED", "0:0"]:
            terminal = matches[0]
            break
        time.sleep(.4)
    require(terminal is not None and "gres/" not in terminal[5]
            and set(query("scontrol", "show", "hostnames", terminal[4])) == set(eligible),
            "Missing numbered zero-GRES terminal")
    approved = {node: dict(parent=parent, pci=row["PCIs"]) for node, row in rows.items()
                if "probe_error" not in row and len(row["PCIs"]) == len(set(row["PCIs"])) == 8
                and all(PCI.fullmatch(pci) for pci in row["PCIs"])
                and row["kfd_process_count"] == row["own_device_fd_process_count"] == 0
                and row["occupied_requested_ports"] == []}
    source = save(workspace / (parent + "-admission.json"), dict(accounting=terminal, actual=rows,
                  approved=approved, stdout_SHA256=digest(result.stdout), stderr_SHA256=digest(result.stderr)))
    return approved, dict(admitted=len(approved), rejected=sorted(set(eligible)-approved.keys()),
                          active_veto=sorted(live.intersection(targets)), step=step, SHA256=source)


def run(claim, publication):
    require(re.fullmatch(r"[a-f0-9]{40}", publication) and re.fullmatch(r"[a-f0-9]{32}", claim["request"])
            and H64.fullmatch(claim["original_snapshot_SHA256"]) and set(claim["targets"]) == set(PARENTS),
            "Unbound refill scope")
    for targets in claim["targets"].values():
        require(0 < len(targets) <= 32 and len(targets) == len(set(targets))
                and all(NODE_ID.fullmatch(node) and node not in EXCLUDED for node in targets), "Unsafe candidate scope")
    raw = {}
    for path, pin in PINS.items():
        require(path.resolve(strict=True) == path and 0 < path.stat().st_size <= 256 << 10, "Pinned source unavailable")
        raw[path.name] = path.read_bytes()
        require(digest(raw[path.name]) == pin, "Pinned source changed")
    work = ROOT / ("training80-observed-expiry-wave-" + claim["request"])
    work.mkdir(mode=0o700)
    save(work / "intent.json", dict(claim=claim, publication=publication, replay=False))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        trials = {p: pool.submit(probe_group, p, claim["targets"][p], digest((claim["request"] + p).encode())[:32],
                                 work) for p in PARENTS}
        admissions = {}
        for parent, trial in trials.items():
            try:
                admissions[parent] = trial.result()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                admissions[parent] = {}, {"admitted": 0, "error": type(exc).__name__, "detail_SHA256": digest(str(exc).encode())}
    results = {}
    for parent, (scope, evidence) in admissions.items():
        results[parent] = {"evidence": evidence}
        if not scope:
            continue
        require(time.monotonic() - start < 55, "Physical evidence too old to dispatch")
        _, active = current(parent)
        selected = {node: data for node, data in scope.items() if node not in active}
        if not selected:
            continue
        group = digest((claim["request"] + ":group:" + parent).encode())[:32]
        checks = dict(pilot_seconds=120, production_min_seconds=10500, production_seconds=10800,
                      Slurm_cap_seconds=11100, gpu_per_node=8, synthetic=True, worker_SHA256=PINS[SOURCE / "standby_worker.py"])
        payload = dict(request=group, publication=publication, worker=raw["standby_worker.py"], claim=checks,
                       seconds=10800, admission_SHA256=evidence["SHA256"], nodes=selected)
        entry = ("INPUT=" + repr(payload) + "\n").encode() + raw["standby_controller.py"]
        envelope = dict(request=group, entry=b64encode(entry).decode(), entry_SHA256=digest(entry),
                        bridge=b64encode(raw["bridge.py"]).decode(), bridge_SHA256=digest(raw["bridge.py"]))
        save(work / (parent + "-bridge-intent.json"), dict(group=group, nodes=sorted(selected), request=claim["request"]))
        proc = subprocess.run([PYTHON, "-I", "-S", "-B", str(BRIDGE), "start", "--launch-only"],
                              input=wire(envelope), capture_output=True, timeout=15)
        result = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout else None
        results[parent].update(group=group, selected=len(selected), bridge_rc=proc.returncode, reply=result,
                               stderr_SHA256=digest(proc.stderr))
        if result is not None:
            receipt = ROOT / ("training80-dcu-standby-" + group) / "submission-receipt.json"
            until = time.monotonic() + 12
            while not receipt.is_file() and time.monotonic() < until:
                time.sleep(.3)
            results[parent]["receipt_visible_before_next_bridge"] = receipt.is_file()
    result = dict(request=claim["request"], batches=results, running_confirmed=False, declared_synthetic=True)
    save(work / "outcome.json", result)
    return result


if __name__ == "__main__":
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard", LC_ALL="C")
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_bytes()), sys.argv[2]), sort_keys=True), flush=True)
