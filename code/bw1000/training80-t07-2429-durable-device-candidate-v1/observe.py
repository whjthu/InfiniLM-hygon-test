"""Take one T07 zero-GRES device observation using the previously qualified node worker."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PROBE = ROOT / "training80-admission-standby-9cfcaff1ae2342868d72861af5392acd"
PYTHON = "/usr/local/python3.12/bin/python3"
NODE, PARENT = "f13r2n07", "1103606"
REQUEST = "f6b4bc0aeb2244fa8a22cdc831aafa05"
GPU_REQUEST = "8dcecfc1023c4bb5bc05f4bccb850b1f"
OLDER_ADMISSION = "42f59a4f4f004d6db5ac2c4560afc034"
PREVIOUS = "1103606.2429"
PREVIOUS_BRIDGE = "1103606.2428"
NAMESPACE = ROOT / f"training80-t07-device-candidate-{REQUEST}"
LABEL = "training80-t07-f13r2n07-candidate-" + REQUEST[:12]
PINS = {
    "collect_training80_dispatch_snapshot.py": "635c29d46aed1924bfa2642c532d30d1152d779acd4baf1d0d7cc38b8098183c",
    "read_training_slurm_membership.py": "9b3b0ca57c7ade5c28874faf46aa6925178ec0536f666842a085a76825d06de7",
    "sample_training80_busy.py": "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324",
    "training_dispatch_allocation.py": "e603bbbb1ef6dd539d667d547675022c2a5236fba2d7bf873dbff44950ed80b6",
    "tuning_queue.py": "1efeacc8331d7d230a1fc53a8512c82f462b67e69993df43ce196066aef37d45",
}
PCI = tuple(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
ACCOUNTING = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=8)


def fields(line):
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def terminal(step, expected_label, node, gres):
    output = query("sacct", "-n", "-P", "-j", step, "-o", ACCOUNTING)
    rows = [row for line in output.splitlines() if (row := line.split("|"))[0] == step]
    need(len(rows) == 1 and len(rows[0]) == 9, "exact terminal accounting missing")
    row = rows[0]
    need(row[1:4] == [expected_label, "COMPLETED", "0:0"] and row[5] == node and row[-1] != "Unknown"
         and {key for key in row[6].split(",") if key.startswith("gres/")} == gres,
         "original step or observer did not reconcile C0")
    return row


def active_target():
    owner = fields(query("scontrol", "show", "job", "-o", PARENT))
    need(owner["JobId"] == PARENT and owner["JobState"] == "RUNNING" and owner["UserId"].endswith(f"({os.getuid()})")
         and "gres/dcu=320" in owner["AllocTRES"].split(",")
         and NODE in query("scontrol", "show", "hostnames", owner["NodeList"]).splitlines(),
         "parent or owned node changed")
    cache = {}
    only = set()
    def add(step, members):
        if step in {PARENT + ".batch", PARENT + ".extern"}:
            return
        need(re.fullmatch(PARENT + r"\.[0-9]+", step) and members not in {None, "", "(null)", "None", "Unknown", "N/A"},
             "unknown numbered step or membership")
        if members not in cache:
            nodes = [members] if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", members) else query(
                "scontrol", "show", "hostnames", members).splitlines()
            need(nodes and len(nodes) == len(set(nodes)) and all(re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", n) for n in nodes),
                 "unknown expanded membership")
            cache[members] = frozenset(nodes)
        if NODE in cache[members]:
            only.add(step)
    for line in query("scontrol", "show", "step", PARENT, "--oneliner").splitlines():
        row = fields(line)
        add(row.get("StepId", ""), row.get("NodeList"))
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N").splitlines():
        add(*line.split("|"))
    return sorted(only)


def old_namespaces_absent():
    return not any(os.path.lexists(ROOT / name) for name in (
        f"training80-replacement-admission-{OLDER_ADMISSION}", f"training80-standby-compute-{GPU_REQUEST}",
        f"training80-dcu-standby-{GPU_REQUEST}", f".training80-t07-successor-admission-{GPU_REQUEST}"))


def write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def run():
    os.umask(0o077)
    mode, source = sys.argv[1:]
    need(mode in {"check-dependencies", "observe"} and re.fullmatch(r"[0-9a-f]{64}", source), "unexpected observer mode or SHA")
    matched = {file: sha256((PROBE / file).read_bytes()).hexdigest() for file in PINS}
    # The worker stage may contain either of two already recorded membership-reader versions.
    valid_member = {PINS["read_training_slurm_membership.py"], "8d071a48cad3a36ae0f917a2b5c7714abecc749cca55c8a38d1942fb2cc679be"}
    need(all(matched[k] == PINS[k] for k in PINS if k != "read_training_slurm_membership.py")
         and matched["read_training_slurm_membership.py"] in valid_member, "qualified installed zero-GRES dependency differs")
    if mode == "check-dependencies":
        return dict(request=REQUEST, observed_epoch=time.time(), status="DEPENDENCIES_MATCH", dependencies=matched)
    with (ROOT / ".training80-dcu-standby.lock").open("ab") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = terminal(PREVIOUS, "training80-dcu-standby-5f03331c61d9-f13r2n07", NODE, {"gres/dcu=8"})
        prior_bridge = terminal(PREVIOUS_BRIDGE, "training80-standby-compute-5f03331c61d9", "f12r2n13", set())
        previous_observer = terminal("1103606.2588", "training80-t07-f13r2n07-zero-gres-783da17f2c93", NODE, set())
        need(not active_target() and old_namespaces_absent(), "target or an original request is still active")
        NAMESPACE.mkdir(mode=0o700)
        write(NAMESPACE / "intent.json", (json.dumps(dict(request=REQUEST, gpu_request=GPU_REQUEST, node=NODE, parent=PARENT,
              observer_SHA256=source, dependencies_SHA256=matched, epoch=time.time(), replay_allowed=False), sort_keys=True) + "\n").encode())
        begin = time.time()
        args = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
                "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=256M", "--gres=none",
                "--time=00:01:00", "--kill-on-bad-exit=0", "--nodelist=" + NODE, PYTHON, "-E", "-s", "-B", "-u",
                str(PROBE / "collect_training80_dispatch_snapshot.py"), "node", REQUEST, PARENT, "[]"]
        observed = subprocess.run(args, capture_output=True, check=False, timeout=15)
        after = time.time()
        need(len(observed.stdout) <= 8192 and len(observed.stderr) <= 8192, "bounded zero-GRES worker evidence")
        write(NAMESPACE / "worker.stdout.private", observed.stdout)
        write(NAMESPACE / "worker.stderr.private", observed.stderr)
        need(observed.returncode == 0 and not observed.stderr, "independent zero-GRES worker failed or warned")
        row = json.loads(observed.stdout)
        need(row["request_id"] == REQUEST and row["node"] == NODE and row["parent"] == PARENT and row["observer_uid"] == os.getuid()
             and re.fullmatch(PARENT + r"\.[0-9]+", row["step"]) and sorted(row["PCIs"]) == list(PCI)
             and row["ports_checked"] == [] and row["occupied_requested_ports"] == [] and "probe_error" not in row
             and row["kfd_process_count"] == row["own_device_fd_process_count"] == 0, "fresh PCI, UID or process ownership differs")
        certificate = dict(request_id=REQUEST, node=NODE, parent=PARENT, PCIs=list(PCI), observed_epoch=after,
           payload_SHA256=sha256(observed.stdout).hexdigest(), device_probe_step=row["step"], device_probe_observed=True,
           observer_uid=row["observer_uid"], KFD_clean=True, PID_clean=True, ports_clean=True, membership_valid=True,
           kfd_process_count=0, own_device_fd_process_count=0, active_steps=[], probe_error=None)
        result = dict(request=REQUEST, gpu_request=GPU_REQUEST, status="ZERO_GRES_DEVICE_CANDIDATE", epoch=time.time(),
                      certified_observation=certificate, original_predecessor=prior, old_bridge=prior_bridge,
                      previously_finished_probe=previous_observer, current_probe_srun_rc=observed.returncode,
                      observer_SHA256=source, srun_seconds=after-begin)
        write(NAMESPACE / "candidate.json", (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True, allow_nan=False), flush=True)
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as e:
        print(json.dumps(dict(request=REQUEST, error=type(e).__name__, detail_SHA256=sha256(str(e).encode()).hexdigest())), flush=True)
        raise SystemExit(1)
