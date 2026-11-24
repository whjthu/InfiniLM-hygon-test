"""Verify and run one read-only, zero-GRES device inspection on f13r1n02."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
CLAIM = "coordination/tasks/training80-f13r1n02-group13-zero-gres-clearance-v1.json"
SCOPE = "training80-f13r1n02-group13-zero-gres-clearance-v1"
NODE = "f13r1n02"
PARENT = "1103606"
PREVIOUS = "1103606.2974"
ORIGINAL = "e0efd68bbc4d429eb87cf3ad255972cd"
CALENDAR = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
PORTS = [43843, 43844]
HELPER = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
HELPER_PIN = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
PUBLISHER_PIN = "e900334ac5be3a74fbde60f5c99457706dd9e802b7f749f0c375a1ecca48424e"
RUNNER_PIN = "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73"
JOURNAL = LOCAL / "training80-1738-ordinary-renewal-dispatch"
TOKEN = LOCAL / (SCOPE + "-send.json")
REPLY = LOCAL / (SCOPE + "-reply.json")

HOST = r'''
import fcntl, json, os, re, subprocess, time
from hashlib import sha256
from pathlib import Path

RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PROBE = RESULTS / "training80-admission-standby-9cfcaff1ae2342868d72861af5392acd"
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
STEP = re.compile(re.escape(PARENT) + r"\.[0-9]+")
HOSTNAME = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
ZERO = "cpu=1,mem=256M,node=1"
FMT = "JobIDRaw,JobName%110,State,ExitCode,NodeList,AllocTRES,End"
DEADLINE = time.monotonic() + 48


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(argv, limit=6):
    left = DEADLINE - time.monotonic()
    require(left > 0, "one HOST budget exhausted")
    reply = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=min(left, limit))
    require(len(reply.stdout) <= 1048576 and len(reply.stderr) <= 65536, "process output exceeds bound")
    return reply


def query(*argv):
    reply = run(argv)
    require(reply.returncode == 0, "scheduler read failed")
    return reply.stdout.decode().strip().splitlines()


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def account(step):
    rows = [row for line in query("sacct", "-nP", "-j", step, "-o", FMT)
            if len(row := line.split("|")) == 7 and row[0] == step]
    require(len(rows) <= 1, "duplicate materially relevant accounting")
    return rows[0] if rows else None


def check_previous(row):
    require(row is not None and row[:4] == [PREVIOUS, PREVIOUS_LABEL, "COMPLETED", "0:0"]
            and row[4:6] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"]
            and row[6] == PREVIOUS_END, "old exact C0/End changed")


def scope(own=None):
    expansions = {}

    def hosts(expression):
        if expression not in expansions:
            names = [expression] if HOSTNAME.fullmatch(expression) else query("scontrol", "show", "hostnames", expression)
            require(names and len(set(names)) == len(names) and all(HOSTNAME.fullmatch(n) for n in names),
                    "unbound Slurm hostname expansion")
            expansions[expression] = set(names)
        return expansions[expression]

    line, = query("scontrol", "show", "job", "-o", PARENT)
    owner = fields(line)
    parent_nodes = hosts(owner["NodeList"])
    require(owner["JobId"] == PARENT and owner["JobState"] == "RUNNING" and owner["UserId"].endswith(f"({os.getuid()})")
            and owner["NumNodes"] == "40" and "gres/dcu=320" in owner["AllocTRES"].split(",")
            and len(parent_nodes) == 40 and NODE in parent_nodes, "owned parent or placement changed")
    current = {}
    for line in query("scontrol", "show", "step", "-o", PARENT):
        item = fields(line)
        step = item["StepId"]
        if step in (PARENT + ".batch", PARENT + ".extern"):
            continue
        require(STEP.fullmatch(step) and hosts(item["NodeList"]) <= parent_nodes, "unbound scheduler step")
        if NODE in hosts(item["NodeList"]):
            require(step not in current, "duplicate target controller step")
            current[step] = item

    seen_queue = set()
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        step, expression = line.split("|")
        if step in (PARENT + ".batch", PARENT + ".extern"):
            continue
        require(STEP.fullmatch(step) and hosts(expression) <= parent_nodes, "unbound queue membership")
        if NODE in hosts(expression):
            seen_queue.add(step)
    require((set(current) | seen_queue) <= ({OBSERVER_STEP, own} if own else {OBSERVER_STEP}),
            "additional target step or unresolved placement")
    if own in current:
        item = current[own]
        require(item["Name"] == LABEL and item["State"] == "COMPLETED" and item["TRES"] == ZERO
                and hosts(item["NodeList"]) == {NODE}, "own zero-GRES step was not fully released")
    require(own not in seen_queue, "own zero-GRES step remains pending")

    observer = account(OBSERVER_STEP)
    item = current.get(OBSERVER_STEP)
    require(observer is not None and observer[:2] == [OBSERVER_STEP, OBSERVER_LABEL]
            and observer[5] == "cpu=40,mem=10G,node=40" and hosts(observer[4]) == parent_nodes,
            "frozen observer accounting scope changed")
    if item is not None:
        require(item["Name"] == OBSERVER_LABEL and item["TRES"] == observer[5]
                and hosts(item["NodeList"]) == parent_nodes, "frozen observer controller scope changed")
    if observer[2:4] == ["RUNNING", "0:0"]:
        require(item is not None and item["State"] == "RUNNING", "observer controller/accounting disagree")
        require(ALLOW_OBSERVER_OVERLAP, "fixed-hour observer has priority until release")
        disposition = "EXACT_ZERO_GRES_CO_RESIDENT"
    else:
        require(observer[2] in TERMINAL and observer[6] not in {"", "Unknown", "None", "N/A"}
                and (item is None or item["State"] in TERMINAL) and OBSERVER_STEP not in seen_queue,
                "observer release not independently confirmed")
        disposition = "TERMINAL_AND_RELEASED"
    return dict(step=OBSERVER_STEP, disposition=disposition, accounting=observer,
                in_squeue=OBSERVER_STEP in seen_queue)


def save(path, data):
    with path.open("x", encoding="utf8") as file:
        json.dump(data, file, sort_keys=True, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def validate_sample(sample):
    step = sample["step"]
    require(set(sample) == {"request_id", "parent", "step", "node", "ports_checked", "observer_uid", "PCIs",
                            "kfd_process_count", "own_device_fd_process_count", "occupied_requested_ports"}
            and sample["request_id"] == REQUEST and sample["parent"] == PARENT and sample["node"] == NODE
            and sample["observer_uid"] == os.getuid() and STEP.fullmatch(step) and step not in {PREVIOUS, OBSERVER_STEP}
            and sample["ports_checked"] == PORTS and sample["PCIs"] == PCI
            and type(sample["kfd_process_count"]) is int and sample["kfd_process_count"] >= 0
            and type(sample["own_device_fd_process_count"]) is int and sample["own_device_fd_process_count"] >= 0
            and type(sample["occupied_requested_ports"]) is list
            and set(sample["occupied_requested_ports"]) <= set(PORTS), "wrong target or device receipt scope")
    return step


def main():
    require(all(not (PROBE / name).is_symlink() and sha256((PROBE / name).read_bytes()).hexdigest() == pin
                for name, pin in PROBE_PINS.items()), "existing qualified probe bytes changed")
    os.umask(0o077)
    root = RESULTS / SCOPE
    root.mkdir(mode=0o700)
    outcome = dict(request=REQUEST, node=NODE, parent=PARENT, label=LABEL, GPU_requested=0,
                   replay_allowed=False, srun_attempted=False, GPU_dispatch_eligible=False,
                   other_UID_device_FDs_observed=False)
    save(root / "intent.json", outcome)
    try:
        with (RESULTS / ".training80-dcu-standby.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_previous(account(PREVIOUS))
            outcome["observer_before"] = scope()
            require(not any(line.split("|")[1:2] == [LABEL]
                            for line in query("sacct", "-nP", "-j", PARENT, "-o", "JobIDRaw,JobName%110")),
                    "original unique sampling label already exists")
            save(root / "srun-intent.json", dict(outcome, srun_attempted=True, argv=ARGV))
            outcome["srun_attempted"] = True
            reply = run(ARGV, 18)
            outcome.update(srun_rc=reply.returncode, stdout_SHA256=sha256(reply.stdout).hexdigest(),
                           stderr_SHA256=sha256(reply.stderr).hexdigest())
            require(reply.returncode == 0 and not reply.stderr and 0 < len(reply.stdout) <= 8192,
                    "new observer returned no warning-free bounded receipt")
            sample = json.loads(reply.stdout)
            step = validate_sample(sample)
            outcome.update(probe_step=step, observation=sample)
            save(root / "sample.json", sample)
            for _ in range(5):
                current = account(step)
                if current and current[:4] == [step, LABEL, "COMPLETED", "0:0"]:
                    break
                require(not current or current[2] in {"RUNNING", "COMPLETING"}, "observer failed")
                require(time.monotonic() + .5 < DEADLINE, "step accounting deadline expired")
                time.sleep(.5)
            require(current and current[:4] == [step, LABEL, "COMPLETED", "0:0"]
                    and current[4:6] == [NODE, ZERO] and current[6] not in {"", "Unknown", "None", "N/A"},
                    "new zero-GRES terminal is unverified")
            outcome["observer_after"] = scope(own=step)
            outcome.update(accounting=current, state="OBSERVED",
                           scoped_device_clear=(sample["kfd_process_count"] == 0
                                                and sample["own_device_fd_process_count"] == 0
                                                and not sample["occupied_requested_ports"]))
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        outcome.update(state="UNKNOWN_RECONCILE" if outcome["srun_attempted"] else "REJECTED_PRE_SRUN",
                       error_class=type(error).__name__, error_SHA256=sha256(str(error).encode()).hexdigest())
    save(root / "result.json", outcome)
    print(json.dumps(outcome, sort_keys=True), flush=True)
    return int(outcome["state"] != "OBSERVED")


raise SystemExit(main())
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, pin):
    with path.open("rb") as stream:
        raw = stream.read(131073)
    require(0 < len(raw) <= 131072 and digest(raw) == pin, "original bytes differ")
    return raw


def prior_evidence(claim):
    pinned = claim["original_evidence"]
    data = {name: checked(REPO / item["path"], item["SHA256"]) for name, item in pinned.items()}
    marker, child, outcome, remote, fresh = (json.loads(data[name]) for name in (
        "legacy_journal", "legacy_child", "legacy_outcome", "legacy_terminal", "fleet_0745"))
    require(marker["node"] == NODE and marker["predecessor"] == PREVIOUS and marker["calendar_SHA256"] == CALENDAR
            and child == dict(batch=13, calendar_SHA256=CALENDAR, nodes=[NODE], replay_allowed=False,
                              wrapper_invocation=ORIGINAL)
            and marker["wrapper_invocation"] == outcome["wrapper_invocation"] == ORIGINAL
            and marker["batch"] == outcome["batch"] == 13 and marker["replay_allowed"] is False
            and marker["terminal_receipt"] == outcome["terminal_receipt"] == Path(pinned["legacy_terminal"]["path"]).name
            and marker["terminal_receipt_SHA256"] == outcome["terminal_receipt_SHA256"] == digest(data["legacy_terminal"])
            and outcome["terminal_ready_nodes"] == [NODE] and outcome["refill_rc"] == 255
            and outcome["refill_attempted"] is True and outcome["publication_prechecked_before_terminal"] is True
            and outcome["no_GPU_send_confirmed"] is False and outcome["stdout_SHA256"] == digest(data["legacy_stdout"])
            and outcome["transport_observation"] == "REFILL_PROCESS_RETURNED", "original group 13 identity differs")
    require(re.fullmatch(rb"Bounded admission did not complete 255 fe33e26c82872922e259aad676bf569ad9969c999d74f5ebb391b4f18d009a7f\r?\n",
                         data["legacy_stdout"]), "original exact pre-bridge stdout differs")
    launcher = checked(REPO.parent / "run-dcu-standby-cycle-20261123.ps1", RUNNER_PIN).decode("utf-8-sig")
    embedded = launcher.replace("\r\n", "\n").split("\n@'\n", 1)[1].split("\n'@ | & ", 1)[0]
    probe_exit = 'if a.returncode:\n print("Bounded admission did not complete",a.returncode,sha256(a.stderr).hexdigest())\n sys.exit(a.returncode)'
    transport_send = "q=subprocess.run(inner+[command],input=transport_body,capture_output=True,timeout=120)"
    require(embedded.count(probe_exit) == embedded.count(transport_send) == 1
            and embedded.index(probe_exit) < embedded.index(transport_send), "original send ordering changed")
    publisher = REPO / "tools/run_standby_renewal_with_pinned_publication.py"
    publisher_raw = checked(publisher, PUBLISHER_PIN)
    source = ModuleType("qualified_publisher_static_renderer")
    source.__file__ = str(publisher)
    exec(compile(publisher_raw, source.__file__, "exec"), vars(source))
    source.render()
    entry, = remote["observed"]["targets"]
    acct, = fresh["observed"]["targets"][NODE]["original_accounting"]
    require(remote["rc"] == 0 and entry["node"] == NODE and entry["step"] == PREVIOUS and acct[0] == PREVIOUS
            and acct[2:4] == ["COMPLETED", "0:0"] and acct[4:6] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"]
            and entry["accounting"][8] == acct[6] and fresh["rc"] == 0
            and set(fresh["observed"]["targets"][NODE]["numbered_steps"]) == {"1103606.3088"},
            "historical 07:45 anchor differs")
    return acct[1], acct[6]


def local_holds():
    sys.path.insert(0, str(REPO / "tools"))
    import report_training80_backlog as report
    holds = report.read_queue_holds(LOCAL / "tuning/queue.sqlite3")
    jobs = json.loads((REPO / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    known = set(report.allocation(report.TRAINING80_LEDGER.read_bytes())[2])
    known.update(n for job in jobs for n in job["placement"])
    original, errors, _ = report.read_original_holds(REPO, jobs, known, holds)
    require(not any(NODE in item["nodes"] for item in (*holds, *original.values()))
            and not any(NODE in job["placement"] and job["state"] == "ready" for job in jobs)
            and not any(job["id"] in errors and NODE in job["placement"] for job in jobs), "target contains a competing work hold")
    require(all(not (JOURNAL / f"{NODE}-{PREVIOUS}-{slot}.json").exists() for slot in (1, 2)),
            "a later original ordinary reservation appeared")


def prepare(claim, *, allow_observer_overlap=False):
    require((claim["scope"], claim["node"], claim["parent"], claim["GPU_requested"], claim["max_invocations"],
             claim["replay_allowed"], claim["ports"]) == (SCOPE, NODE, PARENT, 0, 1, False, PORTS), "claim scope differs")
    request = claim["request"]
    require(re.fullmatch(r"[a-f0-9]{32}", request), "invalid unique claim request")
    previous_label, previous_end = prior_evidence(claim)
    pci_pin = claim["pci_pins"]
    pci = json.loads(checked(REPO / pci_pin["path"], pci_pin["SHA256"]))[NODE]
    require(pci == sorted(set(pci)) and len(pci) == 8 and all(re.fullmatch(r"[a-f0-9]{4}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]", p) for p in pci),
            "frozen node PCI set differs")
    observer = claim["observer"]
    require(observer["step"] == "1103606.3088" and observer["label"] == "training80-long-busy-0e1a294fabfc4854854c09ed6b4d6f83-1103606",
            "observer identity differs")
    label = "training80-n02-clearance-" + request[:12]
    program = ("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
               "training80-admission-standby-9cfcaff1ae2342868d72861af5392acd/collect_training80_dispatch_snapshot.py")
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--overlap", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=256M", "--gres=none",
            "--time=00:01:00", "--kill-on-bad-exit=0", "--nodelist=" + NODE,
            "/usr/local/python3.12/bin/python3", "-E", "-s", "-B", "-u", program, "node", request, PARENT, json.dumps(PORTS)]
    config = dict(SCOPE=SCOPE, NODE=NODE, PARENT=PARENT, PREVIOUS=PREVIOUS, REQUEST=request, LABEL=label,
                  PREVIOUS_LABEL=previous_label, PREVIOUS_END=previous_end, PORTS=PORTS, PCI=pci,
                  OBSERVER_STEP=observer["step"], OBSERVER_LABEL=observer["label"],
                  ALLOW_OBSERVER_OVERLAP=allow_observer_overlap, PROBE_PINS=claim["staged_probe_SHA256"], ARGV=argv)
    envelope = "\n".join(f"{key}={value!r}" for key, value in config.items()) + "\n" + HOST
    compile(envelope, "<f13r1n02 one zero-GRES HOST>", "exec")
    local_holds()
    require(not TOKEN.exists() and not REPLY.exists(), "existing observation request requires exact reconciliation")
    return envelope, argv


def verify_publication(revision, claim_raw):
    require(re.fullmatch(r"[a-f0-9]{40}", revision), "exact ordinary publication required")
    command = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    subprocess.run([*command, "merge-base", "--is-ancestor", revision, "refs/remotes/origin/bw1000-experiments"],
                   cwd=REPO, check=True, capture_output=True, timeout=8)
    source = Path(__file__).resolve().relative_to(REPO).as_posix()
    for name, raw in ((CLAIM, claim_raw), (source, Path(__file__).read_bytes())):
        published = subprocess.check_output([*command, "cat-file", "blob", revision + ":" + name], cwd=REPO,
                                            stderr=subprocess.PIPE, timeout=8)
        require(published == raw, "published source or claim differs")


def save(path, data):
    with path.open("x", encoding="utf8") as file:
        json.dump(data, file, sort_keys=True, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    parser.add_argument("--allow-observer-overlap", action="store_true")
    args = parser.parse_args()
    claim_raw = (REPO / CLAIM).read_bytes()
    claim = json.loads(claim_raw)
    host, argv = prepare(claim, allow_observer_overlap=args.allow_observer_overlap)
    helper_raw = checked(HELPER, HELPER_PIN)
    if args.check:
        print(json.dumps(dict(claim_SHA256=digest(claim_raw), host_SHA256=digest(host.encode()), argv=argv,
                              allow_observer_overlap=args.allow_observer_overlap,
                              request=claim["request"], Git_calls=0, SCNet_calls=0, Slurm_steps=0, GPU_requested=0)))
        return 0
    verify_publication(args.publication, claim_raw)
    helper = ModuleType("existing_strict_transport")
    exec(compile(helper_raw, str(HELPER), "exec"), vars(helper))
    intent = dict(request=claim["request"], publication=args.publication, claim_SHA256=digest(claim_raw),
                  host_SHA256=digest(host.encode()), replay_allowed=False)
    save(TOKEN, intent)
    remote = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-"])
    try:
        call = subprocess.run([*helper.ssh(REPO.parent), remote], input=host.encode(), capture_output=True, timeout=58)
        rc, out, err = call.returncode, call.stdout, call.stderr
    except subprocess.TimeoutExpired as error:
        rc, out, err = None, error.stdout or b"", error.stderr or b""
    except OSError as error:
        rc, out, err = None, b"", type(error).__name__.encode()
    try:
        report = json.loads(out) if 0 < len(out) <= 16384 else None
    except (UnicodeError, ValueError):
        report = None
    if not isinstance(report, dict) or report.get("request") != claim["request"]:
        report = None
    outcome = dict(intent, rc=rc, stdout_SHA256=digest(out), stderr_SHA256=digest(err), observed=report,
                   state=report["state"] if report else "UNKNOWN_RECONCILE", GPU_dispatch_eligible=False)
    save(REPLY, outcome)
    print(json.dumps(dict(file=REPLY.name, SHA256=digest(REPLY.read_bytes()), rc=rc, state=outcome["state"])))
    return int(rc != 0 or outcome["state"] != "OBSERVED")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(kind=type(error).__name__, detail_SHA256=digest(str(error).encode()))))
        raise SystemExit(1)
