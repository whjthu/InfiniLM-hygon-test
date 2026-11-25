"""Guard one f13r1n02 successor around the unchanged standby-v2 worker and launcher."""

import argparse
import ast
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import shlex
import sqlite3
import subprocess
import sys
from contextlib import closing
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
NODE = "f13r1n02"
PARENT = "1103606"
STEM = "training80-f13r1n02-group13-guarded-bf16-v1"
CLAIM = f"coordination/tasks/{STEM}.json"
OLD_SCOPE = REPO / "tools/run_training80_n02_group13_clearance_once.py"
OLD_SCOPE_PIN = "d655c46eef67dd9b8f3dc501d184ab5764d6a8b51b50f89fb3f5843e10fed1d5"
OLD_CLAIM = REPO / "coordination/tasks/training80-f13r1n02-group13-zero-gres-clearance-v1.json"
OLD_CLAIM_PIN = "33431059cf33a06aab2b335429911d216c2a065673ed5646f0f38053adcc90a6"
CLEARANCE = LOCAL / "training80-f13r1n02-group13-zero-gres-clearance-v1-reply.json"
CLEARANCE_PIN = "61fbd559e0cff4051b8d6ab07496d7f5e56d26dd62dfaa4136cf9366f2896e20"
ADOPTION_SHA = "d9ea8816084974ba439f5fca0e96fc40daf159d9b0d892bb2c3305a80d283573"
ADOPTION = LOCAL / "training80-vnext-0841-ordinary58-sole-adoption-v1.json"
ADOPT_TOOL = REPO / "tools/run_training80_ordinary58_adoption_once.py"
ADOPT_TOOL_SHA = "bd377144cf837ef34cd566b685abefd350a7c491e50cb67efe61c0e64e57ca65"
HELPER = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
HELPER_SHA = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
TRANSPORT = REPO / "tools/run_hygon_bound_payload.py"
TRANSPORT_SHA = "16563ae8141d79ef83990f8c0c16baf2ad4505a013dbbf4c8bfb9395d038d172"
STANDBY = REPO / "code/bw1000/fleet-dcu-standby-v2"
SOURCE_PINS = {"CLAIM.json": "1f441531cc4c9fdbf3226fb887242d693475f4e4eca603cf0ce69ccc7f7299cd",
               "controller.py": "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba",
               "worker.py": "dc8820e17b8d4d2886325c6093ad8884e6a5a1085e0eedd40eb53cfb12235059"}
TOKEN = LOCAL / (STEM + "-send.json")
INTENT = LOCAL / (STEM + "-intent.json")
REPLY = LOCAL / (STEM + "-reply.json")

NODE_PROBE = r'''
import json, os, re, socket, sys
from hashlib import sha256
from pathlib import Path

probe, request, parent, node, ports = sys.argv[1:]
step = os.environ["SLURM_STEP_ID"]
assert os.environ["SLURM_JOB_ID"] == parent and step.isdecimal()
assert socket.gethostname().split(".", 1)[0] == node
ports = json.loads(ports)
sys.path.insert(0, probe)
from collect_training80_dispatch_snapshot import KFD_PROCESSES, device_status

mounts = [line.split(" - ", 1) for line in Path("/proc/self/mountinfo").read_text().splitlines()]
[sysfs] = [row for row in mounts if row[0].split()[4] == "/sys"]
assert sysfs[0].split()[3] == "/" and sysfs[1].split()[0] == "sysfs"
kfd = KFD_PROCESSES.resolve(strict=True)
assert kfd.is_relative_to(Path("/sys/devices")) and kfd.name == "proc"
assert not any(kfd.is_relative_to(Path(row[0].split()[4])) for row in mounts
               if row[0].split()[4].startswith("/sys/"))
before = device_status(ports)
after = device_status(ports)
cgroup = Path("/proc/self/cgroup").read_text().splitlines()
cgroup_owned = any({"job_" + parent, "step_" + step} <= set(line.split(":", 2)[-1].strip("/").split("/"))
                   for line in cgroup)
boot = Path("/proc/sys/kernel/random/boot_id").read_bytes()
assert re.fullmatch(rb"[a-f0-9-]{36}\n", boot)
print(json.dumps(dict(request=request, node=node, parent=parent, step=parent + "." + step,
                      observer_uid=os.getuid(), ports_checked=ports, first=before, second=after,
                      global_driver_KFD_queried=True, other_UID_device_FDs="UNOBSERVED",
                      own_probe_cgroup=cgroup_owned, boot_SHA256=sha256(boot).hexdigest()), sort_keys=True))
'''

WORKER_CGROUP = r'''
import json, os, socket, sys
from pathlib import Path

directory = Path(sys.argv[1])
pid = int(sys.argv[2])
expected = json.loads((directory / "expected.json").read_bytes())
node = socket.gethostname().split(".", 1)[0]
parent = os.environ["SLURM_JOB_ID"]
numeric = os.environ["SLURM_STEP_ID"]
assert node == expected["node"] and parent == expected["parent"] and numeric.isdecimal()
proc = Path("/proc") / str(pid)
assert proc.stat().st_uid == os.getuid()
groups = ((line.split(":", 2)[-1]).strip("/").split("/")
          for line in (proc / "cgroup").read_text().splitlines())
assert any({"job_" + parent, "step_" + numeric} <= set(parts) for parts in groups)
ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
with (directory / "worker-cgroup.json").open("x", encoding="utf8") as out:
    json.dump(dict(node=node, step=parent + "." + numeric, worker_pid=pid, worker_start_ticks=ticks,
                   same_UID=True, exact_job_and_step_cgroup=True), out, sort_keys=True)
    out.flush()
    os.fsync(out.fileno())
'''

HOST_SCOPE = r'''
import base64, runpy, shlex, sys

def no_foreign_allocation():
    rows = query("squeue", "-h", "-w", NODE, "-o", "%i|%T|%N")
    require(len(rows) == 1, "unexpected or unknown Slurm owner on target")
    job, state, expression = rows[0].split("|")
    require(job == PARENT and state == "RUNNING" and NODE in query("scontrol", "show", "hostnames", expression),
            "target allocation does not belong solely to its parent")
    line, = query("scontrol", "show", "node", "-o", NODE)
    node = fields(line)
    require(node["NodeName"] == NODE and dcu_count(node["AllocTRES"]) == dcu_count(node["CfgTRES"]) == 8,
            "target does not bind exactly the eight DCUs")


def check_first_clearance(row):
    require(row is not None and row[:4] == [CLEAR_STEP, CLEAR_LABEL, "COMPLETED", "0:0"]
            and row[4:] == [NODE, ZERO, CLEAR_END], "original zero-GRES clearance accounting drift")


def bind_probe(value):
    require(value["request"] == REQUEST and value["node"] == NODE and value["parent"] == PARENT
            and value["observer_uid"] == os.getuid() and value["ports_checked"] == PORTS
            and STEP.fullmatch(value["step"]) and value["step"] not in {PREVIOUS, OBSERVER_STEP, CLEAR_STEP}
            and re.fullmatch(r"[a-f0-9]{64}", value["boot_SHA256"]), "fresh probe scope differs")
    return value["step"]


def probe_sample(value):
    bind_probe(value)
    expected = dict(PCIs=PCI, kfd_process_count=0, own_device_fd_process_count=0, occupied_requested_ports=[])
    require(value["first"] == value["second"] == expected and value["own_probe_cgroup"] is True
            and value["global_driver_KFD_queried"] is True and value["other_UID_device_FDs"] == "UNOBSERVED",
            "fresh cgroup, PCI, global KFD, same-UID FD or socket evidence blocked")


def run_guarded():
    global dcu_count
    os.umask(0o077)
    audit = RESULTS / SCOPE
    audit.mkdir(mode=0o700)
    outcome = dict(request=REQUEST, node=NODE, parent=PARENT, state="PRECHECK", GPU_requested=8,
                   zero_GRES_call_attempted=False, GPU_srun_attempted=False, replay_allowed=False)
    save(audit / "intent.json", outcome)
    try:
        require(all(not (PROBE / name).is_symlink() and sha256((PROBE / name).read_bytes()).hexdigest() == pin
                    for name, pin in PROBE_PINS.items()), "original deployed observer source changed")
        dcu_count = runpy.run_path(str(PROBE / "read_training_slurm_membership.py"))["allocated_dcu"]
        raw = {name: base64.b64decode(value, validate=True) for name, value in SOURCES.items()}
        require({name: sha256(value).hexdigest() for name, value in raw.items()} == PINS, "published standby source differs")
        claim = json.loads(raw["CLAIM.json"])
        require(claim["gpu_per_node"] == 8 and claim["preemptible"] is True
                and claim["production_min_seconds"] == 10500 and claim["production_seconds"] == 10800
                and claim["Slurm_cap_seconds"] == 11100 and claim["worker_SHA256"] == PINS["worker.py"]
                and claim["controller_SHA256"] == PINS["controller.py"], "standby production scope changed")
        controller = {"__name__": "published_standby_v2_exact"}
        exec(compile(raw["controller.py"], "<frozen standby-v2 controller>", "exec"), controller)
        shell = controller["SHELL"]
        exec_worker = 'exec python3 -B -u "$1" "$2"'
        require(shell.count(exec_worker) == 1, "original worker exec boundary changed")
        controller["SHELL"] = shell.replace(exec_worker,
            'python3 -B -c ' + shlex.quote(WORKER_CGROUP) + ' "$2" "$$"; ' + exec_worker, 1)
        root = RESULTS / ("training80-dcu-standby-" + REQUEST)
        with (RESULTS / ".training80-dcu-standby.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            outcome["mutex_acquired"] = True
            require(not os.path.lexists(root), "fixed successor namespace already exists")
            old_label, gpu_label = LABEL, "training80-dcu-standby-" + REQUEST[:12] + "-" + NODE
            require(not any(row.split("|")[1:2] in ([old_label], [gpu_label])
                            for row in query("sacct", "-n", "-P", "-j", PARENT, "-o", "JobIDRaw,JobName%110")),
                    "fixed successor label already exists")
            check_previous(account(PREVIOUS))
            check_first_clearance(account(CLEAR_STEP))
            scope()
            no_foreign_allocation()
            outcome["zero_GRES_call_attempted"] = True
            save(audit / "fresh-probe-intent.json", outcome)
            probe_at = time.monotonic()
            observation = run(["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--overlap", "--immediate=5",
                "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=256M", "--gres=none",
                "--time=00:01:00", "--kill-on-bad-exit=0", "--nodelist=" + NODE,
                "/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", NODE_PROBE,
                str(PROBE), REQUEST, PARENT, NODE, json.dumps(PORTS)], 18)
            outcome.update(probe_rc=observation.returncode, probe_stdout_SHA256=sha256(observation.stdout).hexdigest(),
                           probe_stderr_SHA256=sha256(observation.stderr).hexdigest())
            require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,
                    "fresh probe did not return a bounded warning-free receipt")
            sample = json.loads(observation.stdout)
            step = bind_probe(sample)
            save(audit / "fresh-probe.json", dict(sample=sample, payload_SHA256=outcome["probe_stdout_SHA256"]))
            outcome["probe_step"] = step
            outcome["other_UID_device_FDs"] = sample["other_UID_device_FDs"]
            for _ in range(5):
                terminal = account(step)
                if terminal and terminal[2] == "COMPLETED":
                    break
                require(not terminal or terminal[2] in {"RUNNING", "COMPLETING"}, "fresh zero-GRES probe did not complete")
                require(time.monotonic() + .5 < DEADLINE, "fresh probe terminal query exceeded the original deadline")
                time.sleep(.5)
            require(terminal and terminal[:4] == [step, LABEL, "COMPLETED", "0:0"] and terminal[4:6] == [NODE, ZERO]
                    and terminal[6] not in {"", "Unknown", "None", "N/A"}, "fresh zero-GRES accounting not exact")
            outcome["fresh_zero_GRES_accounting"] = terminal
            probe_sample(sample)
            scope(own=step)
            no_foreign_allocation()
            require(time.monotonic() - probe_at <= 30 and DEADLINE - time.monotonic() >= 2, "fresh physical receipt aged")
            root.mkdir(mode=0o700)
            controller["write"](root, "intent.json", dict(request=REQUEST, publication=PUBLICATION, synthetic=True,
                admission_SHA256=outcome["probe_stdout_SHA256"], nodes=[NODE], seconds=10800, epoch=time.time(),
                replay_allowed=False, original_clearance=CLEAR_STEP, new_clearance=step))
            with (root / "worker.py").open("xb") as stream:
                stream.write(raw["worker.py"])
            require(time.monotonic() - probe_at <= 30 and time.monotonic() < DEADLINE, "fresh physical receipt aged before GPU send")
            save(audit / "gpu-send-intent.json", dict(outcome, GPU_srun_attempted=True, label=gpu_label))
            outcome["GPU_srun_attempted"] = True
            record = controller["spawn"](root, NODE, dict(parent=PARENT, pci=PCI), 10800, REQUEST)
            require(record["label"] == gpu_label and 10500 <= record["worker_seconds"] <= 10800,
                    "original standby launcher returned a mismatched record")
            controller["write"](root, "submission-receipt.json", dict(request=REQUEST, publication=PUBLICATION, synthetic=True,
                submit_records=[record], launch_failures=[], card_steps_observed=0))
            outcome.update(state="SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING", receipt=record)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        outcome.update(state="GPU_OUTCOME_UNKNOWN_RECONCILE" if outcome["GPU_srun_attempted"] else "GPU_NOT_SENT",
                       detail_class=type(error).__name__, detail_SHA256=sha256(str(error).encode()).hexdigest())
    save(audit / "outcome.json", outcome)
    print(json.dumps(outcome, sort_keys=True), flush=True)


run_guarded()
'''

OBSERVE = r'''
def original(path):
    if not path.is_file():
        return None
    require(not path.is_symlink(), "unexpected original artifact link")
    with path.open("rb") as stream:
        raw = stream.read(16385)
    require(0 < len(raw) <= 16384, "original artifact outside bound")
    return json.loads(raw), sha256(raw).hexdigest()


def gpu_steps(label):
    ids = []
    for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", "JobIDRaw,JobName%110"):
        row = line.split("|")
        if len(row) == 2 and row[1] == label and STEP.fullmatch(row[0]):
            ids.append(row[0])
    require(len(ids) <= 1, "multiple candidate scheduler steps")
    return ids


def controller_running(step, label):
    [line] = query("scontrol", "show", "step", "-o", step)
    row = fields(line)
    return (row["StepId"], row["Name"], row["NodeList"], row["State"], row["TRES"]) == (
        step, label, NODE, "RUNNING", "cpu=8,gres/dcu=8,mem=16G,node=1")


def sample(folder, worker):
    rows = {}
    for index, address in enumerate(worker["pci"]):
        card = original(folder / f"card-{index}.json")
        if card is None:
            continue
        entry, pin = card
        if (entry["pci"] == address and entry["synthetic"] is True and entry["dtype"] == "BF16"
                and entry["ready"] is True and entry["active"] is True and type(entry["calls"]) is int and entry["calls"] > 0
                and -1 <= time.time() - entry["epoch"] <= 18):
            busy = entry["busy"]
            rows[address] = dict(calls=entry["calls"], epoch=entry["epoch"], SHA256=pin,
                                 busy=busy if type(busy) is int and 0 <= busy <= 100 else None)
    return rows


def observe():
    root = RESULTS / ("training80-dcu-standby-" + REQUEST)
    audit = RESULTS / SCOPE
    label = "training80-dcu-standby-" + REQUEST[:12] + "-" + NODE
    report = dict(request=REQUEST, node=NODE, parent=PARENT, new_scheduling_operations=0,
                  actual_DCU_progress=None, replay_allowed=False, state="ORIGINAL_UNKNOWN_RECONCILE")
    report["original_guard_outcome"] = original(audit / "outcome.json")
    steps = gpu_steps(label)
    if not steps:
        return dict(report, scheduler_label_observed=False)
    [step] = steps
    accounting = account(step)
    require(accounting is not None and accounting[:2] == [step, label]
            and accounting[4:6] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"], "exact GPU TRES or placement missing")
    report.update(step=step, accounting=accounting)
    folder = root / NODE
    if accounting[2] in TERMINAL and accounting[6] not in {"", "Unknown", "None", "N/A"}:
        with_end = original(folder / "terminal.json")
        return dict(report, state="EXACT_TERMINAL_ACCOUNTING", worker_terminal=with_end)
    require(accounting[2:4] == ["RUNNING", "0:0"] and controller_running(step, label),
            "numbered step RUNNING views disagree")
    found, cgroups, expected = [original(folder / name) for name in ("worker.json", "worker-cgroup.json", "expected.json")]
    if not all((found, cgroups, expected)):
        return dict(report, state="RUNNING_NUMBERED_WORKER_EVIDENCE_PENDING")
    worker, worker_pin = found
    group, group_pin = cgroups
    policy, policy_pin = expected
    require(worker["node"] == group["node"] == policy["node"] == NODE and worker["step"] == group["step"] == step
            and group["exact_job_and_step_cgroup"] is True and group["same_UID"] is True
            and worker["worker_pid"] == group["worker_pid"] > 0 and group["worker_start_ticks"] > 0
            and sorted(worker["pci"]) == sorted(policy["pci"]) == PCI and worker["synthetic"] is True
            and policy["request"] == REQUEST and policy["label"] == label and policy["parent"] == PARENT
            and policy["seconds"] == WORK_SECONDS, "worker, placement or cgroup evidence differs")
    first = sample(folder, worker)
    time.sleep(5.8)
    second = sample(folder, worker)
    require(original(folder / "worker.json") == found and original(folder / "worker-cgroup.json") == cgroups
            and controller_running(step, label), "worker identity changed between pulses")
    growth = {address: b["calls"] - first[address]["calls"] for address, b in second.items()
              if address in first and b["epoch"] > first[address]["epoch"] and b["calls"] > first[address]["calls"]}
    budget = worker["epoch"] + WORK_SECONDS
    return dict(report, state="SAME_PCI_COMPUTE_VERIFIED" if len(growth) == 8 else "RUNNING_PARTIAL_EVIDENCE",
                first=first, second=second, actual_DCU_progress=len(growth), counter_deltas=growth,
                worker_SHA256=worker_pin, worker_cgroup_SHA256=group_pin, expected_SHA256=policy_pin,
                actual_worker_start_epoch=worker["epoch"], worker_budget_seconds=WORK_SECONDS,
                next_preparation_epoch=budget - 900, projected_budget_end_epoch=budget)


try:
    print(json.dumps(observe(), sort_keys=True), flush=True)
except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
    print(json.dumps(dict(request=REQUEST, state="ORIGINAL_UNKNOWN_RECONCILE", error_class=type(error).__name__,
                         error_SHA256=sha256(str(error).encode()).hexdigest(), replay_allowed=False)), flush=True)
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(0 < len(raw) <= 262144 and digest(raw) == expected and not path.is_symlink(), "source or evidence differs")
    return raw


def original_scope():
    original = json.loads(pinned(OLD_CLAIM, OLD_CLAIM_PIN))
    pinned(OLD_SCOPE, OLD_SCOPE_PIN)
    scope = runpy.run_path(str(OLD_SCOPE))
    old_label, old_end = scope["prior_evidence"](original)
    proof = json.loads(pinned(CLEARANCE, CLEARANCE_PIN))
    data = proof["observed"]
    row = data["accounting"]
    pci = json.loads(pinned(REPO / original["pci_pins"]["path"], original["pci_pins"]["SHA256"]))[NODE]
    require(proof["rc"] == 0 and proof["claim_SHA256"] == OLD_CLAIM_PIN and proof["request"] == original["request"]
            and not proof["GPU_dispatch_eligible"] and not proof["replay_allowed"] and data["state"] == "OBSERVED"
            and data["scoped_device_clear"] is True and data["other_UID_device_FDs_observed"] is False
            and row[:6] == ["1103606.3089", "training80-n02-clearance-8a2dabb92fe1", "COMPLETED", "0:0", NODE, "cpu=1,mem=256M,node=1"]
            and data["observation"]["PCIs"] == pci and data["observation"]["kfd_process_count"] == 0
            and data["observation"]["own_device_fd_process_count"] == 0 and data["observation"]["occupied_requested_ports"] == []
            and all(data[key]["disposition"] == "TERMINAL_AND_RELEASED" for key in ("observer_before", "observer_after")),
            "sealed historical .3089 clearance differs")
    return original, scope, old_label, old_end, row, pci


def render(claim, publication=None):
    request = claim["request"]
    require((claim["scope"], claim["node"], claim["parent"], claim["GPU_requested"], claim["max_invocations"], claim["replay_allowed"])
            == (STEM, NODE, PARENT, 8, 1, False) and re.fullmatch(r"[a-f0-9]{32}", request), "claim changes fixed exact scope")
    original, scope, old_label, old_end, row, pci = original_scope()
    pinned(ADOPT_TOOL, ADOPT_TOOL_SHA)
    pinned(ADOPTION, ADOPTION_SHA)
    adoption = runpy.run_path(str(ADOPT_TOOL))
    api, roster, _ = adoption["checked_plan"]()
    groups, _ = api["calendar_scope"](api["legacy"](), adoption["CALENDAR"], adoption["CALENDAR_SHA"])
    require(NODE not in {item["node"] for group in groups for item in group} and roster["excluded_nodes"] == 22,
            "f13r1n02 collides with the adopted vNext ordinary roster")
    sources = {name: b64encode(pinned(STANDBY / name, pin)).decode() for name, pin in SOURCE_PINS.items()}
    controller_text = pinned(STANDBY / "controller.py", SOURCE_PINS["controller.py"])
    tree = ast.parse(controller_text)
    [budget] = [value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == "node_seconds"]
    computed = {"sha256": sha256}
    exec(compile(ast.Module(body=[budget], type_ignores=[]), "<frozen standby-v2 budget>", "exec"), computed)
    seconds = computed["node_seconds"](NODE, 10800, request)
    require(10500 <= seconds <= 10800, "long standby budget outside original envelope")
    values = dict(REQUEST=request, PARENT=PARENT, NODE=NODE, SCOPE=STEM, PUBLICATION=publication, PREVIOUS="1103606.2974",
                  PREVIOUS_LABEL=old_label, PREVIOUS_END=old_end, CLEAR_STEP=row[0], CLEAR_LABEL=row[1], CLEAR_END=row[6],
                  LABEL="training80-n02-fresh-" + request[:12], PORTS=original["ports"], PCI=pci,
                  OBSERVER_STEP=original["observer"]["step"], OBSERVER_LABEL=original["observer"]["label"],
                  ALLOW_OBSERVER_OVERLAP=False, PROBE_PINS=original["staged_probe_SHA256"],
                  SOURCES=sources, PINS=SOURCE_PINS, NODE_PROBE=NODE_PROBE, WORKER_CGROUP=WORKER_CGROUP, WORK_SECONDS=seconds)
    prefix = "\n".join(f"{key}={value!r}" for key, value in values.items()) + "\n"
    frozen = scope["HOST"].split("\ndef main():", 1)[0]
    launch = prefix + frozen + HOST_SCOPE
    observe = prefix + frozen + OBSERVE
    for definition in (launch, observe, NODE_PROBE, WORKER_CGROUP):
        compile(definition, "<pinned standalone n02 successor>", "exec")
    return dict(launch=launch, observe=observe, worker_seconds=seconds, adoption=adoption, roster=roster, scope=scope)


def current_local_authority(config):
    config["scope"]["local_holds"]()
    adoption = config["adoption"]
    api = runpy.run_path(str(adoption["COORDINATOR"]))
    api["calendar58"].require_adoption(ADOPTION, ADOPTION_SHA, adoption["WAVE"], adoption["LEDGER"], config["roster"], LOCAL)
    adoption["ledger_identity"](adoption["LEDGER"])
    db_uri = adoption["LEDGER"].as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(db_uri, uri=True, timeout=5)) as db:
        require(db.execute("SELECT count(*) FROM targets WHERE node=?", (NODE,)).fetchone() == (0,), "existing vNext ledger claim touches n02")
    adoption["local_census"]()


def verify_publication(revision, claim_raw):
    require(re.fullmatch(r"[a-f0-9]{40}", revision), "exact ordinary publication required")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    subprocess.run([*git, "merge-base", "--is-ancestor", revision, "refs/remotes/origin/bw1000-experiments"],
                   cwd=REPO, capture_output=True, timeout=8, check=True)
    paths = [Path(__file__).resolve(), REPO / CLAIM, *(STANDBY / name for name in SOURCE_PINS)]
    for path in paths:
        published = subprocess.check_output([*git, "cat-file", "blob", revision + ":" + path.relative_to(REPO).as_posix()],
                                            stderr=subprocess.PIPE, cwd=REPO, timeout=8)
        require(published == (claim_raw if path == REPO / CLAIM else path.read_bytes()), "original publication bytes differ")


def launch(claim, claim_raw, config, publication):
    require(not any(path.exists() for path in (INTENT, TOKEN, REPLY)), "original unique request requires reconciliation")
    verify_publication(publication, claim_raw)
    current_local_authority(config)
    helper = runpy.run_path(str(HELPER))
    transport = runpy.run_path(str(TRANSPORT))
    intent = dict(request=claim["request"], publication=publication, claim_SHA256=digest(claim_raw),
                  exact_program_SHA256=digest(config["launch"].encode()), GPU_requested=8, replay_allowed=False)
    config["scope"]["save"](INTENT, intent)
    receipt = transport["send"](helper["ssh"](REPO.parent), request=claim["request"], program=config["launch"], data=b"",
                                send_token=TOKEN, authority_seconds=18, outcome_seconds=82)
    config["scope"]["save"](REPLY, dict(intent, transport=receipt))
    return dict(file=REPLY.name, SHA256=digest(REPLY.read_bytes()), transport=receipt)


def observe(claim, config):
    require(INTENT.is_file() and TOKEN.is_file(), "original send must reconcile only by its fixed request")
    intent = json.loads(INTENT.read_bytes())
    require(intent["request"] == claim["request"] and not intent["replay_allowed"], "original local intent changed")
    helper = runpy.run_path(str(HELPER))
    ssh = helper["ssh"](REPO.parent)
    command = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-"])
    try:
        result = subprocess.run([*ssh, command], input=config["observe"].encode(), capture_output=True, timeout=48)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    except OSError as error:
        rc, stdout, stderr = None, b"", type(error).__name__.encode()
    try:
        body = json.loads(stdout) if rc == 0 and 0 < len(stdout) <= 16384 else None
    except (ValueError, UnicodeError):
        body = None
    if body:
        require(body["request"] == claim["request"], "foreign observation request")
    out = LOCAL / (STEM + "-independent-" + uuid4().hex + ".json")
    record = dict(rc=rc, observed=body, stdout_SHA256=digest(stdout), stderr_SHA256=digest(stderr), replay_allowed=False)
    config["scope"]["save"](out, record)
    return dict(file=out.name, SHA256=digest(out.read_bytes()), **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    mode.add_argument("--observe-original", action="store_true")
    args = parser.parse_args()
    raw = (REPO / CLAIM).read_bytes()
    claim = json.loads(raw)
    config = render(claim, args.publication)
    pinned(HELPER, HELPER_SHA)
    pinned(TRANSPORT, TRANSPORT_SHA)
    if args.check:
        result = dict(claim_SHA256=digest(raw), request=claim["request"], mode="PREPARED_ONLY", actual_gpu_attempts=0, Slurm=0,
                      Git=0, SCNet=0, worker_seconds=config["worker_seconds"], slurm_seconds=config["worker_seconds"] + 300,
                      roster_SHA256=config["roster"]["group_roster_SHA256"], launch_check_SHA256=digest(config["launch"].encode()),
                      observe_SHA256=digest(config["observe"].encode()))
    elif args.observe_original:
        result = observe(claim, config)
    else:
        result = launch(claim, raw, config, args.publication)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(json.dumps(dict(local_error=type(error).__name__, detail_SHA256=digest(str(error).encode()))))
        raise SystemExit(1)
