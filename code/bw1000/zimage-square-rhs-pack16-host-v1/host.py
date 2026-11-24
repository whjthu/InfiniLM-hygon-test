"""One admitted Pack16 HOST step, with read-only local argv rendering."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import time


PARENT = "1103606"
REQUEST = "a31421758fc74d319923dbce7045cc12"
LABEL = "O01-zimage-square-rhs-pack16-v1"
PUBLICATION = "c4da359b16d2eea35300547f42e9967dd3199b86"
RUNTIME_SHA = "999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
HISTORIC_PRE = "1103606.2662"
SOURCE = "code/bw1000/zimage-square-rhs-pack16-v1"
ORIGINAL = "code/bw1000/zimage-square-rhs-pack16-target-preflight-v1"
ACCOUNT_HELPER = "code/bw1000/GLM474-transport-ordinal150-full-model-v1/slurm_accounting.py"
DEP_PATH = "results/bw1000/2026-11-24/O01-pack16-json-v3-original2662-zero-GRES-dependencies-pass.json"
HOST_ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
GATE_SHA = "22e25c3e34bdee42430f2051f5ad1b3f9aba247cae1d86c3fba8637d374782d8"
ORDINARY_ENTRY = "tools/run_training80_0544_ordinary_refill_once.py"
ORDINARY_DRIVER = "tools/run_training80_2347_ordinary_refill_once.py"
ORDINARY_CALENDAR = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
TARGET_SITE = HOST_ROOT / "results/training-t06-sdxl-runtime-v1"
CONTAINER = HOST_ROOT / "containers/dtk26-sandbox-r1"
LOCK = HOST_ROOT / "results/.training80-dcu-standby.lock"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
DENOMINATOR = {"routes": 2, "pairs": 3, "arms": 6, "measured_mm_calls": 384, "measured_copies": 12}
PROTECTED_NODES = {"f12r2n11"}
PROTECTED_STEPS = {"1103606.2559"}
CAP_LOG = 131072
SHA = re.compile("[0-9a-f]{64}")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
NODE = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
STEP = re.compile(r"1103606[.][0-9]+")
BOOT = re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}")
FIXED = {
    SOURCE + "/SOURCE_FILES.json": "a4b14d7b6d84cd8876f5205daa6ec61f97ed086aacff76d103f90fa28289ff22",
    DEP_PATH: "1bfb518c35e1dc58f277345d33cf48fe591a51097ef59fd27102d0afcfb5b353",
    "coordination/tasks/O01-zimage-pack16-json-v3-current-2615.json":
        "2efc1c226905d112521c924e26dcacb9187a934fac7765661816bf3e8d84989e",
    ACCOUNT_HELPER: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    ORIGINAL + "/precheck.py": "2d6152e99c3401dbd05c2c63d64212807df855e98f01bfe030af08e6e51928a0",
    ORIGINAL + "/read_result.py": "822942ad0bb4c7cb23893d4a48e960e15649e44ab20c3ebd7a6e8bd40e776216",
    ORDINARY_ENTRY: "2fb4649087e1e2162f706fe61b095cc434c120758bff4e60f843ebc973de7a0b",
    ORDINARY_DRIVER: "f5cbd175c5b3bca49475c8310cb552f2bf180ca1953565cafca4ce646541ce24",
}


def require(value, gate):
    if not value:
        raise ValueError(gate)


def pin(payload):
    return {"bytes": len(payload), "SHA256": sha256(payload).hexdigest()}


def read(path, maximum, expected=None):
    require(path.is_file() and path.resolve(strict=True) == path.absolute(), "ALIASED_OR_MISSING_INPUT")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (expected is None or pin(raw)["SHA256"] == expected), "INPUT_PIN")
    return raw


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def exclusive(path, value):
    raw = encode(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


def helper(repo, relative):
    path = repo / relative
    source = read(path, 131072, FIXED[relative])
    spec = importlib.util.spec_from_file_location("pack16_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), vars(module))
    return module


def frozen(repo, runtime, original):
    pins = {item: pin(read(repo / item, 131072, wanted)) for item, wanted in FIXED.items()}
    source = json.loads(read(repo / SOURCE / "SOURCE_FILES.json", 16384, FIXED[SOURCE + "/SOURCE_FILES.json"]))
    for name, wanted in source["files"].items():
        require(pin(read(repo / SOURCE / name, 65536, wanted["sha256"]))["bytes"] == wanted["bytes"], "FROZEN_SOURCE_SIZE")
    prior = json.loads(read(repo / DEP_PATH, 16384, FIXED[DEP_PATH]))
    actual = json.loads(read(original, 65536, prior["raw_transport"]["sha256"]))
    evidence = prior["precheck"]
    response = actual["response"]
    require(actual["request"] == response["request"] == prior["request"]
            and actual["rc"] == response["srun_rc"] == 0 and actual["remote_received"]
            and response["step_result"] == evidence and response["accounting"] == [prior["accounting"]]
            and prior["accounting"][:4] == [HISTORIC_PRE, "O01-pack16-deps-json-c0243062d1f2", "COMPLETED", "0:0"]
            and "gres/" not in prior["accounting"][6]
            and evidence["precheck_passed"] and not evidence["target_GPU_GO"]
            and evidence["runtime"] == {"manifest_SHA256": RUNTIME_SHA, "members": 5}, "HISTORICAL_DEPENDENCIES")
    verified = helper(repo, ORIGINAL + "/precheck.py").runtime(runtime)
    return {"frozen_pins": pins, "runtime": verified, "original_2662": pin(read(original, 65536)),
            "independent_reader": FIXED[ORIGINAL + "/read_result.py"]}


def path_arg(raw):
    path = PurePosixPath(raw)
    require(path.is_absolute() and str(path) == raw and ".." not in path.parts
            and re.fullmatch(r"/[A-Za-z0-9_./-]+", raw), "HOST_PATH")
    return path


def check_scope(scope, now):
    epoch, deadline = scope["epoch"], scope["deadline_epoch"]
    require(scope["request"] == REQUEST and scope["label"] == LABEL and scope["parent"] == PARENT
            and scope["source_publication"] == PUBLICATION and scope["runtime_SHA256"] == RUNTIME_SHA
            and scope["root_admitted"] is True and scope["gfx"] == "gfx936"
            and scope["torch_version"] == "2.10.0" and scope["torch_hip"] == "6.3.26113", "ROOT_SCOPE")
    protected = scope["protected_nodes"]
    node = scope["node"]
    require(NODE.fullmatch(node) and isinstance(protected, list) and all(NODE.fullmatch(n) for n in protected)
            and node not in set(protected) | PROTECTED_NODES and PCI.fullmatch(scope["PCI"])
            and BOOT.fullmatch(scope["boot_id"]), "PHYSICAL_SCOPE")
    require(STEP.fullmatch(scope["pre_step"]) and scope["pre_step"] not in PROTECTED_STEPS | {HISTORIC_PRE}
            and SHA.fullmatch(scope["pre_SHA256"]) and SHA.fullmatch(scope["protection_authority_SHA256"]), "FRESH_AUTHORITY")
    require(type(epoch) in (float, int) and math.isfinite(epoch) and deadline == epoch + 240
            and epoch <= now < epoch + 30, "ORIGINAL_EPOCH")


def protection_gate(authority, scope, now):
    require(authority["parent"] == PARENT and authority["protected_nodes"] == scope["protected_nodes"]
            and len(set(authority["protected_nodes"])) == len(authority["protected_nodes"])
            and PROTECTED_NODES <= set(authority["protected_nodes"])
            and PROTECTED_STEPS <= set(authority["protected_steps"])
            and authority["selected_node"] == scope["node"]
            and authority["selected_node"] not in authority["protected_nodes"]
            and 0 <= now - authority["observed_epoch"] <= 90
            and now < authority["valid_until_epoch"] <= authority["observed_epoch"] + 240,
            "PROTECTED_WORK_OR_EXPIRED_AUTHORITY")


def cpu_resources(row, node, step):
    return (len(row) == 7 and row[0] == step and STEP.fullmatch(step) and row[2:4] == ["COMPLETED", "0:0"]
            and row[4].isdecimal() and row[5] == node and not any(
                any(kind in field for kind in ("gpu", "dcu", "gres/")) for field in row[6].lower().split(",")))


def clean_inventory(receipt, scope, phase, now, gpu_row=None):
    item = receipt["inventory"]
    row = receipt["accounting"]
    stamp = receipt["observed_epoch"]
    require(receipt["phase"] == phase and scope["epoch"] - 120 <= stamp <= now
            and item["node"] == scope["node"] and cpu_resources(row, item["node"], item["step"])
            and item["boot_id_SHA256"] == sha256((scope["boot_id"] + "\n").encode()).hexdigest()
            and len(item["PCIs"]) == len(set(item["PCIs"])) == 8 and scope["PCI"] in item["PCIs"]
            and all(PCI.fullmatch(pci) for pci in item["PCIs"])
            and item["kfd_process_count"] == item["own_device_fd_process_count"] == 0, "INDEPENDENT_CLEAN_INVENTORY")
    if phase == "pre":
        require(item["step"] == scope["pre_step"] and 0 <= now - stamp <= 90
                and receipt["assigned_PCI"] == scope["PCI"] and receipt["gfx"] == scope["gfx"]
                and receipt["exclusive_1_DCU_lease"] is True, "FRESH_PHYSICAL_ADMISSION")
    else:
        # An independent reader binds its zero-GRES observation to the archived GPU terminal.
        require(item["step"] not in {scope["pre_step"], gpu_row[0]}
                and receipt["after_gpu_step"] == gpu_row[0] and receipt["after_terminal"] is True
                and receipt["terminal_row_SHA256"] == pin(encode(gpu_row))["SHA256"], "INDEPENDENT_POST_ORDER")
    return item


def ordinary_exclusion(repo, calendar, node):
    entry = read(repo / ORDINARY_ENTRY, 65536, FIXED[ORDINARY_ENTRY])
    driver = read(repo / ORDINARY_DRIVER, 65536, FIXED[ORDINARY_DRIVER])
    require(ORDINARY_CALENDAR.encode() in entry and FIXED[ORDINARY_DRIVER].encode() in entry, "ORDINARY_SCOPE_SOURCE")
    plan = json.loads(read(calendar, 131072, ORDINARY_CALENDAR))
    declared = set(plan["special_scopes_excluded"]) | set(plan["existing_held_fences"])
    bindings = [item.value for item in ast.parse(driver).body if isinstance(item, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "FENCES" for t in item.targets)]
    require(len(bindings) == 1 and ast.literal_eval(bindings[0]) == declared, "ORDINARY_SCOPE_SOURCE")
    planned = {row["node"] for group in plan["ordinary_candidates"] for row in group["targets"]}
    require(len(planned) == 70 and not declared & planned, "ORDINARY_SCOPE_SOURCE")
    require(node in declared and node not in planned, "EXACT_ORDINARY_REFILL_EXCLUSION_UNAVAILABLE")


def build_argv(scope, scope_sha, target_runtime, target_scope, target_output, now, target_gate=None):
    check_scope(scope, now)
    require(SHA.fullmatch(scope_sha), "SCOPE_SHA")
    runtime, scoped, output = map(path_arg, (target_runtime, target_scope, target_output))
    gate = path_arg(target_gate) if target_gate is not None else scoped.with_name("worker_gate.py")
    require(all(path.is_relative_to(HOST_ROOT) for path in (runtime, scoped, output, gate))
            and gate not in (runtime, scoped, output) and output not in gate.parents
            and output not in runtime.parents and runtime not in output.parents and output != runtime
            and output not in scoped.parents and scoped not in output.parents and output != scoped, "WORKSPACE_ISOLATION")
    # Keep time for owned-client reaping and terminal observation within this same epoch.
    seconds = int(scope["deadline_epoch"] - now) - 34
    worker = ["python3", "-B", "-u", "/pair-package/worker.py", "--scope", "/pack16-scope.json",
              "--scope-sha256", scope_sha, "--out", "/pack16-output/worker"]
    worker_timeout = shlex.quote("import time; print(int(" + repr(scope["deadline_epoch"]) + "-time.time())-40)")
    shell = (
        "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export PYTHONPATH=/training-runtime/site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
        "unset ROCBLAS_LAYER HIPBLASLT_LOG_LEVEL HIPBLASLT_LOG_MASK; "
        f'LEFT=$(python3 -B -c {worker_timeout}); test "$LEFT" -gt 75; '
        'exec /usr/bin/timeout --kill-after=5s "$LEFT" ' + shlex.join(worker)
    )
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + scope["node"], "--cpus-per-task=4",
            "--threads-per-core=1", "--cpu-bind=threads", "--mem=8192M", "--gres=dcu:1",
            "--time=00:04:00", "--kill-on-bad-exit=1", HOST_PYTHON, "-I", "-S", "-B", str(gate),
            "--scope", str(scoped), "--scope-sha256", scope_sha, "--",
            APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/"]
    for source, target, mode in (
        (runtime, "/pair-package", "ro"), (scoped, "/pack16-scope.json", "ro"),
        (output, "/pack16-output", "rw"), (TARGET_SITE, "/training-runtime", "ro"),
        ("/usr/local/hyhal", "/opt/hyhal", "ro"),
    ):
        argv.extend(("--bind", f"{source}:{target}:{mode}"))
    argv.extend((str(CONTAINER), "/bin/bash", "--noprofile", "--norc", "-c", shell))
    return argv, seconds


def fields(line):
    return dict(piece.split("=", 1) for piece in line.split() if "=" in piece)


def query(args, deadline):
    left = min(4, deadline - time.monotonic())
    require(left > 0, "HOST_QUERY_DEADLINE")
    output = subprocess.check_output(args, timeout=left)
    require(len(output) <= 512 * 1024, "HOST_QUERY_SIZE")
    return output.decode().splitlines()


def accounting_row(deadline):
    rows = query(["sacct", "-nP", "-j", PARENT, "--format", ",".join(COLUMNS)], deadline)
    exact = [line.split("|") for line in rows if "|" + LABEL + "|" in line]
    require(len(exact) <= 1 and all(len(row) == len(COLUMNS) for row in exact), "AMBIGUOUS_ORIGINAL_LABEL")
    return exact[0] if exact else None


def gpu_resources(row, node):
    resources = row[6].split(",")
    return (row[1] == LABEL and STEP.fullmatch(row[0]) and row[0] not in PROTECTED_STEPS
            and row[5] == node and len(resources) == len(set(resources))
            and set(resources) in ({"cpu=4", "gres/dcu=1", "mem=8G", "node=1"},
                                   {"cpu=4", "gres/dcu=1", "mem=8192M", "node=1"}))


def idle_owned(scope, deadline, final_states, authority=None):
    result = query(["scontrol", "show", "job", PARENT, "--oneliner"], deadline)
    require(len(result) == 1, "PARENT_OWNERSHIP")
    job = fields(result[0])
    require(job["JobId"] == PARENT and job["JobState"] == "RUNNING"
            and job["UserId"].endswith(f"({os.getuid()})"), "PARENT_OWNERSHIP")
    nodes = query(["scontrol", "show", "hostnames", job["NodeList"]], deadline)
    require(len(set(nodes)) == 40 and scope["node"] in nodes and "gres/dcu=320" in job["AllocTRES"].split(","), "PARENT_ALLOC")
    target = query(["scontrol", "show", "node", scope["node"], "--oneliner"], deadline)
    require(len(target) == 1, "NODE_STATE")
    live_node = fields(target[0])
    require(live_node["NodeName"] == scope["node"] and int(live_node["CPUTot"]) >= 4
            and int(live_node["RealMemory"]) >= 8192
            and not set(live_node["State"].rstrip("*~#").split("+")) & {"DOWN", "DRAIN", "DRAINED", "FAIL", "MAINT", "NO_RESPOND"}
            and not live_node["State"].endswith(("*", "~", "#")), "NODE_STATE")
    live = query(["scontrol", "show", "step", PARENT, "--oneliner"], deadline)
    for line in live:
        item = fields(line)
        if not STEP.fullmatch(item.get("StepId", "")) or item.get("State") in final_states:
            continue
        hosts = item.get("NodeList", "")
        require(hosts not in ("", "(null)"), "UNRESOLVED_NUMBERED_STEP")
        members = [hosts] if NODE.fullmatch(hosts) else query(["scontrol", "show", "hostnames", hosts], deadline)
        require(scope["node"] not in members, "CANDIDATE_NODE_OCCUPIED")
    if scope["node"] == "f12r2n10":
        require(authority is not None and authority.get("released_predecessor_step") == "1103606.2930",
                "EXACT_PREDECESSOR_REQUIRED")
        raw = query(["sacct", "-nP", "-j", "1103606.2930", "--format", ",".join(COLUMNS)], deadline)
        prior = [line.split("|") for line in raw if line.startswith("1103606.2930|")]
        require(len(prior) == 1 and len(prior[0]) == len(COLUMNS)
                and prior[0][2].split()[0].rstrip("+") in final_states and prior[0][5] == scope["node"]
                and "gres/dcu=8" in prior[0][6].split(","), "PREDECESSOR_TERMINAL_NOT_SEEN")
    require(accounting_row(deadline) is None, "ORIGINAL_LABEL_ALREADY_EXISTS")


@contextmanager
def shared_lock(path):
    import fcntl
    with path.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def spawn_owned(argv, log):
    with log.open("xb") as stream:
        env = {k: v for k, v in os.environ.items()
               if k in {"PATH", "LANG", "LC_ALL", "HOME", "USER", "LOGNAME", "TMPDIR"} or k.startswith("SLURM_")}
        return subprocess.Popen(argv, cwd="/", stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)


def collect_owned(child, log, end):
    forced = False
    try:
        while child.poll() is None and time.monotonic() < end and log.stat().st_size <= CAP_LOG:
            time.sleep(min(.35, max(0, end - time.monotonic())))
    finally:
        if child.poll() is None:
            forced = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    continue
    return {"owned_srun_rc": child.poll(), "forced_client_reap": forced,
            "log_bytes": log.stat().st_size, "log_within_bound": log.stat().st_size <= CAP_LOG}


def visible_step(scope, deadline):
    durable_rows = query(["scontrol", "show", "step", PARENT, "--oneliner"], deadline)
    rows = [fields(line) for line in durable_rows]
    exact = [row for row in rows if row.get("Name") == LABEL]
    require(len(exact) <= 1, "START_VISIBILITY_AMBIGUOUS")
    if not exact:
        return None
    row = exact[0]
    probe = [row.get("StepId", ""), LABEL, row.get("State"), "", "", row.get("NodeList"), row.get("TRES", "")]
    require(gpu_resources(probe, scope["node"]), "START_STEP_RESOURCES")
    return row["StepId"] if row.get("State") == "RUNNING" else None


def submit_under_lock(lock, prepare, start, observe, *, clock=time.monotonic, sleep=time.sleep):
    """Release the global lock after at most six seconds of ownership observation."""
    with lock:
        argv, worker_end = prepare()
        child = start(argv)
        stop = min(worker_end, clock() + 6)
        numeric_step, last_error = None, None
        for _ in range(5):
            if clock() >= stop:
                break
            try:
                numeric_step = observe(stop)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                last_error = type(error).__name__
                break
            if numeric_step or child.poll() is not None:
                break
            sleep(min(.3, max(0, stop - clock())))
        return child, worker_end, numeric_step, last_error


def try_once(repo, runtime, scope_path, scope_pin, scope, pre, authority, calendar, gate, output):
    # A successful exclusion must be recognized by the actual ordinary driver.
    ordinary_exclusion(repo, calendar, scope["node"])
    read(gate, 16384, GATE_SHA)
    require(sys.platform == "linux" and output.parent.resolve(strict=True) == output.parent.absolute()
            and not output.exists() and not output.is_symlink(), "HOST_RUN_LOCATION")
    require(Path(str(CONTAINER)).is_dir() and Path(str(TARGET_SITE / "site")).is_dir()
            and Path(APPTAINER).is_file(), "HOST_RUNTIME_LOCATIONS")
    watcher = helper(repo, ACCOUNT_HELPER)
    now = time.time()
    hard_end = time.monotonic() + scope["deadline_epoch"] - now
    result = {"request": REQUEST, "scope_SHA256": scope_pin, "state": "UNKNOWN",
              "planned": DENOMINATOR, "accounting_columns": COLUMNS,
              "accounting": [], "independent_post": "UNVERIFIED", "promotion_allowed": False}
    log = output / "srun.private.log"

    def prepare():
        now = time.time()
        check_scope(scope, now)
        clean_inventory(pre, scope, "pre", now)
        protection_gate(authority, scope, now)
        idle_owned(scope, min(hard_end - 34, time.monotonic() + 10), watcher.TERMINAL, authority)
        now = time.time()
        clean_inventory(pre, scope, "pre", now)
        protection_gate(authority, scope, now)
        argv, limit = build_argv(scope, scope_pin, runtime.as_posix(), scope_path.as_posix(), output.as_posix(), now,
                                 gate.as_posix())
        output.mkdir(mode=0o700)
        exclusive(output / "intent.json", {"request": REQUEST, "scope_SHA256": scope_pin,
                  "state": "SUBMISSION_POSSIBLE", "argv_SHA256": pin(encode(argv))["SHA256"], "planned": DENOMINATOR})
        descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return argv, time.monotonic() + limit

    def start(argv):
        return spawn_owned(argv, log)

    try:
        child, work_end, numeric_step, startup_error = submit_under_lock(
            shared_lock(Path(str(LOCK))), prepare, start, lambda deadline: visible_step(scope, deadline))
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if not (output / "intent.json").is_file():
            raise
        result["launcher_error_type"] = type(error).__name__
    else:
        result["startup_numeric_step"] = numeric_step
        result["startup_error_type"] = startup_error
        exclusive(output / "startup.json", {"request": REQUEST, "numeric_step": numeric_step,
                  "state": "VISIBLE" if numeric_step else "UNKNOWN", "replay_allowed": False})
        result.update(collect_owned(child, log, work_end if numeric_step else time.monotonic()))
    observations = []

    def observe():
        row = accounting_row(hard_end)
        observations.append(row)
        return row

    try:
        terminal = watcher.wait_terminal(observe, hard_end, interval=1, max_observations=14)
        row = terminal["row"]
        if row:
            same_step = result.get("startup_numeric_step") in (None, row[0])
            result.update(state="TERMINAL", accounting=[row],
                          actual_resources_match=bool(same_step and gpu_resources(row, scope["node"])),
                          terminal_observed_epoch=time.time())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["accounting_error_type"] = type(error).__name__
    result["accounting_observations"] = observations
    exclusive(output / "host-terminal.json", result)
    return result


def independently_finish(repo, scope, scope_sha, pre, post, host, host_path, host_pin, runtime, scope_path, output):
    require(host["request"] == REQUEST and host["scope_SHA256"] == scope_sha and host["state"] == "TERMINAL"
            and host["actual_resources_match"] and host["accounting_columns"] == list(COLUMNS)
            and len(host["accounting"]) == 1, "HOST_NOT_INDEPENDENTLY_FINALIZABLE")
    row = host["accounting"][0]
    require(gpu_resources(row, scope["node"]) and post["observed_epoch"] >= host["terminal_observed_epoch"], "POST_BEFORE_TERMINAL")
    after = clean_inventory(post, scope, "post", time.time(), row)
    require(after["PCIs"] == pre["inventory"]["PCIs"], "PCI_INVENTORY_MOVED")
    require(post["independent_gpu_accounting"] == row, "INDEPENDENT_GPU_ACCOUNTING_DIFFERS")
    reader = helper(repo, ORIGINAL + "/read_result.py")
    original_precheck = repo / ORIGINAL / "precheck.py"
    require(Path(reader.pins.__file__).resolve() == original_precheck.resolve(), "RESULT_READER_PROVIDER")
    read(original_precheck, 131072, FIXED[ORIGINAL + "/precheck.py"])
    audited = reader.read_result(output / "worker", runtime, scope_path, scope_sha, host_path, host_pin, row[0])
    qualified = (host.get("startup_numeric_step") == row[0] and host.get("owned_srun_rc") == 0 and not host.get("forced_client_reap")
                 and host.get("log_within_bound") and audited["component_timing_candidate"])
    return {"request": REQUEST, "actual_gpu_step": row[0], "planned": DENOMINATOR,
            "raw_worker_audit": audited, "component_timing_candidate": bool(qualified),
            "independent_post": "CLEAN", "training_performance": False, "promotion_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run", "finish"))
    for name in ("repo", "runtime", "original-dependencies", "scope", "pre", "protection"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--target-runtime", required=True)
    parser.add_argument("--target-scope", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--target-gate")
    parser.add_argument("--ordinary-calendar", type=Path)
    parser.add_argument("--observed-now", type=float)
    for option in ("host-terminal", "post"):
        parser.add_argument("--" + option, type=Path)
        parser.add_argument("--" + option + "-sha256")
    args = parser.parse_args()
    scope_sha = args.scope_sha256
    require(SHA.fullmatch(scope_sha), "SCOPE_SHA")
    scope = json.loads(read(args.scope, 16384, scope_sha))
    pre = json.loads(read(args.pre, 16384, scope["pre_SHA256"]))
    authority = json.loads(read(args.protection, 16384, scope["protection_authority_SHA256"]))
    proof = frozen(args.repo.resolve(strict=True), args.runtime, args.original_dependencies)
    if args.mode == "dry-run":
        now = args.observed_now if args.observed_now is not None else time.time()
        clean_inventory(pre, scope, "pre", now)
        protection_gate(authority, scope, now)
        argv, limit = build_argv(scope, scope_sha, args.target_runtime, args.target_scope, args.target_output, now, args.target_gate)
        print(json.dumps({"dry_run": True, "argv": argv, "client_timeout_seconds": limit,
                          "target_calls": 0, "planned": DENOMINATOR, "sources": proof}, sort_keys=True))
        return 0
    require(args.observed_now is None and args.runtime.as_posix() == args.target_runtime
            and args.scope.as_posix() == args.target_scope, "NONLOCAL_OR_OVERRIDE_TARGET")
    out = Path(path_arg(args.target_output))
    if args.mode == "run":
        require(args.ordinary_calendar and args.target_gate, "EXACT_ORDINARY_AND_WORKER_GATE_REQUIRED")
        gate = Path(path_arg(args.target_gate))
        result = try_once(args.repo.resolve(strict=True), args.runtime, args.scope, scope_sha, scope, pre, authority,
                          args.ordinary_calendar, gate, out)
        print(json.dumps({"state": result["state"], "owned_srun_rc": result.get("owned_srun_rc"), "final_pass": False}))
        passed = (result["state"] == "TERMINAL" and result["actual_resources_match"]
                  and result.get("startup_numeric_step") and result.get("owned_srun_rc") == 0
                  and not result.get("forced_client_reap")
                  and result["accounting"][0][2:4] == ["COMPLETED", "0:0"])
        return int(not passed)
    check_scope(scope, scope["epoch"])
    clean_inventory(pre, scope, "pre", scope["epoch"])
    protection_gate(authority, scope, scope["epoch"])
    require(args.post and args.host_terminal and SHA.fullmatch(args.post_sha256 or "")
            and SHA.fullmatch(args.host_terminal_sha256 or "") and args.host_terminal == out / "host-terminal.json",
            "POST_PINS_REQUIRED")
    post = json.loads(read(args.post, 32768, args.post_sha256))
    host = json.loads(read(args.host_terminal, 131072, args.host_terminal_sha256))
    audit = independently_finish(args.repo.resolve(strict=True), scope, scope_sha, pre, post, host,
                                 args.host_terminal, args.host_terminal_sha256, args.runtime, args.scope, out)
    print(json.dumps({"finish": exclusive(out / "independent-final.json", audit), "state": audit["independent_post"],
                      "component_timing_candidate": audit["component_timing_candidate"]}))
    return int(not audit["component_timing_candidate"])


if __name__ == "__main__":
    raise SystemExit(main())
