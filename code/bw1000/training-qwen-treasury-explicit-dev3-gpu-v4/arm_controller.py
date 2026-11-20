"""Run the frozen development comparison on one DCU and reconcile terminal ownership."""

import ctypes
import re
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

import snapshot as devices
from owned_cache import DIRECTORIES as CACHE_DIRECTORIES, observe, prepare
from slurm_accounting import wait_terminal
from snapshot_io import separate_logs, read_host_snapshot
from scope import PACKAGE, SRC, CPU, MEDIA, ARMS, verify_prerequisites, verify_model, seconds_left, stage_limits


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
ARM_NAME = sys.argv[1]
if ARM_NAME not in ARMS:
    raise ValueError("unknown frozen independent arm")
RUN = PACKAGE / "arms" / ARM_NAME
PREP = CPU
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
ADAPTER = Path(ARMS[ARM_NAME]["adapter"]) if ARMS[ARM_NAME]["adapter"] else None
PARENT, NODE = ARMS[ARM_NAME]["parent"], ARMS[ARM_NAME]["node"]
LABEL = PACKAGE.name + "-" + ARM_NAME
devices.NODE = NODE
devices.STEP = re.compile(PARENT + r"\.\d+")
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_bytes())


def save(name, value):
    with (RUN / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pin(path):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink()
                and all(getattr(before, f) == getattr(after, f) for f in fields), "asset changed during validation")
    return dict(bytes=before.st_size, SHA256=digest.hexdigest())


def query(*args, terminal_deadline=None):
    deadline = read(RUN / "intent.json")["deadline"]
    timeout = seconds_left(deadline, time.time(), 12)
    if terminal_deadline is not None:
        timeout = min(timeout, terminal_deadline - time.monotonic())
    require(timeout > 0, "own terminal window exhausted")
    return subprocess.check_output(args, text=True, timeout=timeout).strip().splitlines()


def accounting_for(parent, label, terminal_deadline=None):
    rows = [row for line in query("sacct", "-n", "-P", "-j", parent, "-o", COLUMNS, terminal_deadline=terminal_deadline)
            if len(row := line.split("|")) == 7 and row[1] == label]
    require(len(rows) <= 1, "duplicate exact step label")
    return rows[0] if rows else None


def accounting(label, terminal_deadline=None):
    return accounting_for(PARENT, label, terminal_deadline)



def admit():
    alloc = query("squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N")
    require(len(alloc) == 1, "parent not visible")
    state, count, nodes = alloc[0].split("|")
    require((state, count) == ("RUNNING", "40") and NODE in query("scontrol", "show", "hostnames", nodes),
            "node allocation changed")
    for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        step = fields.get("StepId", "")
        if step.startswith(PARENT + ".") and step.rsplit(".", 1)[1].isdigit():
            require(NODE not in query("scontrol", "show", "hostnames", fields["NodeList"]),
                    "node already has a numbered step")


def verify_assets(claim):
    verify_prerequisites(accounting_for)
    require((claim["arms"][ARM_NAME]["parent"], claim["arms"][ARM_NAME]["node"]) == (PARENT, NODE),
            "frozen independent placement changed")
    require(claim["arms"][ARM_NAME]["adapter"] == ARMS[ARM_NAME]["adapter"],
            "claim and actual canonical adapter path differ")


def start_step(phase, args, deadline):
    gpu = phase == "gpu"
    label = LABEL + "-" + phase
    require(accounting(label) is None, "original step exists; no replay")
    limits = stage_limits(deadline, time.time(), phase)
    if gpu:
        work_start = time.time()
        work_budget = min(900, limits["slurm_seconds"] - 60, deadline - work_start - 180)
        require(work_budget > 0, "no original load/generation budget remains")
        work_deadline = work_start + work_budget
        save("work-intent.json", dict(original_epoch=read(RUN / "intent.json")["epoch"],
                                     budget_started=work_start, deadline=work_deadline,
                                     load_and_generation_budget_seconds=work_budget,
                                     actual_Slurm_cap_seconds=limits["slurm_seconds"]))
        args = container(work_deadline)
    require(limits["slurm_seconds"] <= deadline - time.time() - (180 if gpu or phase == "pre" else 10),
            "stage preparation consumed original reserved budget")
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--nodes=1", "--ntasks=1",
           "--nodelist=" + NODE, "--exact", "--immediate=5", "--cpus-per-task=" + ("8" if gpu else "2"),
           "--mem=" + ("65536M" if gpu else "4096M"), "--gres=" + ("dcu:1" if gpu else "none"),
           "--time=" + limits["slurm_time"], *args]
    rc = None
    with separate_logs(RUN / (phase + ".stdout"), RUN / (phase + ".stderr")) as (stdout, stderr):
        child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
        handle = dict(PID=child.pid, limits=limits)
        try:
            fields = Path("/proc", str(child.pid), "stat").read_text().rsplit(") ", 1)[1].split()
            handle.update(PGID=int(fields[2]), startticks=int(fields[19]))
        except FileNotFoundError:
            handle["exited_before_identity_observation"] = True
        save(phase + "-owned-srun.json", handle)
        try:
            rc = child.wait(timeout=min(limits["client_seconds"], seconds_left(deadline, time.time(), 1000, 5)))
        except subprocess.TimeoutExpired:
            pass
    terminal_deadline = time.monotonic() + min(35, max(0, deadline - time.time() - 5))
    accounting_errors = []
    def observe():
        try:
            return accounting(label, terminal_deadline)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            accounting_errors.append(type(error).__name__)
            return None
    result = wait_terminal(observe, terminal_deadline, max_observations=17)
    row = result["row"]
    result.update(rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"]
                                   and row[5] == NODE
                                   and (("gres/dcu=1" in row[6].split(",")) if gpu else "gres/dcu=" not in row[6])))
    result["accounting_errors"] = accounting_errors
    save(phase + "-accounting.json", result)
    return result


def runtime_bind_source(host):
    host = Path(host)
    if host == Path("/opt/hyhal"):
        target = Path("/usr/local/hyhal")
        require(host.resolve(strict=True) == target and target.is_dir() and not target.is_symlink(),
                "system hyhal alias identity changed")
        return target
    require(host.is_dir() and not host.is_symlink(), "required runtime bind missing or linked")
    return host


def container(deadline):
    runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--rocm", "--no-home", "--pwd", "/"]
    binds = [(RUN, RUN, "rw"), (SRC, SRC, "ro"), (MODEL, MODEL, "ro"), (PREP, PREP, "ro"),
             (MEDIA, PREP / "dev/media", "ro"),
             (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
             (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
             (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
             (Path("/opt/hyhal"), "/opt/hyhal", "ro")]
    if ADAPTER is not None:
        binds.append((ADAPTER, ADAPTER, "ro"))
    for host, target, mode in binds:
        runtime.extend(("--bind", f"{runtime_bind_source(host)}:{target}:{mode}"))
    for subdir, target in (("tmp","/tmp"),("var_tmp","/var/tmp")):
        runtime.extend(("--bind", f"{RUN}/cache/{subdir}:{target}:rw"))
    owned_environment = {key: str(RUN/"cache"/name) for key, name in CACHE_DIRECTORIES.items()}
    cache_export = " ".join(f"{key}={shlex.quote(value)}" for key,value in owned_environment.items())
    inside_seconds = int(seconds_left(deadline, time.time(), 900))
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:" + str(SRC) + "; "
             "export " + cache_export + "; cd -- " + shlex.quote(str(RUN)) + "; "
             "exec /usr/bin/timeout --kill-after=10s " + str(inside_seconds) + "s " +
             shlex.join(["python3", "-B", "-u", str(SRC / "arm_controller.py"), ARM_NAME, "worker", str(deadline)]))
    return runtime + [str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def snapshot():
    snap = devices.snapshot(False)
    require(snap["kfd_holder_count"] == 0 and snap["full_fd_inventory_performed"], "GPU holder remains")
    if (RUN / "assigned.json").is_file():
        assigned = read(RUN / "assigned.json")
        process = Path("/proc") / str(assigned["worker_pid"]) / "stat"
        if process.is_file():
            data = process.read_text().split(") ", 1)[1].split()
            require(int(data[19]) != assigned["worker_startticks"] or data[0] == "Z", "assigned worker remains")
    print(json.dumps(snap), flush=True)


def leased_kfd(pci):
    matches = []
    denied = 0
    for node in (devices.KFD / "topology/nodes").iterdir():
        try:
            gpu_id = int((node / "gpu_id").read_text())
            if not gpu_id:
                continue
            props = dict(row.split(maxsplit=1) for row in (node / "properties").read_text().splitlines())
        except PermissionError:
            denied += 1
            continue
        reported = devices.pci_for_render("renderD" + str(int(props["drm_render_minor"])))
        if reported == pci:
            domain, bus, slot_function = reported.split(":")
            slot, function = slot_function.split(".")
            location = int(bus, 16) << 8 | int(slot, 16) << 3 | int(function, 16)
            require(int(props["domain"]) == int(domain, 16) and int(props["location_id"]) == location,
                    "leased KFD and DRM disagree")
            matches.append(gpu_id)
    require(len(matches) == 1 and matches[0] > 0, "leased HIP PCI has no unique accessible KFD identity")
    return matches[0], denied


def worker(deadline):
    intent = read(RUN / "intent.json")
    require(time.time() < deadline <= intent["deadline"] - 180 and deadline - intent["epoch"] <= 1800,
            "worker exceeds original global epoch")
    save("worker-progress.json", dict(arm=ARM_NAME, planned_rows=108, topic_denominator=36, max_new_tokens=64,
                                     original_epoch=intent["epoch"], work_deadline=deadline, actual_training_updates=0))
    observe(RUN)
    # The exact native entry is checked before Torch import, HIP CDLL, or model hashing/loading.
    require("torch" not in sys.modules, "Torch imported before real input-entry gate")
    from native_generation import bound_inputs
    input_root = (PREP / "dev").resolve(strict=True)
    inventory = PREP / "inventory.json"
    visual, text = bound_inputs(input_root, inventory)
    require(len(visual) == len(text) == 108 and "torch" not in sys.modules, "real native input-entry gate failed")
    require(Path(bound_inputs.__code__.co_filename) == SRC / "check_explicit_dev_inputs.py",
            "different bound_inputs provider")
    input_pins = read(inventory)
    save("input-entry-gate.json", dict(actual_native_entry="native_generation.bound_inputs",
         inventory_SHA256=pin(inventory)["SHA256"], input_root=str(input_root), actual_files=len(input_pins),
         visual_rows=len(visual), text_only_rows=len(text), torch_imported=False,
         native_generation_SHA256=pin(SRC / "native_generation.py")["SHA256"],
         bound_inputs_SHA256=pin(SRC / "check_explicit_dev_inputs.py")["SHA256"],
         member_SHA256={name: pin(input_root / name)["SHA256"] for name in sorted(input_pins)}))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "ms-swift", "transformers", "qwen-vl-utils")}
    require(versions == {"torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580", "ms-swift":"4.4.2",
                         "transformers":"5.12.1", "qwen-vl-utils":"0.0.14"}, "node provider versions differ")
    first = read(RUN / "before.json")
    node = socket.gethostname().split(".")[0]
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(node == NODE and devices.STEP.fullmatch(step), "wrong Slurm compute node or step")
    current = dict(node=node, step=step, pci=devices.drm_inventory(), kfd_holder_count=len(devices.kernel_holders()),
                   boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), monotonic_ns=time.monotonic_ns())
    require(first["pci"] == current["pci"] and first["boot_id"] == current["boot_id"]
            and 0 <= current["monotonic_ns"] - first["monotonic_ns"] <= 45_000_000_000
            and current["kfd_holder_count"] == 0, "device claim changed after host check")
    hip = ctypes.CDLL("libamdhip64.so")
    count = ctypes.c_int()
    pci = ctypes.create_string_buffer(32)
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
            and hip.hipDeviceGetPCIBusId(pci, len(pci), 0) == 0, "unexpected HIP visibility")
    pci = pci.value.decode().lower()
    pci = "0000:" + pci if len(pci) == 7 else pci
    require(pci in current["pci"] and len(os.sched_getaffinity(0)) >= 8, "unexpected CPU or PCI range")
    kfd, denied = leased_kfd(pci)
    process = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
    save("assigned.json", dict(current, visible_pci=pci, visible_kfd_id=kfd, inaccessible_topology_nodes=denied,
                               worker_pid=os.getpid(), worker_startticks=int(process[19])))
    model_validation_start = time.monotonic()
    verify_model()
    save("provider.json", dict(versions=versions, model_manifest_SHA256=pin(SRC / "MODEL_ASSETS.json")["SHA256"],
                              model_validation_seconds=time.monotonic()-model_validation_start))
    sys.argv = [str(SRC / "native_generation.py"), "--arm", ARM_NAME, "--inputs", str(PREP / "dev"),
                "--inventory", str(PREP / "inventory.json"), "--model", str(MODEL),
                "--output", str(RUN / "outputs"), "--deadline", str(deadline)]
    if ADAPTER is not None:
        sys.argv.extend(("--adapter", str(ADAPTER)))
    from runpy import run_path
    run_path(sys.argv[0], run_name="__main__")


def controller():
    intent, claim = read(RUN / "intent.json"), read(SRC / "CLAIM.json")
    require(pin(SRC / "CLAIM.json")["SHA256"] == intent["claim_SHA256"]
            and (claim["arms"][ARM_NAME]["parent"], claim["arms"][ARM_NAME]["node"], claim["label"] + "-" + ARM_NAME) == (PARENT, NODE, LABEL), "scope changed")
    deadline = intent["deadline"]
    budget = int(deadline - time.time())
    require(0 < budget <= 1800, "controller deadline expired")
    def expired(*_):
        raise TimeoutError("controller deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    for name, sha in claim["source_SHA256"].items():
        require(pin(SRC / name)["SHA256"] == sha, "source pin differs")
    for phase in ("pre", "gpu", "post"):
        require(accounting(LABEL + "-" + phase) is None, "step exists, reconcile without replay")
    admit()
    verify_assets(claim)
    admit()
    stage = {}
    snap = [HOST_PYTHON, "-B", str(SRC / "arm_controller.py"), ARM_NAME, "snapshot"]
    stage["pre"] = start_step("pre", snap, deadline)
    if stage["pre"]["passed"]:
        before = read_host_snapshot(RUN / "pre.stdout", NODE, stage["pre"]["row"][0])
        save("before.json", before)
        prepare(RUN)
        stage["gpu"] = start_step("gpu", None, deadline)
        if stage["gpu"]["terminal_visible"]:
            stage["post"] = start_step("post", snap, deadline)
            if stage["post"]["passed"]:
                after = read_host_snapshot(RUN / "post.stdout", NODE, stage["post"]["row"][0])
                require(before["boot_id"] == after["boot_id"] and before["pci"] == after["pci"], "postflight identity changed")
                save("after.json", after)
    receipt_path = RUN / "outputs/generation-outcome.json"
    receipt = read(receipt_path) if receipt_path.is_file() else None
    verified = bool(receipt and receipt["arm"] == ARM_NAME and receipt["generated"]["rows"] == 108)
    passed = all(stage.get(p, {}).get("passed") for p in ("pre", "gpu", "post")) and verified
    save("controller-outcome.json", dict(passed=passed, stages=stage, generation_verified=verified))
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[2] == "worker":
        try:
            worker(float(sys.argv[3]))
        except BaseException as error:
            save("worker-failure.json", dict(arm=ARM_NAME, planned_rows=108, actual_training_updates=0, error_class=type(error).__name__))
            raise
    elif sys.argv[2] == "snapshot":
        snapshot()
    else:
        raise SystemExit(controller())
