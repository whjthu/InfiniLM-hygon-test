"""Run the frozen development comparison on one DCU and reconcile terminal ownership."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import time

import snapshot as devices
from slurm_accounting import wait_terminal


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
RUN = RESULTS / "training-qwen-treasury-dev-pair-hip-v1"
PREP = RESULTS / "training-qwen-treasury-dev-native-CPU-v1"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
ADAPTER = RESULTS / "training-qwen-treasury-sft-v3/continuous/train/checkpoint-5"
PARENT, NODE, LABEL = "1087656", "a01r4n14", RUN.name
devices.NODE = NODE
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


def query(*args):
    return subprocess.check_output(args, text=True, timeout=12).strip().splitlines()


def accounting(label):
    rows = [r for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS)
            if len(r := line.split("|")) == 7 and r[1] == label]
    require(len(rows) <= 1, "duplicate step label")
    return rows[0] if rows else None


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
    require(pin(PREP / "outcome.json")["SHA256"] == claim["dev_CPU_terminal_SHA256"],
            "native CPU development proof changed")
    outcome = read(PREP / "outcome.json")
    require(outcome["passed"] and outcome["rows"] == [accounting(PREP.name)], "CPU proof is not terminal")
    require(pin(PREP / "inventory.json")["SHA256"] == claim["inventory_SHA256"], "development input changed")
    require(pin(RUN / "src/MODEL_ASSETS.json")["SHA256"] == claim["model_manifest_SHA256"],
            "model manifest changed")
    for name, identity in read(RUN / "src/MODEL_ASSETS.json")["model"].items():
        require(pin(MODEL / name) == {"bytes": identity["bytes"], "SHA256": identity["SHA256"]},
                "base artifact changed: " + name)
    for name, identity in claim["adapter_files"].items():
        require(pin(ADAPTER / name) == identity, "candidate adapter changed: " + name)
    require(read(ADAPTER / "trainer_state.json")["global_step"] == 5, "candidate is not step five")


def start_step(phase, args, deadline):
    gpu = phase == "gpu"
    label = LABEL + "-" + phase
    require(accounting(label) is None, "existing step requires reconciliation")
    seconds = min(1110 if gpu else 60, int(deadline - time.time()) - (100 if gpu else 30))
    require(seconds > 0, "stage deadline expired")
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--nodes=1", "--ntasks=1",
           "--nodelist=" + NODE, "--exact", "--immediate=5", "--cpus-per-task=" + ("8" if gpu else "2"),
           "--mem=" + ("65536M" if gpu else "4096M"), "--gres=" + ("dcu:1" if gpu else "none"),
           "--time=" + ("18" if gpu else "2"), *args]
    rc = None
    with (RUN / (phase + ".log")).open("xb") as stream:
        try:
            rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            pass
    result = wait_terminal(lambda: accounting(label),
                           time.monotonic() + min(35, max(0, deadline - time.time())), max_observations=17)
    row = result["row"]
    result.update(rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"]
                                   and row[5] == NODE
                                   and (("gres/dcu=1" in row[6]) if gpu else "gres/dcu=" not in row[6])))
    save(phase + "-accounting.json", result)
    return result


def container(deadline):
    runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
    binds = ((RUN, RUN, "rw"), (MODEL, MODEL, "ro"), (PREP, PREP, "ro"), (ADAPTER, ADAPTER, "ro"),
             (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
             (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
             (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
             ("/opt/hyhal", "/opt/hyhal", "ro"))
    for host, target, mode in binds:
        runtime.extend(("--bind", f"{host}:{target}:{mode}"))
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site; "
             "exec " + shlex.join(["python3", "-B", "-u", str(Path(__file__).resolve()), "worker", str(deadline)]))
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


def worker(deadline):
    first, current = read(RUN / "before.json"), devices.snapshot(True)
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
    process = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
    save("assigned.json", dict(current, visible_pci=pci, worker_pid=os.getpid(), worker_startticks=int(process[19])))
    sys.argv = [str(RUN / "src/dev_pair_generation.py"), "--inputs", str(PREP / "dev"), "--inventory",
                str(PREP / "inventory.json"), "--model", str(MODEL), "--adapter", str(ADAPTER),
                "--output", str(RUN / "outputs"), "--deadline", str(deadline)]
    from runpy import run_path
    run_path(sys.argv[0], run_name="__main__")


def controller():
    intent, claim = read(RUN / "intent.json"), read(RUN / "claim.json")
    require(pin(RUN / "claim.json")["SHA256"] == intent["claim_SHA256"]
            and (claim["parent"], claim["node"], claim["label"]) == (PARENT, NODE, LABEL), "scope changed")
    deadline = intent["deadline"]
    budget = int(deadline - time.time())
    require(0 < budget <= 2700, "controller deadline expired")
    def expired(*_):
        raise TimeoutError("controller deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    for name, sha in claim["source_SHA256"].items():
        require(pin(RUN / "src" / name)["SHA256"] == sha, "source pin differs")
    for phase in ("pre", "gpu", "post"):
        require(accounting(LABEL + "-" + phase) is None, "step exists, reconcile without replay")
    admit()
    verify_assets(claim)
    admit()
    stage = {}
    snap = [HOST_PYTHON, "-B", str(Path(__file__).resolve()), "snapshot"]
    stage["pre"] = start_step("pre", snap, deadline)
    if stage["pre"]["passed"]:
        before = read(RUN / "pre.log")
        save("before.json", before)
        stage["gpu"] = start_step("gpu", container(min(deadline - 110, time.time() + 1020)), deadline)
        if stage["gpu"]["terminal_visible"]:
            stage["post"] = start_step("post", snap, deadline)
            if stage["post"]["passed"]:
                after = read(RUN / "post.log")
                require(before["boot_id"] == after["boot_id"] and before["pci"] == after["pci"], "postflight identity changed")
                save("after.json", after)
    receipt_path = RUN / "outputs/generation-outcome.json"
    receipt = read(receipt_path) if receipt_path.is_file() else None
    verified = bool(receipt and receipt["base"]["rows"] == receipt["domain"]["rows"] == 108)
    passed = all(stage.get(p, {}).get("passed") for p in ("pre", "gpu", "post")) and verified
    save("controller-outcome.json", dict(passed=passed, stages=stage, generation_verified=verified))
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(float(sys.argv[2]))
    elif sys.argv[1] == "snapshot":
        snapshot()
    else:
        raise SystemExit(controller())
