"""Check the original probe loader with an isolated single device assigned by Slurm."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
import time

import snapshot
from assigned_kfd import assigned_kfd
from slurm_accounting import wait_terminal


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "Deep-C4-causal-consumer-r483-runtime-owner-GPU-v2"
PACKAGE = Path(__file__).resolve().parent
RUN = (Path("/refactor") if PACKAGE.is_relative_to("/refactor") else R) / "results" / LABEL
HOST_PY = "/usr/local/python3.12/bin/python3"
PARENT, NODE = "1103606", "f13r2n01"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
snapshot.NODE = NODE
snapshot.STEP = re.compile(PARENT + r"\.\d+")


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    with (RUN / name).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def collect(label):
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS], timeout=14, text=True)
    rows = [r for line in raw.splitlines() if len(r := line.split("|")) == 7 and r[1] == label]
    require(len(rows) <= 1, "duplicate Slurm label")
    return rows[0] if rows else None


def step(phase, args, deadline):
    label = LABEL + "-" + phase
    gpu = phase == "gpu"
    require(collect(label) is None, "reconcile original step rather than replay")
    seconds = min(130 if gpu else 50, int(deadline - time.time()) - (65 if gpu else 20))
    require(seconds > 4, "scope deadline expired")
    command = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--cpu-bind=none", "-N1", "-n1", "-c4",
               "--mem=8192M", "--gres=" + ("dcu:1" if gpu else "none"), "-w", NODE,
               "-t", "00:02:00" if gpu else "00:01:00", "--job-name=" + label, *args]
    with (RUN / (phase + ".log")).open("xb") as log:
        try:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            rc = None
    result = wait_terminal(lambda: collect(label), time.monotonic() + min(24, max(0, deadline - time.time())),
                           max_observations=12)
    row = result["row"]
    result.update(rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                                   and (("gres/dcu=1" in row[6]) if gpu else "gres/dcu=" not in row[6])))
    save(phase + "-accounting.json", result)
    return result


def holder_snapshot():
    snap = snapshot.snapshot(False)
    require(snap["kfd_holder_count"] == 0 and snap["full_fd_inventory_performed"], "device holder remains")
    assigned = RUN / "assignment.json"
    if assigned.is_file():
        owned = read(assigned)
        try:
            row = (Path("/proc") / str(owned["pid"]) / "stat").read_text().rsplit(") ", 1)[1].split()
            require(row[0] == "Z" or int(row[19]) != owned["startticks"], "owned process still visible")
        except FileNotFoundError:
            pass
    print(json.dumps(snap), flush=True)


def worker():
    claim = read(PACKAGE / "CLAIM.json")
    before = read(RUN / "before.json")
    node = socket.gethostname().split(".")[0]
    handle = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    pci_inventory = snapshot.drm_inventory()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(node == NODE and snapshot.STEP.fullmatch(handle) and pci_inventory == before["pci"]
            and boot == before["boot_id"] and len(snapshot.kernel_holders()) == 0
            and 0 <= time.monotonic_ns() - before["monotonic_ns"] <= 45_000_000_000,
            "assigned compute node differs from host snapshot")
    hip = C.CDLL("libamdhip64.so")
    count, raw = C.c_int(), C.create_string_buffer(32)
    require(hip.hipGetDeviceCount(C.byref(count)) == 0 and count.value == 1
            and hip.hipDeviceGetPCIBusId(raw, len(raw), 0) == 0, "wrong HIP lease")
    pci = raw.value.decode().lower()
    pci = "0000:" + pci if len(pci) == 7 else pci
    require(pci in pci_inventory, "wrong HIP PCI")
    owned = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    save("assignment.json", dict(node=node, step=handle, pci=pci_inventory, boot_id=boot, assigned=assigned_kfd(pci),
                                 pid=os.getpid(), startticks=int(owned[19])))
    owner = runpy.run_path(str(PACKAGE / "check_runtime_owner.py"))
    owner["identity"](owner["BUILD_AUDIT"], claim["build_audit_SHA256"])
    owner["identity"](owner["BINDING"], claim["binding_SHA256"])
    audit = read(owner["BUILD_AUDIT"])
    binding = runpy.run_path(str(owner["BINDING"]))
    probe, api, report = owner["load_and_verify"](owner["PROBE"], audit, binding, owner["ROOT"])
    require(probe and len(api) == 13, "live ABI unavailable")
    report.update(actual_step=handle, runtime_provenance_only=True, device_numerics_qualified=False)
    save("native.json", report)
    print(json.dumps({"actual_step": handle, "resolved_entries": len(api)}), flush=True)


def controller():
    intent = read(RUN / "intent.json")
    claim = read(PACKAGE / "CLAIM.json")
    deadline = intent["deadline"]
    require(sha(PACKAGE / "CLAIM.json") == intent["claim_SHA256"] and
            all(sha(PACKAGE / name) == digest for name, digest in claim["files"].items()), "published package differs")
    original = R / "incoming/Deep-C4-causal-consumer-r483/CPU-v2"
    require(sha(original / "launch_CPU.py") == claim["parent_launcher_SHA256"]
            and sha(original / "CPU_outer.sh") == claim["parent_outer_SHA256"], "original admission helper differs")
    runpy.run_path(str(original / "launch_CPU.py"))["verify_membership"]()
    prior = collect("Deep-C4-causal-consumer-r483-runtime-owner-CPU-v1")
    require(prior and prior[0] == "1103606.157" and prior[2:4] == ["FAILED", "6:0"], "prior CPU run unsettled")
    stages = {}
    snap = [HOST_PY, "-B", str(Path(__file__).resolve()), "snapshot"]
    stages["pre"] = step("pre", snap, deadline)
    if stages["pre"]["passed"]:
        before = read(RUN / "pre.log")
        save("before.json", before)
        params = ["env", "RUNTIME_OWNER_DEADLINE=" + repr(deadline),
                  "WAVE_INSIDE_R2323=Deep-C4-causal-consumer-r483/runtime-owner-v2/GPU_inside.sh",
                  "bash", str(original / "CPU_outer.sh"), "probe"]
        stages["gpu"] = step("gpu", params, deadline)
        if stages["gpu"]["terminal_visible"]:
            stages["post"] = step("post", snap, deadline)
            if stages["post"]["passed"]:
                after = read(RUN / "post.log")
                require(before["pci"] == after["pci"] and before["boot_id"] == after["boot_id"], "postflight identity changed")
                save("after.json", after)
    passed = all(stages.get(k, {}).get("passed") for k in ("pre", "gpu", "post"))
    save("outcome.json", dict(passed=passed, stages=stages))
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        holder_snapshot()
    elif sys.argv[1] == "worker":
        worker()
    else:
        raise SystemExit(controller())
