"""Run nine-pair VGGT and one fresh checkpoint20 resume in one isolated Slurm step."""

import base64
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = R / "results"
LABEL = "training-vggt-tum-three-recording-native25-resume20-v1"
REQUEST = "7b07ac6c8bd3e86e63e7528fd1c885f0"
PARENT, NODE = "1103606", "f13r4n09"
CPU, MEMORY = 4, 65536
CANDIDATE = R / "candidates/training-vggt-tum-three-recording-native-preflight-v2-3653b95925860e0dd3f9bbfb9f1b1fde"
UPSTREAM = RESULTS / "training-vggt-native-provider-HIP-CPU-v1/upstream"
WEIGHTS = RESULTS / "training-vggt-original-public-model-v1/model"
WEIGHT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
WEIGHT_STAT = (5026874952, 20701942394, 1794489337000000000)
BUNDLE_SHA = '647b78e286687cc1d55670f64472fbe0f395c1b81cbeb928999fcf415f6a59f7'
OVERLAY_SHA = '762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
HOST_PY = "/usr/local/python3.12/bin/python3"
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest):
    require(path.is_file() and not path.is_symlink(), "missing or aliased input: " + path.name)
    raw = path.read_bytes()
    require(sha(raw) == digest, "input digest differs: " + path.name)
    return raw


def save(directory, name, data):
    with (directory / name).open("x", encoding="utf8") as out:
        json.dump(data, out, indent=2, sort_keys=True, allow_nan=False)


def weight_identity(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and (value.st_size, value.st_ino, value.st_mtime_ns) == WEIGHT_STAT,
            "qualified model file identity changed")


def accounting(label, *, timeout=10, parent=PARENT):
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", parent, "-o", COLS], timeout=timeout, text=True)
    rows = [row for line in output.splitlines() if len(row := line.split("|")) == 7 and row[1] == label]
    require(len(rows) <= 1, "duplicate exact Slurm step label")
    return rows[0] if rows else None


def step(root, phase, args, deadline):
    from slurm_accounting import wait_terminal
    from run_contract import allocation_matches, step_budget

    name = LABEL + "-" + phase
    require(accounting(name) is None, "original Slurm label already exists")
    native = phase == "gpu"
    limit, slurm_limit = step_budget(phase, time.time(), deadline)
    cmd = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--cpu-bind=none", "-N1", "-n1",
           "-c" + str(CPU if native else 2), "--mem=" + str(MEMORY if native else 2048) + "M",
           "--gres=" + ("dcu:1" if native else "none"), "-w", NODE,
           "-t", slurm_limit, "--job-name=" + name, *args]
    with (root / (phase + ".log")).open("xb") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            rc = None
    expires = time.monotonic() + min(22, max(0, deadline - time.time()))
    def observe():
        remaining = expires - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return accounting(name, timeout=min(8, remaining))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
    result = wait_terminal(observe, expires, max_observations=9)
    row = result["row"]
    result.update(rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                                   and allocation_matches(row[6], native)))
    save(root, phase + "-accounting.json", result)
    return result


def host_snapshot(root):
    import snapshot

    snapshot.NODE = NODE
    snapshot.STEP = __import__("re").compile(PARENT + r"\.\d+")
    row = snapshot.snapshot(False)
    require(row["kfd_holder_count"] == 0 and row["full_fd_inventory_performed"], "node still holds KFD")
    owner = root / "assignment.json"
    if owner.is_file():
        assigned = json.loads(owner.read_bytes())
        try:
            fields = Path(f"/proc/{assigned['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
            require(fields[0] == "Z" or int(fields[19]) != assigned["startticks"], "native process still live")
        except FileNotFoundError:
            pass
    print(json.dumps(row), flush=True)


def native(root):
    import snapshot
    from assigned_kfd import assigned_kfd

    before = json.loads((root / "before.json").read_bytes())
    intent = json.loads((root / "intent.json").read_bytes())
    require(time.time() < intent["deadline"] - 45 and os.environ["SLURM_JOB_ID"] == PARENT
            and socket.gethostname().split(".")[0] == NODE and len(os.sched_getaffinity(0)) >= CPU,
            "wrong native allocation")
    require(snapshot.drm_inventory() == before["pci"] and not snapshot.kernel_holders()
            and before["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            and 0 <= time.monotonic_ns() - before["monotonic_ns"] <= 45_000_000_000, "stale or occupied device snapshot")
    hip = ctypes.CDLL("libamdhip64.so")
    count, pci_raw = ctypes.c_int(), ctypes.create_string_buffer(32)
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
            and hip.hipDeviceGetPCIBusId(pci_raw, len(pci_raw), 0) == 0, "one HIP device required")
    pci = pci_raw.value.decode().lower()
    pci = "0000:" + pci if len(pci) == 7 else pci
    require(pci in before["pci"], "HIP PCI does not belong to the node")
    save(root, "assignment.json", dict(step=PARENT+"."+os.environ["SLURM_STEP_ID"], node=NODE, pci=pci,
                                        kfd=assigned_kfd(pci), pid=os.getpid(), startticks=int(
                                            Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19])))
    require(importlib.metadata.version("torch") ==
            "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "qualified Torch provider differs")
    env = dict(os.environ, VGGT_DOMAIN_EPOCH=str(intent["epoch"]),
               VGGT_DOMAIN_DEADLINE=str(intent["deadline"] - 100))
    receipts = {}
    for arm in ("continuous", "resume20"):
        require(time.time() < intent["deadline"] - 125, "original training budget exhausted")
        args = [sys.executable, "-B", "-u", "/pilot/src/native_worker.py", "--arm", arm,
                "--data", "/sealed/data", "--weights", "/weights", "--upstream", "/vggt-source",
                "--output-root", "/pilot/arms", "--bundle-sha256", BUNDLE_SHA, "--overlay-sha256", OVERLAY_SHA]
        with (root / (arm + ".log")).open("xb") as log:
            rc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                timeout=max(1, int(intent["deadline"] - time.time()) - 110)).returncode
        require(rc == 0, "native training arm failed: " + arm)
        proof = root / "arms" / arm / "receipt.json"
        receipts[arm] = dict(SHA256=sha(proof.read_bytes()), value=json.loads(proof.read_bytes()))
        weight_identity(Path("/weights/model.pt"))
    from compare_resume import compare
    comparison = compare({arm: proof["value"] for arm, proof in receipts.items()})
    save(root, "native-result.json", dict(receipts=receipts, comparison=comparison,
                                          actual_DCU=1, formal_CO3D_eligible=False,
                                          formal_performance_eligible=False))
    require(comparison["comparison_passed"], "fresh-process recovery differs; evidence retained")


def controller(payload):
    os.umask(0o077)
    root = RESULTS / LABEL
    require(payload["request"]==REQUEST and not root.exists(),"new request identity or original namespace differs")
    epoch = time.time()
    deadline = epoch + 1800
    def deadline_reached(*_):
        raise TimeoutError("Original HOST experiment epoch expired")
    signal.signal(signal.SIGALRM, deadline_reached)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
    row = dict(item.split("=", 1) for item in subprocess.check_output(
        ["scontrol", "show", "job", "-o", PARENT], timeout=8, text=True).split() if "=" in item)
    members = subprocess.check_output(
        ["scontrol", "show", "hostnames", row["NodeList"]], timeout=8, text=True).split()
    require(all(accounting(LABEL + "-" + phase) is None for phase in ("pre", "gpu", "post")), "original label exists")
    for line in subprocess.check_output(["scontrol", "show", "step", PARENT, "-o"],
                                        timeout=8, text=True).splitlines():
        fields = dict(field.split("=", 1) for field in line.split() if "=" in field)
        suffix = fields["StepId"].removeprefix(PARENT + ".")
        if suffix in ("batch", "extern"):
            continue
        require(suffix.isdecimal(), "unknown current step")
        active = subprocess.check_output(["scontrol", "show", "hostnames", fields["NodeList"]],
                                         timeout=8, text=True).split()
        require(bool(active) and NODE not in active, "active step on target node")
    weight_identity(WEIGHTS / "model.pt")
    require(shutil.disk_usage(RESULTS).free > 10 * 1024**3, "insufficient experiment storage")
    root.mkdir()
    save(root, "intent.json", dict(publication=payload["publication"], request=payload["request"],
                                   epoch=epoch, deadline=deadline, parent=PARENT, node=NODE))
    print(json.dumps(dict(stage="intent", label=LABEL, request=payload["request"], epoch=epoch)), flush=True)
    for name, item in payload["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and len(item["b64"]) < 6_000_000, "invalid staged member")
        raw = base64.b64decode(item["b64"], validate=True)
        require(len(raw) == item["bytes"] and sha(raw) == item["sha"], "staged input differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            out.write(raw)
    sys.path.insert(0, str(root / "src"))
    from preflight_gate import bind_preflight
    from run_contract import parent_matches

    require(parent_matches(row,members,os.getuid(),PARENT,NODE),"parent allocation or ownership changed")
    inherited=bind_preflight(RESULTS,root/"src/actual-922-preflight.json",CANDIDATE)
    save(root,"preflight-binding.json",inherited)
    (root / "arms").mkdir()
    work = root / "src/run_tum_continuous_once.py"
    before = step(root, "pre", [HOST_PY, "-B", str(work), "snapshot"], deadline)
    stages = {"pre": before}
    if before["passed"]:
        save(root, "before.json", json.loads((root / "pre.log").read_bytes()))
        args = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
        for bind in (f"{root}:/pilot:rw", f"{CANDIDATE}:/sealed:ro", f"{WEIGHTS}:/weights:ro", f"{UPSTREAM}:/vggt-source:ro",
                     f"{RESULTS/'training-t06-sdxl-runtime-v1'}:/training-runtime:ro", "/opt/hyhal:/opt/hyhal:ro"):
            args += ["--bind", bind]
        shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
                 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
                 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
                 'PYTHONPATH=/training-runtime/site:/sealed/framework/generation/src:/sealed/framework; '
                 'exec /usr/bin/timeout --kill-after=5s 1300s python3 -B -u /pilot/src/run_tum_continuous_once.py native')
        stages["gpu"] = step(root, "gpu", [*args, str(R / "containers/dtk26-sandbox-r1"),
                                           "/bin/bash", "--noprofile", "--norc", "-c", shell], deadline)
        if stages["gpu"]["terminal_visible"]:
            stages["post"] = step(root, "post", [HOST_PY, "-B", str(work), "snapshot"], deadline)
            if stages["post"]["passed"]:
                after = json.loads((root / "post.log").read_bytes())
                original = json.loads((root / "before.json").read_bytes())
                require(after["pci"] == original["pci"] and after["boot_id"] == original["boot_id"], "post identity differs")
                save(root, "after.json", after)
    native_result = root / "native-result.json"
    result = dict(request=payload["request"], publication=payload["publication"], stages=stages,
                  passed=all(stages.get(phase, {}).get("passed") for phase in ("pre", "gpu", "post")) and native_result.is_file(),
                  result_SHA256=sha(native_result.read_bytes()) if native_result.is_file() else None)
    save(root, "outcome.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) == 2 else "controller"
    if mode == "snapshot":
        host_snapshot(Path(__file__).resolve().parents[1])
    elif mode == "native":
        native(Path("/pilot"))
    else:
        raise SystemExit(controller(PAYLOAD))
