"""Run eight-pair VGGT training and one fresh-process resume in one isolated Slurm step."""

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = R / "results"
LABEL = "training-vggt-tum-eight-pairs25-recovery-v2"
PREDECESSOR = "training-vggt-tum-eight-pairs25-recovery-v1"
PARENT, NODE = "1103606", "f13r4n09"
CPU, MEMORY = 4, 65536
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
FRAMEWORK_MANIFEST = FRAMEWORK.parent / "source/FRAMEWORK_FILES.json"
FRAMEWORK_SHA = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
UPSTREAM = RESULTS / "training-vggt-native-provider-HIP-CPU-v1/upstream"
WEIGHTS = RESULTS / "training-vggt-original-public-model-v1/model"
WEIGHT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
WEIGHT_STAT = (5026874952, 20701942394, 1794489337000000000)
BUNDLE_SHA = 'fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c'
BACKEND_REL = "generation/src/gencuda/backends/vggt_training.py"
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


def accounting(label, deadline=None, job=PARENT):
    remaining = 10 if deadline is None else min(10, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError("original accounting deadline exhausted")
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", job, "-o", COLS], timeout=remaining, text=True)
    rows = [row for line in output.splitlines() if len(row := line.split("|")) == 7 and row[1] == label]
    require(len(rows) <= 1, "duplicate exact Slurm step label")
    return rows[0] if rows else None


def phase_budget(phase, deadline, now):
    require(phase in ("pre", "gpu", "post"), "unknown step phase")
    require(math.isfinite(deadline) and math.isfinite(now), "invalid original clock")
    post_reserve = 90 if phase != "post" else 0
    final_write_reserve = 2
    terminal_seconds = 22
    launch_seconds = 5
    client_cap = 1440 if phase == "gpu" else 65
    available = math.floor(deadline - now) - post_reserve - final_write_reserve
    # Slurm rounds fractional minutes upward. Request only whole minutes that fit.
    slurm_seconds = 60 * math.floor(min(client_cap - launch_seconds,
                                      available - terminal_seconds - launch_seconds) / 60)
    require(slurm_seconds >= 60, "original epoch has no complete step and cleanup budget")
    client_seconds = slurm_seconds + launch_seconds
    return dict(slurm_seconds=slurm_seconds, client_seconds=client_seconds,
                slurm_time=f"{slurm_seconds // 3600:02d}:{slurm_seconds // 60 % 60:02d}:00",
                terminal_seconds=terminal_seconds, post_reserve_seconds=post_reserve,
                latest_terminal_wall=deadline-post_reserve-final_write_reserve)


def step(root, phase, args, deadline):
    from slurm_accounting import wait_terminal

    name = LABEL + "-" + phase
    phase_budget(phase, deadline, time.time())
    require(accounting(name, deadline) is None, "original Slurm label already exists")
    native = phase == "gpu"
    budget = phase_budget(phase, deadline, time.time())
    cmd = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--cpu-bind=none", "-N1", "-n1",
           "-c" + str(CPU if native else 2), "--mem=" + str(MEMORY if native else 2048) + "M",
           "--gres=" + ("dcu:1" if native else "none"), "-w", NODE,
           "-t", budget["slurm_time"], "--job-name=" + name, *args]
    with (root / (phase + ".log")).open("xb") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=budget["client_seconds"]).returncode
        except subprocess.TimeoutExpired:
            rc = None
    terminal_wall = min(time.time()+budget["terminal_seconds"], budget["latest_terminal_wall"])
    timed_out_reads = []
    def read_terminal():
        if time.time() >= terminal_wall:
            return None
        try:
            return accounting(name, terminal_wall)
        except (subprocess.TimeoutExpired, TimeoutError):
            timed_out_reads.append(True)
            return None
    result = wait_terminal(read_terminal, time.monotonic()+max(0, terminal_wall-time.time()), max_observations=9)
    row = result["row"]
    result.update(rc=rc, original_deadline=deadline, phase_budget=budget, terminal_deadline=terminal_wall,
                  accounting_timeout_reads=len(timed_out_reads),
                  passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                                   and (("gres/dcu=1" in row[6]) if native else "gres/dcu=" not in row[6])))
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
                "--data", "/pilot/data", "--weights", "/weights", "--upstream", "/vggt-source",
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


def controller():
    os.umask(0o077)
    root = RESULTS / LABEL
    intent = json.loads((root / "intent.json").read_bytes())
    epoch, deadline = intent["epoch"], intent["deadline"]
    require(deadline == epoch + 1800 and epoch <= time.time() < epoch + 10,
            "unexpected controller launch window")
    manifest = pinned(root / "SOURCE_FILES.json", intent["manifest_SHA256"])
    for name, spec in json.loads(manifest)["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root), "source escaped frozen run")
        require(len(pinned(path, spec["SHA256"])) == spec["bytes"], "frozen source size differs")
    claim = json.loads((root/"CLAIM.json").read_bytes())
    require((claim["request"], claim["label"], claim["parent"], claim["node"])
            == (intent["request"], LABEL, PARENT, NODE), "launch identity differs")
    row = dict(item.split("=", 1) for item in subprocess.check_output(
        ["scontrol", "show", "job", "-o", PARENT], timeout=8, text=True).split() if "=" in item)
    require(row["JobState"] == "RUNNING" and row["NumNodes"] == "40"
            and row["UserId"].endswith(f"({os.getuid()})") and "gres/dcu=320" in row["AllocTRES"].split(",")
            and NODE in subprocess.check_output(
                ["scontrol", "show", "hostnames", row["NodeList"]], timeout=8, text=True).split(),
            "parent allocation changed")
    require(all(accounting(LABEL+"-"+phase, deadline) is None for phase in ("pre", "gpu", "post")),
            "new label already recorded")
    require(not (RESULTS/PREDECESSOR).exists() and
            all(accounting(PREDECESSOR+"-"+phase, deadline, job="1087656") is None
                for phase in ("pre", "gpu", "post")), "original pending request now has a namespace or Slurm row")
    for phase, expected in (("gpu", "1103606.978"), ("post", "1103606.979")):
        original = accounting("training-vggt-tum-near7-sequence-holdout25-v1-"+phase, deadline)
        require(original and original[0] == expected and original[2:4]==["COMPLETED","0:0"]
                and original[5] == NODE, "prior same-node qualification differs")
    pci_raw = pinned(RESULTS/"training80-busy-34e44d3913754d7ab69c09f75bd91335/pci.json",
                     claim["physical_roster_SHA256"])
    expected_pci = json.loads(pci_raw)[NODE]
    require(len(set(expected_pci)) == 8, "prior physical PCI roster differs")
    weight_identity(WEIGHTS / "model.pt")
    print(json.dumps(dict(stage="intent", label=LABEL, request=intent["request"], epoch=epoch)), flush=True)
    sys.path.insert(0, str(root / "src"))
    from domain_overlay import stage, verify_bundle

    verify_bundle(root / "data", BUNDLE_SHA)
    (root / "arms").mkdir()
    files = json.loads(pinned(FRAMEWORK_MANIFEST, FRAMEWORK_SHA))["files"]
    require(len(files) == 393, "framework file count differs")
    for name, item in files.items():
        raw = pinned(FRAMEWORK / name, item["SHA256"])
        target = root / "framework" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for item in json.loads((root / "src/UPSTREAM_FILES.json").read_bytes())["files"]:
        pinned(UPSTREAM / item["name"], item["SHA256"])
    stage(root / "framework" / BACKEND_REL, root / "src/tum_vggt_pair.py", root / "delta")
    for source in (root / "delta").rglob("*.py"):
        target = root / "framework/generation/src/gencuda" / source.relative_to(root / "delta")
        target.write_bytes(source.read_bytes())
    pinned(root / "framework" / BACKEND_REL, OVERLAY_SHA)
    work = root / "src/run_tum_continuous_once.py"
    before = step(root, "pre", [HOST_PY, "-B", str(work), "snapshot"], deadline)
    stages = {"pre": before}
    if before["passed"]:
        pre = json.loads((root / "pre.log").read_bytes())
        require(pre["node"] == NODE and pre["step"] == before["row"][0] and
                set(pre["pci"]) == set(expected_pci), "fresh device roster differs")
        save(root, "before.json", pre)
        args = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
        for bind in (f"{root}:/pilot:rw", f"{WEIGHTS}:/weights:ro", f"{UPSTREAM}:/vggt-source:ro",
                     f"{RESULTS/'training-t06-sdxl-runtime-v1'}:/training-runtime:ro", "/opt/hyhal:/opt/hyhal:ro"):
            args += ["--bind", bind]
        shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
                 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
                 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
                 'PYTHONPATH=/training-runtime/site:/pilot/framework/generation/src:/pilot/framework; '
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
    result = dict(request=intent["request"], publication=intent["publication"], stages=stages,
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
        raise SystemExit(controller())
