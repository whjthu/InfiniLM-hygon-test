"""Run one scoped Treasury continuous arm under the existing T01 allocation."""

import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

import mid_epoch_recovery
import owned_processes
from qwen_domain_recipe import DATASET, DATASET_SHA256, make_plan
from slurm_accounting import is_terminal, wait_terminal
from snapshot import snapshot


RUN = Path(__file__).resolve().parents[1]
REFACTOR = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = REFACTOR / "results"
BASE = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1"
DATA = RESULTS / "training-qwen-treasury-pilot-data-v1"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
FRAMEWORK = RUN / "framework"
ARM = RUN / "continuous"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
NODE, PARENT, PORT = "a01r4n16", "1087656", 29907
LABEL = "training-qwen-treasury-continuous-hip-v2"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
PACKAGE_SECONDS = 1620
WORKER_SECONDS = 900


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def port_available():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    return True


def asset(path, size, digest):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size == size and sha(path) == digest,
            f"asset identity differs: {path.name}")
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(info, field) == getattr(after, field) for field in fields), "asset changed while reading")


def prepare():
    cfg = read(RUN / "intent.json")
    require(cfg["node"] == NODE and cfg["parent"] == PARENT and cfg["arm"] == "continuous",
            "original resource intent differs")
    manifest = BASE / "source/FRAMEWORK_FILES.json"
    require(sha(manifest) == "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
            "original common framework manifest differs")
    files = read(manifest)["files"]
    require(len(files) == 393, "common framework denominator differs")
    for name, pin in files.items():
        path = BASE / "framework" / name
        asset(path, pin["bytes"], pin["SHA256"])
        require(stat.S_IMODE(path.stat().st_mode) == pin["mode"], "common framework mode differs")
    updates = mid_epoch_recovery.candidate_sources(BASE / "framework")
    FRAMEWORK.mkdir(mode=0o700, exist_ok=False)
    for name in files:
        target = FRAMEWORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in updates:
            target.write_bytes(updates[name])
            target.chmod(files[name]["mode"])
        else:
            shutil.copy2(BASE / "framework" / name, target)
        expected = hashlib.sha256(updates[name]).hexdigest() if name in updates else files[name]["SHA256"]
        require(sha(target) == expected, "candidate framework file changed")
    original = BASE / "source/original-launch.json"
    binding = BASE / "source/QWEN_TARGET_NATIVE_BINDING.json"
    require(sha(binding) == "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067",
            "original trainable binding differs")
    initial = read(binding)["initial_state"]
    asset(Path(initial["path"]), initial["bytes"], initial["SHA256"])
    require(sha(DATA / "data/inputs.jsonl") == DATASET_SHA256, "domain dataset changed")
    stage = read(DATA / "original-stage-receipt.json")
    require(stage["archive_SHA256"] == "6ef7428bb1f3f60732edf2624ec2bee824cde701da73347ac747fd7dc93e6383",
            "original data archive differs")
    image_pins = RUN / "src/image-pins.jsonl"
    require(sha(image_pins) == "275428f4a74843d3e8d55c1775adb7257d4d0f76e909b2d084c26ef08b87d68a",
            "original image inventory differs")
    images = dict(json.loads(line) for line in image_pins.read_bytes().splitlines())
    require(len(images) == 72, "original image denominator differs")
    for identity, digest in images.items():
        require(re.fullmatch(r"[0-9-]{10}_[0-9-]{10}", identity) is not None, "invalid image identity")
        path = DATA / "data/train" / (identity + ".png")
        asset(path, path.stat().st_size, digest)
    require(sha(DATA / "resume-cpu-v1-outcome.json") == "0f435ca9ff8d5bcf1dbbd816366749cd483e6fd7f0bb90a74b575d60dd9b94a1",
            "original CPU skip result differs")
    plan = make_plan(FRAMEWORK, original.read_bytes(), (RUN / "src/native-input-CPU.json").read_bytes(),
                     PurePosixPath(str(RUN)), PurePosixPath(str(RUN / "src/qwen_treasury_witness.py")),
                     PORT, "continuous")
    for name, pin in read(RUN / "src/MODEL_ASSETS.json")["model"].items():
        asset(MODEL / name, pin["bytes"], pin["SHA256"])
    for plugin in plan["required_assets"]["plugins"]:
        require(Path(plugin).is_file(), "planned callback missing")
    save(RUN / "native-plan.json", plan)
    require(shutil.disk_usage(RUN).free >= 16 * 1024**3, "insufficient checkpoint space")
    save(RUN / "witness-config.json", {"original_binding": str(binding)})


def command(args):
    return subprocess.check_output(args, text=True, timeout=14).strip()


def accounting(label):
    rows = [line.split("|") for line in command(["sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS]).splitlines()]
    selected = [row for row in rows if len(row) == 7 and row[1] == label]
    require(len(selected) <= 1, "duplicate Slurm step identity")
    return selected[0] if selected else None


def admit():
    record = command(["squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N"]).split("|")
    require(record[:2] == ["RUNNING", "40"] and NODE in command(["scontrol", "show", "hostnames", record[2]]).splitlines(),
            "claim node is absent from the active parent")
    shown = command(["scontrol", "-o", "show", "step", PARENT])
    for line in shown.splitlines():
        fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
        if re.fullmatch(PARENT + r"\.\d+", fields.get("StepId", "")):
            require(NODE not in command(["scontrol", "show", "hostnames", fields["NodeList"]]).splitlines(),
                    "claimed node already has a compute step")


def step(name, args, gpu, seconds):
    label = LABEL + "-" + name
    require(accounting(label) is None, "existing step requires reconciliation")
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--nodes=1", "--ntasks=1",
           "--nodelist=" + NODE, "--exact", "--immediate=5", "--cpus-per-task=" + ("64" if gpu else "2"),
           "--mem=" + ("224000M" if gpu else "4096M"), "--gres=" + ("dcu:8" if gpu else "none"),
           "--time=" + ("18" if gpu else "2"), *args]
    rc = None
    with (RUN / (name + ".log")).open("xb") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            pass
    status = wait_terminal(lambda: accounting(label), time.monotonic() + 40, interval=2, max_observations=18)
    row = status["row"]
    result = dict(status, rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"]
                  and row[5] == NODE and ("gres/dcu=8" in row[6] if gpu else "gres/dcu=" not in row[6])))
    save(RUN / (name + "-accounting.json"), result)
    return result


def cpu_snapshot():
    data = snapshot(False)
    require(data["kfd_holder_count"] == 0 and data["full_fd_inventory_performed"] is True,
            "host GPU ownership is not empty")
    data["port_available"] = port_available()
    if (ARM / "native-outcome.json").exists():
        held = read(ARM / "native-outcome.json")["owners"]
        current = owned_processes.snapshot()
        data["owned_remaining"] = sum(int(pid) in current and current[int(pid)]["startticks"] == row["startticks"]
                                       and current[int(pid)]["state"] != "Z" for pid, row in held.items())
        require(data["owned_remaining"] == 0, "native owned children remain")
    print(json.dumps(data), flush=True)


def worker():
    ARM.mkdir(mode=0o700, exist_ok=False)
    before = read(RUN / "before.json")
    current = snapshot(True)
    require(current["pci"] == before["pci"] and current["boot_id"] == before["boot_id"]
            and current["kfd_holder_count"] == 0 and 0 <= current["monotonic_ns"] - before["monotonic_ns"] <= 45_000_000_000,
            "compute GPU ownership changed")
    port_available()
    hip = ctypes.CDLL("libamdhip64.so")
    n = ctypes.c_int()
    require(hip.hipGetDeviceCount(ctypes.byref(n)) == 0 and n.value == 8, "wrong HIP device count")
    pci = []
    for index in range(8):
        label = ctypes.create_string_buffer(32)
        require(hip.hipDeviceGetPCIBusId(label, len(label), index) == 0, "HIP PCI query failed")
        ident = label.value.decode().lower()
        pci.append("0000:" + ident if len(ident) == 7 else ident)
    require(sorted(pci) == current["pci"] and len(os.sched_getaffinity(0)) >= 64, "actual lease differs")
    save(ARM / "assigned.json", dict(current, visible_pci=pci, cpus=len(os.sched_getaffinity(0))))
    plan = read(RUN / "native-plan.json")
    config = RUN / "witness-config.json"
    intent = read(RUN / "intent.json")
    epoch = intent["epoch"]
    deadline = min(epoch + PACKAGE_SECONDS - 150, time.time() + WORKER_SECONDS)
    env = dict(os.environ, **plan["environment_overrides"])
    env.update(MASTER_ADDR="127.0.0.1", QWEN_TREASURY_CONFIG=str(config),
               QWEN_TREASURY_CONFIG_SHA256=sha(config), QWEN_TREASURY_EPOCH=str(epoch),
               QWEN_TREASURY_DEADLINE=str(deadline), QWEN_TREASURY_MAX_SECONDS=str(PACKAGE_SECONDS))
    require(not any(k in env for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")), "worker must start unranked")
    save(ARM / "native-intent.json", dict(plan_SHA256=sha(RUN / "native-plan.json"), deadline=deadline))
    owned, rc, failed = {}, None, None
    try:
        with (ARM / "native.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", *plan["command"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            observed = owned_processes.snapshot().get(child.pid)
            if observed:
                owned[child.pid] = {k: observed[k] for k in ("PID", "PPID", "PGID", "startticks", "state")}
            while child.poll() is None:
                owned_processes.refresh(owned, ARM)
                require(time.time() < deadline, "native continuous deadline expired")
                time.sleep(1)
            rc = child.returncode
            quiet_until = time.monotonic() + 5
            while rc == 0 and owned_processes.refresh(owned, ARM) and time.monotonic() < quiet_until:
                time.sleep(.25)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        failed = type(error).__name__ + ": " + str(error)[:220]
    cleaned = owned_processes.cleanup(owned, ARM, grace=5)
    finished = []
    for rank in range(8):
        p = ARM / "train/treasury-witness" / f"rank-{rank:02d}" / "result.json"
        if p.is_file():
            entry = read(p)
            if entry["rank"] == rank and entry["completed_global_step"] == 5:
                finished.append(dict(rank=rank, SHA256=sha(p)))
    passed = bool(rc == 0 and failed is None and len(finished) == 8 and not cleaned["termination_needed"])
    save(ARM / "native-outcome.json", dict(rc=rc, error=failed, cleanup=cleaned, owners=owned,
                                           finished=finished, passed=passed))
    return 0 if passed else 1


def controller():
    expected = read(RUN / "intent.json")
    require(RUN.name == "training-qwen-treasury-sft-v2" and RUN.parent == RESULTS, "wrong run root")
    budget = int(expected["epoch"] + PACKAGE_SECONDS - time.time())
    require(0 < budget <= PACKAGE_SECONDS, "expired original launch epoch")
    def expired(*_):
        raise TimeoutError("original host controller deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    admitted = False
    stages = {}
    failure = None
    try:
        admit()
        prepare()
        admit()
        shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
                 "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH=" +
                 shlex.quote("/vision:/minicpm-transformers:/swift-runtime/site:" + str(FRAMEWORK) + ":" + str(RUN / "src")) +
                 "; exec python3 -B -u " + shlex.quote(str(Path(__file__).resolve())) + " worker")
        runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        for source, target, mode in (
                (RUN, RUN, "rw"), (BASE / "source", BASE / "source", "ro"), (MODEL, MODEL, "ro"),
                (RESULTS / "training-qwen-initial-trainables-v1", RESULTS / "training-qwen-initial-trainables-v1", "ro"),
                (RESULTS / "training-vlm-original-assets-v1", RESULTS / "training-vlm-original-assets-v1", "ro"),
                (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
                (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
                (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
                (DATA / "data", DATASET.parent, "ro"), ("/opt/hyhal", "/opt/hyhal", "ro")):
            runtime.extend(["--bind", f"{source}:{target}:{mode}"])
        runtime.extend([str(REFACTOR / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell])
        snapshot_cmd = [HOST_PYTHON, "-B", str(Path(__file__).resolve()), "snapshot"]
        stages["pre"] = step("pre", snapshot_cmd, False, 60)
        require(stages["pre"]["passed"], "original pre-snapshot failed")
        before = json.loads((RUN / "pre.log").read_bytes())
        save(RUN / "before.json", before)
        admitted = True
        stages["gpu"] = step("gpu", runtime, True, 1040)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        failure = type(error).__name__ + ": " + str(error)[:280]
    if admitted and is_terminal(accounting(LABEL + "-gpu")):
        stages["post"] = step("post", snapshot_cmd, False, 75)
        if stages["post"]["passed"]:
            after = json.loads((RUN / "post.log").read_bytes())
            require(after["pci"] == before["pci"] and after["boot_id"] == before["boot_id"],
                    "post snapshot identities differ")
            save(RUN / "after.json", after)
    passed = bool(failure is None and all(stages.get(k, {}).get("passed") for k in ("pre", "gpu", "post"))
                  and (ARM / "native-outcome.json").exists() and read(ARM / "native-outcome.json")["passed"])
    save(RUN / "controller-outcome.json", dict(stages=stages, failure=failure, passed=passed, finished=time.time()))
    return 0 if passed else 1


if __name__ == "__main__":
    entry = {"host": controller, "worker": worker, "snapshot": cpu_snapshot}[sys.argv[1]]
    raise SystemExit(entry())
