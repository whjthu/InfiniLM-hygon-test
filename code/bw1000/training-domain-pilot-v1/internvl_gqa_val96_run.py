"""Admit one eight-DCU paired GQA validation step and reconcile its original evidence."""

from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import internvl_gqa_val96 as data
from internvl_gqa_val96 import need
import internvl_gqa_val96_native as native


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
R = ROOT / "results"
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REQUEST = "9b386eceea2e4e75b16886fb49a29d43"
LABEL = "training-internvl-gqa-val96-paired-hip-v2"
PARENT, NODE = "1103606", "f13r1n14"
PREDECESSOR = PARENT + ".4390"
STANDBY = "training80-dcu-standby-de51eae65bd247278b9e2c7e624f4773"
STANDBY_LABEL = "training80-dcu-standby-de51eae65bd2-f13r1n14"
INPUTS = R / "training-internvl-gqa-val96-user-only-v1/input"
INPUT_PROOF = R / "training-internvl-gqa-val96-cpu-native-v2/work/native.json"
CHECKPOINT = R / "training-internvl-gqa64-continuous25-hip-v1/train/checkpoint-25"
ORIGINAL = R / "training-internvl-gqa64-continuous25-hip-v1-source-e23c26259c4344bc837a693ae9c2f11a"
OLD_SOURCE_SHA = "52444c213cac9529bb150e8b5d10aa107c8cc9c29a162a6b490060223ad92ed7"
EXPORT_SHA = "0006bf1e58e7e84e4a13a6266762f51f603fd0814bd7a880f71402d14882a603"
FRAMEWORK = R / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
RUNTIME = R / "training-vlm-swift-offline-runtime-v1"
UPLOAD_PIN = "bd65e3b2cfe4779168ffb937e5aaf8559594c9f8d8307486cc031870f9f3ac46"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
WHOLE = 2160
GPU_LIMIT = 1800


def save(name, value):
    raw = data.encode(value)
    data.write(WORK / name, raw)
    return data.digest(raw)


def read(path, cap=65536, pin=None):
    return json.loads(data.read(path, cap, pin))


def query(argv):
    result = subprocess.check_output(argv, text=True, timeout=10)
    return result.splitlines()


def fields(line):
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


def account(handle, label=None):
    rows = [line.split("|") for line in query(["sacct", "-nP", "-j", handle, "-o", COLUMNS])]
    if label is None:
        return [row for row in rows if len(row) == 9 and row[0] == handle]
    return [row for row in rows if len(row) == 9 and row[1] == label and re.fullmatch(PARENT + r"\.\d+", row[0])]


def terminal(rows):
    return len(rows) == 1 and rows[0][2].split()[0] in TERMINAL and rows[0][8] not in {"", "Unknown"}


def placement():
    [job] = map(fields, query(["scontrol", "show", "job", "--oneliner", PARENT]))
    need(job["UserId"].endswith(f"({os.getuid()})") and job["JobState"] == "RUNNING"
         and "gres/dcu=320" in job["AllocTRES"].split(","), "Parent resources or owner differ")
    cache = {}

    def covers(spec):
        if spec in {"", "None", "N/A", "Unknown", "(null)"}:
            return False
        if spec not in cache:
            cache[spec] = [spec] if re.fullmatch(r"f\d+r\d+n\d+", spec) else query(["scontrol", "show", "hostnames", spec])
        return NODE in cache[spec]

    need(covers(job["NodeList"]), "Node no longer belongs to the held parent")
    active = {}
    for line in query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        row = fields(line)
        identity = row["StepId"]
        if re.fullmatch(PARENT + r"\.\d+", identity) and covers(row["NodeList"]):
            active[identity] = (row["Name"], row["State"])
    pending = set()
    for line in query(["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"]):
        identity, nodes = line.split("|")
        if re.fullmatch(PARENT + r"\.\d+", identity) and covers(nodes):
            pending.add(identity)
    return active, pending


def confirm_submission(epoch):
    need(time.time() < epoch + WHOLE, "Original controller deadline expired")
    manifest = read(HERE / "SOURCE.json")
    need(manifest["request"] == REQUEST and set(manifest["files"]) == {
        "internvl_gqa_val96.py", "internvl_gqa_val96_native.py", "internvl_gqa_val96_run.py"},
        "Paired evaluation source population differs")
    for name, stamp in manifest["files"].items():
        need(data.fields(data.read(HERE / name, 65536)) == stamp, "Published source changed")


def original():
    manifest = read(ORIGINAL / "SOURCE_FILES.json", 16384, OLD_SOURCE_SHA)
    need(len(manifest["files"]) == 16, "Original training package population differs")
    for name, stamp in manifest["files"].items():
        need(Path(name).name == name and data.fields(data.read(ORIGINAL / name, 1 << 20)) == stamp,
             "Original qualified source bytes differ")
    sys.path.insert(0, str(ORIGINAL))
    import node
    import snapshot
    need(snapshot.NODE == NODE, "Original device inventory is for another node")
    return node, snapshot


def node_phase(phase, epoch):
    confirm_submission(epoch)
    need(socket.gethostname().split(".")[0] == NODE and os.environ["SLURM_JOB_ID"] == PARENT,
         "Compute worker placement differs")
    node, inventory = original()
    if phase == "prepare":
        a, plan, views = node.actual_model_plan(EXPORT_SHA)
        a.verify_originals(plan)
        need(len(plan["binds"]) == 18 and len(views) == 20, "Underlying base model mounts differ")
        data.read(INPUT_PROOF, 1 << 17, native.PROCESSED_INPUT_SHA)
        observed = {"step": PARENT + "." + os.environ["SLURM_STEP_ID"], "node": NODE,
                    "pci": inventory.drm_inventory(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "model_views": len(views), "mount_members": len(plan["binds"]),
                    "cpu_sha": native.PROCESSED_INPUT_SHA, "full_fd_inventory_performed": False}
        save("prepare-devices.json", observed)
        return 0
    if phase in {"pre", "post"}:
        observed = inventory.snapshot(False)
        need(observed["full_fd_inventory_performed"] is True, "Original complete ownership inventory required")
        need(observed["kfd_holder_count"] == 0, "Unexpected GPU device holders in the independent HOST check")
        save(phase + "-devices.json", observed)
        return 0

    step = PARENT + "." + os.environ["SLURM_STEP_ID"]
    available = inventory.snapshot(True)
    before = read(WORK / "pre-devices.json")
    need(available["pci"] == before["pci"] and available["boot_id"] == before["boot_id"]
         and available["kfd_holder_count"] == 0, "GPU step does not own the previously clear physical scope")
    a, plan, views = node.actual_model_plan(EXPORT_SHA)
    a.verify_originals(plan)
    need(len(plan["binds"]) == 18 and len(views) == 20, "Base model export scope differs")
    binds = []
    for binding in plan["binds"]:
        source, target, mode = binding.rsplit(":", 2)
        need(target == "/internvl-model" or target.startswith("/internvl-model/"), "Unexpected original model mount")
        need(mode == "ro", "Original base must remain read-only")
        binds.append(f"{source}:{native.MODEL}{target[len('/internvl-model'):]}:ro")
    save("gpu-devices.json", {"step": step, "PCI": available["pci"], "boot_id": available["boot_id"],
                              "request": REQUEST, "base_mounts": len(binds), "localized_views": len(views)})
    fs = subprocess.check_output(["findmnt", "--noheadings", "-o", "FSTYPE", "--target", "/tmp"],
                                 text=True, timeout=5).strip()
    need(fs == "ext4", "Expected previously qualified node-local filesystem")
    with tempfile.TemporaryDirectory(prefix="internvl-gqa-val96-", dir="/tmp") as scratch:
        specs = [*binds, f"{HERE}:/gqa-source:ro", f"{INPUTS}:/gqa-input:ro",
                 f"{INPUT_PROOF}:/gqa-native-inputs.json:ro", f"{CHECKPOINT}:/gqa-domain25:ro",
                 f"{WORK}:/gqa-run:rw", f"{FRAMEWORK}:/gqa-framework:ro",
                 f"{RUNTIME}:/swift-runtime:ro", f"{scratch}:/tmp:rw", "/usr/local/hyhal:/opt/hyhal:ro"]
        argv = [APPTAINER, "exec", "--rocm", "--contain", "--no-home", "--pwd", "/"]
        for spec in specs:
            argv.extend(("--bind", spec))
        shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; '
                 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; '
                 'export HF_HOME=/tmp/hf XDG_CACHE_HOME=/tmp/xdg TOKENIZERS_PARALLELISM=false PYTHONPATH=/gqa-framework:/swift-runtime/site; '
                 'exec /usr/bin/python3 -B -u "$@"')
        args = [*argv, str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
                "-c", shell, "gqa96-gpu", "/gqa-source/internvl_gqa_val96_run.py",
                "ranks", str(epoch), step]
        with (WORK / "container.log").open("xb") as log:
            child = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=max(1, epoch + GPU_LIMIT - time.time() - 8))
        a.verify_originals(plan)
        return child.returncode


def rank_launcher(epoch, step):
    work = Path("/gqa-run")
    children = []
    started = time.monotonic()
    codes = []
    try:
        for rank in range(8):
            script = Path("/gqa-source/internvl_gqa_val96_native.py")
            argv = [sys.executable, "-B", str(script), "--request", REQUEST, "--step", step,
                    "--input", "/gqa-input", "--input-proof", "/gqa-native-inputs.json",
                    "--output", f"/gqa-run/rank-{rank}", "--epoch", str(epoch), "--deadline", str(epoch + GPU_LIMIT - 20),
                    "--rank", str(rank)]
            with (work / f"rank-{rank}.log").open("xb") as log:
                children.append(subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
        for child in children:
            codes.append(child.wait(timeout=max(1, epoch + GPU_LIMIT - time.time() - 12)))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    data.write(work / "ranks-exit.json", data.encode(
        {"returncodes": codes, "runtime_seconds": time.monotonic() - started, "request": REQUEST, "step": step}))
    return int(codes != [0] * 8)


def wait_account(label, duration):
    until = time.monotonic() + duration
    while True:
        found = account(PARENT, label)
        need(len(found) <= 1, "Duplicate named numbered stage")
        if terminal(found) or time.monotonic() >= until:
            return found
        time.sleep(1)


def launch_phase(name, epoch, *, detached=False):
    gpu = name == "gpu"
    label = LABEL + "-" + name
    need(not account(PARENT, label), "Numbered stage already exists")
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--immediate=6", "-N1", "-n1",
            "--nodelist=" + NODE, "--cpus-per-task=" + ("48" if gpu else "2"),
            "--mem=" + ("224000M" if gpu else "4G"), "--gres=" + ("dcu:8" if gpu else "none"),
            "--cpu-bind=" + ("none" if gpu else "cores"), "--time=" + ("00:29:00" if gpu else "00:01:00"),
            HOST_PYTHON, "-B", str(HERE / "internvl_gqa_val96_run.py"), name, str(epoch)]
    options = {k: v for k, v in os.environ.items() if k not in {"SLURM_OVERLAP", "SRUN_OVERLAP"}}
    with (WORK / (name + "-srun.log")).open("xb") as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=options)
        if detached:
            save("gpu-srun.json", {"pid": child.pid, "epoch": epoch, "request": REQUEST, "label": label})
            return child
        rc = child.wait(timeout=60)
    rows = wait_account(label, 15)
    save(name + "-accounting.json", {"rows": rows, "srun_rc": rc})
    need(rc == 0 and terminal(rows) and rows[0][2:4] == ["COMPLETED", "0:0"] and rows[0][5] == NODE
         and "gres/" not in rows[0][6], "CPU verifier did not finish on the qualified node")
    proof = read(WORK / (name + "-devices.json"))
    need(proof["step"] == rows[0][0], "Verifier accounting and physical receipt differ")
    return proof


def release_standby(epoch, pretest):
    live, pending = placement()
    [row] = account(PREDECESSOR)
    need(row[1] == STANDBY_LABEL and row[5:7] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"]
         and live == {PREDECESSOR: (STANDBY_LABEL, "RUNNING")} and pending <= {PREDECESSOR}
         and row[2] == "RUNNING", "Only the exact active project standby can be replaced")
    directory = R / STANDBY / NODE
    expected, worker = (read(directory / name) for name in ("expected.json", "worker.json"))
    need(expected["label"] == STANDBY_LABEL and expected["node"] == worker["node"] == NODE
         and expected["parent"] == PARENT and worker["step"] == PREDECESSOR
         and worker["synthetic"] is True and sorted(worker["pci"]) == sorted(expected["pci"]) == pretest["pci"],
         "Standby process and physical ownership differ")
    need(epoch <= time.time() < epoch + 120, "Standby handoff budget expired")
    save("signal-intent.json", {"request": REQUEST, "target": PREDECESSOR, "prior": row})
    signal = subprocess.run(["scancel", "--signal=TERM", PREDECESSOR], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=10)
    save("signal-result.json", {"rc": signal.returncode, "error_SHA256": data.digest(signal.stderr)})
    need(signal.returncode == 0, "Exact standby signal was not acknowledged")
    until = time.monotonic() + 40
    while time.monotonic() < until:
        rows = account(PREDECESSOR)
        if terminal(rows) and placement() == ({}, set()):
            save("standby-terminal.json", {"row": rows[0], "request": REQUEST})
            need(rows[0][2:4] == ["COMPLETED", "0:0"], "Standby termination was not clean in Slurm")
            return
        time.sleep(1)
    raise RuntimeError("Original standby terminal or clear placement has not been established")


def merge_rank_predictions(folder, output, step, pcis):
    arms = {arm: {} for arm in data.ARMS}
    paired_devices = set()
    for rank in range(8):
        source = folder / f"rank-{rank}"
        result = read(source / "outcome.json")
        witness = read(source / "device.json")
        need((result["step"], result["request"], result["rank"], result["PCI"]) ==
             (step, REQUEST, rank, witness["PCI"]) and result["fresh_process"] is True
             and result["model_loads"] == 1 and result["planned"] == 24 and result["gold_read"] is False
             and result["CPU_processed_input_proof_SHA256"] == native.PROCESSED_INPUT_SHA
             and result["live_checkpoint_tensors_verified"] == 504 and result["failure"] is None
             and witness["PCI"] in pcis and witness["PCI"] not in paired_devices
             and (witness["step"], witness["request"], witness["rank"]) == (step, REQUEST, rank),
             "Physical device, original checkpoint or rank coverage differs")
        paired_devices.add(witness["PCI"])
        seal = read(source / "SEALED.json", 65536, result["seal_SHA256"])
        need(seal["input_SHA256"] == native.INPUT_SHA and set(seal["arms"]) == set(data.ARMS), "Rank seal scope differs")
        for arm in data.ARMS:
            raw = data.read(source / (arm + ".json"), 1 << 20)
            need(data.fields(raw) == seal["arms"][arm], "Rank predictions differ from its own seal")
            for row in data.prediction_rows(raw, arm, native.INPUT_SHA, native.rank_slots(rank, 8)):
                need(row["slot"] not in arms[arm] and row["status"] in {"cap", "eos"}, "Duplicate or incomplete rank row")
                arms[arm][row["slot"]] = row
    expected_slots = range(1, data.COUNT + 1)
    need(paired_devices == set(pcis) and all(list(sorted(group)) == list(expected_slots) for group in arms.values()),
         "Eight independent physical cards and all paired slots required")
    need(all(arms["base"][slot]["processed_input_SHA256"] == arms["domain25"][slot]["processed_input_SHA256"]
             for slot in expected_slots), "Native comparison input identity differs")
    output.mkdir(mode=0o700)
    pins = {}
    for arm, rows in arms.items():
        raw = data.encode({"schema": "INTERNVL_GQA_VAL96_PREDICTION_V1", "arm": arm,
                           "input_SHA256": native.INPUT_SHA, "rows": [rows[slot] for slot in expected_slots]})
        data.prediction_rows(raw, arm, native.INPUT_SHA)
        data.write(output / (arm + ".json"), raw)
        pins[arm] = data.fields(raw)
    seal = data.encode({"schema": "INTERNVL_GQA_VAL96_SEALED_V1", "input_SHA256": native.INPUT_SHA, "arms": pins})
    data.write(output / "SEALED.json", seal)
    return {"seal_SHA256": data.digest(seal), "devices": len(paired_devices), "model_loads": 8,
            "planned": 192, "counts": {arm: dict(Counter(row["status"] for row in rows.values()))
                                       for arm, rows in arms.items()}}


def controller(epoch):
    import fcntl

    confirm_submission(epoch)
    need(data.digest(data.read(INPUTS.parent / "verified-upload.json", 65536)) == UPLOAD_PIN,
         "Actual 98-member no-answer upload required")
    predecessor = R / "training-internvl-gqa-val96-paired-hip-v1"
    failed = account("1103606.4393")
    need(terminal(failed) and failed[0][2:4] == ["FAILED", "1:0"]
         and not any((predecessor / name).exists() for name in
                     ("signal-intent.json", "gpu-srun.json", "gpu-visible.json")), "Original failed scope is unresolved")
    since = time.monotonic()
    with (R / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live, pending = placement()
        need(live == {PREDECESSOR: (STANDBY_LABEL, "RUNNING")} and pending <= {PREDECESSOR},
             "No eligible standby-to-paired-evaluation handoff")
        prepare = launch_phase("prepare", epoch)
        release_standby(epoch, prepare)
        before = launch_phase("pre", epoch)
        need(prepare["pci"] == before["pci"] and prepare["boot_id"] == before["boot_id"]
             and placement() == ({}, set()), "Physical placement changed after independent device check")
        child = launch_phase("gpu", epoch, detached=True)
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline:
            live, pending = placement()
            names = [step for step, (label, _) in live.items() if label == LABEL + "-gpu"]
            if names:
                save("gpu-visible.json", {"request": REQUEST, "step": names[0], "observed_epoch": time.time()})
                break
            if child.poll() is not None:
                break
            time.sleep(1)
        else:
            raise RuntimeError("No authoritative GPU launch visibility under the original reservation")
    rc = child.wait(timeout=max(1, epoch + GPU_LIMIT - time.time() + 45))
    rows = wait_account(LABEL + "-gpu", 25)
    save("gpu-accounting.json", {"rows": rows, "srun_rc": rc})
    need(terminal(rows) and rows[0][5] == NODE and "gres/dcu=8" in rows[0][6], "GPU accounting is unresolved")
    after = launch_phase("post", epoch)
    need(after["boot_id"] == before["boot_id"] and after["pci"] == before["pci"], "Original device set changed")
    summary = merge_rank_predictions(WORK, WORK / "predictions", rows[0][0], before["pci"]) if (
        rc == 0 and rows[0][2:4] == ["COMPLETED", "0:0"]) else None
    outcome = {"request": REQUEST, "actual_step": rows[0][0], "accounting": rows[0], "post_step": after["step"],
               "fresh_clean_post": True, "sealed_paired_predictions": summary, "gold_read": False,
               "wall_seconds": time.monotonic() - since, "domain_quality_verified": False}
    save("outcome.json", outcome)
    return int(summary is None)


def main():
    phase, raw_epoch, *args = sys.argv[1:]
    epoch = float(raw_epoch)
    if phase == "ranks":
        return rank_launcher(epoch, args[0])
    if phase != "controller":
        return node_phase(phase, epoch)
    try:
        return controller(epoch)
    except Exception as error:
        save("controller-failure.json", {"request": REQUEST, "class": type(error).__name__,
             "reason": str(error) if isinstance(error, (ValueError, RuntimeError)) else None,
             "frames": [{"file": Path(f.filename).name, "line": f.lineno, "method": f.name}
                        for f in traceback.extract_tb(error.__traceback__)], "observed_epoch": time.time()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
