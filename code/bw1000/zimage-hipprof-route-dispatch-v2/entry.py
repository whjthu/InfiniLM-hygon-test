"""Schedule one admitted GEMM pair with independent pre and post inventories."""

from base64 import b64decode
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
LABEL = "O01-zimage-square-mm-hipprof-route-v2"
REQUEST = "9429f548ca3346c683f7361037079f75"
RUN = RESULTS / (LABEL + "-run")
PACKAGE = RESULTS / (LABEL + "-package")
PARENT, NODE, CONTROL = "1103606", "f12r2n11", "f12r2n13"
STANDBY, STANDBY_NAME = "1103606.1873", "training80-dcu-standby-0e8fc17697b0-f12r2n11"
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
ENDING = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
          "PREEMPTED", "DEADLINE", "BOOT_FAIL"}
PACKAGE_RESULT = "77a3d8ddf8ccce33b1f37e011377fc7b62b1854337d2e7510191838ac2dad161"
PACKAGE_MANIFEST = "284ea93f46d806f036bdeea2580015979112dcdcc94848eefc6db65ad0dc8a89"
SHELL = ("set -euo pipefail; source /opt/dtk/env.sh; "
         "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
         "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site; "
         "unset ROCBLAS_LAYER HIPBLASLT_LOG_LEVEL HIPBLASLT_LOG_MASK; "
         "exec python3 -B -u /pair-run/device_entry.py")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def save(name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    write(RUN / name, raw)
    return digest(raw)


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=7)


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def parent_owned():
    [raw] = query("scontrol", "show", "job", PARENT, "--oneliner").splitlines()
    job = fields(raw)
    members = set(query("scontrol", "show", "hostnames", job["NodeList"]).splitlines())
    require(job["JobState"] == "RUNNING" and job["UserId"].endswith(f"({os.getuid()})")
            and len(members) == 40 and {NODE, CONTROL} <= members
            and "gres/dcu=320" in job["AllocTRES"].split(","), "Original parent ownership differs")


def target_live():
    active = []
    for line in query("scontrol", "show", "step", PARENT, "--oneliner").splitlines():
        row = fields(line)
        if not re.fullmatch(PARENT + r"\.[0-9]+", row.get("StepId", "")) or row.get("State") in ENDING:
            continue
        expression = row.get("NodeList")
        require(expression not in (None, "", "(null)"), "Unresolved numbered placement")
        nodes = ([expression] if re.fullmatch(r"f\d+r\d+n\d+", expression) else
                 query("scontrol", "show", "hostnames", expression).splitlines())
        if NODE in nodes:
            active.append(row)
    return active


def pinned_standby():
    [row] = target_live()
    require((row["StepId"], row["Name"], row["State"], row["NodeList"]) ==
            (STANDBY, STANDBY_NAME, "RUNNING", NODE)
            and "gres/dcu=8" in row["TRES"].split(","), "Original project standby moved")


def accounting(name):
    out = query("sacct", "-nP", "-j", PARENT, "--format", ",".join(COLUMNS))
    rows = [line.split("|") for line in out.splitlines() if "|" + name + "|" in line]
    require(len(rows) <= 1, "Original accounting label duplicated")
    return rows[0] if rows else None


def wait_terminal(name, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        r = accounting(name)
        if r and r[2].rstrip("+") in ENDING:
            return r
        time.sleep(0.45)
    raise TimeoutError("Original Slurm step terminal state unobserved")


def verify_package():
    stage = (PACKAGE / "stage-result.json").read_bytes()
    manifest = (PACKAGE / "SOURCE_FILES.json").read_bytes()
    require(digest(stage) == PACKAGE_RESULT and digest(manifest) == PACKAGE_MANIFEST,
            "Deployed operator source receipt differs")
    for name, pin in json.loads(manifest)["files"].items():
        raw = (PACKAGE / name).read_bytes()
        require(len(raw) == pin["bytes"] and digest(raw) == pin["SHA256"], "Deployed operator file drift")


def srun(name, node, cpus, memory, gpu, seconds):
    return ["srun", "--jobid=" + PARENT, "--job-name=" + name, "--exact", "--immediate=6",
            "--nodes=1", "--ntasks=1", "--nodelist=" + node, "--threads=1", "--cpus-per-task=" + str(cpus),
            "--mem=" + memory, "--gres=" + ("dcu:1" if gpu else "none"),
            "--time=" + time.strftime("%H:%M:%S", time.gmtime(seconds)), "--kill-on-bad-exit=1"]


def start(data):
    release = data["publication"]
    require(re.fullmatch(r"[0-9a-f]{40}", release), "Missing published launch scope")
    files = {name: b64decode(raw, validate=True) for name, raw in data["files"].items()}
    require(set(files) == {"entry.py", "device_entry.py", "node_inventory.py", "DISPATCH.json"},
            "Wrong execution envelope members")
    claim = json.loads(files["DISPATCH.json"])
    require(claim["request"] == REQUEST and claim["parent"] == PARENT and claim["target_node"] == NODE
            and claim["control_node"] == CONTROL and claim["handoff_step"] == STANDBY,
            "Wrong published execution scope")
    for name, pin in claim["files"].items():
        require(len(files[name]) == pin["bytes"] and digest(files[name]) == pin["SHA256"],
                "Wrong self-contained execution source")
    require(claim["handoff_job"] == STANDBY_NAME
            and claim["handoff_observed_epoch"] <= time.time() < claim["handoff_observed_epoch"] + 300,
            "Fresh standby identity observation has expired")
    parent_owned()
    pinned_standby()
    verify_package()
    require(all(accounting(LABEL + "-" + phase) is None for phase in ("control", "pre", "gpu", "post")),
            "Existing attempt must be reconciled")
    os.umask(0o077)
    RUN.mkdir()
    for name, raw in files.items():
        write(RUN / name, raw)
    epoch = time.time()
    save("intent.json", dict(request=REQUEST, label=LABEL, publication=release, parent=PARENT, node=NODE,
         control_node=CONTROL, epoch=epoch, deadline_epoch=epoch + 240, standby=STANDBY, replay_allowed=False))
    cmd = [*srun(LABEL + "-control", CONTROL, 2, "4096M", False, 260),
           "--output=" + str(RUN / "control.stdout"), "--error=" + str(RUN / "control.stderr"),
           PYTHON, "-I", "-S", "-B", str(RUN / "entry.py"), "control"]
    os.execvp(cmd[0], cmd)


def control():
    require(socket.gethostname().split(".")[0] == CONTROL and os.environ["SLURM_JOB_ID"] == PARENT
            and os.environ["SLURM_STEP_ID"].isdecimal(), "Wrong control step")
    owner = PARENT + "." + os.environ["SLURM_STEP_ID"]
    save("control-owner.json", {"step": owner, "node": CONTROL})
    for key in tuple(os.environ):
        if key.startswith(("SLURM_", "PMI_", "PMIX_")):
            del os.environ[key]
    intent = json.loads((RUN / "intent.json").read_bytes())
    deadline = intent["deadline_epoch"]
    require(intent["request"] == REQUEST and deadline == intent["epoch"] + 240
            and intent["epoch"] <= time.time() < intent["epoch"] + 45, "Wrong original epoch")
    result = dict(request=REQUEST, control_step=owner, gpu_attempted=False, standby_cleared=False,
                  post_clean=False, passed=False, phases={})

    def step(phase, command, gpu):
        name = LABEL + "-" + phase
        require(accounting(name) is None, "Planned numbered step already exists")
        seconds = min(170 if gpu else 25, int(deadline - time.time()) - (38 if gpu else 5))
        require(seconds >= (40 if gpu else 10), "Original epoch has no admissible remaining budget")
        args = [*srun(name, NODE, 4 if gpu else 2, "8192M" if gpu else "4096M", gpu, seconds), *command]
        log_path = RUN / (phase + ".log")
        with log_path.open("xb") as log:
            if gpu:
                result["gpu_attempted"] = True
                save("gpu-attempt.json", dict(request=REQUEST, label=name, attempted_epoch=time.time()))
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = child.wait(timeout=seconds + 1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait(timeout=4)
        row = wait_terminal(name, 20)
        result["phases"][phase] = {"accounting": row, "rc": rc}
        require(row[0].startswith(PARENT + ".") and row[5] == NODE
                and (("gres/dcu=1" in row[6].split(",")) if gpu else "gres/" not in row[6]),
                "Actual step resources differ")
        return rc == 0 and row[2:4] == ["COMPLETED", "0:0"]

    inventory_cmd = [PYTHON, "-I", "-S", "-B", str(RUN / "node_inventory.py"), PARENT, NODE]

    def inventory(phase):
        require(step(phase, inventory_cmd, False), "Independent inventory failed")
        [data] = [json.loads(line) for line in (RUN / (phase + ".log")).read_bytes().splitlines()
                  if line.startswith(b"{")]
        require(data["step"] == result["phases"][phase]["accounting"][0] and len(set(data["PCIs"])) == 8
                and data["kfd_process_count"] == data["own_device_fd_process_count"] == 0,
                "Independent device inventory not clear")
        save(phase + ".json", data)
        return data

    lock = (RESULTS / ".training80-dcu-standby.lock").open("ab")
    pre = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_owned()
        verify_package()
        pinned_standby()
        save("handoff-intent.json", {"request": REQUEST, "step": STANDBY, "node": NODE})
        query("scancel", "--signal=TERM", STANDBY)
        standby = wait_terminal(STANDBY_NAME, 28)
        require(standby[0] == STANDBY and standby[5] == NODE and not target_live(),
                "Original standby did not release target")
        result.update(standby_cleared=True, standby_accounting=standby)
        pre = inventory("pre")
        require(not target_live(), "Target acquired an intervening step")
        app = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/",
               "--bind", f"{RUN}:/pair-run:rw", "--bind", f"{PACKAGE}:/pair-package:ro",
               "--bind", str(RESULTS / "training-t06-sdxl-runtime-v1") + ":/training-runtime:ro",
               "--bind", "/usr/local/hyhal:/opt/hyhal:ro", str(PROJECT / "containers/dtk26-sandbox-r1"),
               "/bin/bash", "--noprofile", "--norc", "-c", SHELL]
        result["GPU_step_passed"] = step("gpu", app, True)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:220])
    finally:
        if result["standby_cleared"] and (not result["gpu_attempted"] or "gpu" in result["phases"]):
            try:
                require(not target_live(), "Active numbered step blocks independent post")
                after = inventory("post")
                result["post_clean"] = pre is not None and set(pre["PCIs"]) == set(after["PCIs"]) and (
                    pre["boot_id_SHA256"] == after["boot_id_SHA256"])
            except Exception as error:
                result["post_error"] = str(error)[:220]
        result["passed"] = result.get("GPU_step_passed", False) and result["post_clean"]
        save("outcome.json", result)
        lock.close()
    return int(not result["passed"])


if __name__ == "__main__":
    if sys.argv[1:] == ["control"]:
        raise SystemExit(control())
    start(BOUND_PAYLOAD)
