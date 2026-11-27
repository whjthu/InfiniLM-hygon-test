"""Bind a published CP20 continuation to its private placement and exact inputs."""

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback

import internvl_gqa_val96_run as scheduling
from scope import HERE, LABEL, NODE, PARENT, REQUEST, RESULTS, RUN, need, pin


DISPATCH = HERE.parent
PREPARE = LABEL + "-prepare"
CP20 = RESULTS / "training-internvl-gqa64-continuous25-hip-v1/train/checkpoint-20"


def private():
    claim = json.loads(pin(HERE / "CLAIM.json"))
    deployment = json.loads(pin(DISPATCH / "handoff.json", expected=claim["private_deployment_SHA256"]))
    need(claim["request"] == REQUEST and all(re.fullmatch(PARENT + r"\.\d+", deployment[key])
         for key in ("audit_step", "standby_step")), "Original private placement differs")
    return claim, deployment


def prepare():
    import node
    import snapshot

    claim, deployment = private()
    need(os.environ["SLURM_JOB_ID"] == PARENT and socket.gethostname().split(".")[0] == NODE,
         "Original compute precheck placement differs")
    assembler, plan, views = node.actual_model_plan(claim["exports_SHA256"])
    assembler.verify_originals(plan)
    need(len(views) == 20, "Original model mount inventory differs")
    for name, record in deployment["checkpoint_files"].items():
        path = CP20 / name
        need(path.parent == CP20 and path.resolve(strict=True) == path and path.is_file() and
             path.stat().st_size == record["bytes"], "Checkpoint member shape differs from isolated CPU preflight")
        sha = sha256()
        with path.open("rb") as stream:
            old = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                sha.update(chunk)
            end = os.fstat(stream.fileno())
        need((old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) ==
             (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns) and sha.hexdigest() == record["SHA256"],
             "Original CP20 file changed")
    need(len(deployment["checkpoint_files"]) == 13, "Complete CP20 required")
    scheduling.save("prepare-receipt.json", {
        "request": REQUEST, "step": PARENT + "." + os.environ["SLURM_STEP_ID"], "pci": snapshot.drm_inventory(),
        "cp20_files": 13, "model_views": 20})


def dispatch():
    claim, deployment = private()
    epoch = time.time()
    intent = json.loads(pin(DISPATCH / "dispatch-intent.json"))
    need(not RUN.exists() and not (RESULTS / (LABEL + "-controller")).exists(), "Recovery namespace is occupied")
    need(re.fullmatch(r"[0-9a-f]{40}", intent["publication"]) and intent["request"] == REQUEST,
         "Unsigned recovery source")
    previous = RESULTS / "training-internvl-gqa64-cp20-resume-hip-v1-controller/outcome.json"
    failed = json.loads(pin(previous, expected=deployment["prior_attempt_outcome_SHA256"]))
    old_rows = scheduling.account(deployment["prior_attempt_gpu_step"])
    need(scheduling.terminal(old_rows) and old_rows[0][2:4] == ["FAILED", "1:0"]
         and failed["phases"]["gpu"]["row"][0] == old_rows[0][0] and failed["resource_clean"] is True,
         "Prior GPU attempt is unresolved")
    audit = json.loads(pin(RESULTS / "training-internvl-gqa-cp20-resume-cpu-v2/result.json",
                           expected=deployment["cpu_audit_SHA256"]))
    need(audit["rc"] == 0 and audit["summary"]["step"] == deployment["audit_step"]
         and audit["summary"]["global_step"] == 20 and audit["summary"]["optimizer_update_counts"] == {"20": 504}
         and len(audit["summary"]["rng"]) == 8 and audit["gpu_restore_executed"] is False,
         "Actual CPU resume inspection is missing")
    rows = scheduling.account(deployment["audit_step"])
    need(scheduling.terminal(rows) and rows[0][2:4] == ["COMPLETED", "0:0"]
         and "gres/" not in rows[0][6], "CPU inspection accounting is not terminal")
    standby = deployment["standby_step"]
    label = f'training80-dcu-standby-{deployment["standby_request"][:12]}-{NODE}'
    with (RESULTS / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, pending = scheduling.placement()
        need(active == {standby: (label, "RUNNING")} and pending <= {standby}
             and not scheduling.account(PARENT, PREPARE), "Current recovery placement differs or has an earlier attempt")
        args = ["srun", "--jobid=" + PARENT, "--job-name=" + PREPARE, "--exact", "--immediate=6", "-N1", "-n1",
                "--nodelist=" + NODE, "--cpus-per-task=2", "--mem=4G", "--gres=none", "--time=00:01:00",
                sys.executable, "-B", str(Path(__file__).resolve()), "prepare"]
        with (DISPATCH / "prepare.log").open("xb") as stream:
            child = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=62)
        rows = scheduling.wait_account(PREPARE, 12)
        actual = scheduling.read(DISPATCH / "prepare-receipt.json")
        need(child.returncode == 0 and scheduling.terminal(rows) and rows[0][2:4] == ["COMPLETED", "0:0"]
             and "gres/" not in rows[0][6] and actual["step"] == rows[0][0] and actual["request"] == REQUEST,
             "Compute precheck did not finish")
        scheduling.save("prepare-accounting.json", {"rows": rows, "checked_cp20_members": actual["cp20_files"]})
        scheduling.release_standby(epoch, actual, deployment)
        args = [sys.executable, "-B", str(HERE / "control_once.py"), "--publication", intent["publication"],
                "--exports-sha256", claim["exports_SHA256"], "--claim-sha256", sha256(pin(HERE / "CLAIM.json")).hexdigest(),
                "--manifest-sha256", sha256(pin(HERE / "SOURCE_FILES.json")).hexdigest()]
        with (DISPATCH / "training-start.log").open("xb") as stream:
            child = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=25)
        need(child.returncode == 0, "Recovery controller was not successfully started")
        control = RESULTS / (LABEL + "-controller")
        until = time.monotonic() + 18
        while time.monotonic() < until:
            active, _ = scheduling.placement()
            if any(name == LABEL + "-gpu" for name, _ in active.values()):
                break
            if (control / "outcome.json").is_file():
                break
            time.sleep(1)


if __name__ == "__main__":
    need(len(sys.argv) == 2 and sys.argv[1] in {"prepare", "dispatch"}, "Recovery phase required")
    scheduling.WORK = DISPATCH
    scheduling.REQUEST = REQUEST
    if sys.argv[1] == "prepare":
        prepare()
    else:
        try:
            dispatch()
        except Exception as error:
            scheduling.save("dispatch-failure.json", {"request": REQUEST, "class": type(error).__name__,
                "reason": str(error) if isinstance(error, ValueError) else None,
                "frames": [(Path(frame.filename).name, frame.lineno, frame.name)
                           for frame in traceback.extract_tb(error.__traceback__)]})
            raise SystemExit(1)
