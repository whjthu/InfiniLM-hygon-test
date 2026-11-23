"""Admit one eight-card pilot and wait for its actual CPU preflight accounting."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from launch_plan import PILOT_REQUEST, derive
from slurm_accounting import wait_terminal


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE = "1103606", "f12r2n13"
SCOPE = PROJECT / "results" / ("GLM474-fresh-kv-headshare-pilot-" + PILOT_REQUEST)
PACKAGE = SCOPE / "package"
BUILD = PROJECT / "candidates/GLM474-fresh-kv-headshare2-gfx936-CPU-v2/CPU-result.json"
BUILD_MANIFEST = "99a4dda85f2485a0d4fa61c82b6699e115d1b5dedb260e36960ab71c962bab75"
BUILD_RECEIPT = "869fa66e2c61b66f25f87921344e6fc5338001b9de738450ca49714e2873dcfd"
INVENTORY = PROJECT / "results/training40-dispatch-inventory-0e97f89c5ba543bbb81739d5b8351380"
PRE = "GLM-H2-admit-" + PILOT_REQUEST[:12]
GPU = "GLM474-fresh-kv-h2-" + PILOT_REQUEST[:12] + "-00"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def call(*args, timeout=8):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=timeout).stdout


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def hosts(expression):
    return call("scontrol", "show", "hostnames", expression).splitlines()


def owner_available():
    job = fields(call("scontrol", "show", "job", PARENT, "-o"))
    nodes = hosts(job["NodeList"])
    need(job["JobState"] == "RUNNING" and job["UserId"].endswith(f"({os.getuid()})")
         and "gres/dcu=320" in job["AllocTRES"].split(",") and len(set(nodes)) == 40 and NODE in nodes,
         "Pilot parent is no longer available")
    for line in call("scontrol", "show", "step", PARENT, "--oneliner").splitlines():
        step = fields(line)
        if step["StepId"] not in (PARENT + ".batch", PARENT + ".extern"):
            need(re.fullmatch(PARENT + r"\.\d+", step["StepId"]) and NODE not in hosts(step["NodeList"]),
                 "Pilot node is in use")


def accounting(label):
    rows = [row.split("|") for row in call("sacct", "-nP", "-j", PARENT, "--format", COLUMNS).splitlines()]
    matches = [row for row in rows if row[1] == label]
    need(len(matches) <= 1, "Ambiguous pilot accounting")
    return matches[0] if matches else None


def save(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def before():
    probe = (
        "import os, socket, json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from collect_training80_dispatch_snapshot import device_status; row=device_status([]); "
        "row.update(step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], "
        "node=socket.gethostname().split('.')[0], "
        "boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()); print(json.dumps(row))"
    )
    argv = ["srun", f"--jobid={PARENT}", "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
            f"--nodelist={NODE}", "--cpus-per-task=1", "--mem=256M", "--gres=none", "--time=1",
            f"--job-name={PRE}", sys.executable, "-B", "-c", probe, str(INVENTORY)]
    row = json.loads(call(*argv, timeout=18))
    pins = json.loads((INVENTORY / "pci-pins.json").read_bytes())
    need(row["node"] == NODE and len(set(row["PCIs"])) == 8 and sorted(row["PCIs"]) == sorted(pins[NODE])
         and row["kfd_process_count"] == row["own_device_fd_process_count"] == 0, "Pilot device is held or differs")
    save(SCOPE / "before.json", row)
    acct = wait_terminal(lambda: accounting(PRE), time.monotonic() + 21, interval=1, max_observations=14)
    save(SCOPE / "before-accounting.json", acct)
    record = acct["row"]
    need(acct["terminal_visible"] and record[0] == row["step"] and record[2:4] == ["COMPLETED", "0:0"]
         and record[5] == NODE and "gres/dcu" not in record[6], "Original CPU preflight terminal failed")
    return row


def launch():
    need(Path(__file__).resolve().parent == PACKAGE, "Wrong pilot namespace")
    need(sha256(BUILD.read_bytes()).hexdigest() == BUILD_RECEIPT, "Actual build receipt differs")
    build = json.loads(BUILD.read_bytes())
    need(build["passed"] and build["original_step"] == "1103606.1446" and build["manifest_SHA256"] == BUILD_MANIFEST
         and sha256(BUILD.with_name("libGLM_fresh_kv_headshare.so").read_bytes()).hexdigest() == build["shared_SHA256"],
         "Actual build is unqualified")
    owner_available()
    need(accounting(PRE) is None and accounting(GPU) is None, "Pilot must not be replayed")
    preflight_start = time.monotonic()
    row = before()
    owner_available()
    need(time.monotonic() - preflight_start <= 42, "Fresh pilot admission expired")
    epoch = time.time()
    common = {"node": NODE, "parent": PARENT, "request": PILOT_REQUEST, "pre_step": row["step"],
              "root_admitted": True, "epoch": epoch, "deadline_epoch": epoch + 150}
    scope_sha = save(SCOPE / (NODE + ".json"), {**common, "PCI": row["PCIs"], "boot_id": row["boot_id"],
        "build_step": build["original_step"], "build_source_manifest_SHA256": BUILD_MANIFEST})
    plan = {"kind": "pilot", "request": PILOT_REQUEST, "package": str(PACKAGE), "campaign": str(SCOPE),
            "epoch": epoch, "deadline_epoch": epoch + 150, "project_owned": True, "fresh_protected_excluded": True,
            "build_receipt": str(BUILD), "build_receipt_SHA256": BUILD_RECEIPT,
            "nodes": {NODE: {**common, "scope_SHA256": scope_sha}}}
    [shard] = derive(plan)["shards"]
    need(shard["label"] == GPU and shard["planned_cards"] == 8, "Rendered pilot allocation differs")
    response = {**common, "label": GPU, "scope_SHA256": scope_sha}
    save(SCOPE / "launch-intent.json", response)
    env = {key: value for key, value in os.environ.items()
           if key not in ("LD_PRELOAD", "LD_AUDIT", "BASH_ENV", "ENV", "PYTHONPATH")
           and not key.startswith(("APPTAINERENV_", "SINGULARITYENV_"))}
    with (SCOPE / "client.log").open("xb") as log:
        child = subprocess.Popen(shard["argv"], start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
    stat = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
    response.update(pid=child.pid, start_ticks=int(stat[19]))
    save(SCOPE / "started.json", response)
    return response


if __name__ == "__main__":
    print(json.dumps(launch(), sort_keys=True))
