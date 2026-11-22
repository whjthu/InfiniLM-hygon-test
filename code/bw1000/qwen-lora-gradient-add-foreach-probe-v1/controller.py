"""Check the assigned O01 node, run one device step, and reconcile cleanup."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time

from slurm_accounting import wait_terminal


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PARENT, NODE = "1103606", "f12r2n13"
LABEL = "qwen-lora-gradient-add-foreach-probe-v1"
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=8).splitlines()


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def accounting(label):
    rows = [line.split("|") for line in query("sacct", "-nP", "-j", PARENT, "--format", COLUMNS)]
    matches = [row for row in rows if len(row) == 7 and row[1] == label]
    need(len(matches) <= 1, "Duplicate original step label")
    return matches[0] if matches else None


def placement():
    [raw] = query("scontrol", "show", "job", PARENT, "-o")
    parent = fields(raw)
    need(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING" and parent["NumNodes"] == "40"
         and parent["UserId"].endswith(f"({os.getuid()})") and "gres/dcu=320" in parent["AllocTRES"].split(",")
         and NODE in query("scontrol", "show", "hostnames", parent["NodeList"]), "Assigned parent has changed")
    for raw in query("scontrol", "show", "step", PARENT, "-o"):
        step = fields(raw)
        if step["StepId"] not in (PARENT + ".batch", PARENT + ".extern"):
            need(re.fullmatch(PARENT + r"\.\d+", step["StepId"])
                 and NODE not in query("scontrol", "show", "hostnames", step["NodeList"]),
                 "Assigned O01 node already has an owned step")


def run(phase, gpu, payload, seconds):
    label = LABEL + "-" + phase
    need(accounting(label) is None, "Original phase already has accounting")
    argv = ["srun", f"--jobid={PARENT}", f"--job-name={label}", "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", f"--nodelist={NODE}",
            f"--cpus-per-task={24 if gpu else 2}", f"--mem={49152 if gpu else 2048}M",
            "--gres=dcu:8" if gpu else "--gres=none",
            f"--time={seconds // 60}", *payload]
    with (ROOT / (phase + ".log")).open("xb") as log:
        try:
            client = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=seconds + 8)
            rc = client.returncode
        except subprocess.TimeoutExpired:
            rc = None
    receipt = wait_terminal(lambda: accounting(label), time.monotonic() + 19, interval=1, max_observations=12)
    receipt["client_rc"] = rc
    save(phase + "-accounting.json", receipt)
    row = receipt["row"]
    need(row and re.fullmatch(PARENT + r"\.\d+", row[0]) and row[5] == NODE
         and ("gres/dcu=8" in row[6].split(",")) == gpu, "Unknown terminal accounting or GPU resource shape")
    return row


def inventory(phase):
    payload = [PYTHON, "-I", "-S", "-B", str(ROOT / "node_inventory.py"), PARENT, NODE]
    row = run(phase, False, payload, 60)
    need(row[2:4] == ["COMPLETED", "0:0"], "Original physical inventory failed")
    [result] = [json.loads(line) for line in (ROOT / (phase + ".log")).read_bytes().splitlines()
                if line.startswith(b"{")]
    need(result["step"] == row[0] and len(set(result["PCIs"])) == 8
         and result["kfd_process_count"] == result["own_device_fd_process_count"] == 0, "DCU holders are present")
    return result


def check_lease():
    import split_admission

    split_admission.NODE = NODE
    split_admission.STEP = re.compile(PARENT + r"\.\d+")
    physical = split_admission.snapshot(True)
    before = json.loads((ROOT / "before.json").read_bytes())
    need(physical["pci"] == before["PCIs"] and physical["kfd_holder_count"] == 0
         and len(set(physical["pci_to_kfd"].values())) == 8, "Leased DCU roster has changed")
    boot = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    need(sha256(boot).hexdigest() == before["boot_id_SHA256"], "Leased node boot differs")
    save("lease.json", physical)
    app = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
    env = 'set -euo pipefail; source /opt/dtk/env.sh; exec python3 -B -u "$1/worker.py" "$1"'
    args = [app, "exec", "--rocm", "--no-home", "--bind", f"{ROOT}:{ROOT}",
            "--bind", "/opt/hyhal:/opt/hyhal:ro", str(PROJECT / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", env, "gradient-add", str(ROOT)]
    os.execv(app, args)


def main():
    claim = json.loads((ROOT / "CLAIM.json").read_bytes())
    need((claim["label"], claim["node"], claim["parent"]) == (LABEL, NODE, PARENT)
         and claim["operator_GO"] and not claim["Model_GO"] and claim["cards"] == 8, "Unexpected claim")
    for name, item in claim["files"].items():
        data = (ROOT / name).read_bytes()
        need(sha256(data).hexdigest() == item["SHA256"] and len(data) == item["bytes"], "Published source drift")
    ledger = json.loads((ROOT / "ledger.json").read_bytes())
    need(NODE in ledger["groups"]["O01"]["nodes"] and ledger["parent"] == PARENT, "Wrong node ownership")
    placement()
    save("intent.json", {"request": claim["request"], "epoch": time.time(), "deadline_seconds": 365})
    first = inventory("pre")
    save("before.json", first)
    placement()
    gpu = run("gpu", True, [PYTHON, "-B", "-u", str(ROOT / "controller.py"), "leased"], 180)
    post = inventory("post")
    need(first["PCIs"] == post["PCIs"] and first["boot_id_SHA256"] == post["boot_id_SHA256"], "Post inventory differs")
    result = {"request": claim["request"], "gpu_accounting": gpu, "post_clean": True, "cards": [],
              "component_passed": False, "formal_training": False, "GPU_busy": None}
    if gpu[2:4] == ["COMPLETED", "0:0"]:
        for index in range(8):
            result["cards"].append(json.loads((ROOT / f"card-{index}.json").read_bytes()))
        need({row["PCI"] for row in result["cards"]} == set(first["PCIs"])
             and all(row["step"] == gpu[0] and row["exact_tensors"] == 504 for row in result["cards"]),
             "Per-card numeric denominator differs")
        result["component_passed"] = True
    save("outcome.json", result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["leased"]:
        check_lease()
    else:
        main()
