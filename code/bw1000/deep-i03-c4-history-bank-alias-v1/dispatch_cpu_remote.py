"""Dispatch the published C4 history-bank alias single-object CPU build once."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

data = json.loads(base64.b64decode(WIRE, validate=True))
claim_raw = base64.b64decode(data["claim"], validate=True)
claim_sha = hashlib.sha256(claim_raw).hexdigest()
claim = json.loads(claim_raw)
assert claim_sha == data["claim_sha256"]
assert (claim["label"], claim["parent"], claim["node"]) == (
    "i03-deep-c4-history-alias-cpu-v1", 1087656, "f12r3n12")
assert claim["CPU_GO"] is True and claim["GPU_GO"] is False and claim["once"] is True
assert claim["hard_s"] == 300 and (claim["cores"], claim["mem_MiB"]) == (4, 8192)
files = {name: base64.b64decode(content, validate=True) for name, content in data["files"].items()}
assert set(files) == set(claim["files_SHA256"])
for name, raw in files.items():
    assert Path(name).name == name and hashlib.sha256(raw).hexdigest() == claim["files_SHA256"][name]

root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
work = root / "candidates/Deep-C4-history-bank-alias-CPU-v1"
parent = str(claim["parent"])
node = claim["node"]
label = claim["label"]


def check(*args):
    return subprocess.check_output(args, timeout=15, text=True).splitlines()


def accounting():
    return [row.split("|") for row in check("sacct", "-n", "-P", "-j", parent,
                                             "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw")
            if "|" + label + "|" in row]


assert not accounting() and not work.exists()
row, = check("squeue", "-h", "-j", parent, "-o", "%T|%D|%N")
state, count, hosts = row.split("|")
assert state == "RUNNING" and count == "40" and node in check("scontrol", "show", "hostnames", hosts)
for row in check("squeue", "--steps", "-h", "-j", parent, "-o", "%i|%j|%N"):
    step, name, hosts = row.split("|")
    if step in {parent + ".batch", parent + ".extern"}:
        continue
    assert name != label
    if hosts and hosts != "(null)":
        assert node not in check("scontrol", "show", "hostnames", hosts)
work.mkdir(mode=0o700)
source = work / "source"
source.mkdir()
for name, raw in dict(files, **{"CPU_RESOURCE_CLAIM.json": claim_raw}).items():
    with (source / name).open("xb") as stream:
        stream.write(raw)
epoch = time.time()
deadline = epoch + claim["hard_s"]
command = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--nodes=1", "--ntasks=1",
           "--cpus-per-task=4", "--nodelist=" + node, "--gres=none", "--mem=8192M", "--immediate=5",
           "--time=00:05:00", "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home",
           "--pwd", "/", "--bind", str(root) + ":/refactor:ro",
           "--bind", "/public/home/aclsylqidf/hygon-inference-20261013:/old:ro",
           "--bind", "/public/home/aclsylqidf/shared:/shared:ro", "--bind", "/opt/hyhal:/opt/hyhal:ro",
           "--bind", str(work) + ":/build:rw", str(root/"containers/dtk26-sandbox-r1"),
           "/usr/bin/env", "DEEP_HISTORY_CLAIM_SHA=" + claim_sha,
           "DEEP_HISTORY_PUBLICATION=" + data["commit"],
           "/bin/bash", "--noprofile", "--norc", "/build/source/CPU_inside.sh", str(deadline)]
print(json.dumps(dict(dispatch_intent=True, epoch=epoch, deadline=deadline, label=label,
                      claim_SHA256=claim_sha)), flush=True)
run = subprocess.run(command, capture_output=True, timeout=310)
rows = []
for _ in range(21):
    rows = accounting()
    assert len(rows) <= 1
    if rows and rows[0][2] not in ("PENDING", "RUNNING", "COMPLETING"):
        break
    time.sleep(1)
receipt = work/"artifacts/CPU-build.json"
verified = None
if receipt.is_file():
    raw = receipt.read_bytes()
    q = json.loads(raw)
    verified = {name: q.get(name) for name in ("completed", "GPU_initialized", "Native_dlopen",
        "actual_step", "replaced", "reused", "elapsed_s", "library", "library_SHA256",
        "old_strong_exports", "new_strong_exports", "dynamic_NEEDED_unchanged")}
    verified["receipt_SHA256"] = hashlib.sha256(raw).hexdigest()
    if not q["completed"]:
        verified["error_type"] = q.get("error", "").partition("(")[0]
    if q["completed"]:
        actual = work/"artifacts"/Path(q["library"]).name
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == q["library_SHA256"]
print(json.dumps(dict(result=True, rc=run.returncode, accounting=rows, verified=verified,
                      process_stdout_SHA256=hashlib.sha256(run.stdout).hexdigest(),
                      process_stderr_SHA256=hashlib.sha256(run.stderr).hexdigest())),flush=True)
raise SystemExit(0 if run.returncode==0 and verified and verified["completed"]
                 and len(rows)==1 and rows[0][0]==verified["actual_step"]
                 and rows[0][2:4]==["COMPLETED","0:0"] else 1)
