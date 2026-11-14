"""Run one bounded CPU-only visibility diagnostic under the existing Slurm allocation."""
import base64
import hashlib
import os
import signal
import sys
import json
from pathlib import Path
import re
import subprocess
import time

label = "GLM474-TP16-file-visibility-CPU-v1"
parent = "1103606"
nodes = {"f13r1n08", "f13r1n09"}
root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
run = root / "results" / label
columns = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList"]
epoch = time.time()
anchor = time.monotonic()


def budget():
    return min(epoch + 105 - time.time(), 105 - (time.monotonic() - anchor))


def query(*args):
    remaining = min(8, budget())
    if remaining <= 0:
        raise TimeoutError("diagnostic deadline expired")
    return subprocess.check_output(args, text=True, timeout=remaining).strip().splitlines()


def accounting():
    lines = query("sacct", "-n", "-P", "-j", parent, "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList")
    return [line.split("|") for line in lines if "|" + label + "|" in line]


assert not accounting()
job, = query("squeue", "-h", "-j", parent, "-o", "%i|%T|%D|%N")
handle, state, count, members = job.split("|")
assert (handle, state, count) == (parent, "RUNNING", "40")
assert nodes <= set(query("scontrol", "show", "hostnames", members))
for line in query("scontrol", "show", "step", parent, "--oneliner"):
    fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
    if not fields["StepId"].endswith(".extern"):
        assert not nodes.intersection(query("scontrol", "show", "hostnames", fields["NodeList"]))
run.mkdir(mode=0o700)
for name, item in FILES.items():
    assert Path(name).name == name
    raw = base64.b64decode(item["data"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    with (run / name).open("xb") as f:
        f.write(raw)
(run / "existing.bin").write_bytes(bytes(64))
intent = dict(request_id=REQUEST, claim=CLAIM, epoch=epoch, deadline=epoch + 105, GPU_cards=0)
with (run / "intent.json").open("x") as f:
    json.dump(intent, f)
scope = "/refactor/results/" + label
args = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--nodes=2",
        "--ntasks=2", "--ntasks-per-node=1", "--distribution=block:block",
        "--nodelist=f13r1n08,f13r1n09", "--immediate=5", "--kill-on-bad-exit=1",
        "--cpus-per-task=1", "--mem=1024M", "--gres=none", "--time=2",
        "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/",
        "--bind", f"{root}:/refactor:ro", "--bind", f"{run}:{scope}:rw",
        str(root / "containers/dtk26-sandbox-r1"),
        "/usr/bin/python3", "-B", scope + "/worker.py", scope, str(epoch)]
forced = False
with (run / "run.log").open("xb") as log:
    result = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        fields = Path(f"/proc/{result.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        with (run / "srun-owner.json").open("x") as f:
            json.dump(dict(pid=result.pid, pgid=int(fields[2]), startticks=int(fields[19])), f)
        result.wait(timeout=max(.01, min(75, budget() - 20)))
    except subprocess.TimeoutExpired:
        forced = True
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if result.poll() is not None:
                break
            try:
                os.killpg(result.pid, sig)
            except ProcessLookupError:
                pass
            try:
                result.wait(timeout=4)
            except subprocess.TimeoutExpired:
                forced = True
sys.path.insert(0, str(run))
from slurm_accounting import wait_terminal
def terminal_row():
    rows = accounting()
    assert len(rows) <= 1
    return rows[0] if rows else None
terminal = wait_terminal(terminal_row, time.monotonic() + min(12, budget() - 4), interval=.5, max_observations=24)
row = terminal["row"]
records = []
if row and row[2:4] == ["COMPLETED", "0:0"] and result.returncode == 0 and not forced:
    assert re.fullmatch(parent + r"\.\d+", row[0])
    from terminal_receipts import collect_receipts
    found = collect_receipts(run, ["rank-0.json", "rank-1.json"],
                             time.monotonic() + min(12, budget() - 2), max_attempts=24)
    records = [entry["value"] for entry in found["receipts"].values()]
    records.sort(key=lambda entry: entry["rank"])
    assert all(r["step"] == row[0] for r in records)
out = dict(**intent, passed=len(records) == 2, row=row, records=records, accounting_columns=columns,
           returncode=result.returncode, forced=forced, wall_seconds=time.monotonic() - anchor)
with (run / "outcome.json").open("x") as f:
    json.dump(out, f)
print(json.dumps(out), flush=True)
