"""Run the published one-node Qwen collective once under its existing Slurm parent."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-qwen-hip-input-agreement-tcp-v2"
ROOT = R / "results" / LABEL
PARENT = "1087656"
NODE = "a01r4n16"
COLUMNS = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList"]
PHASES = ("pre", "gpu", "post")
EPOCH = time.time()
ANCHOR = time.monotonic()
WHOLE = 390
WORK = 285


def left(limit=WHOLE):
    return min(EPOCH + limit - time.time(), ANCHOR + limit - time.monotonic())


def query(args):
    budget = min(8, left() - 5)
    if budget <= 0:
        raise TimeoutError("one-shot deadline exhausted")
    return subprocess.check_output(args, text=True, timeout=budget).strip().splitlines()


def one_row(phase):
    label = LABEL + "-" + phase
    rows = query(["sacct", "-n", "-P", "-j", PARENT, "--format="
                  "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"])
    matching = [row.split("|") for row in rows if "|" + label + "|" in row]
    assert len(matching) <= 1
    if not matching:
        return None
    row = matching[0]
    assert len(row) == len(COLUMNS) and re.fullmatch(PARENT + r"\.\d+", row[0])
    return row


def check_node():
    rows = query(["squeue", "-h", "-j", PARENT, "-o", "%i|%T|%D|%N"])
    assert len(rows) == 1
    parent, state, count, hosts = rows[0].split("|")
    assert (parent, state, count) == (PARENT, "RUNNING", "40")
    assert NODE in query(["scontrol", "show", "hostnames", hosts])
    for line in query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        step = fields["StepId"]
        assert step.startswith(PARENT + ".")
        if step == PARENT + ".extern":
            continue
        assert NODE not in query(["scontrol", "show", "hostnames", fields["NodeList"]])


def save(name, value):
    with (ROOT / name).open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def owner(pid):
    status = Path("/proc", str(pid), "stat").read_text().rsplit(") ", 1)[1].split()
    return dict(host=socket.gethostname(), pid=pid, pgid=int(status[2]), startticks=int(status[19]))


def phase_run(phase, limit):
    assert one_row(phase) is None
    gpu = phase == "gpu"
    label = LABEL + "-" + phase
    command = [
        "srun", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--nodes=1",
        "--ntasks=1", "--nodelist=" + NODE, "--immediate=5", "--kill-on-bad-exit=1",
        "--time=" + ("5" if gpu else "1"), "--cpus-per-task=" + ("16" if gpu else "1"),
        "--mem=" + ("32768M" if gpu else "1024M"), "--gres=" + ("dcu:8" if gpu else "none"),
        "/usr/local/python3.12/bin/python3", "-B", str(ROOT / "src" / "host_entry.py"),
        str(ROOT), "gpu-gate" if gpu else phase,
    ]
    if gpu:
        shell = (
            'set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
            'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 '
            'OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=INFO PYTHONPATH=/minicpm-transformers:/swift-runtime/site; '
            'exec python3 -B -u "$1/src/worker.py" "$1"'
        )
        command += [
            "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec",
            "--rocm", "--no-home", "--pwd", "/", "--bind", f"{ROOT}:{ROOT}:rw",
            "--bind", f"{R}/results/training-vlm-swift-offline-runtime-v1:/swift-runtime:ro",
            "--bind", f"{R}/results/training-minicpm-hygon-processor-r2/cpu/runtime-site:/minicpm-transformers:ro",
            "--bind", "/opt/hyhal:/opt/hyhal:ro", str(R / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", shell, "qwen-collective", str(ROOT),
        ]
    budget = left(limit)
    if budget <= 0:
        raise TimeoutError("phase deadline exhausted")
    forced = False
    errors = []
    logfile = ROOT / (phase + ".log")
    with logfile.open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            save(phase + "-owner.json", owner(child.pid))
            child.wait(timeout=max(0.1, left(limit)))
        except Exception as error:
            forced = True
            errors.append(type(error).__name__)
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if child.poll() is not None:
                    break
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    forced = True
    observed = wait_terminal(lambda: one_row(phase), time.monotonic() + max(0, min(20, left()-10)),
                             interval=1, max_observations=20)
    row = observed["row"]
    raw = logfile.read_bytes()
    passed = bool(row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                  and child.returncode == 0 and not forced)
    receipt = dict(terminal=observed, passed=passed, returncode=child.returncode, forced=forced,
                   errors=errors, log_sha256=hashlib.sha256(raw).hexdigest())
    save(phase + "-receipt.json", receipt)
    return receipt


def read(name):
    path = ROOT / name
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def main():
    for phase in PHASES:
        assert one_row(phase) is None
    check_node()
    ROOT.mkdir(mode=0o700)
    save("intent.json", dict(claim=CLAIM, request_id=REQUEST, node=NODE, parent=PARENT,
         epoch=EPOCH, deadline=EPOCH+WHOLE, work_deadline=EPOCH+WORK, owner=owner(os.getpid()),
         source_sha256={name: blob["sha256"] for name, blob in FILES.items()}))
    for folder in ("src", "pre", "gpu-gate", "post", "ranks"):
        (ROOT / folder).mkdir()
    for name, blob in FILES.items():
        assert Path(name).name == name
        payload = base64.b64decode(blob["data"], validate=True)
        assert hashlib.sha256(payload).hexdigest() == blob["sha256"]
        with (ROOT / "src" / name).open("xb") as f:
            f.write(payload)
    sys.path.insert(0, str(ROOT / "src"))
    global wait_terminal
    from slurm_accounting import wait_terminal

    phases = {}
    errors = []
    try:
        phases["pre"] = phase_run("pre", 40)
        if phases["pre"]["passed"]:
            check_node()
            phases["gpu"] = phase_run("gpu", WORK+12)
    except Exception as error:
        errors.append(dict(stage="dispatch", type=type(error).__name__, message=str(error)[:200]))
    finally:
        try:
            if phases.get("gpu", {}).get("terminal", {}).get("terminal_visible"):
                phases["post"] = phase_run("post", 365)
        except Exception as error:
            errors.append(dict(stage="post", type=type(error).__name__, message=str(error)[:200]))

    rows = [read("ranks/" + str(rank) + ".json") for rank in range(8)]
    pre, gate, post = [read(f"{key}/{NODE}.json") for key in ("pre", "gpu-gate", "post")]
    assigned = read("leased.json")
    marks = []
    for rank in range(8):
        prefix = f"ranks/{rank}"
        before = read(prefix + "-before-init.json")
        after = read(prefix + "-after-init.json")
        votes = [[read(prefix + f"-vote-{index}-{boundary}.json") for boundary in ("start", "end")]
                 for index in range(9)]
        marks.append(dict(before=before, after=after, votes=votes))
    passed = bool(not errors and all(phases.get(phase, {}).get("passed") for phase in PHASES)
                  and pre and gate and post and assigned and all(rows))
    if passed:
        snapshots = [x["value"] for x in (pre, gate, post)]
        devices = assigned["value"]
        results = [x["value"] for x in rows]
        gpu_step = phases["gpu"]["terminal"]["row"][0]
        passed = (len(devices) == len(set(devices.values())) == 8 and
                  sorted(r["pci"] for r in results) == sorted(devices) and
                  len({x["boot_id"] for x in snapshots}) == 1 and
                  all(x["pci"] == sorted(devices) and x["kfd_holder_count"] == 0 for x in snapshots) and
                  all(r["rank"] == rank and r["step"] == gpu_step and r["kfd"] == devices[r["pci"]]
                      and r["process_group_destroyed"] and r["backend"] == "nccl"
                      and r["observations"] == [dict(rejecting_rank=rejector, local_vote=int(rank != rejector),
                                                      agreed=rejector is None)
                                               for rejector in [None, *range(8)]]
                      for rank, r in enumerate(results)))
    if passed:
        start_ports = set()
        for rank, marks_for_rank in enumerate(marks):
            before, after = marks_for_rank["before"], marks_for_rank["after"]
            if not before or not after or not all(all(pair) for pair in marks_for_rank["votes"]):
                passed = False
                break
            start = before["value"]
            stop = after["value"]
            start_ports.add(start["master_port"])
            chain = [start, stop, *(record["value"] for pair in marks_for_rank["votes"] for record in pair)]
            if (any(row["rank"] != rank for row in chain) or
                    any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(chain, chain[1:])) or
                    [pair[1]["value"]["vote"] for pair in marks_for_rank["votes"]] != results[rank]["observations"]):
                passed = False
                break
        passed = passed and len(start_ports) == 1
    outcome = dict(request_id=REQUEST, claim=CLAIM, component_passed=passed, errors=errors,
                   accounting_columns=COLUMNS, phases=phases, pre=pre, gate=gate, post=post,
                   assigned=assigned, rank_results=rows, rank_marks=marks, launcher=read("launcher.json"), model_loaded=False,
                   training_updates=0, formal_performance=False)
    save("outcome.json", outcome)
    print(json.dumps(outcome), flush=True)


main()
