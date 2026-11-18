"""Run one owned T09 replay and preserve real Slurm accounting and post-device evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from check_hunyuan_sft_replay_inputs import inspect as inspect_inputs
from framework_binding import COMMIT, DEPLOYMENT, exact_bytes, verify_framework
from review_native_terminal import COLUMNS, inspect as review_terminal
from slurm_accounting import wait_terminal
from slurm_command import NODE, PARENT, ROOT, command

LABEL = "training-hunyuanvideo-sft-native-replay-v3"
WHOLE_SECONDS = 1440
WORK_SECONDS = 1240
CLEANUP_SECONDS = 120


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


class Budget:
    def __init__(self):
        self.origin = time.time()
        self.started = time.monotonic()

    def left(self, offset=WHOLE_SECONDS):
        return max(0, min(self.origin + offset - time.time(), self.started + offset - time.monotonic()))

    def query(self, argv):
        left = self.left()
        if left <= 0:
            raise TimeoutError("Original controller deadline expired")
        return subprocess.check_output(argv, text=True, timeout=min(left, 12)).splitlines()


def expand(budget, nodelist):
    hosts = budget.query(["scontrol", "show", "hostnames", nodelist])
    distinct = set(hosts)
    if not hosts or len(hosts) != len(distinct) or any(not host for host in hosts):
        raise ValueError("Slurm node list is empty or contains duplicate nodes")
    return distinct


def slurm_fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def admit(budget):
    lines = budget.query(["scontrol", "show", "job", PARENT, "--oneliner"])
    if len(lines) != 1:
        raise ValueError("Original long allocation is missing or ambiguous")
    job = slurm_fields(lines[0])
    if (job.get("JobId") != PARENT or job.get("JobState") != "RUNNING"
            or job.get("NumNodes") != "40" or "gres/dcu=320" not in job.get("AllocTRES", "").split(",")
            or not job.get("NodeList")):
        raise ValueError("Original parent allocation or node has changed")
    hosts = expand(budget, job["NodeList"])
    if len(hosts) != 40 or NODE not in hosts:
        raise ValueError("Original parent allocation or node has changed")
    seen = set()
    for line in budget.query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        if not line.strip():
            continue
        step = slurm_fields(line)
        handle = step.get("StepId", "")
        if handle in (PARENT + ".batch", PARENT + ".extern"):
            continue
        if (not re.fullmatch(re.escape(PARENT) + r"\.[0-9]+", handle) or handle in seen
                or not step.get("NodeList") or not step.get("Nodes", "").isdigit()):
            raise ValueError("Slurm step membership is incomplete or ambiguous")
        seen.add(handle)
        members = expand(budget, step["NodeList"])
        if len(members) != int(step["Nodes"]) or not members <= hosts:
            raise ValueError("Slurm step membership differs from its actual parent")
        if NODE in members:
            raise ValueError("Requested node already has an active step: " + handle)


def accounting(budget, phase):
    rows = budget.query(["sacct", "-n", "-P", "-j", PARENT,
                         "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"])
    matches = [line.split("|") for line in rows]
    matches = [row for row in matches if len(row) == len(COLUMNS) and row[1] == LABEL + "-" + phase]
    if len(matches) > 1:
        raise ValueError("Original Slurm label is not unique")
    if matches and (not re.fullmatch(r"1103606\.[0-9]+", matches[0][0])
                    or matches[0][5] not in (NODE, "None assigned", "")):
        raise ValueError("Slurm row does not belong to this run")
    return matches[0] if matches else None


def host_step(workspace, phase):
    return [
        "srun", f"--jobid={PARENT}", f"--job-name={LABEL}-{phase}", "--exact",
        "--nodes=1", "--ntasks=1", f"--nodelist={NODE}", "--immediate=5", "--gres=none",
        "--cpus-per-task=1", "--mem=1024M", "--time=1",
        "/usr/local/python3.12/bin/python3", "-B", str(workspace / "src/host_entry.py"),
        str(workspace), phase,
    ]


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def run_phase(workspace, budget, phase, argv):
    seconds = min(45, budget.left() - 25) if phase != "gpu" else budget.left(WORK_SECONDS)
    if seconds <= 0:
        raise TimeoutError(f"No time remains for owned {phase} step")
    with (workspace / f"{phase}.log").open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            proc = Path(f"/proc/{child.pid}/stat")
            try:
                fields = proc.read_text().rsplit(") ", 1)[1].split()
                owner = dict(pid=child.pid, pgid=int(fields[2]), start_ticks=int(fields[19]))
            except FileNotFoundError:
                owner = dict(pid=child.pid, exited_before_observed=True)
            save(workspace / f"{phase}-srun.json", owner)
            child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop_child(child)
    end = time.monotonic() + min(20, budget.left() - 5)
    state = wait_terminal(lambda: accounting(budget, phase), end, interval=1, max_observations=20)
    row = state["row"]
    passed = (not timed_out and child.returncode == 0 and row is not None and row[2:4] == ["COMPLETED", "0:0"]
              and row[5] == NODE)
    receipt = dict(phase=phase, returncode=child.returncode, timed_out=timed_out, passed=passed, **state)
    save(workspace / f"{phase}-accounting.json", receipt)
    return receipt


def sequence(launch):
    receipts = {"pre": launch("pre")}
    if receipts["pre"]["passed"]:
        receipts["gpu"] = launch("gpu")
        if receipts["gpu"]["terminal_visible"]:
            receipts["post"] = launch("post")
    return receipts


def verify_package(source_root, manifest_sha):
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise ValueError("Expected the source manifest SHA-256")
    path = source_root / "SOURCE_FILES.json"
    length = path.lstat().st_size
    if not 0 < length <= 16384:
        raise ValueError("Unexpected source manifest extent")
    raw, _ = exact_bytes(path, length)
    if hashlib.sha256(raw).hexdigest() != manifest_sha:
        raise ValueError("Replay source manifest differs")
    manifest = json.loads(raw)
    files = manifest["files"]
    if manifest["case"] != "T09.sft" or manifest["framework_commit"] != COMMIT or len(files) != 14:
        raise ValueError("Replay package inventory differs")
    expected = set(files) | {"SOURCE_FILES.json", "transformer-audit.json", "execution-authority.json"}
    if {p.name for p in source_root.iterdir()} != expected:
        raise ValueError("Replay source directory inventory differs")
    for name, pin in files.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("Unsafe replay source path")
        data, _ = exact_bytes(source_root / name, pin["bytes"])
        if hashlib.sha256(data).hexdigest() != pin["SHA256"]:
            raise ValueError(f"Replay source bytes differ: {name}")


def run(workspace, audit_sha256, source_manifest_sha256, execution_authority_sha256):
    budget = Budget()
    results = Path(ROOT) / "results"
    if workspace != results / LABEL or not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("Expected the frozen private workspace")
    verify_package(workspace / "src", source_manifest_sha256)
    if (workspace / "intent.json").exists() or any(accounting(budget, p) is not None for p in ("pre", "gpu", "post")):
        raise ValueError("Prior intent or original Slurm step exists")
    manifest = results / DEPLOYMENT / "source/FRAMEWORK_FILES.json"
    framework = verify_framework(manifest.parent.parent / "framework", manifest)
    adoption = json.loads((workspace / "src/CLAIM_GPU.json").read_bytes())
    if not (adoption["root_GO"] is adoption["GPU_GO"] is True and adoption["max_invocations"] == 1):
        raise ValueError("New native caller is SOURCE_ONLY until separate root GPU adoption")
    authority = workspace / "src/execution-authority.json"
    authority_raw = authority.read_bytes()
    if len(authority_raw) > 65536 or hashlib.sha256(authority_raw).hexdigest() != execution_authority_sha256:
        raise ValueError("Actual independent execution-mount authority differs")
    audit = workspace / "src/transformer-audit.json"
    inputs = inspect_inputs(results, audit, audit_sha256)
    if not inputs["pre_dispatch_inputs_ready"]:
        raise ValueError("Frozen target replay inputs are not ready")
    admit(budget)
    if budget.left(WORK_SECONDS) < CLEANUP_SECONDS:
        raise TimeoutError("Insufficient original budget after preflight")
    save(workspace / "intent.json", dict(epoch=budget.origin, deadline=budget.origin + WHOLE_SECONDS,
         framework=framework, inputs=inputs, parent=PARENT, node=NODE, gpu_count=1,
         source_manifest_SHA256=source_manifest_sha256, transformer_audit_SHA256=audit_sha256))
    for folder in ("pre", "gpu-gate", "post", "locks"):
        (workspace / folder).mkdir(mode=0o700)
    receipts = {}
    result = dict(case="T09.sft", terminal_review=None, gpu_step=None, post_step=None,
                  accounting_columns=COLUMNS, accounting=[])
    try:
        def launch(phase):
            if phase == "gpu":
                admit(budget)
                argv = command(workspace=workspace, audit_sha256=audit_sha256,
                               deadline=budget.origin + WORK_SECONDS, execution_authority_sha256=execution_authority_sha256)
            else:
                argv = host_step(workspace, phase)
            receipt = run_phase(workspace, budget, phase, argv)
            receipts[phase] = receipt
            return receipt

        sequence(launch)
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result["controller_error"] = type(error).__name__
    for phase, receipt in receipts.items():
        row = receipt["row"]
        if row:
            result["accounting"].append(row)
            if phase in ("gpu", "post"):
                result[f"{phase}_step"] = row[0]
    result["receipts"] = receipts
    if result["gpu_step"] and result["post_step"]:
        try:
            result["terminal_review"] = review_terminal(workspace, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["review_error"] = type(error).__name__
    review = result["terminal_review"]
    result["execution_passed"] = bool(review and review["native_evidence_complete"]
                                      and all(receipts[p]["passed"] for p in ("pre", "gpu", "post")))
    save(workspace / "outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    args = parser.parse_args()
    result = run(args.workspace, args.audit_sha256, args.source_manifest_sha256, args.execution_authority_sha256)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
