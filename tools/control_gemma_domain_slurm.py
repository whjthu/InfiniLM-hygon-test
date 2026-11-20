"""Drive one preflight, one native eight-card step, and independent cleanup."""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from compare_gemma_domain_recovery import read_receipt, require
from read_gemma_domain_terminal import LABEL, NODE, PARENT, bound_step, numbered_step, token
from read_training_slurm_membership import allocated_dcu, fields
from run_gemma_domain_pair import pinned_module, write_json
from snapshot_gemma_pair_step import seal_accounting


R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
WAIT_HELPER = (
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py",
    "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
)
FILES = {"pre": "before.json", "gpu": "gpu-step-binding.json", "post": "after.json"}


def arm_timeline(step, seal):
    stages = {"pre": step("pre")}
    if stages["pre"]["passed"]:
        stages["gpu"] = step("gpu")
        gpu = stages["gpu"]
        if gpu["terminal_visible"] and gpu["original_step_bound"]:
            stages["post"] = step("post")
            if stages["post"]["terminal_visible"] and stages["post"]["original_step_bound"]:
                seal({phase: result["row"] for phase, result in stages.items()})
    return stages


def remaining(deadline, now, maximum, reserve=0):
    if not all(math.isfinite(v) for v in (deadline, now, maximum, reserve)) or maximum <= 0 or reserve < 0:
        raise ValueError("invalid original admission clock")
    budget = min(maximum, deadline - now - reserve)
    if budget <= 0:
        raise TimeoutError("original Gemma admission deadline exhausted")
    return budget


def srun_command(phase, payload, seconds):
    gpu = phase == "gpu"
    hour, remainder = divmod(seconds, 3600)
    minute, second = divmod(remainder, 60)
    return ["srun", f"--jobid={PARENT}", f"--job-name={LABEL}-{phase}", "--nodes=1", "--ntasks=1",
            f"--nodelist={NODE}", "--exact", "--immediate=5", "--cpus-per-task=" + ("64" if gpu else "2"),
            "--mem=" + ("224000M" if gpu else "4096M"), "--gres=" + ("dcu:8" if gpu else "none"),
            f"--time={hour:02d}:{minute:02d}:{second:02d}", *payload]


def gpu_payload(repo, root, intent_sha, intent):
    sites = R / "results"
    binds = {
        root: (root, "rw"),
        repo: (repo, "ro"),
        sites / "training-vlm-swift-offline-runtime-v1": (Path("/swift-runtime"), "ro"),
        sites / "training-minicpm-hygon-processor-r2/cpu/runtime-site": (Path("/minicpm-transformers"), "ro"),
        sites / "training-gemma-usgs-render72-native-CPU-v1/data": (
            sites / "training-gemma-usgs-render72-native-CPU-v1/data", "ro"),
        sites / "training-gemma-original-model-v1": (sites / "training-gemma-original-model-v1", "ro"),
        Path("/opt/hyhal"): (Path("/opt/hyhal"), "ro"),
    }
    initial = Path(intent["initial_trainables"]).resolve(strict=True).parent
    require(initial.is_relative_to(sites) and initial != sites, "initial state is outside project storage")
    binds[initial] = (initial, "ro")
    command = [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/"]
    for src, (dest, mode) in binds.items():
        command.extend(["--bind", f"{src}:{dest}:{mode}"])
    env_path = f"/minicpm-transformers:/swift-runtime/site:{root / 'framework'}"
    native = ["python3", "-B", "-u", str(repo / "tools/run_gemma_domain_pair.py"), str(root), "--repo", str(repo),
              "--intent-sha", intent_sha]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; "
             f"export PYTHONPATH={shlex.quote(env_path)}; exec {shlex.join(native)}")
    return [*command, str(R / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def controller(repo, root, intent_sha):
    repo = repo.resolve(strict=True)
    root = root.resolve(strict=True)
    require(root == R / "results" / LABEL and not repo.is_relative_to(root),
            "original run root and frozen source package must be distinct")
    evidence = {}
    intent = read_receipt(root, Path("intent.json"), evidence)
    require(evidence["intent.json"] == intent_sha and intent["label"] == LABEL and "steps" not in intent
            and token(intent["request_id"], 32) and token(intent["publication"], 40),
            "original Gemma launch intent differs")
    epoch, deadline, native = (intent[name] for name in ("epoch", "deadline", "native_deadline"))
    now = time.time()
    require(all(math.isfinite(v) for v in (epoch, deadline, native)) and 0 < epoch <= now < native < deadline
            and deadline - epoch <= 2400 and deadline - native >= 120,
            "invalid original deadline or missing cleanup reserve")
    installed = read_receipt(root, Path("installation-complete.json"), evidence)
    cpu_input = read_receipt(root, Path("native-input-cpu-result.json"), evidence)
    initial = Path(intent["initial_trainables"])
    require(evidence["installation-complete.json"] == intent["installation_receipt_SHA256"]
            and installed["installed_files_verified"] == 396
            and evidence["native-input-cpu-result.json"] == intent["native_cpu_receipt_SHA256"]
            and cpu_input["passed"] is True and (cpu_input["unique_rows"], cpu_input["encode_calls"],
                cpu_input["collator_calls"]) == (72, 144, 72)
            and initial.is_file() and not initial.is_symlink() and initial.stat().st_size == 119430531,
            "target install, original initialization or native CPU input requirement is incomplete")
    wait_terminal = pinned_module(repo, WAIT_HELPER, "gemma_slurm_accounting")["wait_terminal"]

    def query(*argv):
        return subprocess.check_output(argv, text=True, timeout=remaining(deadline, time.time(), 10)).strip()

    def accounting(phase):
        rows = [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT, "-o", ",".join(COLUMNS)).splitlines()]
        match = [row for row in rows if len(row) == len(COLUMNS) and row[1] == f"{LABEL}-{phase}"]
        require(len(match) <= 1, "ambiguous original Slurm step; reconciliation required")
        return match[0] if match else None

    def admit():
        parent = fields(query("scontrol", "show", "job", PARENT, "-o"))
        queue = query("squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N").split("|")
        require(parent["UserId"].endswith(f"({os.getuid()})") and allocated_dcu(parent["AllocTRES"]) == 320
                and queue[:2] == ["RUNNING", "40"]
                and NODE in query("scontrol", "show", "hostnames", queue[2]).splitlines(),
                "original allocation, ownership or node membership changed")
        for line in query("scontrol", "-o", "show", "step", PARENT).splitlines():
            stage = fields(line)
            if numbered_step(stage.get("StepId")):
                require(NODE not in query("scontrol", "show", "hostnames", stage["NodeList"]).splitlines(),
                        "original target node already has a live numbered step")
        require(all(accounting(phase) is None for phase in FILES), "existing step label requires reconciliation")

    def snapshot_command(phase):
        return [HOST_PYTHON, "-B", str(repo / "tools/snapshot_gemma_pair_step.py"), phase, str(root),
                "--repo", str(repo), "--intent-sha", intent_sha]

    def step(phase):
        require(accounting(phase) is None, "existing step label requires reconciliation")
        maximum = 2100 if phase == "gpu" else 90
        reserve = 145 if phase in {"pre", "gpu"} else 15
        seconds = int(remaining(deadline, time.time(), maximum, reserve))
        require(seconds >= 1, "original phase budget is below one second")
        payload = gpu_payload(repo, root, intent_sha, intent) if phase == "gpu" else snapshot_command(phase)
        rc = None
        with (root / f"{phase}.log").open("xb") as log:
            try:
                rc = subprocess.run(srun_command(phase, payload, seconds), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=remaining(deadline, time.time(), seconds + 5, reserve)).returncode
            except subprocess.TimeoutExpired:
                pass
        write_json(root / f"{phase}-client-outcome.json", dict(rc=rc, remote_outcome_unknown=rc is None))
        status = wait_terminal(lambda: accounting(phase),
            time.monotonic() + remaining(deadline, time.time(), 40, 5), interval=2, max_observations=18)
        row = status["row"]
        bound = False
        if row and (root / FILES[phase]).exists():
            saved = read_receipt(root, Path(FILES[phase]), evidence)
            bound = bound_step(saved, intent["request_id"], intent_sha) == row[0]
        result = dict(status, rc=rc, original_step_bound=bound,
            passed=bool(rc == 0 and bound and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE
                        and allocated_dcu(row[6]) == (8 if phase == "gpu" else 0)))
        write_json(root / f"{phase}-accounting.json", result)
        return result

    admit()
    write_json(root / "controller-entered.json", {"request_id": intent["request_id"], "intent_SHA256": intent_sha})
    try:
        stages = arm_timeline(step, lambda rows: seal_accounting(root, intent_sha, {"columns": COLUMNS, "rows": rows}))
    except (OSError, ValueError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
        result = dict(passed=False, failure=type(error).__name__, reconcile_original=True)
    else:
        result = dict(stages=stages, passed=all(stages.get(phase, {}).get("passed") for phase in FILES))
    write_json(root / "controller-outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--intent-sha", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if controller(args.repo, args.root, args.intent_sha)["passed"] else 1)


if __name__ == "__main__":
    main()
