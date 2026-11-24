"""Run one bounded CPU tokenizer pair beside the verified T08 standby."""

from hashlib import sha256
import json
from pathlib import Path
import shlex
import signal
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
CONTAINER = ROOT / "containers/dtk26-sandbox-r1"
RUNTIME = ROOT / "results/training-t06-sdxl-runtime-v1"
QWEN = Path("/public/home/aclsylqidf/shared/Qwen3-8B")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"


def run(claim, files):
    from base64 import b64decode

    helpers_raw = b64decode(files["controller_helpers.py"], validate=True)
    helper_pin = claim["payload_files"]["controller_helpers.py"]
    if sha256(helpers_raw).hexdigest() != helper_pin["SHA256"]:
        raise ValueError("qualified controller helpers differ")
    helpers = {"__name__": "qualified_cpu_controller_helpers"}
    exec(compile(helpers_raw, "<qualified_cpu_controller_helpers>", "exec"), helpers)
    data = helpers["get_files"](claim["payload_files"], files)
    accounting_scope = {"__name__": "qualified_slurm_accounting"}
    exec(compile(data["slurm_accounting.py"], "<qualified_slurm_accounting>", "exec"), accounting_scope)
    epoch, anchor = time.time(), time.monotonic()
    deadline, work_deadline = epoch + 300, epoch + 220
    label, parent, node = (claim[k] for k in ("label", "parent", "node"))
    if (label, parent, node, claim["cpu_cores"], claim["gpu_cards"], claim["memory_MiB"]) != (
            "training-hymotion-qwen3-tokenizer-cpu-v1", "1103606", "f13r2n09", 4, 0, 8192):
        raise ValueError("published CPU scope differs")

    def left(reserve=0):
        remaining = min(deadline - time.time(), 300 - (time.monotonic() - anchor)) - reserve
        if remaining <= 0:
            raise TimeoutError("the original CPU budget expired")
        return remaining

    def ask(*argv):
        return subprocess.check_output(argv, text=True, timeout=min(9, left(4))).strip()

    def original_row():
        rows = [line.split("|") for line in ask(
                "sacct", "-nP", "-j", parent, "--format=" + ",".join(helpers["ACCOUNTING"])).splitlines()]
        matched = [r for r in rows if len(r) == 7 and r[1] == label]
        if len(matched) > 1:
            raise ValueError("duplicate original accounting label")
        return matched[0] if matched else None

    work = ROOT / "results" / label
    if work.exists() or original_row():
        raise ValueError("original label or namespace already exists")
    for path in (QWEN, CONTAINER, RUNTIME / "site", Path(APPTAINER)):
        if not path.exists():
            raise ValueError("known target dependency path is absent")
    before = helpers["verify_live"](ask, claim)
    write = helpers["write_json"]
    source, evidence = work / "source", work / "evidence"
    work.mkdir(mode=0o700)
    source.mkdir()
    evidence.mkdir()
    for name, raw in data.items():
        if name not in ("controller_helpers.py", "slurm_accounting.py"):
            with (source / name).open("xb") as stream:
                stream.write(raw)
    write(source / "CLAIM.json", claim)
    write(work / "intent.json", dict(request=claim["request"], epoch=epoch, deadline=deadline,
                                      claim_SHA256=CLAIM_SHA256, standby_before=before))
    print(json.dumps(dict(request=claim["request"], event="REMOTE_INTENT_CREATED", epoch=epoch)), flush=True)
    shell = (
        "set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
        "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/training-runtime/site; "
        "LEFT=$(python3 -B -c " + shlex.quote(f"import time;print(int({work_deadline}-time.time()))") + "); "
        'test "$LEFT" -gt 15; exec /usr/bin/timeout --kill-after=5s "$LEFT" python3 -B -u /input-stage/inside.py'
    )
    apptainer = [APPTAINER, "exec", "--no-home", "--pwd", "/",
                 "--bind", f"{source}:/input-stage:ro", "--bind", f"{evidence}:/evidence:rw",
                 "--bind", f"{QWEN}:/input-qwen:ro", "--bind", f"{RUNTIME}:/training-runtime:ro",
                 "--bind", "/opt/hyhal:/opt/hyhal:ro", str(CONTAINER), "/bin/bash", "--noprofile", "--norc", "-c", shell]
    argv = ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--nodes=1", "--ntasks=1",
            "--nodelist=" + node, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--immediate=5", "--time=4", *apptainer]
    with (work / "CPU.log").open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            child.wait(timeout=left(38))
        except subprocess.TimeoutExpired:
            timed_out = True
            import os
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if sig == signal.SIGKILL:
                        raise
    terminal = accounting_scope["wait_terminal"](original_row, time.monotonic() + min(17, left(12)),
                                                  interval=.6, max_observations=25)
    row = terminal["row"]
    try:
        after = helpers["verify_live"](ask, claim)
    except (OSError, ValueError, subprocess.SubprocessError):
        after = None
    value = None
    p = evidence / "CPU-result.json"
    if p.is_file():
        with p.open("rb") as stream:
            raw = stream.read(12289)
        if 0 < len(raw) <= 12288:
            value = dict(SHA256=sha256(raw).hexdigest(), value=json.loads(raw))
    passed = bool(row and row[2:4] == ["COMPLETED", "0:0"] and row[5] == node and "gres/" not in row[6]
                  and child.returncode == 0 and not timed_out and before == after
                  and value and value["value"]["actual_step"] == row[0] and value["value"]["passed"])
    report = dict(request=claim["request"], passed=passed, accounting_columns=helpers["ACCOUNTING"], terminal=terminal,
                  CPU_result=value, standby_before=before, standby_after=after,
                  child_returncode=child.returncode, timed_out=timed_out, GPU_cards=0)
    write(work / "outcome.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(run(CLAIM, FILES))
