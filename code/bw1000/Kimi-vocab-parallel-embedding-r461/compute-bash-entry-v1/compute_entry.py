"""Isolate Apptainer-to-Bash entry on a Kimi compute node with no model work."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
D = Path(__file__).resolve().parent
OLD = "/public/home/aclsylqidf/hygon-inference-20261013"
SHARED = "/public/home/aclsylqidf/shared"
APP = "/public/software/apps/apptainer/1.3.4/bin/apptainer"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    raw = (D / "CPU_RESOURCE_CLAIM.json").read_bytes()
    claim_sha = sha256(raw).hexdigest()
    assert claim_sha == os.environ["KIMI_COMPUTE_ENTRY_CLAIM_SHA"]
    q = json.loads(raw)
    assert q["CPU_GO"] and q["once"] and not q["GPU_GO"] and not q["Model_GO"]
    assert digest(Path(__file__)) == q["entry_sha256"]
    assert (os.environ["SLURM_JOB_ID"], socket.gethostname().split(".")[0]) == (q["parent_job"], q["node"])
    assert all(os.environ[k] == "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    out = R / "results" / q["label"]
    out.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_")) and k not in ("LD_PRELOAD", "BASH_ENV", "ENV")}
    argv = [APP, "exec", "--home", OLD + "/build-deps/root:/root",
            "--bind", str(R) + ":/refactor", "--bind", SHARED + ":/shared:ro",
            "--bind", OLD + ":/old:ro", "--bind", "/dev/shm:/local",
            str(R / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
            "-c", "printf '%s\\n' KIMI_COMPUTE_BASH_ENTRY_OK"]
    start = time.monotonic()
    timed_out = False
    signals = []
    with (out / "stdout.txt").open("xb") as stdout, (out / "stderr.txt").open("xb") as stderr:
        p = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                             start_new_session=True)
        try:
            p.wait(timeout=q["work_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not alive(p.pid):
                break
            try:
                os.killpg(p.pid, sig)
                signals.append(sig.name)
            except ProcessLookupError:
                break
            end = min(start + q["inside_seconds"], time.monotonic() + 4)
            while time.monotonic() < end:
                p.poll()
                if not alive(p.pid):
                    break
                time.sleep(0.05)
    gone = not alive(p.pid)
    mark = (out / "stdout.txt").read_bytes() == b"KIMI_COMPUTE_BASH_ENTRY_OK\n"
    passed = mark and p.poll() == 0 and gone and not timed_out and not signals
    result = dict(passed=passed, timeout=timed_out, bash_entry_observed=mark, parent_rc=p.poll(),
                  process_group_gone=gone, signals=signals, elapsed_s=time.monotonic()-start,
                  actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  claim_sha256=claim_sha, child_PID=p.pid, node=q["node"],
                  stdout_sha256=digest(out / "stdout.txt"), stderr_sha256=digest(out / "stderr.txt"),
                  GPU=0, SDK_calls=0, checkpoint_header_reads=0,
                  UTC=datetime.now(timezone.utc).isoformat())
    with (out / "result.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
