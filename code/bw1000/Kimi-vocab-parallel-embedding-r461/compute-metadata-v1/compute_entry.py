"""Run the immutable Kimi metadata payload on the verified compute-node route."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
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
    masks = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    # Slurm's zero-GRES step may remove empty inherited visibility variables.
    assert not os.environ.get("SLURM_STEP_GPUS")
    assert all(os.environ.get(name, "") in ("", "-1", "NoDevFiles") for name in masks)
    prior = R / "results/Kimi461-compute-bash-mask-CPU-v2/result.json"
    assert digest(prior) == q["compute_bash_receipt_sha256"]
    namespace = R / "incoming/Kimi-vocab-parallel-embedding-r461"
    metadata = namespace / "metadata-exit-v3"
    device = namespace / "device-v1"
    for package, pin in ((metadata, q["metadata_manifest_sha256"]), (device, q["device_manifest_sha256"])):
        mf = package / "SOURCE_MANIFEST.json"
        assert digest(mf) == pin
        for name, expected in json.loads(mf.read_bytes())["files_SHA256"].items():
            item = package / name
            assert item.resolve().is_relative_to(package.resolve()) and digest(item) == expected
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(metadata))
    from diagnose_once import summarize_exit
    out = R / "results" / q["label"]
    out.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_")) and k not in ("LD_PRELOAD", "BASH_ENV", "ENV")}
    env.update({name: "" for name in masks})
    env["K461_METADATA_ONLY"] = "1"
    argv = [APP, "exec", "--home", OLD + "/build-deps/root:/root",
            "--bind", str(R) + ":/refactor", "--bind", SHARED + ":/shared:ro",
            "--bind", OLD + ":/old:ro", "--bind", "/dev/shm:/local",
            str(R / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
            str(Path("/refactor") / metadata.relative_to(R) / "metadata_inside.sh")]
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
    result = dict(timeout=timed_out, parent_rc=p.poll(), group_gone=gone, signals=signals,
                  terminal_UNKNOWN=(not gone or p.poll() is None), elapsed_s=time.monotonic()-start,
                  actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  claim_sha256=claim_sha, child_PID=p.pid, node=q["node"],
                  stdout_sha256=digest(out / "stdout.txt"), stderr_sha256=digest(out / "stderr.txt"),
                  GPU=0, metadata_manifest_sha256=q["metadata_manifest_sha256"],
                  explicit_Native_or_SDK_calls=0,
                  original_LD_PRELOAD_mapping_or_initialization="UNKNOWN",
                  UTC=datetime.now(timezone.utc).isoformat())
    stdout = (out / "stdout.txt").read_bytes()
    stderr = (out / "stderr.txt").read_bytes()
    assert len(stdout) <= 262144 and len(stderr) <= 262144
    result["exit_boundary"] = summarize_exit(stdout, stderr, result)
    result["bash_entry_observed"] = any(line.startswith(b"META_PHASE bash.entry ") for line in stderr.splitlines())
    passed = result["bash_entry_observed"] and result["exit_boundary"]["passed"]
    result["passed"] = passed
    with (out / "result.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
