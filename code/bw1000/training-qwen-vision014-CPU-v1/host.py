"""Materialize the pinned framework and execute one bounded CPU step."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-qwen-vision014-CPU-v1"
PARENT = "1087656"
NODE = "a01r4n16"
W = R / "results" / LABEL


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(name, value):
    with (W / name).open("x") as f:
        json.dump(value, f, indent=2)


def query(*argv):
    return subprocess.check_output(argv, timeout=10, text=True).strip()


def accounting():
    return [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT,
        "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList").splitlines()
        if "|"+LABEL+"|" in line]


def main():
    epoch = time.time()
    started = time.monotonic()
    payload = json.load(sys.stdin)
    require(set(payload) == {"request","claim","compute","vision"}, "Unexpected payload")
    require(not W.exists() and not accounting(), "Existing scope requires reconciliation")
    state, count, hosts = query("squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N").split("|")
    nodes = query("scontrol", "show", "hostnames", hosts).splitlines()
    require(state == "RUNNING" and int(count) == len(set(nodes)) == 40 and NODE in nodes,
            "Frozen CPU node is unavailable")
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N").splitlines():
        handle, hosts = line.split("|")
        if not handle.endswith((".batch", ".extern")) and hosts not in ("", "(null)"):
            require(NODE not in query("scontrol", "show", "hostnames", hosts).splitlines(),
                    "CPU node has an active step")
    W.mkdir(mode=0o700)
    write("intent.json", dict(epoch=epoch, deadline=epoch+290, request=payload["request"],
        claim=payload["claim"], pid=os.getpid(), parent=PARENT, node=NODE, GPU_cards=0))
    prior = R/"results/training-qwen-a68952c-swift-initial-CPU-v1"
    assets = prior/"source"
    runtime = prior/"framework"
    original = R/"results/training-qwen-native25-hip-v2/outcome.json"
    require(hashlib.sha256(original.read_bytes()).hexdigest() ==
            "92ee1d4292e379413b326b936fafa36fc70d4410a024e547d9d1f33c8e95d1ee",
            "Original GPU terminal differs")
    pins = VISION_PINS
    require(set(payload["vision"]) == set(pins), "Vision inventory differs")
    vision = W/"vision"
    for name, pin in pins.items():
        path = vision/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(payload["vision"][name], validate=True)
        require(len(data)==pin["bytes"] and hashlib.sha256(data).hexdigest()==pin["SHA256"], name)
        with path.open("xb") as f:
            f.write(data)
    fix = W/"fix"
    fix.mkdir()
    compute = base64.b64decode(payload["compute"], validate=True)
    require(hashlib.sha256(compute).hexdigest()=="5ac3225d0950bf2538a7fb3b7f9e2907321d47d51754231a7a67d29d4d6483e5", "Compute payload SHA differs")
    with (fix/"compute.py").open("xb") as f:
        f.write(compute)
    evidence = W/"evidence"
    evidence.mkdir()
    require(time.monotonic() - started < 55 and time.time() < epoch+55, "Admission budget exhausted")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 "
        "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/vision:/framework:/minicpm-transformers:/swift-runtime/site QWEN_CPU_DEADLINE="
        + shlex.quote(str(epoch+230)) + " QWEN_CPU_EPOCH=" + shlex.quote(str(epoch))
        + "; exec python3 -B -u /fix/compute.py")
    command = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    binds = [(vision, "/vision", "ro"), (fix, "/fix", "ro"), (assets, "/stage", "ro"), (runtime, "/framework", "ro"), (evidence, "/evidence", "rw"),
        (R/"results/training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
        (R/"results/training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro")]
    for relative in ("training-qwen-original-public-model-v1/model",
                     "training-qwen-initial-trainables-v1", "training-vlm-original-assets-v1"):
        source = R/"results"/relative
        binds.append((source, str(source), "ro"))
    binds.append((Path("/opt/hyhal"), "/opt/hyhal", "ro"))
    for source, target, mode in binds:
        command += ["--bind", f"{source}:{target}:{mode}"]
    command += [str(R/"containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]
    step = ["srun", "--jobid="+PARENT, "--job-name="+LABEL, "--exact", "--nodes=1", "--ntasks=1",
        "--nodelist="+NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none",
        "--immediate=5", "--time=4", *command]
    forced = False
    with (W/"CPU.log").open("xb") as log:
        p = subprocess.Popen(step, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            stat = Path(f"/proc/{p.pid}/stat").read_text().split(") ",1)[1].split()
            write("owned-srun.json", {"pid":p.pid, "pgid":int(stat[2]), "startticks":int(stat[19])})
            p.wait(timeout=max(1,min(epoch+242-time.time(),242-(time.monotonic()-started))))
        finally:
            if p.poll() is None:
                forced = True
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)
                except ProcessLookupError:
                    p.wait(timeout=5)
            rows = []
            while time.monotonic()-started < 265 and time.time() < epoch+265:
                rows = accounting()
                if rows and rows[0][2] not in ("RUNNING","PENDING","COMPLETING"):
                    break
                time.sleep(0.5)
            receipt = evidence/"compute.json"
            compute = json.loads(receipt.read_bytes()) if receipt.is_file() else None
            passed = bool(compute and len(rows)==1 and rows[0][0]==compute["step"] and
                          rows[0][2:4]==["COMPLETED","0:0"] and p.returncode==0 and not forced)
            value = dict(passed=passed, rows=rows, rc=p.returncode, forced=forced, compute=compute,
                         log_SHA256=hashlib.sha256((W/"CPU.log").read_bytes()).hexdigest(),
                         request=payload["request"], claim=payload["claim"])
            write("outcome.json",value)
            print(json.dumps(value),flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
