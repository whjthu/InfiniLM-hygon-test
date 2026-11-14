"""Materialize the pinned framework and execute one bounded CPU step."""
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
LABEL = "training-qwen-a68952c-swift-initial-CPU-v2"
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
    require(set(payload) == {"request","claim"}, "Unexpected payload")
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
    intent = (prior/"intent.json").read_bytes()
    framework = (prior/"framework.json").read_bytes()
    require(hashlib.sha256(intent).hexdigest() ==
            "2247c1fda3f251bba90e38ce558d65448d81734f4a61e50a54ac25cb90b6c48f" and
            hashlib.sha256(framework).hexdigest() ==
            "ff3d901135d0e94e03931a293329eae53d2db5e1941ba650985f74b0471eb643",
            "Original materialization identity differs")
    require(not (prior/"owned-srun.json").exists() and not (prior/"CPU.log").exists()
            and "|"+prior.name+"|" not in query("sacct", "-n", "-P", "-j", PARENT,
            "--format=JobIDRaw,JobName%100,State"), "Original CPU submit requires reconciliation")
    runtime = prior/"framework"
    assets = prior/"source"
    manifest = (assets/"FRAMEWORK_FILES.json").read_bytes()
    require(hashlib.sha256(manifest).hexdigest() ==
            "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
            "Framework manifest differs")
    files = json.loads(manifest)["files"]
    require(len(files)==393 and {p.relative_to(runtime).as_posix() for p in runtime.rglob("*")
            if p.is_file()} == set(files), "Staged framework inventory differs")
    for name, pin in files.items():
        path = runtime/name
        require(not path.is_symlink() and path.stat().st_size == pin["bytes"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == pin["SHA256"], "Framework differs: "+name)
    expected = {
        "compute.py":"bacfcfbcf950acaa345d52dbec461bd1da2a55e17b4887741ac2fff1c7affec7",
        "recipe.py":"c251e21dbaa7d4258add5275f1a959af2536128cac7d86374d1bf7de83bb5806",
        "original-launch.json":"28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400",
        "QWEN_TARGET_NATIVE_BINDING.json":"9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067",
    }
    for name, digest in expected.items():
        path = assets/name
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest()==digest, "Input differs: "+name)
    evidence = W/"evidence"
    evidence.mkdir()
    write("framework.json",json.loads(framework))
    require(time.monotonic() - started < 55 and time.time() < epoch+55, "Admission budget exhausted")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 "
        "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; "
        "export PYTHONPATH=/framework:/minicpm-transformers:/swift-runtime/site QWEN_CPU_DEADLINE="
        + shlex.quote(str(epoch+230)) + "; exec python3 -B -u /stage/compute.py")
    command = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    binds = [(assets, "/stage", "ro"), (runtime, "/framework", "ro"), (evidence, "/evidence", "rw"),
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
