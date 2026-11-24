"""Install one pinned public source bundle and run one owned CPU-only decoder check."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "trellis-upstream75fb-cpu-decode-source-v1"
STAGE = BASE / "results" / LABEL
WEIGHTS = BASE / "results/trellis-native-generation-microsoft-v1"
PROOF = "a518b396b411c54aabb59905f0faf0782a79b6455b5495d5c22ac2c7dcec12d1"
PARENT, NODE = "1103606", "f13r4n05"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(raw, pin):
    require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "source payload identity differs")


def save(name, value):
    with (STAGE / name).open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def query(*args):
    return subprocess.check_output(args, text=True, timeout=8).strip()


def accounting():
    lines = query("sacct", "-n", "-P", "-j", PARENT, "--format=" + COLUMNS).splitlines()
    return [row for line in lines if len(row := line.split("|")) == 7 and row[1] == LABEL]


def git_at(root, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
           "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    command = ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false",
               "-c", "submodule.recurse=false", *args]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=26)
    if result.returncode:
        raise RuntimeError(f"git {args[0]} exit={result.returncode}")
    return result.stdout.strip()


def checkout(bundle, proof):
    root = STAGE / "checkout"
    root.mkdir()
    git_at(root, "init", "--quiet", "--template=")
    source = proof["source_commit"]
    (root / ".git/shallow").write_text(source + "\n", encoding="ascii")
    git_at(root, "bundle", "unbundle", str(bundle))
    git_at(root, "checkout", "--detach", "--quiet", source)
    git_at(root, "fsck", "--full", "--no-reflogs")
    require(git_at(root, "rev-parse", "HEAD") == source and
            git_at(root, "rev-parse", "HEAD^{tree}") == proof["tree"] and
            git_at(root, "rev-parse", "--is-shallow-repository") == "true" and
            not git_at(root, "status", "--porcelain", "--untracked-files=normal"), "checkout identity differs")
    for name, pin in proof["files"].items():
        checked((root / name).read_bytes(), pin)
    return root


def execute_cpu(root, end):
    script = (
        "set -euo pipefail; source /opt/dtk/env.sh; "
        "export PYTHONNOUSERSITE=1 PYTHONPATH=/candidate/src TORCH_DEVICE_BACKEND_AUTOLOAD=0 "
        "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null; "
        "exec python3 -B -u /candidate/decoder_construct.py"
    )
    allowance = min(190, int(end - time.monotonic()) - 25)
    require(allowance >= 60, "insufficient original CPU budget")
    command = [
        "srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--nodes=1", "--ntasks=1",
        "--nodelist=" + NODE, "--cpus-per-task=2", "--mem=8192M", "--gres=none", "--exact",
        "--overlap", "--immediate=5", "--time=00:04:00", "/public/software/apps/apptainer/1.3.4/bin/apptainer",
        "exec", "--no-home", "--pwd", "/", "--bind", str(STAGE) + ":/candidate:ro",
        "--bind", str(root) + ":/checkout:ro", "--bind", str(WEIGHTS / "model") + ":/trellis-weights:ro",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(BASE / "containers/dtk26-sandbox-r1"),
        "/usr/bin/timeout", "-k", "5", str(allowance), "/bin/bash", "--noprofile", "--norc", "-c", script,
    ]
    forced = False
    with (STAGE / "CPU.log").open("xb") as out:
        worker = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        proc = Path("/proc", str(worker.pid), "stat").read_text().split(") ", 1)[1].split()
        save("cpu-intent.json", {"PID": worker.pid, "startticks": int(proc[19]), "node": NODE, "parent": PARENT})
        try:
            worker.wait(timeout=allowance + 10)
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=4)
    rows = []
    while time.monotonic() + 9 < end and len(rows) < 6:
        try:
            observation = accounting()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            observation = []
        rows.append(observation)
        if len(observation) == 1 and observation[0][2].split()[0] in TERMINAL:
            break
        time.sleep(1)
    body = (STAGE / "CPU.log").read_bytes()
    prefix = b"TRELLIS_PINNED_DECODER_RECEIPT="
    matches = [line[len(prefix):] for line in body.splitlines() if line.startswith(prefix)]
    receipt = json.loads(matches[0]) if len(matches) == 1 else None
    actual = rows[-1] if rows else []
    success = (worker.returncode == 0 and not forced and len(actual) == 1 and actual[0][2:4] == ["COMPLETED", "0:0"]
               and actual[0][5] == NODE and "gres/" not in actual[0][6] and receipt is not None
               and receipt["step"] == actual[0][0])
    return {"CPU_passed": success, "CPU_rc": worker.returncode, "force_timeout": forced,
            "accounting_columns": COLUMNS, "accounting_observations": rows, "log_SHA256": sha256(body).hexdigest(),
            "decoder_receipt": receipt, "error_tail": None if success else body[-2200:].decode(errors="replace")}


def main():
    start = time.monotonic()
    epoch = time.time()
    end = start + 270
    require(len(sys.argv) == 2 and re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]), "publication is required")
    message = json.loads(sys.stdin.buffer.readline(512 * 1024 + 1))
    raw = base64.b64decode(message["proof"], validate=True)
    require(sha256(raw).hexdigest() == PROOF, "source proof differs")
    proof = json.loads(raw)
    paths = {**{"src/" + k: v for k, v in proof["framework_source_pins"].items()},
             "decoder_construct.py": proof["driver_pin"]}
    require(set(paths) == set(message["files"]), "framework runtime source set differs")
    payload = {k: base64.b64decode(message["files"][k], validate=True) for k in paths}
    for name, data in payload.items():
        checked(data, paths[name])
    require(sha256((WEIGHTS / "outcome.json").read_bytes()).hexdigest() ==
            "3eee0c16539c8946bd5256d4028cabc5263dbe3409fde503d1a2d63a9138d362", "decoder weight stage differs")
    current = query("squeue", "-h", "-j", PARENT, "-o", "%i|%T|%D|%N").split("|")
    owner = re.search(r"(?:^|\s)UserId=[^()\s]+\(([0-9]+)\)", query("scontrol", "show", "job", PARENT, "--oneliner"))
    require(len(current) == 4 and current[:3] == [PARENT, "RUNNING", "40"] and owner is not None
            and int(owner[1]) == os.geteuid(), "owned parent differs")
    require(NODE in query("scontrol", "show", "hostnames", current[3]).splitlines() and not accounting(),
            "node membership or prior step differs")
    STAGE.mkdir(mode=0o700)
    save("intent.json", {"epoch": epoch, "deadline_epoch": epoch + 270, "publication": sys.argv[1],
                         "proof_SHA256": PROOF, "bundle": proof["bundle"], "GPU": 0})
    result = {"publication": sys.argv[1], "proof_SHA256": PROOF, "source_verified": False, "CPU_passed": False}
    try:
        bundle = STAGE / "upstream.bundle"
        data = sys.stdin.buffer.read(proof["bundle"]["bytes"] + 1)
        checked(data, proof["bundle"])
        with bundle.open("xb") as stream:
            stream.write(data)
        del data
        root = checkout(bundle, proof)
        result.update(source_verified=True, source_commit=proof["source_commit"], source_tree=proof["tree"])
        save("source-verified.json", {"commit": proof["source_commit"], "tree": proof["tree"], "files": proof["files"]})
        for name, data in payload.items():
            file = STAGE / name
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.open("xb") as stream:
                stream.write(data)
        result.update(execute_cpu(root, end))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)[:240]
    result["elapsed_seconds"] = time.monotonic() - start
    save("outcome.json", result)
    print(json.dumps(result))
    return 0 if result["CPU_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
