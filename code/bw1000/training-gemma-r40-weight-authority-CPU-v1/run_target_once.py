"""One zero-GRES run of the corrected Gemma asset gate in the production image."""
import base64
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE, POST = "1103606", "f13r1n12", "1103606.3418"
LABEL = "training-gemma-r40-weight-authority-zero-GRES-v1"
FILES = {
    "assets.py": "2cd0c3a463d2efeebd6d41fb05308f87da998380b9d305ee5b3c07825a90626b",
    "check_target.py": "8af09606811774e18f98345db4d40696e587e2910d03931839bb827ca8d36dc1",
    "SOURCE_FILES.json": "8028e49104a3cc546e88a57828c0c7b7b5fb3365557bbfc4382a3a8465e1523d",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def query(*command):
    done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5, check=True)
    require(not done.stderr, "R40_WEIGHT_QUERY_STDERR")
    return done.stdout.splitlines()


def accounting(job):
    fields = "JobIDRaw,JobName%110,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
    rows = [line.split("|") for line in query("sacct", "-nP", "-j", job, "-o", fields)]
    require(all(len(row) == 7 for row in rows), "R40_WEIGHT_EXACT_ACCOUNTING_COLUMNS")
    return rows


def persist(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return sha256(raw).hexdigest()


def compile_argv(directory, root, epoch):
    sandbox = R / "containers/dtk26-sandbox-r1"
    original = root / "training-gemma-usgs-r39-two-updates-native-v1/src"
    authority = root / "training-gemma-usgs-r39-two-updates-v1/6d14e9df9832432b97f1f46f032b49f3/authority"
    model = root / "training-gemma-original-model-v1"
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5", "-N1", "-n1",
            "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--time=00:02:00",
            "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--no-home", "--pwd", "/"]
    for path in (directory, original, authority, model):
        require(path.resolve(strict=True) == path, "R40_WEIGHT_CANONICAL_BIND")
        argv.extend(("--bind", f"{path}:{path}:ro"))
    command = ["python3", "-I", "-S", "-B", str(directory / "check_target.py"), "--epoch", str(epoch)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; "
             "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; exec " + shlex.join(command))
    return argv + [str(sandbox), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def main(envelope):
    require(set(envelope) == {"request", "publication", "files"}
            and re.fullmatch("[0-9a-f]{32}", envelope["request"])
            and re.fullmatch("[0-9a-f]{40}", envelope["publication"]), "R40_WEIGHT_EXACT_ENVELOPE")
    require(set(envelope["files"]) == set(FILES), "R40_WEIGHT_EXACT_SOURCE_SET")
    content = {key: base64.b64decode(value, validate=True) for key, value in envelope["files"].items()}
    require(all(sha256(content[key]).hexdigest() == value for key, value in FILES.items()), "R40_WEIGHT_EXACT_SOURCE_BYTES")
    root = R / "results"
    directory = root / LABEL
    with (root / ".training80-dcu-standby.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(root.resolve(strict=True) == root and not os.path.lexists(directory), "R40_WEIGHT_ORIGINAL_NAMESPACE")
        rows = [line.split("|", 2) for line in query("squeue", "-h", "-w", NODE, "-o", "%i|%T|%N")]
        require(len(rows) == 1 and rows[0][:2] == [PARENT, "RUNNING"]
                and NODE in query("scontrol", "show", "hostnames", rows[0][2]), "R40_WEIGHT_CURRENT_OWNED_PARENT")
        post = [row for row in accounting(POST) if row[0] == POST]
        require(len(post) == 1 and post[0][2:4] == ["COMPLETED", "0:0"]
                and post[0][5] == NODE and "gres/" not in post[0][6], "R40_WEIGHT_PRIOR_INDEPENDENT_POST")
        require(not any(row[1] == LABEL for row in accounting(PARENT)), "R40_WEIGHT_ACCOUNTING_EXISTS")
        for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
            handle, hosts = line.split("|")
            require(re.fullmatch(r"1103606\.(?:[0-9]+|batch|extern)", handle), "R40_WEIGHT_UNKNOWN_STEP")
            if handle.rsplit(".", 1)[-1].isdecimal():
                require(NODE not in query("scontrol", "show", "hostnames", hosts), "R40_WEIGHT_NUMBERED_OWNER_PRESENT")
        directory.mkdir(mode=0o700)
        for name, raw in content.items():
            with (directory / name).open("xb") as handle:
                handle.write(raw)
        epoch = time.time()
        argv = compile_argv(directory, root, epoch)
        identity = {"request": envelope["request"], "publication": envelope["publication"], "label": LABEL,
                    "epoch": epoch, "GPU_requested": 0, "replay_allowed": False}
        persist(directory / "intent.json", identity | {"argv": argv})
        with (directory / "stdout").open("xb") as out, (directory / "stderr").open("xb") as err:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        persist(directory / "srun.json", identity | {"client_pid": proc.pid})
    try:
        rc = proc.wait(timeout=38)
    except subprocess.TimeoutExpired:
        return identity | {"state": "ORIGINAL_OUTCOME_UNKNOWN"}
    result = identity | {"srun_rc": rc}
    for _ in range(4):
        exact = [row for row in accounting(PARENT) if row[1] == LABEL]
        require(len(exact) <= 1, "R40_WEIGHT_DUPLICATE_NUMERIC_SCOPE")
        if exact and exact[0][2] not in ("PENDING", "RUNNING", "COMPLETING"):
            result["accounting"] = exact[0]
            break
        time.sleep(.5)
    with (directory / "stdout").open("rb") as out, (directory / "stderr").open("rb") as err:
        raw_out, raw_err = out.read(8193), err.read(8193)
    require(max(len(raw_out), len(raw_err)) <= 8192, "R40_WEIGHT_EXACT_OUTPUT_BOUND")
    result.update(stdout_SHA256=sha256(raw_out).hexdigest(), stderr_SHA256=sha256(raw_err).hexdigest())
    try:
        report = json.loads(raw_out) if raw_out else {}
    except ValueError:
        report = {}
    keys = ("passed", "step", "node", "epoch", "CPU", "target_GPU", "native_same_entry", "members",
            "historic_full_file_shas_bound", "new_shard_payload_bytes_rehashed", "actual_model_loaded",
            "optimizer_updates", "current_assets_identity_SHA256", "predicate", "error_class")
    report = {key: report[key] for key in keys if key in report}
    row = result.get("accounting")
    passed = bool(rc == 0 and row and row[0] == report.get("step") and row[2:4] == ["COMPLETED", "0:0"]
                  and row[5] == NODE and "gres/" not in row[6] and "cpu=4" in row[6].split(",")
                  and report.get("passed") is True and report.get("historic_full_file_shas_bound") is True
                  and report.get("members") == 16 and report.get("new_shard_payload_bytes_rehashed") == 0
                  and report.get("optimizer_updates") == 0)
    final = result | {"state": "COMPLETED_CHECK" if passed else "ORIGINAL_TERMINAL_OR_GATE_UNQUALIFIED", "report": report}
    return final | {"on_target_result_SHA256": persist(directory / "result.json", final)}


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin)), sort_keys=True))
