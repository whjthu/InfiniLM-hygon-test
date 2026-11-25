"""Run one Pack16 isolated-import check in an owned zero-GRES Slurm step."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE, POST = "1103606", "f12r2n10", "1103606.3407"
PYTHON = "/usr/local/python3.12/bin/python3"
OWNER = "0d0cea41f5124c719f122ffb481a12a3"
LABEL = "training-O01-pack16-r40-import0-" + OWNER[:12]
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
FILES = {
    "gate.py": "be7933c5d38561714509d26e33498e56af93ab650ea42f01aa1b55c11677f534",
    "step_evidence.py": "05113bd795355ac88851efbc6b870d9adfbbed3347746b4f2bb96267c810cf98",
    "SOURCE_FILES.json": "1bf431c00c0ea7b6e2a9eb5c7861b3ef0b8275cdd272cb782bba8f76bd1f438d",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(predicate, reason):
    if not predicate:
        raise ValueError(reason)


def lines(argv):
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=True, text=True)
    require(not proc.stderr, "ZERO_GRES_QUERY_STDERR")
    return proc.stdout.splitlines()


def accounting(job):
    rows = [row.split("|") for row in lines(["sacct", "-nP", "-j", job, "-o", COLUMNS])]
    require(all(len(row) == 7 for row in rows), "ZERO_GRES_ACCOUNTING_COLUMNS")
    return rows


def save(path, doc):
    raw = (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
    return sha(raw)


def main(envelope):
    require(set(envelope) == {"request", "publication", "files"}
            and re.fullmatch("[0-9a-f]{32}", envelope["request"])
            and re.fullmatch("[0-9a-f]{40}", envelope["publication"]), "ZERO_GRES_FIXED_ENVELOPE")
    require(set(envelope["files"]) == set(FILES), "ZERO_GRES_FROZEN_FILE_SET")
    payload = {name: base64.b64decode(value, validate=True) for name, value in envelope["files"].items()}
    require(all(sha(payload[name]) == value for name, value in FILES.items()), "ZERO_GRES_FROZEN_SOURCE_HASH")
    results = R / "results"
    require(results.resolve(strict=True) == results, "ZERO_GRES_CANONICAL_RESULTS")
    directory = results / LABEL
    with (results / ".training80-dcu-standby.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not os.path.lexists(directory), "ZERO_GRES_ORIGINAL_NAMESPACE_EXISTS")
        parents = [row.split("|", 2) for row in lines(["squeue", "-h", "-w", NODE, "-o", "%i|%T|%N"])]
        require(len(parents) == 1 and parents[0][:2] == [PARENT, "RUNNING"]
                and NODE in lines(["scontrol", "show", "hostnames", parents[0][2]]), "ZERO_GRES_OWNED_PARENT")
        post = [row for row in accounting(POST) if row[0] == POST]
        require(len(post) == 1 and post[0][2:4] == ["COMPLETED", "0:0"]
                and post[0][5] == NODE and "gres/" not in post[0][6], "ZERO_GRES_ORIGINAL_CLEAN_POST")
        require(not any(row[1] == LABEL for row in accounting(PARENT)), "ZERO_GRES_NUMERIC_LABEL_EXISTS")
        for row in lines(["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"]):
            handle, hosts = row.split("|")
            require(re.fullmatch(r"1103606\.(?:[0-9]+|batch|extern)", handle), "ZERO_GRES_UNKNOWN_STEP")
            if handle.rsplit(".", 1)[-1] not in ("batch", "extern"):
                require(NODE not in lines(["scontrol", "show", "hostnames", hosts]), "ZERO_GRES_NODE_HAS_NUMBERED_STEP")
        directory.mkdir(mode=0o700)
        for name, raw in payload.items():
            with (directory / name).open("xb") as out:
                out.write(raw)
        argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=4",
                "-N1", "-n1", "--nodelist=" + NODE, "--cpus-per-task=1", "--mem=1024M", "--gres=none",
                "--time=00:01:00", PYTHON, "-I", "-S", "-B", str(directory / "gate.py"), "--check-pinned-step-evidence"]
        identity = {"request": envelope["request"], "owner": OWNER, "label": LABEL,
                    "publication": envelope["publication"], "GPU_requested": 0, "replay_allowed": False}
        save(directory / "intent.json", identity | {"argv": argv, "recorded_epoch": time.time()})
        with (directory / "stdout").open("xb") as out, (directory / "stderr").open("xb") as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        save(directory / "srun.json", identity | {"client_pid": process.pid})
    try:
        rc = process.wait(timeout=23)
    except subprocess.TimeoutExpired:
        # Preserve the original request and scheduler identity for reconciliation.
        return identity | {"state": "ORIGINAL_OUTCOME_UNKNOWN", "complete": False}
    row = None
    for attempt in range(4):
        found = [x for x in accounting(PARENT) if x[1] == LABEL]
        require(len(found) <= 1, "ZERO_GRES_MULTIPLE_EXACT_LABEL")
        if found and found[0][2] not in ("RUNNING", "PENDING", "COMPLETING"):
            row = found[0]
            break
        time.sleep(.5)
    with (directory / "stdout").open("rb") as f:
        out = f.read(8193)
    with (directory / "stderr").open("rb") as f:
        err = f.read(8193)
    require(len(out) <= 8192 and len(err) <= 8192, "ZERO_GRES_LOG_BOUND")
    evidence = None
    if rc == 0:
        evidence = json.loads(out)
        require(evidence == {"request": OWNER, "helper_SHA256": FILES["step_evidence.py"], "check_only": True,
                             "scheduler_queries": 0, "worker_or_HIP_entry": False, "module_loaded": True,
                             "sys_path_unchanged": True}, "ZERO_GRES_EXACT_HELPER_CHECK")
    complete = bool(rc == 0 and row and re.fullmatch(r"1103606\.[0-9]+", row[0])
                    and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE and "gres/" not in row[6])
    result = identity | {"state": "COMPLETED_CHECK" if complete else "ORIGINAL_ACCOUNTING_OR_IMPORT_UNQUALIFIED",
                         "complete": complete, "srun_rc": rc, "accounting": row,
                         "stderr_SHA256": sha(err), "stdout_SHA256": sha(out), "import_evidence": evidence}
    return result | {"on_target_result_SHA256": save(directory / "result.json", result)}


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin)), sort_keys=True))
