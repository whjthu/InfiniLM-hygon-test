"""Run one real zero-GRES Dataset.map and Unix socket lifecycle on the production image."""
import base64
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1103606"
NODE = "f13r1n12"
LABEL = "training-gemma-r42-live-empty-pool-retirement-zero-gres-v2"
SCOPE = "95f9932f11bc48f49bdab17953da995b"
SOURCE_SHA = "3dcb33a44f412771dbad36c2cef2c1f9a0bc38cde6f7fff38679c463cec7c8e4"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"


def require(value, code):
    if not value:
        raise ValueError(code)


def query(*argv):
    result = subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL, timeout=5, check=True)
    require(not result.stderr and len(result.stdout) <= 1 << 20, "G42_HOST_SCHEDULER_QUERY_BOUND")
    return result.stdout.decode("ascii").splitlines()


def rows(job):
    result = [line.split("|") for line in query("sacct", "-nP", "-j", job, "-o", COLUMNS)]
    require(all(len(row) == 7 for row in result), "G42_HOST_ACCOUNTING_COLUMNS")
    return result


def nodes(value):
    if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", value):
        return [value]
    return query("scontrol", "show", "hostnames", value)


def write(path, doc):
    raw = (json.dumps(doc, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def other_records():
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        handle, hostlist = line.split("|")
        require(re.fullmatch(PARENT + r"\.(?:[0-9]+|batch|extern|interactive)", handle), "G42_HOST_UNKNOWN_STEP")
        if handle.rsplit(".", 1)[1].isdecimal():
            require(NODE not in nodes(hostlist), "G42_HOST_NODE_HAS_AN_ACTIVE_STEP")
    for line in query("scontrol", "show", "step", PARENT, "-o"):
        entry = dict(token.split("=", 1) for token in line.split() if "=" in token)
        handle = entry.get("StepId", "")
        require(re.fullmatch(PARENT + r"\.(?:[0-9]+|batch|extern|interactive)", handle), "G42_HOST_UNKNOWN_CONTROLLER_STEP")
        if handle.rsplit(".", 1)[1].isdecimal() and entry.get("State") not in (
                "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            require(NODE not in nodes(entry["NodeList"]), "G42_HOST_NODE_HAS_AN_ACTIVE_CONTROLLER_STEP")


def main(data):
    require(set(data) == {"request", "publication", "source"}
            and re.fullmatch("[0-9a-f]{32}", data["request"])
            and re.fullmatch("[0-9a-f]{40}", data["publication"]), "G42_HOST_INVALID_ENVELOPE")
    source = base64.b64decode(data["source"], validate=True)
    require(len(source) == 14741 and sha256(source).hexdigest() == SOURCE_SHA, "G42_HOST_FROZEN_SOURCE")
    directory = ROOT / LABEL
    with (ROOT / ".training80-dcu-standby.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(ROOT.resolve(strict=True) == ROOT and not os.path.lexists(directory), "G42_HOST_EXACT_NEW_NAMESPACE")
        parent = [x.split("|") for x in query("squeue", "-h", "-w", NODE, "-o", "%i|%T|%N")]
        require(len(parent) == 1 and parent[0][:2] == [PARENT, "RUNNING"]
                and NODE in nodes(parent[0][2]), "G42_HOST_CURRENT_OWNED_PARENT")
        prior = [row for row in rows("1103606.3436") if row[0] == "1103606.3436"]
        require(len(prior) == 1 and prior[0][2:4] == ["COMPLETED", "0:0"] and prior[0][5] == NODE
                and "gres/" not in prior[0][6], "G42_HOST_PRIOR_INDEPENDENT_POST")
        earlier = [row for row in rows("1103606.3438") if row[0] == "1103606.3438"]
        require(len(earlier) == 1 and earlier[0][2:4] == ["FAILED", "1:0"]
                and earlier[0][5] == NODE and "gres/" not in earlier[0][6], "G42_HOST_PRIOR_CPU_TERMINAL")
        require(not any(row[1] == LABEL for row in rows(PARENT)), "G42_HOST_PRIOR_CURRENT_LABEL")
        other_records()
        directory.mkdir(mode=0o700)
        member = directory / "short_tmp.py"
        with member.open("xb") as output:
            output.write(source)
        scratch = directory / "socket-scratch"
        scratch.mkdir(mode=0o700)
        os.chmod(scratch, 0o700)
        marker = scratch / ".scope"
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write((SCOPE + "\n").encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        inode = scratch.stat().st_ino
        swift = ROOT / "training-vlm-swift-offline-runtime-v1"
        processor = ROOT / "training-minicpm-hygon-processor-r2/cpu/runtime-site"
        framework = ROOT / "training-gemma-usgs72-fresh-initial-domain25-v1/framework"
        hyhal = Path("/opt/hyhal")
        for path in (swift, processor, framework, scratch, member):
            require(path.resolve(strict=True) == path, "G42_HOST_CANONICAL_MOUNT")
        require(hyhal.is_dir(), "G42_HOST_ACTUAL_HYHAL")
        epoch = time.time()
        identity = {key: data[key] for key in ("request", "publication")} | {
            "label": LABEL, "epoch": epoch, "GPU_requested": 0, "replay_allowed": False}
        argv = ["srun", "--jobid=" + PARENT, "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
                "--nodelist=" + NODE, "--cpus-per-task=4", "--threads-per-core=1", "--cpu-bind=threads",
                "--mem=8192M", "--gres=none", "--time=00:02:00", "--job-name=" + LABEL,
                "/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--contain", "--no-home", "--pwd", "/"]
        for path, alias, mode in ((scratch, "/tmp/g42", "rw"), (member, "/g42-short-tmp.py", "ro"),
                                  (swift, "/swift-runtime", "ro"), (processor, "/minicpm-transformers", "ro"),
                                  (framework, str(framework), "ro"), (hyhal, "/opt/hyhal", "ro")):
            argv.extend(("--bind", f"{path}:{alias}:{mode}"))
        command = ["python3", "-B", "/g42-short-tmp.py", "--epoch", str(epoch),
                   "--scope", SCOPE, "--scratch-inode", str(inode)]
        shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; "
                 "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TMPDIR=/tmp/g42 TMP=/tmp/g42 TEMP=/tmp/g42 "
                 "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 "
                 "TRANSFORMERS_OFFLINE=1; export PYTHONPATH=" + shlex.quote(
                     "/minicpm-transformers:/swift-runtime/site:" + str(framework)) + "; exec " + shlex.join(command))
        argv.extend((str(ROOT.parent / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell))
        write(directory / "intent.json", identity | {"source_SHA256": SOURCE_SHA, "argv": argv})
        with (directory / "stdout").open("xb") as out, (directory / "stderr").open("xb") as err:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        write(directory / "client.json", identity | {"client_PID": child.pid})
    try:
        rc = child.wait(timeout=65)
    except subprocess.TimeoutExpired:
        return identity | {"state": "ORIGINAL_OUTCOME_UNKNOWN"}
    match = []
    for _ in range(4):
        match = [row for row in rows(PARENT) if row[1] == LABEL]
        require(len(match) <= 1, "G42_HOST_DUPLICATE_ACCOUNTING_LABEL")
        if match and match[0][2] not in ("PENDING", "RUNNING", "COMPLETING"):
            break
        time.sleep(.4)
    with (directory / "stdout").open("rb") as out, (directory / "stderr").open("rb") as err:
        stdout, stderr = out.read(8193), err.read(8193)
    require(max(len(stdout), len(stderr)) <= 8192, "G42_HOST_PROBE_OUTPUT_BOUND")
    reports = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("request") == SCOPE:
            reports.append(row)
    require(len(reports) <= 1, "G42_HOST_UNIQUE_CHILD_REPORT")
    probe = reports[0] if reports else None
    row = match[0] if match else None
    keys = ("request", "passed", "target_GPU", "native_optimizer_updates", "actual_training_rows_opened",
            "actual_weight_files_opened", "step", "epoch", "phase", "arrow_SHA256", "pool_SHA256",
            "actual_manager_provider_SHA256", "actual_dataset_map_fixture_rows", "IPC",
            "active_provider_children_after_exit", "private_mount_clean_after", "retirement", "error_class", "predicate", "errno")
    projected = {key: probe[key] for key in keys if key in probe} if probe else None
    marker = scratch / ".scope"
    host_clean = bool({entry.name for entry in scratch.iterdir()} == {".scope"} and marker.is_file()
                      and marker.read_bytes() == (SCOPE + "\n").encode("ascii")
                      and stat.S_IMODE(scratch.stat().st_mode) == 0o700 and scratch.stat().st_ino == inode)
    passed = bool(rc == 0 and probe and probe["passed"] is True and row
                  and probe["step"] == row[0] and row[1] == LABEL and row[2:4] == ["COMPLETED", "0:0"]
                  and row[5] == NODE and row[4].isdigit() and int(row[4]) <= 120
                  and set(row[6].split(",")) in ({"cpu=4", "mem=8G", "node=1"}, {"cpu=4", "mem=8192M", "node=1"})
                  and probe["target_GPU"] == probe["native_optimizer_updates"] == probe["actual_training_rows_opened"]
                  == probe["actual_weight_files_opened"] == probe["active_provider_children_after_exit"] == 0
                  and probe["actual_dataset_map_fixture_rows"] == 2 and host_clean and probe["private_mount_clean_after"]
                  and probe["IPC"]["Dataset.map"]["binds"] > 0
                  and probe["IPC"]["Dataset.map"]["connects_to_live_socket"] > 0
                  and probe["IPC"]["Dataset.map"]["maximum_path_bytes"] <= 107
                  and probe["retirement"]["state"] == "MARKER_ONLY"
                  and probe["retirement"]["children_reaped"] is True
                  and probe["retirement"]["after"] == {"marker_valid": True, "candidate_directories": 0, "unexpected": 0})
    result = identity | {"state": "LIFECYCLE_AND_ORIGINAL_ACCOUNTING_PASSED" if passed else "ORIGINAL_CHECK_UNQUALIFIED",
                         "rc": rc, "probe": projected, "accounting": row, "private_mount_clean_host": host_clean,
                         "stdout_SHA256": sha256(stdout).hexdigest(), "stderr_SHA256": sha256(stderr).hexdigest()}
    return result | {"target_result_SHA256": write(directory / "result.json", result)}


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin)), sort_keys=True))
