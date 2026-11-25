"""Dispatch the exact published zero-GRES MIOpen file-bind experiment once."""

import base64
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


SCOPE = "c6025678606b4170ba7f9fbb5bf0a34d"
LABEL = "t09-miopen-single-bind-" + SCOPE[:12]
PARENT, NODE = "1103606", "f13r3n13"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
STAGE = ROOT / "incoming" / LABEL
AUDIT = ROOT / "results" / (LABEL + "-controller")
AUTHORITY = STAGE.with_name(LABEL + ".authority.json")
ARCHIVE = "29240578320fe40a1497bd0c45c7bde095e7ff6923acc838e68663c26433f7d3"
MANIFEST = "ceae819436335cb5370f2bbc83669ab7bb209d4d9c80aa6cedfb85635596970e"
DEPENDENCY = "e515abfbb4e589e28e0b31bb6452031dfd95412f21e018329e22341a1fc44789"
MEMBERS = {"host.py", "inside.py", "maps_probe.py", "mount_observation.py", "scope.py"}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def bundle(payload):
    require(set(payload) == {"request", "publication", "sources", "scheduler_guard"}
            and re.fullmatch("[0-9a-f]{32}", payload["request"])
            and re.fullmatch("[0-9a-f]{40}", payload["publication"]), "EXACT_TRANSPORT_BUNDLE")
    require(set(payload["sources"]) == MEMBERS | {"SOURCE_FILES.json"}, "EXACT_SOURCE_SET")
    sources = {name: base64.b64decode(raw, validate=True) for name, raw in payload["sources"].items()}
    raw = sources["SOURCE_FILES.json"]
    require(digest(raw) == MANIFEST, "EXACT_PACKAGE_MANIFEST")
    meta = json.loads(raw)
    require(meta["request"] == SCOPE and meta["label"] == LABEL
            and meta["actual_GPU_planned"] == 0 and meta["old_3598_archive_SHA256"] == ARCHIVE
            and set(meta["files"]) == MEMBERS, "EXACT_MODEL_FREE_SCOPE")
    for name in MEMBERS:
        raw = sources[name]
        require(meta["files"][name] == {"bytes": len(raw), "SHA256": digest(raw)}, "EXACT_SOURCE_MEMBER")
    guard = base64.b64decode(payload["scheduler_guard"], validate=True)
    require(digest(guard) == DEPENDENCY, "EXACT_EXISTING_SCHEDULER_GUARD")
    functions = {"__name__": "scheduler_guard"}
    exec(compile(guard, "<pinned local scheduler guard>", "exec"), functions)
    return sources, functions


def launch_argv(manifest, authority):
    return [
        "srun", "--jobid=" + PARENT, "--exact", "--nodes=1", "--ntasks=1", "--immediate=5",
        "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--time=00:02:00",
        "--job-name=" + LABEL, "/usr/local/python3.12/bin/python3", "-B", (STAGE / "host.py").as_posix(),
        "run-once", "--manifest-sha", manifest, "--authority-sha", authority,
    ]


def terminal(read_rows, terminal_states, epoch, deadline):
    limit = min(deadline, time.time() + 10)
    while time.time() < limit:
        matches = [row for row in read_rows() if row[1] == LABEL]
        require(len(matches) <= 1, "ONE_EXACT_JOB_NAME")
        if matches and matches[0][2] in terminal_states:
            row = matches[0]
            require(re.fullmatch(PARENT + r"\.\d+", row[0]) and row[5] == NODE
                    and len(row[6].split(",")) == 3 and set(row[6].split(",")) in (
                        {"cpu=4", "mem=8G", "node=1"}, {"cpu=4", "mem=8192M", "node=1"}),
                    "EXACT_TERMINAL_ZERO_GRES")
            start,end = [datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() for value in row[7:9]]
            require(epoch - 1 <= start <= end <= deadline + 1, "TERMINAL_SAME_EPOCH")
            return row
        time.sleep(0.4)
    return None


def execute(payload):
    sources, tools = bundle(payload)
    result = {"request": payload["request"], "scope": SCOPE, "state": "PRECHECK",
              "srun_attempted": False, "replay_allowed": False}
    owns_audit = False
    lock = os.open(ROOT / "results/.training80-dcu-standby.lock",
                   os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        require(stat.S_ISREG(os.fstat(lock).st_mode), "EXACT_REGULAR_STANDBY_LOCK")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tools["node_clear"]()
        require(not any(os.path.lexists(p) for p in (AUDIT, STAGE, AUTHORITY))
                and not any(row[1] == LABEL for row in tools["rows"]()), "ORIGINAL_SCOPE_ALREADY_EXISTS")
        AUDIT.mkdir(mode=0o700)
        owns_audit = True
        STAGE.mkdir(mode=0o700)
        for name, raw in sources.items():
            tools["write"](STAGE / name, raw)
            require(tools["owned_file"](STAGE / name) == raw, "SOURCE_READBACK_DIFFERS")
        epoch = time.time()
        deadline = epoch + 120
        result["epoch"] = epoch
        authority = dict(request=SCOPE, node=NODE, parent=PARENT, epoch=epoch, deadline=deadline,
                         manifest_SHA256=MANIFEST, archive_SHA256=ARCHIVE,
                         root_approved_zero_GRES=True, root_approved_single_file_bind=True)
        auth_sha = tools["write"](AUTHORITY, encoded(authority))
        argv = launch_argv(MANIFEST, auth_sha)
        result["intent_SHA256"] = tools["write"](AUDIT / "intent.json",
            encoded(dict(authority_SHA256=auth_sha, argv_SHA256=digest(encoded(argv)),
                         scope=SCOPE, request=payload["request"], publication=payload["publication"],
                         epoch=epoch, deadline=deadline, resource="zero_GRES", replay_allowed=False)))
        with (AUDIT / "stdout").open("xb") as out, (AUDIT / "stderr").open("xb") as err:
            result["srun_attempted"] = True
            child = subprocess.Popen(argv, env=tools["ENV"], stdout=out, stderr=err)
            tools["write"](AUDIT / "srun.json", encoded(dict(pid=child.pid, epoch=epoch)))
            result["srun_rc"] = child.wait(timeout=max(1, deadline - time.time() - 9))
        result["stdout_SHA256"] = digest(tools["owned_file"](AUDIT / "stdout", 131072))
        result["stderr_SHA256"] = digest(tools["owned_file"](AUDIT / "stderr", 131072))
        result["actual_terminal"] = terminal(tools["rows"], tools["TERMINAL"], epoch, deadline)
        outcome_path = STAGE / "work-once" / "outcome.json"
        if outcome_path.exists():
            raw = tools["owned_file"](outcome_path, 131072)
            body = json.loads(raw)
            require(body["request"] == SCOPE and body["authority_SHA256"] == auth_sha
                    and body["epoch"] == epoch and body["deadline"] == deadline
                    and (not result["actual_terminal"] or body["step"] == result["actual_terminal"][0]),
                    "EXACT_SAME_PRIVATE_ORIGINAL")
            keys = ("step", "passed", "mapped_copy_identity_verified", "mapping_outcome", "error_type",
                    "error_code", "copy_phase_SHA256", "compute_host_copy_readback_SHA256", "import_phase_SHA256")
            result["worker"] = {k: body[k] for k in keys if k in body}
            result["worker_outcome_SHA256"] = digest(raw)
        result["state"] = "ORIGINAL_TERMINAL_OBSERVED" if result["actual_terminal"] else "ORIGINAL_TERMINAL_UNKNOWN"
    except (OSError, TypeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
        result["error_code"] = str(error) if re.fullmatch("[A-Z0-9_]+", str(error)) else None
        result["state"] = "ORIGINAL_OUTCOME_UNKNOWN" if result["srun_attempted"] else "BEFORE_SRUN_REFUSED"
    finally:
        if owns_audit:
            result["controller_outcome_SHA256"] = tools["write"](AUDIT / "outcome.json", encoded(result))
        os.close(lock)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
