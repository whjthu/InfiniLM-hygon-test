"""Dispatch the exact published zero-GRES MIOpen bind or trace-detach experiment once."""

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
ARCHIVE = "29240578320fe40a1497bd0c45c7bde095e7ff6923acc838e68663c26433f7d3"
MANIFEST = "ceae819436335cb5370f2bbc83669ab7bb209d4d9c80aa6cedfb85635596970e"
DEPENDENCY = "e515abfbb4e589e28e0b31bb6452031dfd95412f21e018329e22341a1fc44789"
MEMBERS = {"host.py", "inside.py", "maps_probe.py", "mount_observation.py", "scope.py"}
DETACH_MANIFEST = "ca8285ae5bb249e15a6e13b2005e1b98aea4d81f5efdde9c49e53809923496a6"
DETACH_SCOPE = "9eb24484e3c846a48b87f662bf3b8e65"
DETACH_LABEL = "t09-trace-detach-" + DETACH_SCOPE[:12]
DETACH_MEMBERS = {"host.py", "guest.py", "mount_observation.py"}
DETACH_ARCHIVES = {"3600": "d356addd3ef90db2dbe92dc1d86831126962260bf97ac5378dd863c03e7a8228",
                   "3605": "b16aaa3df85d75479345cac12d0788853818f4ca7b3175c1e8509bf802d50783"}


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
    raw = base64.b64decode(payload["sources"]["SOURCE_FILES.json"], validate=True)
    manifest_pin = digest(raw)
    require(manifest_pin in (MANIFEST, DETACH_MANIFEST), "EXACT_PACKAGE_MANIFEST")
    is_detach = manifest_pin == DETACH_MANIFEST
    scope, label, members = (DETACH_SCOPE, DETACH_LABEL, DETACH_MEMBERS) if is_detach else (SCOPE, LABEL, MEMBERS)
    require(set(payload["sources"]) == members | {"SOURCE_FILES.json"}, "EXACT_SOURCE_SET")
    sources = {name: base64.b64decode(raw, validate=True) for name, raw in payload["sources"].items()}
    meta = json.loads(raw)
    fixed_lineage = meta.get("archives") == DETACH_ARCHIVES if is_detach else meta.get("old_3598_archive_SHA256") == ARCHIVE
    require(meta["request"] == scope and meta["label"] == label and meta["actual_GPU_planned"] == 0
            and fixed_lineage and set(meta["files"]) == members, "EXACT_MODEL_FREE_SCOPE")
    for name in members:
        raw = sources[name]
        require(meta["files"][name] == {"bytes": len(raw), "SHA256": digest(raw)}, "EXACT_SOURCE_MEMBER")
    guard = base64.b64decode(payload["scheduler_guard"], validate=True)
    require(digest(guard) == DEPENDENCY, "EXACT_EXISTING_SCHEDULER_GUARD")
    functions = {"__name__": "scheduler_guard"}
    exec(compile(guard, "<pinned local scheduler guard>", "exec"), functions)
    return sources, functions, is_detach


def launch_argv(manifest, authority):
    if manifest == DETACH_MANIFEST:
        stage = STAGE.parent / DETACH_LABEL
        return [
            "srun", "--jobid=" + PARENT, "--exact", "--nodes=1", "--ntasks=1", "--nodelist=" + NODE,
            "--immediate=5", "--cpus-per-task=2", "--mem=2048M", "--gres=none", "--time=00:02:00",
            "--job-name=" + DETACH_LABEL, "/usr/local/python3.12/bin/python3", "-B", (stage / "host.py").as_posix(),
            "run-once", "--manifest-sha", manifest, "--authority-sha", authority,
        ]
    return [
        "srun", "--jobid=" + PARENT, "--exact", "--nodes=1", "--ntasks=1", "--immediate=5",
        "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none", "--time=00:02:00",
        "--job-name=" + LABEL, "/usr/local/python3.12/bin/python3", "-B", (STAGE / "host.py").as_posix(),
        "run-once", "--manifest-sha", manifest, "--authority-sha", authority,
    ]


def terminal(read_rows, terminal_states, epoch, deadline, label, cpu, gib):
    limit = min(deadline, time.time() + 10)
    while time.time() < limit:
        matches = [row for row in read_rows() if row[1] == label]
        require(len(matches) <= 1, "ONE_EXACT_JOB_NAME")
        if matches and matches[0][2] in terminal_states:
            row = matches[0]
            require(re.fullmatch(PARENT + r"\.\d+", row[0]) and row[5] == NODE
                    and len(row[6].split(",")) == 3 and set(row[6].split(",")) in (
                        {f"cpu={cpu}", f"mem={gib}G", "node=1"}, {f"cpu={cpu}", f"mem={gib*1024}M", "node=1"}),
                    "EXACT_TERMINAL_ZERO_GRES")
            start,end = [datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() for value in row[7:9]]
            require(epoch - 1 <= start <= end <= deadline + 1, "TERMINAL_SAME_EPOCH")
            return row
        time.sleep(0.4)
    return None


def execute(payload):
    sources, tools, is_detach = bundle(payload)
    scope, label, manifest = (DETACH_SCOPE, DETACH_LABEL, DETACH_MANIFEST) if is_detach else (SCOPE, LABEL, MANIFEST)
    stage, audit = STAGE.parent / label, AUDIT.parent / (label + "-controller")
    bound_authority = stage.with_name(label + ".authority.json")
    result = {"request": payload["request"], "scope": scope, "state": "PRECHECK",
              "srun_attempted": False, "replay_allowed": False}
    owns_audit = False
    lock = os.open(ROOT / "results/.training80-dcu-standby.lock",
                   os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        require(stat.S_ISREG(os.fstat(lock).st_mode), "EXACT_REGULAR_STANDBY_LOCK")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tools["node_clear"]()
        require(not any(os.path.lexists(p) for p in (audit, stage, bound_authority))
                and not any(row[1] == label for row in tools["rows"]()), "ORIGINAL_SCOPE_ALREADY_EXISTS")
        audit.mkdir(mode=0o700)
        owns_audit = True
        stage.mkdir(mode=0o700)
        for name, raw in sources.items():
            tools["write"](stage / name, raw)
            require(tools["owned_file"](stage / name) == raw, "SOURCE_READBACK_DIFFERS")
        epoch = time.time()
        deadline = epoch + 120
        result["epoch"] = epoch
        authority = dict(request=scope, node=NODE, parent=PARENT, epoch=epoch, deadline=deadline)
        if is_detach:
            authority.update(label=label, archives=DETACH_ARCHIVES, source_manifest_SHA256=manifest,
                             ordinary_zero_GRES_GO=True)
        else:
            authority.update(manifest_SHA256=manifest, archive_SHA256=ARCHIVE,
                             root_approved_zero_GRES=True, root_approved_single_file_bind=True)
        auth_sha = tools["write"](bound_authority, encoded(authority))
        argv = launch_argv(manifest, auth_sha)
        result["intent_SHA256"] = tools["write"](audit / "intent.json",
            encoded(dict(authority_SHA256=auth_sha, argv_SHA256=digest(encoded(argv)),
                         scope=scope, request=payload["request"], publication=payload["publication"],
                         epoch=epoch, deadline=deadline, resource="zero_GRES", replay_allowed=False)))
        with (audit / "stdout").open("xb") as out, (audit / "stderr").open("xb") as err:
            result["srun_attempted"] = True
            child = subprocess.Popen(argv, env=tools["ENV"], stdout=out, stderr=err)
            tools["write"](audit / "srun.json", encoded(dict(pid=child.pid, epoch=epoch)))
            result["srun_rc"] = child.wait(timeout=max(1, deadline - time.time() - 9))
        result["stdout_SHA256"] = digest(tools["owned_file"](audit / "stdout", 131072))
        result["stderr_SHA256"] = digest(tools["owned_file"](audit / "stderr", 131072))
        cpu, gib = (2, 2) if is_detach else (4, 8)
        result["actual_terminal"] = terminal(tools["rows"], tools["TERMINAL"], epoch, deadline, label, cpu, gib)
        outcome_path = stage / "work-once" / "outcome.json"
        if outcome_path.exists():
            raw = tools["owned_file"](outcome_path, 131072)
            body = json.loads(raw)
            require(body["request"] == scope and body["authority_SHA256"] == auth_sha
                    and body["epoch"] == epoch and body["deadline"] == deadline
                    and (not result["actual_terminal"] or body["step"] == result["actual_terminal"][0]),
                    "EXACT_SAME_PRIVATE_ORIGINAL")
            keys = ("step", "passed", "mapped_copy_identity_verified", "mapping_outcome", "error_type",
                    "error_code", "copy_phase_SHA256", "compute_host_copy_readback_SHA256", "import_phase_SHA256",
                    "tracer_exit_rc", "actual_container_launcher_exit", "traced_tasks", "detached_tasks",
                    "actual_gpu", "actual_model_calls", "actual_optimizer_updates", "first_trace_match",
                    "final_trace_match", "trace_pin")
            result["worker"] = {k: body[k] for k in keys if k in body}
            result["worker_outcome_SHA256"] = digest(raw)
        result["state"] = "ORIGINAL_TERMINAL_OBSERVED" if result["actual_terminal"] else "ORIGINAL_TERMINAL_UNKNOWN"
    except (OSError, TypeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
        result["error_code"] = str(error) if re.fullmatch("[A-Z0-9_]+", str(error)) else None
        result["state"] = "ORIGINAL_OUTCOME_UNKNOWN" if result["srun_attempted"] else "BEFORE_SRUN_REFUSED"
    finally:
        if owns_audit:
            result["controller_outcome_SHA256"] = tools["write"](audit / "outcome.json", encoded(result))
        os.close(lock)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
