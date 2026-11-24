"""Compare the exact O01 standby worker's procfs view with no GPU operations."""

import argparse
import errno
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
PARENT, NODE = "1103606", "f12r2n11"
TARGET = "1103606.2296"
STANDBY = "2e99b20724104c72b4fc811a616cc247"
TARGET_LABEL = "training80-dcu-standby-2e99b2072410-f12r2n11"
WORKER_SHA = "7156743597d8b660de0020d27230b8dd4b0c645c79a28a44822514337017bfa3"
POINTS_SHA = "3d33ab66101484e866655b9951b0872ffc63df4eb413235f3ff5d092948b08d6"
REQUEST = "8e398fe1bdd343dcaeeb5f42740bf7de"
LABEL = "O01-zimage-fd-visibility-" + REQUEST[:12]
STAGE = PROJECT / "incoming" / (LABEL + "-reviewed")
OUT = RESULTS / LABEL
HOST_PY = "/usr/local/python3.12/bin/python3"
CONTAINER_PY = "/usr/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
DTK = PROJECT / "containers/dtk26-sandbox-r1"
HYHAL = "/usr/local/hyhal"
PROC = Path("/proc")
BOOT = PROC / "sys/kernel/random/boot_id"
DRM = Path("/sys/class/drm")
KFD = Path("/sys/class/kfd/kfd/proc")
PCI = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
FD_LIMIT = 512


def require(value, gate):
    if not value:
        raise ValueError(gate)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, cap=16384):
    require(not path.is_symlink(), "UNEXPECTED_SYMLINK")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap, "UNEXPECTED_FILE_EXTENT")
    return raw


def pinned(path, expected, cap=16384):
    raw = read(path, cap)
    require(digest(raw) == expected, "EVIDENCE_SHA_DIFFERS")
    return raw


def bounded(argv, timeout):
    produced = subprocess.check_output(argv, timeout=timeout)
    require(0 < len(produced) <= 1024**2, "COMMAND_EXTENT")
    return produced.decode("utf-8")


def step_from_env():
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(PARENT + r"\.[0-9]+", step) and step != TARGET
            and socket.gethostname().split(".")[0] == NODE
            and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
            and len(os.sched_getaffinity(0)) == 1, "ZERO_GRES_LOCAL_SCOPE")
    return step


def rows():
    result = bounded(["scontrol", "show", "step", PARENT, "--oneliner"], 4)
    return [dict(token.split("=", 1) for token in line.split() if "=" in token)
            for line in result.splitlines() if line.strip()]


def require_live(steps, step, name, tress):
    matches = [row for row in steps if row.get("StepId") == step]
    require(len(matches) == 1, "NUMBERED_STEP_NOT_UNIQUE")
    row = matches[0]
    actual = row["TRES"].split(",")
    require(row.get("Name") == name and row.get("NodeList") == NODE and row.get("State") == "RUNNING"
            and len(actual) == len(set(actual)) and set(actual) in tress, "ACTUAL_STEP_BINDING")


def live_bindings(step):
    observed = rows()
    require_live(observed, TARGET, TARGET_LABEL,
                 ({"cpu=8", "gres/dcu=8", "node=1", "mem=16G"}, {"cpu=8", "gres/dcu=8", "node=1", "mem=16384M"}))
    require_live(observed, step, LABEL, ({"cpu=1", "node=1", "mem=256M"},))


def belongs_to(step, pid):
    job, ordinal = step.split(".")
    output = bounded(["scontrol", "listpids", step], 4)
    return any(line.split()[:3] == [str(pid), job, ordinal] for line in output.splitlines())


def original():
    producer = RESULTS / ("training80-o01-refill-observer-" + STANDBY)
    record = json.loads(pinned(producer / "evidence.json", POINTS_SHA, 32768))
    worker_path = RESULTS / ("training80-dcu-standby-" + STANDBY) / NODE / "worker.json"
    worker = json.loads(pinned(worker_path, WORKER_SHA))
    fields = ("boot_id_SHA256", "worker_PID", "worker_start_ticks", "worker_cgroup_SHA256")
    first, second = record["first"], record["second"]
    require(record["request"] == STANDBY and worker["step"] == TARGET and worker["node"] == NODE
            and worker["synthetic"] is True and len(worker["pci"]) == 8 and set(worker["pci"]) == PCI
            and worker["worker_pid"] == first["worker_PID"]
            and all(first[k] == second[k] for k in fields)
            and all(p["step"] == TARGET and p["observer_step"] == "1103606.2297" and p["node"] == NODE
                    and p["slurm_listpids_matched"] is True and set(p["rows"]) == PCI for p in (first, second)),
            "ORIGINAL_REFILL_EVIDENCE_DIFFERS")
    return {key: first[key] for key in fields}


def names_step(raw):
    job, ordinal = TARGET.split(".")
    wanted = (f"job_{job}", f"step_{ordinal}")
    paths = []
    for line in raw.decode("utf-8").splitlines():
        hierarchy, _, path = line.split(":", 2)
        parsed = PurePosixPath(path)
        require(hierarchy.isdecimal() and parsed.is_absolute() and ".." not in parsed.parts, "CGROUP_SCHEMA")
        paths.append(parsed.parts)
    return any(wanted in zip(parts, parts[1:]) for parts in paths)


def identity(path, expected):
    def observe():
        start = int(read(path / "stat", 4096).decode().rsplit(") ", 1)[1].split()[19])
        group = read(path / "cgroup")
        uid = path.stat().st_uid
        return start, group, uid

    before = observe()
    after = observe()
    require(before == after and before[0] == expected["worker_start_ticks"] and before[2] == os.getuid()
            and names_step(before[1]) and digest(before[1]) == expected["worker_cgroup_SHA256"]
            and digest(read(BOOT, 128)) == expected["boot_id_SHA256"], "WORKER_IDENTITY_CHANGED_OR_NOT_VISIBLE")


def device_inventory(pid):
    try:
        render = {}
        for entry in DRM.iterdir():
            if re.fullmatch(r"(?:renderD|card)\d+", entry.name):
                render[entry.name] = (entry / "device").resolve(strict=True).name.lower()
        drm_exact = {pci for name, pci in render.items() if name.startswith("renderD")} == PCI
        drm_status = "observed"
    except OSError:
        render, drm_exact, drm_status = {}, False, "unavailable"
    try:
        ids = {path.name for path in KFD.iterdir()}
        kfd_status, kfd_exact = "observed", ids == {str(pid)}
    except OSError:
        kfd_status, kfd_exact = "unavailable", False
    return render, {"drm_status": drm_status, "exact_eight_DRM_PCI": drm_exact,
                    "kfd_status": kfd_status, "KFD_only_bound_worker": kfd_exact}


def fd_visibility(process, render):
    result = {"fd_enumeration": "observed", "fd_enumeration_errno": None,
              "fd_readlink_permission_denied": 0, "fd_readlink_errno": [],
              "fd_races": 0, "fd_readlink_other_errors": 0, "fd_links_read": 0,
              "worker_KFD_FDs": 0, "mapped_DRM_PCI_count": 0, "unknown_device_links": 0}
    entries = []
    try:
        for item in (process / "fd").iterdir():
            if len(entries) >= FD_LIMIT:
                result["fd_enumeration"] = "bound_exceeded"
                return result
            if item.name.isdecimal():
                entries.append(item)
    except OSError as error:
        result["fd_enumeration"] = "permission_denied" if isinstance(error, PermissionError) else "unavailable"
        result["fd_enumeration_errno"] = error.errno
        return result
    mapped = set()
    denied = set()
    for item in entries:
        try:
            target = os.readlink(item)
        except PermissionError as error:
            result["fd_readlink_permission_denied"] += 1
            if error.errno in (errno.EACCES, errno.EPERM):
                denied.add(error.errno)
            continue
        except FileNotFoundError:
            result["fd_races"] += 1
            continue
        except OSError:
            result["fd_readlink_other_errors"] += 1
            continue
        result["fd_links_read"] += 1
        if target == "/dev/kfd":
            result["worker_KFD_FDs"] += 1
        elif target.startswith("/dev/dri/"):
            pci = render.get(Path(target).name)
            if pci in PCI:
                mapped.add(pci)
            else:
                result["unknown_device_links"] += 1
    result["fd_readlink_errno"] = sorted(denied)
    result["mapped_DRM_PCI_count"] = len(mapped)
    return result


def view(expected):
    pid = expected["worker_PID"]
    require(type(pid) is int and pid > 1, "WORKER_IDENTITY_SCHEMA")
    observer_namespace = (PROC / "self/ns/pid").stat()
    process = PROC / str(pid)
    identity(process, expected)
    render, inventory = device_inventory(pid)
    fd = fd_visibility(process, render)
    identity(process, expected)
    return {"namespace": [observer_namespace.st_dev, observer_namespace.st_ino], "identity_bound": True,
            "devices": inventory, "fd": fd}


def public(host, container):
    same_namespace = host["namespace"] == container["namespace"]
    names = ("identity_bound", "devices", "fd")
    return {"same_PID_namespace": same_namespace,
            "same_worker_identity": same_namespace and host["identity_bound"] and container["identity_bound"],
            **{key: {name: value[name] for name in names}
               for key, value in (("host", host), ("container", container))},
            "holder_audit_complete": False, "target_clean": False, "GPU_GO": False}


def container_argv(source, source_sha):
    return [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/",
            "--bind", f"{source.as_posix()}:/fd-visibility.py:ro",
            "--bind", f"{HYHAL}:/opt/hyhal:ro", DTK.as_posix(), CONTAINER_PY,
            "-I", "-S", "-B", "/fd-visibility.py", "inside", "--source-sha256", source_sha]


def container_env():
    allowed = {"PATH", "LANG", "LC_ALL", "USER", "LOGNAME", "HOME", "TMPDIR", "TZ"}
    return {key: value for key, value in os.environ.items() if key in allowed or key.startswith("SLURM_")}


def srun_argv(source, source_sha):
    require(source == STAGE / "probe.py" and re.fullmatch(r"[a-f0-9]{64}", source_sha), "SOURCE_OR_STAGE_DIFFERS")
    return ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--overlap", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=1", "--threads-per-core=1",
            "--cpu-bind=threads", "--gres=none", "--mem=256M", "--time=00:01:00", "--kill-on-bad-exit=1",
            "--output=" + (OUT / "slurm.stdout").as_posix(), "--error=" + (OUT / "slurm.stderr").as_posix(),
            HOST_PY, "-I", "-S", "-B", source.as_posix(), "host", "--source-sha256", source_sha]


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def host(source, source_sha):
    require(source == STAGE / "probe.py" and Path("/opt/hyhal").samefile(HYHAL), "STAGE_OR_HYHAL_DIFFERS")
    step = step_from_env()
    live_bindings(step)
    expected = original()
    require(belongs_to(TARGET, expected["worker_PID"]) and belongs_to(step, os.getpid()), "SLURM_LISTPIDS_DIFFERS")
    host_view = view(expected)
    start = time.monotonic()
    result = subprocess.run(container_argv(source, source_sha), input=encode(expected), cwd="/", env=container_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
    details = {"exit_code": result.returncode, "seconds": time.monotonic() - start,
               "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
               "stderr_SHA256": digest(result.stderr)}
    if result.returncode != 0:
        try:
            denial = json.loads(result.stdout) if 0 < len(result.stdout) <= 4096 else {}
        except (ValueError, UnicodeError):
            denial = {}
        if (isinstance(denial, dict) and denial.get("request") == REQUEST and denial.get("target_step") == TARGET
                and denial.get("failure_type") in {"PermissionError", "FileNotFoundError", "ValueError"}):
            details["worker_view_failure_type"] = denial["failure_type"]
            if denial.get("safe_gate") in {"ZERO_GRES_LOCAL_SCOPE", "WORKER_IDENTITY_CHANGED_OR_NOT_VISIBLE",
                                           "CGROUP_SCHEMA", "UNEXPECTED_SYMLINK", "UNEXPECTED_FILE_EXTENT"}:
                details["worker_view_gate"] = denial["safe_gate"]
        return {"request": REQUEST, "step": step, "target_step": TARGET, "container_start_or_view_failed": True,
                "container_execution": details, "GPU_GO": False, "holder_audit_complete": False,
                "target_clean": False, "host": {k: host_view[k] for k in ("identity_bound", "devices", "fd")}}
    require(0 < len(result.stdout) <= 4096, "CONTAINER_VIEW_EXTENT")
    inside = json.loads(result.stdout)
    identity(PROC / str(expected["worker_PID"]), expected)
    live_bindings(step)
    require(belongs_to(TARGET, expected["worker_PID"]), "SLURM_LISTPIDS_DIFFERS_AFTER_VIEW")
    return {"request": REQUEST, "step": step, "target_step": TARGET,
            "source_SHA256": source_sha, "refill_evidence_SHA256": POINTS_SHA,
            "container_execution": details, **public(host_view, inside)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("host", "inside"))
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    try:
        source = Path(__file__).resolve(strict=True)
        pinned(source, args.source_sha256, 32768)
        if args.phase == "inside":
            step_from_env()
            raw = sys.stdin.buffer.read(4097)
            require(0 < len(raw) <= 4096, "WORKER_EXPECTATION_EXTENT")
            result = view(json.loads(raw))
        else:
            result = host(source, args.source_sha256)
        sys.stdout.buffer.write(encode(result))
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as error:
        print(json.dumps({"request": REQUEST, "target_step": TARGET, "failure_type": type(error).__name__,
                          "safe_gate": str(error) if re.fullmatch(r"[A-Z][A-Z0-9_]+", str(error)) else None,
                          "holder_audit_complete": False, "target_clean": False, "GPU_GO": False}, sort_keys=True))
        return 1
    return int(result.get("container_start_or_view_failed", False))


if __name__ == "__main__":
    raise SystemExit(main())
