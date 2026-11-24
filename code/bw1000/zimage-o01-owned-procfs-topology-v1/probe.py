"""Test procfs visibility for the current observer and explicitly owned CPU children."""

from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys


CONTAINER_PY = "/usr/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
DTK = PROJECT / "containers/dtk26-sandbox-r1"
SOURCE = Path(__file__).resolve()
PROC = Path("/proc")
PARENT, NODE = "1103606", "f12r2n11"


def require(predicate, label):
    if not predicate:
        raise ValueError(label)


def number(text):
    require(type(text) is int and text >= 0, "invalid process metadata")
    return text


def identify(pid, *, proc=PROC):
    entry = proc / str(number(pid))
    data = (entry / "stat").read_text(encoding="utf-8")
    fields = data.rsplit(") ", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "startticks": int(fields[19]),
            "uid": entry.stat().st_uid, "state": fields[0]}


def declared(pid, fd):
    row = identify(pid)
    return {key: row[key] for key in ("pid", "ppid", "startticks", "uid")} | {"fd": fd}


def validate_owner(witness, *, uid, proc):
    require(set(witness) == {"pid", "ppid", "startticks", "uid", "fd"}, "wrong identity fields")
    require(all(number(value) >= 0 for value in witness.values()) and 3 <= witness["fd"] <= 64,
            "invalid sentinel descriptor")
    now = identify(witness["pid"], proc=proc)
    require(witness["uid"] == now["uid"] == uid and now["state"] not in {"Z", "X"}
            and all(witness[key] == now[key] for key in ("pid", "ppid", "startticks")),
            "original owned process identity changed")


def inspect_fd(witness, *, uid, proc=PROC, scandir=os.scandir, readlink=os.readlink):
    validate_owner(witness, uid=uid, proc=proc)
    path = proc / str(witness["pid"]) / "fd"
    states = {}
    try:
        with scandir(path) as entries:
            states["enumerate"] = ("ok" if any(entry.name == str(witness["fd"]) for entry in entries)
                                   else "missing_sentinel")
    except PermissionError as error:
        require(error.errno in {errno.EACCES, errno.EPERM}, "unexpected enumerate permission error")
        states["enumerate"] = f"errno_{error.errno}"
    try:
        states["readlink"] = "ok" if readlink(path / str(witness["fd"])) == "/dev/null" else "foreign_sentinel"
    except PermissionError as error:
        require(error.errno in {errno.EACCES, errno.EPERM}, "unexpected readlink permission error")
        states["readlink"] = f"errno_{error.errno}"
    validate_owner(witness, uid=uid, proc=proc)
    return states


def output(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def stdin_json():
    raw = sys.stdin.buffer.read(2049)
    require(0 < len(raw) <= 2048, "bounded topology payload required")
    return json.loads(raw)


def command(phase):
    return [sys.executable, "-I", "-S", "-B", str(SOURCE), phase]


def identity_payload(value):
    return json.dumps(value, separators=(",", ":")).encode()


def subject():
    with open("/dev/null", "rb") as null:
        output(declared(os.getpid(), null.fileno()))
        require(sys.stdin.buffer.read(1) == b"", "unexpected subject input")


def sibling_observer():
    expected = stdin_json()
    parent, peer = (expected[key] for key in ("parent", "sibling"))
    require(parent["pid"] == os.getppid() == peer["ppid"] and peer["pid"] != os.getpid(),
            "unexpected parent or sibling topology")
    with open("/dev/null", "rb") as null:
        self = declared(os.getpid(), null.fileno())
        output({name: inspect_fd(row, uid=os.getuid())
                for name, row in (("self", self), ("toward_parent", parent), ("toward_sibling", peer))})


def session():
    context = (PROC / "self/ns/pid").stat()
    with ExitStack() as stack:
        null = stack.enter_context(open("/dev/null", "rb"))
        require(stat.S_ISCHR(os.fstat(null.fileno()).st_mode), "sentinel must be a character device")
        parent = declared(os.getpid(), null.fileno())
        child = stack.enter_context(subprocess.Popen(command("subject"), stdin=subprocess.PIPE, stdout=subprocess.PIPE))
        # Normal exits are controlled by EOF. Kill is restricted to this observer's own Popen on failure.
        def finish():
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        stack.callback(finish)
        ready, _, _ = select.select([child.stdout], [], [], 4)
        require(ready, "owned child identity deadline elapsed")
        raw = child.stdout.readline(1025)
        require(0 < len(raw) <= 1024, "subject identity missing")
        sibling = json.loads(raw)
        require(sibling["pid"] == child.pid and sibling["ppid"] == parent["pid"], "wrong owned child")
        views = {"self": inspect_fd(parent, uid=os.getuid()), "toward_child": inspect_fd(sibling, uid=os.getuid())}
        result = subprocess.run(command("sibling"), input=identity_payload({"parent": parent, "sibling": sibling}),
                                capture_output=True, timeout=5, check=True)
        require(0 < len(result.stdout) <= 2048, "bounded sibling observation required")
        row = json.loads(result.stdout)
        require(set(row) == {"self", "toward_parent", "toward_sibling"}, "sibling observation schema differs")
        child.stdin.close()
        require(child.wait(timeout=3) == 0, "owned subject did not complete")
        return {"root_observer": views, "child_observer": row, "namespace": [context.st_dev, context.st_ino]}


def compare(host, container):
    result = {"same_PID_namespace": host["namespace"] == container["namespace"],
              "probe_GPU_calls": 0, "probe_opened_GPU_device_files": 0,
              "full_holder_scan_qualified": False, "original_2292_denied_process_identified": False}
    for name, view in (("host", host), ("container", container)):
        result[name] = {key: view[key] for key in ("root_observer", "child_observer")}
    return result


def cluster():
    import socket

    require(os.environ.get("SLURM_JOB_ID") == PARENT and os.environ.get("SLURM_STEP_ID", "").isdecimal()
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1" and socket.gethostname().split(".")[0] == NODE,
            "wrong bounded zero-card Slurm placement")
    before = session()
    args = [APPTAINER, "exec", "--rocm", "--no-home", "--pwd", "/", "--bind", f"{SOURCE}:/fd-topology.py:ro",
            "--bind", "/usr/local/hyhal:/opt/hyhal:ro",
            str(DTK), CONTAINER_PY, "-I", "-S", "-B", "/fd-topology.py", "session"]
    child = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=22, check=True)
    require(0 < len(child.stdout) <= 2048, "bounded container observation required")
    result = compare(before, json.loads(child.stdout))
    result["step"] = PARENT + "." + os.environ["SLURM_STEP_ID"]
    output(result)


if __name__ == "__main__":
    phase = sys.argv[1]
    {"host": cluster, "session": lambda: output(session()), "subject": subject, "sibling": sibling_observer}[phase]()
