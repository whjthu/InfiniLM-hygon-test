"""Run the frozen per-node standby launcher from one finite zero-GRES Slurm step."""
import argparse
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PYTHON = "/usr/local/python3.12/bin/python3"
PARENT = "1103606"
NODE = "f12r2n13"
PREFIX = "training80-standby-compute"
STANDBY = "training80-dcu-standby"
CONTROLLER_SHA256 = "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"
BRIDGE_SECONDS = 11520
MAX_MESSAGE = 262144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def bounded(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_MESSAGE + 1)
    require(0 < len(data) <= MAX_MESSAGE, "Bridge metadata exceeds its bound")
    return data


def write_json(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def bridge_path(request):
    require(re.fullmatch("[0-9a-f]{32}", request), "Wrong standby bridge request identity")
    return RESULTS / f"{PREFIX}-{request}"


def control_argv(directory):
    return ["srun", "--jobid=" + PARENT, "--job-name=" + PREFIX + "-" + directory.name.split("-")[-1][:12],
            "--exact", "--immediate=8", "--threads=1", "--nodes=1", "--ntasks=1", "--nodelist=" + NODE,
            "--cpus-per-task=2", "--gres=none", "--mem=2048M", "--time=03:12:00", "--kill-on-bad-exit=1",
            PYTHON, "-I", "-S", "-B", str(directory / "bridge.py"), "compute", directory.name.split("-")[-1]]


def start():
    message = sys.stdin.buffer.read(MAX_MESSAGE + 1)
    require(0 < len(message) <= MAX_MESSAGE, "Bridge message exceeds its bound")
    envelope = json.loads(message)
    request = envelope["request"]
    directory = bridge_path(request)
    entry, bridge = (b64decode(envelope[key], validate=True) for key in ("entry", "bridge"))
    require(sha256(entry).hexdigest() == envelope["entry_SHA256"]
            and sha256(bridge).hexdigest() == envelope["bridge_SHA256"]
            and entry.startswith(b"INPUT=")
            and sha256(entry.partition(b"\n")[2]).hexdigest() == CONTROLLER_SHA256, "Bridge source digest differs")
    directory.mkdir(mode=0o700)
    write(directory / "entry.py", entry)
    write(directory / "bridge.py", bridge)
    epoch = time.time()
    write_json(directory / "intent.json", dict(request=request, entry_SHA256=envelope["entry_SHA256"],
               bridge_SHA256=envelope["bridge_SHA256"], epoch=epoch, deadline_epoch=epoch + BRIDGE_SECONDS,
               parent=PARENT, node=NODE, replay_allowed=False))
    with (directory / "stdout.private").open("xb") as out, (directory / "stderr.private").open("xb") as err:
        proc = subprocess.Popen(control_argv(directory), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)
    proc_stat = Path(f"/proc/{proc.pid}/stat")
    try:
        startticks = int(proc_stat.read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        startticks = None
    write_json(directory / "launcher.json", dict(request=request, pid=proc.pid, startticks=startticks))
    result_path = RESULTS / f"{STANDBY}-{request}" / "submission-receipt.json"
    end = time.monotonic() + 36
    while time.monotonic() < end:
        if result_path.is_file() and (directory / "compute.json").is_file():
            raw = bounded(result_path)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                time.sleep(0.2)
                continue
            require(result["request"] == request, "Standby handoff receipt belongs to another request")
            print(raw.decode(), end="", flush=True)
            return 0
        rc = proc.poll()
        if rc is not None:
            write_json(directory / "launcher-terminal.json", dict(request=request, rc=rc))
            return rc if rc else 1
        time.sleep(0.2)
    return 75


class StandbyChildren:
    """Keep only srun children from the frozen launcher under the compute step."""

    DEVNULL = subprocess.DEVNULL
    check_output = staticmethod(subprocess.check_output)

    def __init__(self):
        self.children = []

    def Popen(self, argv, *args, **kwargs):
        require(argv[0] == "srun" and any(arg.startswith("--jobid=") for arg in argv),
                "Unexpected frozen standby child launcher")
        proc = subprocess.Popen([argv[0], "--threads=1", *argv[1:]], *args, **kwargs)
        self.children.append(proc)
        return proc

    def wait(self, deadline):
        exits = []
        for child in self.children:
            remaining = deadline - time.time()
            try:
                exits.append(child.wait(timeout=max(0, remaining)))
            except subprocess.TimeoutExpired:
                exits.append(None)
        return exits


def compute(request):
    directory = bridge_path(request)
    intent = json.loads(bounded(directory / "intent.json"))
    step = os.environ.get("SLURM_STEP_ID", "")
    require(socket.gethostname().split(".")[0] == NODE and os.environ.get("SLURM_JOB_ID") == PARENT
            and step.isascii() and step.isdecimal(), "Compute bridge has no scheduled ownership")
    require(intent["request"] == request and intent["node"] == NODE and intent["parent"] == PARENT
            and intent["deadline_epoch"] - time.time() > 11100, "Bridge intent or residual deadline differs")
    for name in tuple(os.environ):
        if name.startswith(("SLURM_", "PMI_", "PMIX_")):
            del os.environ[name]
    entry = bounded(directory / "entry.py")
    require(sha256(entry).hexdigest() == intent["entry_SHA256"]
            and sha256(bounded(directory / "bridge.py")).hexdigest() == intent["bridge_SHA256"]
            and sha256(entry.partition(b"\n")[2]).hexdigest() == CONTROLLER_SHA256, "Staged bridge source differs")
    namespace = {"__name__": "published_standby_controller"}
    exec(compile(entry, str(directory / "entry.py"), "exec"), namespace)
    inputs = namespace["INPUT"]
    require(inputs["request"] == request and 1 <= len(inputs["nodes"]) <= 32 and NODE not in inputs["nodes"],
            "Compute bridge must launch at most 32 separately owned standby nodes")
    children = StandbyChildren()
    namespace["subprocess"] = children
    write_json(directory / "compute.json", dict(request=request, node=NODE, step=f"{PARENT}.{step}",
               started_epoch=time.time(), child_threads=1, target_nodes=sorted(inputs["nodes"])))
    try:
        namespace["run"](inputs)
    finally:
        exits = children.wait(intent["deadline_epoch"] - 10)
        write_json(directory / "child-terminal.json", dict(request=request, children=len(exits),
                   observed_launcher_exit_codes=exits, synthetic=True))
    return 1 if any(rc is None or rc != 0 for rc in exits) else 0


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "compute"))
    parser.add_argument("request", nargs="?")
    args = parser.parse_args()
    raise SystemExit(start() if args.mode == "start" else compute(args.request))
