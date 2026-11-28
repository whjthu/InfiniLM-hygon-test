"""Release the exact LLaVA target's BF16 standby and enter the first-training owner."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
NODE = "f13r1n14"
PARENT = "1103606"
CONTROL_NODE = "f12r2n13"
PORT = 29871
HOLDER = "1103606.4906"
STANDBY_REQUEST = "5624e701d4c6452597a1adc7a466f166"
STANDBY_LABEL = f"training80-dcu-standby-{STANDBY_REQUEST[:12]}-{NODE}"
STANDBY = RESULTS / f"training80-dcu-standby-{STANDBY_REQUEST}"
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = "JobIDRaw,JobName%128,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160,Start,End"
STAGED = ROOT / "incoming/training-llava-clevr-first2-swift-prehook-c01f84a87d234b89a287be673bfd8d5e"
STAGED_SHA = "58041d441428d424d7a1520d17a989d2b41b4f2d9330250dfb36d804f4b7ce24"
SCIENCE = "b7058dc5ccc472eb2e5eeeb731908166416da7597814c3e8ac9669ad02504e45"
TENSORS = "0b6ba7ff7701c165a114bc5460262dff4c053390312d086034e5019d8ecace93"
SNAPSHOT = "2e8592caa8b573aa676884b63c65d5a378540d77c877d885424d934a5c4ebcb8"
VISION = RESULTS / "training-qwen-vision014-CPU-v2/vision"
HASH = re.compile(r"[0-9a-f]{64}")
NUMERIC_STEP = re.compile(rf"{PARENT}\.\d+")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path, cap=65536, expected=None):
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), "Exact regular input required")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and (expected is None or digest(raw) == expected), "Pinned dependency differs")
    return raw


def save(path, value):
    raw = encode(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def query(*argv):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=7, check=True)
    require(not result.stderr and len(result.stdout) <= 2 << 20, "Unexpected numbered Slurm query response")
    return result.stdout.decode().splitlines()


def fields(line):
    return dict(term.split("=", 1) for term in line.split() if "=" in term)


def hosts(expression):
    if expression in {"(null)", "None", ""}:
        return []
    return [expression] if re.fullmatch(r"f\d+r\d+n\d+", expression) else query("scontrol", "show", "hostnames", expression)


def exact_row(step, *, allow_missing=False):
    rows = [line.split("|") for line in query("sacct", "-nP", "-j", step, "-o", COLUMNS)
            if line.split("|", 1)[0] == step]
    if not rows and allow_missing:
        return None
    require(len(rows) == 1 and len(rows[0]) == 9, "Must observe one corresponding accounting row")
    return rows[0]


def node_occupancy():
    [parent] = map(fields, query("scontrol", "show", "job", PARENT, "--oneliner"))
    require(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING" and NODE in hosts(parent["NodeList"])
            and parent["UserId"].endswith(f"({os.getuid()})"), "Parent lease ownership changed")
    controller = {}
    for row in map(fields, query("scontrol", "show", "step", PARENT, "--oneliner")):
        step = row.get("StepId", "")
        if NUMERIC_STEP.fullmatch(step) and NODE in hosts(row["NodeList"]):
            require(hosts(row["NodeList"]) == [NODE] and step not in controller, "Ambiguous occupied step")
            controller[step] = row
    queue = set()
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        step, listed = line.split("|")
        if NUMERIC_STEP.fullmatch(step) and NODE in hosts(listed):
            queue.add(step)
    return controller, queue


def standby_receipts():
    directory = STANDBY / NODE
    expected_bytes = read(directory / "expected.json")
    worker_bytes = read(directory / "worker.json")
    expected, worker = json.loads(expected_bytes), json.loads(worker_bytes)
    require((expected["request"], expected["parent"], expected["node"], expected["label"])
            == (STANDBY_REQUEST, PARENT, NODE, STANDBY_LABEL), "Standby plan belongs to another owner")
    require(worker["node"] == NODE and worker["step"] == HOLDER and worker["synthetic"] is True
            and len(worker["pci"]) == len(set(worker["pci"])) == 8 and set(worker["pci"]) == set(expected["pci"]),
            "Standby worker is not the verified BF16 holder")
    return digest(expected_bytes), digest(worker_bytes), worker


def release_holder(directory, request):
    plan_sha, worker_sha, worker = standby_receipts()
    before = exact_row(HOLDER)
    controller, queue = node_occupancy()
    require((before[1], before[5], before[6]) == (STANDBY_LABEL, NODE, "cpu=8,gres/dcu=8,mem=16G,node=1")
            and set(controller) <= {HOLDER} and queue <= {HOLDER}, "Unexpected live ownership at handoff")
    terminal = STANDBY / NODE / "terminal.json"
    handoff = before[2] == "RUNNING"
    if handoff:
        require(set(controller) == {HOLDER} and controller[HOLDER]["Name"] == STANDBY_LABEL
                and controller[HOLDER]["State"] == "RUNNING" and not terminal.exists(), "Holder is not live and exact")
        for index, pci in enumerate(worker["pci"]):
            pulse = json.loads(read(STANDBY / NODE / f"card-{index}.json"))
            require(pulse["pci"] == pci and pulse["synthetic"] is True and pulse["active"] is True
                    and pulse["dtype"] == "BF16" and pulse["calls"] > 0
                    and -1 <= time.time() - pulse["epoch"] <= 25, "No recent exact standby computation")
        save(directory / "standby-signal-intent.json", dict(request=request, target=HOLDER, accounting=before))
        signal = subprocess.run(["scancel", "--signal=TERM", HOLDER], stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=8)
        save(directory / "standby-signal-result.json", dict(rc=signal.returncode, stderr_SHA256=digest(signal.stderr)))
        require(signal.returncode == 0, "Exact standby signal failed")
    deadline = time.monotonic() + 80
    while True:
        row = exact_row(HOLDER)
        controller, queue = node_occupancy()
        require(set(controller) <= {HOLDER} and queue <= {HOLDER}, "New work appeared during the handoff")
        if not controller and not queue and row[2:4] == ["COMPLETED", "0:0"] and terminal.is_file():
            break
        require(time.monotonic() < deadline, "The exact standby release has not been proved")
        time.sleep(1.5)
    returned = json.loads(read(terminal))
    require((returned["node"], returned["step"], returned["state"]) == (NODE, HOLDER, "WORKER_RETURNED"),
            "Standby did not return normally")
    return row, worker["pci"], plan_sha, worker_sha, handoff


PHYSICAL = r"""
import json,re,socket,sys
sys.path.insert(0,sys.argv[1])
import snapshot
snapshot.NODE='f13r1n14'
snapshot.STEP=re.compile(r'1103606\.\d+')
value=snapshot.snapshot(False)
with socket.socket() as s:
 s.bind(('127.0.0.1',29871))
value.update(master_port=29871,port_clear=True)
print(json.dumps(value,sort_keys=True),flush=True)
"""


def independent_release(directory, owner_stage, request, pci):
    read(owner_stage / "snapshot.py", 65536, SNAPSHOT)
    label = f"llava-standby-post-{request[:12]}"
    argv = ["srun", "--threads=1", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--immediate=5",
            "-N1", "-n1", "-w", NODE, "-c1", "--mem=256M", "--gres=none", "--time=00:00:45",
            PYTHON, "-B", "-c", PHYSICAL, str(owner_stage)]
    save(directory / "standby-post-intent.json", dict(request=request, label=label))
    [output] = query(*argv)
    physical = json.loads(output)
    step = physical["step"]
    require(NUMERIC_STEP.fullmatch(step) and physical["node"] == NODE and physical["full_fd_inventory_performed"] is True
            and physical["kfd_holder_count"] == 0 and physical["port_clear"] is True
            and len(pci) == 8 and set(physical["pci"]) == set(pci), "Independent eight-PCI/KFD/FD/port release failed")
    until = time.monotonic() + 12
    while True:
        row = exact_row(step, allow_missing=True)
        if row and row[2:4] == ["COMPLETED", "0:0"]:
            break
        require(time.monotonic() < until and
                (row is None or row[2] in {"PENDING", "RUNNING", "COMPLETING"}), "No independent CPU terminal")
        time.sleep(.5)
    require(row[1] == label and row[5] == NODE and row[6] == "cpu=1,mem=256M,node=1"
            and node_occupancy() == ({}, set()), "Independent release scope changed")
    return save(directory / "standby-independent-post.json", dict(request=request, accounting=row, physical=physical))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--owner-stage", type=Path, required=True)
    parser.add_argument("--owner-manifest-sha", required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    import fcntl

    request = args.request
    require(re.fullmatch(r"[0-9a-f]{32}", request) and HASH.fullmatch(args.owner_manifest_sha)
            and args.work.parent == RESULTS and args.work.name == f"llava-t04-orderly-control-{request}"
            and args.work.stat().st_uid == os.getuid() and socket.gethostname().split(".")[0] == CONTROL_NODE
            and os.environ.get("SLURM_JOB_ID") == PARENT and os.environ["SLURM_STEP_ID"].isdigit()
            and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "Exact allocated control CPU scope required")
    control_step = PARENT + "." + os.environ["SLURM_STEP_ID"]
    [step] = [row for row in map(fields, query("scontrol", "show", "step", PARENT, "--oneliner"))
              if row.get("StepId") == control_step]
    cgroup = Path("/proc/self/cgroup").read_text()
    require(re.search(rf"/job_{PARENT}(?:\.scope)?/step_{control_step.split('.')[1]}(?:\.scope)?(?:/|$)",
                      cgroup, re.MULTILINE) is not None, "Control PID is outside the numbered Slurm cgroup")
    require((step["StepId"], step["NodeList"], step["State"], step["TRES"], step["Name"])
            == (control_step, CONTROL_NODE, "RUNNING", "cpu=2,mem=2G,node=1", f"llava-t04-control-{request[:12]}"),
            "Actual controller resources differ")
    for key in tuple(os.environ):
        if key.startswith(("SLURM_", "SRUN_", "PMI_", "PMIX_")):
            os.environ.pop(key)
    os.environ.update(LC_ALL="C", TZ="UTC", SLURM_TIME_FORMAT="standard")
    read(STAGED / "staged-ready.json", expected=STAGED_SHA)
    read(STAGED / "helpers/TENSORS.json", expected=TENSORS)
    owner_manifest = json.loads(read(args.owner_stage / "SOURCE_FILES.json", expected=args.owner_manifest_sha))
    files = owner_manifest["files"]
    require(set(files) == {p.name for p in args.owner_stage.iterdir() if p.name != "SOURCE_FILES.json"},
            "Unexpected T04 owner source files")
    for name, pin in files.items():
        require(Path(name).name == name and len(read(args.owner_stage / name, cap=pin["bytes"], expected=pin["SHA256"]))
                == pin["bytes"], "Exact T04 owner member required")
    label = f"training-llava-clevr-first2-{request[:12]}"
    run = RESULTS / label
    lock = os.open(RESULTS / ".training80-dcu-standby.lock", os.O_RDWR | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(lock).st_mode), "Original shared fleet lock unavailable")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not run.exists(), "Existing T04 request cannot be replayed")
        save(args.work / "release-intent.json", dict(request=request, control_step=control_step, frozen_holder=HOLDER))
        row, pci, plan_sha, worker_sha, handoff = release_holder(args.work, request)
        independent = independent_release(args.work, args.owner_stage, request, pci)
        release_path = args.work / "release.json"
        release_pin = save(release_path, dict(
            request=request, mode="ORDERLY_ROOT_HANDOFF" if handoff else "NATURAL", parent=PARENT, node=NODE,
            holder_step=HOLDER, holder_label=STANDBY_LABEL, accounting=row, worker_SHA256=worker_sha,
            plan_SHA256=plan_sha, independent_post_SHA256=independent, worker_returned=True,
            independent_physical_clean=True, exact_root_handoff_confirmed=handoff))
        authority = args.work / "authority.json"
        epoch = time.time()
        authority_pin = save(authority, dict(
            request=request, parent=PARENT, node=NODE, label=label, port=PORT, epoch=epoch, deadline=epoch + 1500,
            GPU_GO=True, node_exclusive_to_request=True, release_SHA256=release_pin,
            owner_manifest_SHA256=args.owner_manifest_sha, science_manifest_SHA256=SCIENCE,
            TENSORS_SHA256=TENSORS, vision_root=str(VISION)))
        argv = [PYTHON, "-I", "-S", "-B", str(args.owner_stage / "owner.py"), "launch",
                "--run", str(run), "--auth", str(authority), "--auth-sha", authority_pin,
                "--release", str(release_path), "--inherited-lock-fd", str(lock)]
        os.set_inheritable(lock, True)
        os.execv(PYTHON, argv)
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
