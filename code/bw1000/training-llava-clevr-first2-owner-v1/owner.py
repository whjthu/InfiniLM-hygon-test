"""One exact authorized LLaVA CPU pre / GPU / independent CPU post sequence."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT = "1103606"
NODE = "f13r1n14"
UID = 154459
TENSORS = "0b6ba7ff7701c165a114bc5460262dff4c053390312d086034e5019d8ecace93"
SCIENCE = "40b4a9c35489288c633dd6c43a87f84fe315cdcd09b25ad65740d0152bd9a576"
PINNED = {
    "snapshot.py": "2e8592caa8b573aa676884b63c65d5a378540d77c877d885424d934a5c4ebcb8",
    "slurm_accounting.py": "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    "owned_processes.py": "17bdaf431efba6e086a7387c78d158c5312747e9d5db262bbb0d644e940afc58",
}
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
HOST_HYHAL = Path("/usr/local/hyhal")
STEP = re.compile(r"[1-9]\d*\.(?:0|[1-9]\d*)")
HASH = re.compile(r"[a-f0-9]{64}")
SUCCESS = ("COMPLETED", "0:0")
GUARDIAN = "training80-dcu-standby-"
STAGED = ROOT / "incoming/training-llava-clevr-first2-rank-cache-0637f8090c664d27a8a2684b8900fafe"
STAGED_SHA = "4f071529a40c245715167f2355407c1830908c93177e678575163dfa2b5164b5"
INPUT = ROOT / "incoming/training-llava-clevr-native64-cpu-v1-d14768cf8bac/input"
MODEL = ROOT / "results/training-llava-original-model18-v1/model"
RUNTIME = ROOT / "results/training-vlm-swift-offline-runtime-v1"
FRAMEWORK = ROOT / "results/training-qwen-a68952c-swift-initial-CPU-v1/framework"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
VISION = ROOT / "results/training-qwen-vision014-CPU-v2/vision"


def need(ok, label):
    if not ok:
        raise ValueError(label)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pin(path, expected=None, limit=256 * 1024):
    need(path.resolve(strict=True) == path, "CANONICAL_TRUSTED_FILE")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as source:
        before = os.fstat(source.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "REGULAR_BOUNDED_TRUSTED_FILE")
        raw = source.read(limit + 1)
        stamp = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        path_stamp = stamp(path.stat())
        need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(source.fileno()))
             and (stamp(before) == path_stamp or os.name == "nt" and stamp(before)[:-1] == path_stamp[:-1]),
             "STABLE_BOUND_TRUSTED_FILE")
    digest = sha256(raw).hexdigest()
    need(expected is None or expected == digest, "EXACT_TRUSTED_FILE_SHA256")
    return raw, digest


def document(path, expected=None):
    raw, digest = pin(path, expected)
    return json.loads(raw), digest


def write(path, value):
    raw = encode(value)
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def scope(document):
    request = document["request"]
    need(document["parent"] == PARENT and document["node"] == NODE and document["GPU_GO"] is True
         and document["node_exclusive_to_request"] is True and re.fullmatch(r"[a-f0-9]{32}", request)
         and document["label"] == "training-llava-clevr-first2-" + request[:12]
         and type(document["port"]) is int and 20000 <= document["port"] <= 60999
         and type(document["epoch"]) in (int, float) and type(document["deadline"]) in (int, float)
         and 0 < document["deadline"] - document["epoch"] <= 1500
         and document["TENSORS_SHA256"] == TENSORS and document["science_manifest_SHA256"] == SCIENCE,
         "EXACT_FRESH_ROOT_AUTHORITY")
    need(all(HASH.fullmatch(document[key]) for key in (
        "release_SHA256", "owner_manifest_SHA256")), "EXACT_ROOT_AUTHORITY_DIGESTS")
    return request


def here_manifest(path, expected, fixed=None):
    manifest, _ = document(path / "SOURCE_FILES.json", expected)
    members = manifest["files"]
    need(type(members) is dict and set(members) == {entry.name for entry in path.iterdir()
         if entry.name != "SOURCE_FILES.json"}, "EXACT_SOURCE_MEMBER_CLOSURE")
    if fixed:
        need(all(members[name]["SHA256"] == digest for name, digest in fixed.items()),
             "PINNED_SHARED_PROCESS_AND_PHYSICAL_PROVIDERS")
    for name, entry in members.items():
        need(Path(name).name == name and len(pin(path / name, entry["SHA256"])[0]) == entry["bytes"],
             "EXACT_PINNED_SOURCE_MEMBER")


def query(argv, seconds=6):
    reply = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                           timeout=seconds, env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
    need(reply.returncode == 0 and len(reply.stdout) < 2 * 1024**2, "BOUNDED_SLURM_QUERY")
    return [line for line in reply.stdout.splitlines() if line.strip()]


def fields(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/]+)=(\S+)", line))


def on_node(expression):
    need(bool(expression) and expression not in {"(null)", "None"}, "EXACT_STEP_NODELIST")
    if expression == NODE:
        return True
    return NODE in query(["scontrol", "show", "hostnames", expression])


def account(job=PARENT, since=None):
    if since is None:
        since = time.strftime("%Y-%m-%dT00:00:00", time.gmtime(time.time() - 86400))
    rows = [line.split("|") for line in query(["sacct", "-nP", "-j", job, "-S", since, "-o", COLS])]
    need(all(len(row) == 9 for row in rows), "EXACT_NINE_UTC_ACCOUNTING_COLUMNS")
    return rows


def matching(rows, step=None, name=None):
    chosen = [row for row in rows if STEP.fullmatch(row[0])
              and (step is None or row[0] == step) and (name is None or row[1] == name)]
    need(len(chosen) <= 1, "UNIQUE_EXACT_NUMERIC_ACCOUNTING_ROW")
    return chosen[0] if chosen else None


def tres(text, gpu, cpus, mem):
    values = [item.split("=", 1) for item in text.split(",")]
    if any(len(pair) != 2 for pair in values):
        return False
    fields = dict(values)
    if len(fields) != len(values) or fields.get("node") != "1" or fields.get("cpu") != str(cpus):
        return False
    gres = {"gres/dcu": str(gpu)} if gpu else {}
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([MG])", fields.get("mem", ""))
    return fields.keys() == ({"cpu", "node", "mem"} | gres.keys()) and all(fields[key] == n for key, n in gres.items()) \
        and bool(match) and float(match[1]) * (1024 if match[2] == "G" else 1) == mem


def check_release(value, actual):
    require_return = value["mode"] == "ORDERLY_ROOT_HANDOFF"
    need(value["mode"] in {"NATURAL", "ORDERLY_ROOT_HANDOFF"} and value["parent"] == PARENT
         and value["node"] == NODE and value["worker_returned"] is True and value["independent_physical_clean"] is True
         and HASH.fullmatch(value["worker_SHA256"]) and HASH.fullmatch(value["plan_SHA256"])
         and HASH.fullmatch(value["independent_post_SHA256"]) and value["holder_step"].startswith(PARENT + ".")
         and STEP.fullmatch(value["holder_step"]) and value["holder_label"].startswith(GUARDIAN)
         and value["holder_label"].endswith("-" + NODE)
         and (not require_return or value["exact_root_handoff_confirmed"] is True), "EXACT_OWNED_STANDBY_RETURN")
    need(actual is not None and actual == value["accounting"]
         and (actual[0], actual[1], actual[5]) == (value["holder_step"], value["holder_label"], NODE)
         and tuple(actual[2:4]) == SUCCESS and int(actual[4]) >= 0 and tres(actual[6], 8, 8, 16384)
         and all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", stamp)
                 for stamp in actual[7:9]), "EXACT_REAL_STANDBY_TERMINAL")


def clear():
    parents = query(["scontrol", "show", "job", PARENT, "--oneliner"])
    need(len(parents) == 1, "EXACT_PARENT_CONTROLLER_ROW")
    parent = fields(parents[0])
    need(parent["JobId"] == PARENT and parent["JobState"] == "RUNNING" and on_node(parent["NodeList"]),
         "PARENT_RUNNING_OWNS_NODE")
    queue = [line.split("|") for line in query(["squeue", "-h", "-j", PARENT, "-o", "%i|%T|%N"])]
    need(len(queue) == 1 and queue[0][:2] == [PARENT, "RUNNING"] and on_node(queue[0][2]),
         "PARENT_RUNNING_QUEUE_OWNS_NODE")
    for raw in query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        row = fields(raw)
        if STEP.fullmatch(row.get("StepId", "")):
            need(not on_node(row["NodeList"]), "CONTROLLER_NUMBERED_STEP_OCCUPANCY")
    for raw in query(["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%j|%N"]):
        row = raw.split("|")
        need(len(row) == 3, "EXACT_QUEUE_STEP_COLUMNS")
        if STEP.fullmatch(row[0]):
            need(not on_node(row[2]), "QUEUE_NUMBERED_STEP_OCCUPANCY")


def role_label(auth, role):
    return auth["label"] + "-" + role


def local_step(role, auth):
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    need(STEP.fullmatch(step) and step.startswith(PARENT + ".") and os.geteuid() == UID
         and socket.gethostname().split(".")[0] == NODE and len(os.sched_getaffinity(0)) >= (64 if role == "gpu" else 2),
         "REAL_COMPUTE_STEP_UID_NODE_AND_CPU_AFFINITY")
    cgroup = Path("/proc/self/cgroup").read_text()
    need(re.search(rf"/job_{PARENT}(?:\.scope)?/step_{step.split('.')[1]}(?:\.scope)?(?:/|$)",
                   cgroup, re.MULTILINE), "REAL_COMPUTE_NUMBERED_CGROUP")
    rows = [fields(row) for row in query(["scontrol", "show", "step", step, "--oneliner"])]
    exact = [row for row in rows if row.get("StepId") == step]
    cpu, memory, gpu = (64, 224000, 8) if role == "gpu" else (2, 2048, 0)
    need(len(exact) == 1 and exact[0].get("Name", exact[0].get("JobName")) == role_label(auth, role)
         and exact[0]["NodeList"] == NODE and exact[0]["State"] == "RUNNING"
         and tres(exact[0]["TRES"], gpu, cpu, memory), "ACTUAL_WORKER_STEP_NAME_NODE_AND_TRES")
    return step


def providers():
    for name, digest in PINNED.items():
        pin(HERE / name, digest)
    sys.path.insert(0, str(HERE))
    import snapshot
    import slurm_accounting
    import owned_processes
    snapshot.NODE = NODE
    snapshot.STEP = re.compile(rf"{PARENT}\.\d+")
    return snapshot, slurm_accounting, owned_processes


def physical(role, auth, run):
    step = local_step(role, auth)
    if role in {"pre", "gpu"}:
        need(HOST_HYHAL.is_dir(), "MISSING_QUALIFIED_N14_HYHAL_SOURCE")
    snapshot, _, owned = providers()
    data = snapshot.snapshot(role == "gpu")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", auth["port"]))
    need(step == data["step"] and data["kfd_holder_count"] == 0 and len(data["pci"]) == len(set(data["pci"])) == 8,
         "EIGHT_CLEAR_PHYSICAL_PCI")
    data.update(request=auth["request"], role=role, port=auth["port"], port_clear=True)
    if role != "pre":
        pre, pre_pin = document(run / "pre.json")
        need(data["boot_id"] == pre["boot_id"] and data["pci"] == pre["pci"], "SAME_NODE_BOOT_AND_PCI")
        data["pre_SHA256"] = pre_pin
        if role == "gpu":
            need(0 <= data["monotonic_ns"] - pre["monotonic_ns"] <= 65_000_000_000,
                 "FRESH_SAME_BOOT_PHYSICAL_PRE")
        else:
            receipts, _ = document(run / "owned-final.json")
            rows = owned.snapshot()
            alive = [pid for pid, row in receipts["registry"].items() if int(pid) in rows
                     and row["startticks"] == rows[int(pid)]["startticks"] and rows[int(pid)]["state"] != "Z"]
            need(receipts["step"] != step and receipts["request"] == auth["request"] and not alive
                 and receipts["cleanup"] is not None and receipts["cleanup"]["owned_processes_remaining"] == 0,
                 "INDEPENDENT_NO_OWNED_CHILDREN_REMAIN")
            data["owned_processes_remaining"] = len(alive)
    return data


def step_spec(role, auth, run):
    cpu, memory, gpu, minutes = (64, 224000, "dcu:8", 25) if role == "gpu" else (2, 2048, "none", 1)
    return ["srun", "--exact", "--immediate=5", "--threads=1", "--jobid=" + PARENT,
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE,
            "--cpus-per-task=" + str(cpu), "--mem=" + str(memory) + "M", "--gres=" + gpu,
            "--time=" + str(minutes), "--job-name=" + role_label(auth, role), HOST_PYTHON, "-I", "-S", "-B",
            str(HERE / "owner.py"), "compute-" + role, "--run", str(run), "--auth-sha", auth["auth_sha"]]


def runtime_command(auth, run, scratch):
    binds = [
        (STAGED / "source", "/llava-source", "ro"), (STAGED / "helpers", "/llava-helpers", "ro"),
        (INPUT, "/llava-train-only", "ro"), (FRAMEWORK, "/llava-framework", "ro"),
        (RUNTIME, "/swift-runtime", "ro"), (Path(auth["vision_root"]), "/vision", "ro"),
        (MODEL, "/t04-model", "ro"), (run, "/llava-run", "rw"), (scratch, "/tmp", "rw"),
        (HOST_HYHAL, "/opt/hyhal", "ro"),
    ]
    argv = [APPTAINER, "exec", "--rocm", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, target, mode in binds:
        argv.extend(("--bind", f"{source.as_posix()}:{target}:{mode}"))
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; "
             "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 "
             "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "
             "OPENBLAS_NUM_THREADS=1 HF_HOME=/tmp/hf MODELSCOPE_CACHE=/tmp/modelscope "
             "HOME=/tmp/home TMPDIR=/tmp/ipc XDG_CACHE_HOME=/tmp/xdg "
             "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor; unset PYTHONPYCACHEPREFIX; "
             "exec /usr/bin/python3 -B -u \"$@\"")
    return [*argv, IMAGE.as_posix(), "/bin/bash", "--noprofile", "--norc", "-c", shell, "t04-first2",
            "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", "8", "--master-addr", "127.0.0.1",
            "--master-port", str(auth["port"]), "--log-dir", "/tmp/elastic",
            "--", "/llava-source/train.py", "--run", "/llava-run", "--model", "/t04-model",
            "--request", auth["request"], "--fingerprint-sha256", TENSORS]


def compute_gpu(auth, run, started):
    import tempfile

    step = started["step"]
    _, _, owned = providers()
    token = auth["request"].encode()
    owned.native_command = lambda argv, _: b"/llava-source/train.py" in argv and token in argv and b"/llava-run" in argv
    registry = {}
    child = None
    rc = None
    failure = None
    clean = None
    scratch = Path(tempfile.mkdtemp(prefix="t04-clevr-", dir="/tmp"))
    for name in ("home", "hf", "modelscope", "ipc", "xdg", "inductor", "elastic"):
        (scratch / name).mkdir(mode=0o700)
    owner = dict(request=auth["request"], step=step, node=NODE, epoch=auth["epoch"], deadline=auth["deadline"],
                 GPU_GO=True, gpu_count=8, port=auth["port"], fingerprint_sha256=TENSORS)
    owner_sha = write(run / "OWNER.json", owner)

    def interrupt(signum, _):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        env = dict(os.environ, LLAVA_OWNER_SHA256=owner_sha, APPTAINERENV_LLAVA_OWNER_SHA256=owner_sha)
        os.chdir("/")
        with (run / "native.log").open("xb") as out:
            child = subprocess.Popen(runtime_command(auth, run, scratch), env=env, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            row = owned.snapshot().get(child.pid)
            if row:
                registry[child.pid] = {key: row[key] for key in ("PID", "PPID", "PGID", "startticks", "state")}
            while rc is None:
                owned.refresh(registry, run)
                need(time.time() < auth["deadline"] - 30, "OWNED_GPU_WORK_BUDGET")
                try:
                    rc = child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        failure = type(error).__name__
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        try:
            clean = owned.cleanup(registry, run, grace=6)
            if child is not None:
                child.wait(timeout=2)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failure = failure or type(error).__name__
        write(run / "owned-final.json", dict(request=auth["request"], step=step, registry=registry, cleanup=clean,
                                             child_returncode=rc, error_class=failure,
                                             scratch_retained=True))
    return 0 if rc == 0 and failure is None and clean and clean["owned_processes_remaining"] == 0 else 2


def verified_role(auth, role, row):
    cpu, memory, gpu = (64, 224000, 8) if role == "gpu" else (2, 2048, 0)
    need(row[1] == role_label(auth, role) and row[5] == NODE and tres(row[6], gpu, cpu, memory),
         "EXACT_TERMINAL_ROLE_AND_TRES")


def await_role(auth, role, deadline):
    _, accounting, _ = providers()
    name = role_label(auth, role)
    return accounting.wait_terminal(lambda: matching(account(), name=name), deadline, interval=1., max_observations=10)


def receipt(role, auth, run, result, rc):
    write(run / f"{role}-accounting.json", dict(request=auth["request"], client_rc=rc, **result))
    row = result["row"]
    need(result["terminal_visible"] and row is not None, "UNKNOWN_NUMBERED_TERMINAL_NEVER_REPLAY")
    verified_role(auth, role, row)
    if role != "gpu":
        proof, _ = document(run / (role + ".json"))
        need(rc == 0 and tuple(row[2:4]) == SUCCESS and proof["step"] == row[0]
             and proof["request"] == auth["request"] and proof["full_fd_inventory_performed"] is True
             and proof["kfd_holder_count"] == 0 and proof["port_clear"] is True,
             "EXACT_ZERO_GRES_PHYSICAL_PHASE_TERMINAL")
    return row


@contextmanager
def held_lock(inherited=None):
    import fcntl

    path = ROOT / "results/.training80-dcu-standby.lock"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW) if inherited is None else inherited
    acquired = False
    try:
        actual, expected = os.fstat(fd), path.lstat()
        need(stat.S_ISREG(actual.st_mode) and stat.S_ISREG(expected.st_mode)
             and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), "EXACT_SHARED_LOCK_FILE")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def staged(auth):
    vision = Path(auth["vision_root"])
    need(vision == VISION and vision.resolve(strict=True) == vision, "EXACT_APPROVED_VISION_SOURCE")
    pin(STAGED / "staged-ready.json", STAGED_SHA)
    need(len(pin(STAGED / "helpers/TENSORS.json", TENSORS)[0]) == 17260, "ORIGINAL_4878_PRIVATE_TENSOR_WITNESS")
    here_manifest(STAGED / "source", SCIENCE)
    here_manifest(HERE, auth["owner_manifest_SHA256"], PINNED)


def authority(path, digest):
    auth, actual = document(path, digest)
    scope(auth)
    auth["auth_sha"] = actual
    return auth


def exclusive_claim(run, args, auth, release_path):
    need(os.geteuid() == UID and auth["epoch"] <= time.time() < auth["epoch"] + 30
         and run.parent == ROOT / "results" and run.name == auth["label"] and not run.exists(),
         "FRESH_LOCAL_ROOT_OWNED_ONCE_NAMESPACE")
    staged(auth)
    release, _ = document(release_path, auth["release_SHA256"])
    need(release["request"] == auth["request"], "EXACT_RELEASE_BOUND_TO_THIS_SUCCESSOR")
    check_release(release, matching(account(release["holder_step"]), step=release["holder_step"]))
    clear()
    existing = account()
    need(all(matching(existing, name=role_label(auth, role)) is None for role in ("pre", "gpu", "post")),
         "EXCLUSIVE_NEVER_REPLAY_ANY_LABEL")
    run.mkdir(mode=0o700)
    for source, name, digest in ((args, "root-authority.json", auth["auth_sha"]),
                                 (release_path, "original-release.json", auth["release_SHA256"])):
        raw, _ = pin(source, digest)
        with (run / name).open("xb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
    write(run / "launch-intent.json", dict(request=auth["request"], root_authority_SHA256=auth["auth_sha"],
                                          release_SHA256=auth["release_SHA256"], planned_forwards=64,
                                          planned_rank_updates=16, post_required_after_gpu=True))


def run_short(role, auth, run):
    with (run / f"{role}.log").open("xb") as out:
        child = subprocess.run(step_spec(role, auth, run), stdin=subprocess.DEVNULL, stdout=out,
                               stderr=subprocess.STDOUT, timeout=65, env={**os.environ, "TZ": "UTC"})
    result = await_role(auth, role, time.monotonic() + 14)
    return receipt(role, auth, run, result, child.returncode)


def gpu_visible(auth, run, client):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and not (run / "gpu-start.json").is_file():
        time.sleep(.3)
    path = run / "gpu-start.json"
    started = document(path)[0] if path.is_file() else None
    row = matching(account(started["step"] if started else PARENT),
                   step=started["step"] if started else None, name=role_label(auth, "gpu"))
    if row is not None:
        verified_role(auth, "gpu", row)
        return row[0]
    # The node's exact compute HOST gate verifies scontrol RUNNING and TRES before writing this receipt.
    need(started is not None and started["request"] == auth["request"] and STEP.fullmatch(started["step"])
         and started["allocated_controller_running"] is True, "EXACT_COMPUTE_CONTROLLER_GPU_OBSERVED")
    return started["step"]


def launch(auth_path, auth_sha, release_path, run, inherited=None):
    auth = authority(auth_path, auth_sha)
    stage = "ADMISSION"
    gpu_row = post_row = training = attempted = client = None
    claimed = False
    failure = None
    try:
        with held_lock(inherited):
            exclusive_claim(run, auth_path, auth, release_path)
            claimed = True
            stage = "PRE"
            write(run / "pre-intent.json", dict(request=auth["request"]))
            run_short("pre", auth, run)
            stage = "GPU_ADMISSION"
            clear()
            now = time.time()
            need(auth["epoch"] <= now < min(auth["epoch"] + 120, auth["deadline"] - 180)
                 and matching(account(), name=role_label(auth, "gpu")) is None, "FRESH_GPU_ONCE_ADMISSION")
            write(run / "gpu-intent.json", dict(request=auth["request"], observed_epoch=now, allocated_DCU=8))
            attempted = True
            stage = "GPU_VISIBILITY"
            with (run / "gpu.log").open("xb") as out:
                client = subprocess.Popen(step_spec("gpu", auth, run), stdin=subprocess.DEVNULL, stdout=out,
                                          stderr=subprocess.STDOUT, env={**os.environ, "TZ": "UTC"})
                observed = gpu_visible(auth, run, client)
        stage = "GPU_TERMINAL"
        client.wait(timeout=max(1, auth["deadline"] - time.time()))
        result = await_role(auth, "gpu", time.monotonic() + 14)
        gpu_row = receipt("gpu", auth, run, result, client.returncode)
        need(gpu_row[0] == observed, "EXACT_STARTED_GPU_TERMINAL_STEP")
        stage = "POST"
        with held_lock():
            clear()
            need(matching(account(), name=role_label(auth, "post")) is None, "EXCLUSIVE_POST_PHASE")
            write(run / "post-intent.json", dict(request=auth["request"], original_gpu_row=gpu_row))
            post_row = run_short("post", auth, run)
        stage = "SCIENCE"
        result = subprocess.run([HOST_PYTHON, "-B", str(STAGED / "source/verify.py"), str(run), auth["request"]],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
        training = result.returncode == 0 and gpu_row[2:4] == list(SUCCESS) and client.returncode == 0
    except BaseException as error:
        failure = dict(phase=stage, error_class=type(error).__name__,
                       error_SHA256=sha256(str(error).encode()).hexdigest())
    outcome = dict(request=auth["request"], planned_forwards=64, planned_rank_updates=16,
                   gpu_launch_intent=bool(attempted), actual_gpu_step=gpu_row[0] if gpu_row else None,
                   gpu_accounting=gpu_row, post_accounting=post_row, scientific_pass=training is True,
                   release_proven=post_row is not None, error=failure,
                   gpu_never_auto_replay=bool(attempted), launch_claimed=claimed)
    if claimed:
        write(run / "controller-outcome.json", outcome)
    print(encode(outcome).decode(), end="", flush=True)
    return 0 if training and post_row and not failure else 2


def read_only(auth, run):
    """Read labels and extant receipts without sending any steps."""
    rows = account()
    result = dict(request=auth["request"], receipts={}, accounting={})
    for role in ("pre", "gpu", "post"):
        result["accounting"][role] = matching(rows, name=role_label(auth, role))
    for name in ("launch-intent.json", "pre-intent.json", "gpu-intent.json", "gpu-start.json", "post-intent.json",
                 "pre.json", "post.json", "owned-final.json", "independent-first2.json", "controller-outcome.json"):
        path = run / name
        result["receipts"][name] = pin(path)[1] if path.is_file() else None
    print(encode(result).decode(), end="", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "observe", "compute-pre", "compute-gpu", "compute-post"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--auth", type=Path)
    parser.add_argument("--auth-sha", required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--inherited-lock-fd", type=int)
    args = parser.parse_args()
    need(args.run.is_absolute() and HASH.fullmatch(args.auth_sha), "ABSOLUTE_RUN_AND_EXACT_AUTH_SHA")
    if args.mode == "launch":
        need(args.auth is not None and args.release is not None, "AUTHORITATIVE_ROOT_AND_RELEASE")
        need(args.inherited_lock_fd is None or args.inherited_lock_fd > 2, "EXACT_OPTIONAL_SHARED_LOCK_FD")
        return launch(args.auth, args.auth_sha, args.release, args.run, args.inherited_lock_fd)
    auth = authority(args.auth or args.run / "root-authority.json", args.auth_sha)
    need(args.run == ROOT / "results" / auth["label"], "EXACT_REQUEST_RUN")
    if args.mode == "observe":
        return read_only(auth, args.run)
    role = args.mode.removeprefix("compute-")
    need(os.geteuid() == UID and args.run.stat().st_uid == UID
         and stat.S_IMODE(args.run.stat().st_mode) == 0o700, "EXACT_PRIVATE_COMPUTE_RESULT_ROOT")
    need(auth["epoch"] <= time.time() < auth["deadline"] + (90 if role == "post" else 0), "ACTIVE_OWNER_EPOCH")
    evidence_path = args.run / (("gpu" if role == "gpu" else role) + "-intent.json")
    intent, _ = document(evidence_path)
    need(intent["request"] == auth["request"], "EXACT_COMPUTE_INTENT_BINDING")
    if role == "gpu":
        step = local_step(role, auth)
        entered = False
        try:
            data = physical(role, auth, args.run)
            data["allocated_controller_running"] = True
            write(args.run / "gpu-start.json", data)
            staged(auth)
            entered = True
            return compute_gpu(auth, args.run, data)
        finally:
            if not entered and not (args.run / "OWNER.json").exists():
                write(args.run / "owned-final.json", dict(
                    request=auth["request"], step=step, registry={}, child_started=False,
                    cleanup=dict(owned_processes_remaining=0, termination_needed=False)))
    data = physical(role, auth, args.run)
    write(args.run / (role + ".json"), data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
