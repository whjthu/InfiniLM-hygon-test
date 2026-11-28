"""Exclusive LLaVA Val16 source stage, one-card owner, and readonly terminal reader."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
PARENT, NODE, UID = "1103606", "f13r1n14", 154459
PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
MEDIA = ROOT / "incoming/llava-clevr-original5000-val16-c5359ee770ad416f8faf4ebacb7e5c5c"
REQUESTS = "8e3babd94382c39957af6b5209686ea39b62de658cceabf909f52e6b7fe79248"
IMAGES = "1808e416bd6817087fbb6bea68de69d0c578affb2c5b1a6f19d93587a952878b"
NATIVE_MANIFEST = "198f86cb60518786de56d988f2568e88e2c35b3ca0827620be73b058d1bc962d"
CACHE_QUALIFICATION = "7e4bf2dcb4e5cf35ae306a842177ed8f8c3d26aaff1f4327338331de0e44e83a"
NATIVE_WORKER = {"native.py", "contract.py", "repair.py", "two_forward.py"}
NATIVE_SOURCE = NATIVE_WORKER | {"HANDOFF.md", "test_local.py"}
ORIGINAL_REQUEST_PREFIXES = {"330c1477ec87", "acc07cff09e6", "0f17a330cc2b", "354e5ebd0e01"}
RELOAD = "1f5f85c267c47d8a48c569262c3eddff2b0c47962850887abf326b92ac33063b"
CP25 = "b8b8a8b9dfe5012c4ce5f80499a1b3fc5e7b8b140a7a734cada07c4a39b83719"
SUPPORT = {
    "owner.py": "0fcebef4691f43ce6dbda091fb251f63456ff63051999a15e2d8ad9152688bd8",
    "snapshot.py": "2e8592caa8b573aa676884b63c65d5a378540d77c877d885424d934a5c4ebcb8",
    "slurm_accounting.py": "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    "owned_processes.py": "17bdaf431efba6e086a7387c78d158c5312747e9d5db262bbb0d644e940afc58",
}
HELPERS = ROOT / "incoming/training-llava-clevr-first2-rank-cache-0637f8090c664d27a8a2684b8900fafe/helpers"
MODEL = RESULTS / "training-llava-original-model18-v1"
TRAINED = RESULTS / "training-llava-clevr-resume25-0e4a27af103c"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
SWIFT = RESULTS / "training-vlm-swift-offline-runtime-v1"
VISION = RESULTS / "training-qwen-vision014-CPU-v2/vision"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
HEX = re.compile(r"[0-9a-f]{64}")
STEP = re.compile(PARENT + r"\.\d+")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def wire(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def read(path, digest=None, limit=1 << 20):
    need(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), "EXACT_CANONICAL_FILE")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                           getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= limit, "BOUNDED_LOCAL_MEMBER")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "STABLE_LOCAL_MEMBER")
    need(digest is None or HEX.fullmatch(digest) and sha256(raw).hexdigest() == digest, "EXACT_MEMBER_SHA")
    return raw


def label(request):
    need(re.fullmatch(r"[0-9a-f]{32}", request) and request[:12] not in ORIGINAL_REQUEST_PREFIXES,
         "FRESH_REQUEST_ID")
    return "llava-clevr-5000-val16-v4-" + request[:12]


def stage_dir(request):
    return ROOT / "incoming" / (label(request) + "-stage")


def run_dir(request):
    return RESULTS / label(request)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def binder(target):
    for name, pin in SUPPORT.items():
        read(target / "support" / name, pin, 128 << 10)
    old = module(target / "support/owner.py", "qualified_previous_owner")
    sys.path.insert(0, str(target / "support"))
    try:
        snapshot, accounting, owned = old.providers()
    finally:
        sys.path.remove(str(target / "support"))
    return old, snapshot, accounting, owned


def source_dir(path, manifest_sha, selected, stage_members=None):
    raw = read(path / "SOURCE_FILES.json", manifest_sha, 16 << 10)
    declared = json.loads(raw)
    manifest = declared.get("files", declared)
    need(set(manifest) == selected, "EXACT_SOURCE_MANIFEST_CLOSURE")
    members = {name: read(path / name, manifest[name]["SHA256"], 128 << 10)
               for name in (selected if stage_members is None else stage_members)}
    need(all(len(blob) == manifest[name]["bytes"] for name, blob in members.items()), "SOURCE_EXTENT_DIFFERS")
    return dict(members, **{"SOURCE_FILES.json": raw})


def projection(media, target, requests_sha, images_sha):
    roster = json.loads(read(media / "IMAGES.json", images_sha, 16 << 10))
    read(media / "requests.jsonl", requests_sha, 48 << 10)
    names = tuple(f"{slot:02d}.png" for slot in range(16))
    need(len(roster) == len(names) and media.resolve(strict=True) == media
         and (media / "images").resolve(strict=True) == media / "images", "PINNED_MEDIA_CLOSURE")
    for slot, (name, pin) in enumerate(zip(names, roster, strict=True)):
        need(pin["slot"] == slot and pin["name"] in (name, "images/" + name), "PINNED_MEDIA_SLOT")
        need(len(read(media / "images" / name, pin["SHA256"], 2 << 20)) == pin["bytes"], "PINNED_MEDIA_BYTES")
    target.mkdir(mode=0o700)
    (target / "images").mkdir(mode=0o700)
    for relative in ("requests.jsonl", "IMAGES.json", *("images/" + name for name in names)):
        os.link(media / relative, target / relative, follow_symlinks=False)
        need(os.path.samefile(media / relative, target / relative), "EXACT_HARDLINK_PROJECTION")


def cache_proof(path):
    raw = read(path, CACHE_QUALIFICATION, 32 << 10)
    archive = json.loads(raw)
    worker = archive["actual_worker"]
    need(archive["new_accounting"][:4] ==
         ["1103606.5226", "llava-cache-two-forward-354e5ebd0e01", "COMPLETED", "0:0"]
         and archive["actual_native_BF16_CPU_forward_returns"] == 2
         and worker["passed"] is True and worker["heldout_inputs_opened"] == 0
         and worker["facts"]["gpu_initialized"] is False, "EXACT_5226_NATIVE_CPU_CACHE_REPAIR")
    return raw


def staged(request, expected=None):
    target = stage_dir(request)
    raw = read(target / "stage.json", expected, 32 << 10)
    value = json.loads(raw)
    need(value["request"] == request and value["native_manifest_SHA256"] == NATIVE_MANIFEST
         and value["requests_SHA256"] == REQUESTS and value["images_SHA256"] == IMAGES
         and value["cache_qualification_SHA256"] == CACHE_QUALIFICATION
         and value["GPU_GO"] is False, "NEUTRAL_STAGE_ONLY_SCOPE")
    source_dir(target / "native", NATIVE_MANIFEST, NATIVE_SOURCE, NATIVE_WORKER)
    cache_proof(target / "cache-proof.json")
    need({item.name for item in (target / "native").iterdir()} ==
         NATIVE_WORKER | {"llava_clevr_cp25_reload.py", "SOURCE_FILES.json"}, "ONLY_NATIVE_WORKER_STAGE")
    read(target / "native/llava_clevr_cp25_reload.py", RELOAD)
    read(target / "host.py", value["host_SHA256"], 128 << 10)
    return target, value, sha256(raw).hexdigest()


def stage(args):
    target = stage_dir(args.request)
    source = source_dir(args.native_source, NATIVE_MANIFEST, NATIVE_SOURCE, NATIVE_WORKER)
    qualification = cache_proof(args.cache_proof)
    support = {name: read(args.support / name, pin, 128 << 10) for name, pin in SUPPORT.items()}
    reload = read(args.reload_tool, RELOAD)
    host = read(Path(__file__).resolve(), args.host_sha, 128 << 10)
    target.mkdir(mode=0o700)
    for folder, members in (("native", dict(source, **{"llava_clevr_cp25_reload.py": reload})), ("support", support)):
        (target / folder).mkdir(mode=0o700)
        for name, raw in members.items():
            put(target / folder / name, raw)
    put(target / "host.py", host)
    put(target / "cache-proof.json", qualification)
    projection(MEDIA, target / "inputs", REQUESTS, IMAGES)
    need({p.name for p in (target / "inputs").iterdir()} == {"requests.jsonl", "IMAGES.json", "images"},
         "ANSWERLESS_PROJECTION_READBACK")
    record = dict(request=args.request, host_SHA256=args.host_sha, native_manifest_SHA256=NATIVE_MANIFEST,
                  requests_SHA256=REQUESTS, images_SHA256=IMAGES,
                  cache_qualification_SHA256=CACHE_QUALIFICATION, planned=32, GPU_GO=False)
    print(json.dumps(dict(record, stage_SHA256=put(target / "stage.json", wire(record))), sort_keys=True))


def physical(target, auth, role, work, old, snapshot, owned):
    step = os.environ.get("SLURM_JOB_ID", "") + "." + os.environ.get("SLURM_STEP_ID", "")
    cpus, mem, gpu, _ = shape(role)
    need(STEP.fullmatch(step) and os.geteuid() == UID and socket.gethostname().split(".")[0] == NODE
         and len(os.sched_getaffinity(0)) >= cpus
         and re.search(rf"/job_{PARENT}(?:\.scope)?/step_{step.split('.')[1]}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE), "EXACT_NUMBERED_UID_NODE_CGROUP")
    rows = [old.fields(line) for line in old.query(["scontrol", "show", "step", step, "--oneliner"])]
    matched = [row for row in rows if row.get("StepId") == step]
    need(len(matched) == 1 and matched[0]["State"] == "RUNNING" and matched[0]["NodeList"] == NODE
         and matched[0].get("Name", matched[0].get("JobName")) == label(auth["request"]) + "-" + role
         and old.tres(matched[0]["TRES"], gpu, cpus, mem), "EXACT_ACTIVE_STEP_LABEL_TRES")
    data = snapshot.snapshot(False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", auth["port"]))
    need(data["step"] == step and data["kfd_holder_count"] == 0 and data["full_fd_inventory_performed"] is True
         and data["pci"] == auth["pci"] and data["boot_id"] == auth["boot"], "EXACT_PHYSICAL_IDLE_TOPOLOGY")
    if role != "pre":
        pre = json.loads(read(work / "pre.json", auth.get("pre_SHA256")))
        need(pre["request"] == auth["request"] and pre["boot_id"] == data["boot_id"]
             and pre["pci"] == data["pci"], "SAME_BOOT_AND_PRE_ROSTER")
        if role == "gpu":
            need(0 <= data["monotonic_ns"] - pre["monotonic_ns"] <= 65_000_000_000, "FRESH_PHYSICAL_PRE")
        else:
            if (work / "gpu-worker-intent.json").is_file():
                final = json.loads(read(work / "owned-final.json"))
                live = owned.snapshot()
                need(final["request"] == auth["request"] and final["step"] == auth["prior_gpu_step"]
                     and final["cleanup"]["owned_processes_remaining"] == 0
                     and all(int(pid) not in live or live[int(pid)]["startticks"] != row["startticks"]
                             or live[int(pid)]["state"] == "Z" for pid, row in final["registry"].items()),
                     "INDEPENDENT_OWNED_PROCESS_RETURN")
            else:
                need(not (work / "owner/OWNER.json").exists(), "OWNER_WITHOUT_CHILD_INTENT_IS_UNKNOWN")
            data["owned_processes_remaining"] = 0
    data.update(request=auth["request"], role=role, port_clear=True)
    put(work / (role + ".json"), wire(data))
    return step


def shape(role):
    return (8, 64000, 1, 30) if role == "gpu" else (2, 2048, 0, 1)


def step_spec(target, work, request, role, auth_sha):
    cpus, mem, gpu, mins = shape(role)
    return ["srun", "--exact", "--immediate=5", "--threads=1", "--jobid=" + PARENT, "--nodes=1", "--ntasks=1",
            "--nodelist=" + NODE, "--cpus-per-task=" + str(cpus), "--mem=" + str(mem) + "M",
            "--gres=" + ("dcu:1" if gpu else "none"), "--time=" + str(mins),
            "--job-name=" + label(request) + "-" + role, PYTHON, "-I", "-S", "-B", str(target / "host.py"),
            "compute-" + role, "--request", request, "--auth-sha", auth_sha]


def guest(target, work, scratch, args, owner_sha):
    dirs = ((target / "native", "/scope", "ro"), (target / "inputs", "/llava-blind", "ro"),
            (HELPERS, "/llava-helpers", "ro"), (FRAMEWORK, "/llava-framework", "ro"),
            (SWIFT, "/swift-runtime", "ro"), (VISION, "/vision", "ro"), (MODEL, "/llava-base", "ro"),
            (TRAINED, "/llava-trained", "ro"), (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
            (work / "owner", "/owner", "ro"), (work / "eval", "/eval-output", "rw"), (scratch, "/tmp", "rw"))
    argv = [APPTAINER, "exec", "--rocm", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, dest, mode in dirs:
        argv += ["--bind", f"{source.as_posix()}:{dest}:{mode}"]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; "
             "unset PYTHONPYCACHEPREFIX MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_DIR MIOPEN_LOCKFILE_PATH; "
             "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "
             "PYTHONNOUSERSITE=1 HOME=/tmp/home TMPDIR=/tmp/ipc HF_HOME=/tmp/hf MODELSCOPE_CACHE=/tmp/modelscope "
             "XDG_CONFIG_HOME=/tmp/home/.config XDG_CACHE_HOME=/tmp/home/.cache OMP_NUM_THREADS=8 "
             "OPENBLAS_NUM_THREADS=1 MIOPEN_USER_DB_PATH=/tmp/home/.config/miopen/user-db "
             "MIOPEN_CUSTOM_CACHE_DIR=/tmp/home/.cache/miopen; exec /usr/bin/python3 -B -u \"$@\"")
    return [*argv, str(IMAGE), "/bin/bash", "--noprofile", "--norc", "-c", shell, "llava-val16",
            "/scope/native.py", "run", "--input", "/llava-blind", "--requests-sha", REQUESTS, "--images-sha",
            IMAGES, "--owner", "/owner/OWNER.json", "--owner-sha", owner_sha, "--request", args["request"],
            "--output", "/eval-output/once", "--model", "/llava-base/model", "--model-receipt", "/llava-base/receipt.json",
            "--resume-receipt", "/llava-trained/independent-resume25.json", "--checkpoint",
            "/llava-trained/train/checkpoint-25", "--reload-tool", "/scope/llava_clevr_cp25_reload.py"]


def gpu_compute(target, work, auth, step, owned):
    scratch = Path(tempfile.mkdtemp(prefix="llava-val16-", dir="/tmp"))
    for name in ("home", "home/.config", "home/.config/miopen", "home/.config/miopen/user-db",
                 "home/.config/miopen/miopen-lockfiles", "home/.cache", "home/.cache/miopen", "ipc", "hf", "modelscope"):
        (scratch / name).mkdir(mode=0o700)
    owner = dict(request=auth["request"], step=step, node=NODE, epoch=auth["epoch"], deadline=auth["deadline"],
                 gpu_count=1, GPU_GO=True, boot=auth["boot"], pci=auth["pci"], requests_SHA256=REQUESTS,
                 images_SHA256=IMAGES, output="/eval-output/once", checkpoint_SHA256=CP25)
    pin = put(work / "owner/OWNER.json", wire(owner))
    token = auth["request"].encode()
    owned.native_command = lambda argv, _: b"/scope/native.py" in argv and token in argv and b"/eval-output/once" in argv
    child = rc = cleanup = error = None
    registry = {}

    def interrupt(signum, _):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        put(work / "gpu-worker-intent.json", wire(dict(request=auth["request"], step=step, owner_SHA256=pin)))
        with (work / "native.log").open("xb") as out:
            child = subprocess.Popen(guest(target, work, scratch, auth, pin), cwd="/", stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            row = owned.snapshot().get(child.pid)
            if row:
                registry[child.pid] = {key: row[key] for key in ("PID", "PPID", "PGID", "startticks", "state")}
            while rc is None:
                owned.refresh(registry, work)
                need(time.time() < auth["deadline"] - 20, "GPU_CHILD_DEADLINE")
                try:
                    rc = child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as caught:
        error = type(caught).__name__
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        try:
            cleanup = owned.cleanup(registry, work, grace=6)
            if child is not None:
                child.wait(timeout=2)
        except (OSError, ValueError, subprocess.SubprocessError) as caught:
            error = error or type(caught).__name__
        put(work / "owned-final.json", wire(dict(request=auth["request"], step=step, child_rc=rc,
            registry=registry, cleanup=cleanup, error=error, scratch_retained=True)))
    return 0 if rc == 0 and error is None and cleanup["owned_processes_remaining"] == 0 else 2


def scope(auth, request):
    need(set(auth) == {"request", "parent", "node", "epoch", "deadline", "port", "boot", "pci",
         "stage_SHA256", "root_GO", "ordinary_excluded", "idle_release_clean", "prior_holder",
         "prior_release_SHA256"}, "EXACT_ROOT_AUTHORITY_FIELDS")
    need(auth["request"] == request and auth["parent"] == PARENT and auth["node"] == NODE
         and auth["root_GO"] is auth["ordinary_excluded"] is auth["idle_release_clean"] is True
         and all(type(auth[key]) in (int, float) for key in ("epoch", "deadline"))
         and 180 < auth["deadline"] - auth["epoch"] <= 2250
         and type(auth["port"]) is int and 20000 <= auth["port"] <= 60999
         and re.fullmatch(r"[0-9a-f-]{36}", auth["boot"])
         and type(auth["pci"]) is list and sorted(set(auth["pci"])) == auth["pci"] and len(auth["pci"]) == 8
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in auth["pci"])
         and all(HEX.fullmatch(auth[key]) for key in ("stage_SHA256", "prior_release_SHA256")),
         "EXACT_SINGLE_CARD_ROOT_AUTHORITY")
    return auth


def no_occupancy(old):
    parent = [old.fields(line) for line in old.query(["scontrol", "show", "job", PARENT, "--oneliner"])]
    queue = [line.split("|") for line in old.query(["squeue", "-h", "-j", PARENT, "-o", "%i|%T|%N"])]
    need(len(parent) == len(queue) == 1 and parent[0]["JobId"] == PARENT
         and parent[0]["JobState"] == "RUNNING" and parent[0]["UserId"].endswith(f"({UID})")
         and len(queue[0]) == 3 and queue[0][:2] == [PARENT, "RUNNING"]
         and old.on_node(parent[0]["NodeList"]) and old.on_node(queue[0][2]), "EXACT_DUAL_PARENT_UID_AND_MEMBERSHIP")
    records = [(old.fields(line)["StepId"], old.fields(line)["NodeList"])
               for line in old.query(["scontrol", "show", "step", PARENT, "--oneliner"])]
    for line in old.query(["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"]):
        step, nodes = line.split("|")
        records.append((step, nodes))
    for step, nodes in records:
        if step not in (PARENT + ".batch", PARENT + ".extern"):
            need(STEP.fullmatch(step) and not old.on_node(nodes), "NUMBERED_NODE_OCCUPANCY_OR_UNKNOWN_STEP")


def role_row(old, request, role, terminal=False):
    row = old.matching(old.account(), name=label(request) + "-" + role)
    if row is None:
        return None
    cpu, mem, gpu, _ = shape(role)
    need(row[5] == NODE and old.tres(row[6], gpu, cpu, mem), "EXACT_ACCOUNTING_ROLE_SCOPE")
    if terminal:
        need(row[2] in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
             and re.fullmatch(r"\d+:\d+", row[3]) and row[4].isdigit()
             and all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", x) for x in row[7:9])
             and row[7] <= row[8], "ORIGINAL_NUMBERED_TERMINAL_UNKNOWN")
    return row


def prior_holder(auth, old):
    row = auth["prior_holder"]
    need(type(row) is list and len(row) == 9 and STEP.fullmatch(row[0])
         and row[5] == NODE and row[2:4] == ["COMPLETED", "0:0"]
         and (old.tres(row[6], 8, 8, 16384) or old.tres(row[6], 1, 8, 64000)
              or old.tres(row[6], 8, 64, 224000))
         and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", row[8])
         and old.matching(old.account(row[0]), step=row[0], name=row[1]) == row,
         "EXACT_PRIOR_IDLE_RELEASE_ACCOUNTING")


def release_document(auth, data):
    need(type(data) is dict and set(data) == {"request", "parent", "node", "holder", "independent_physical_clean",
         "worker_returned", "source_SHA256"} and data["request"] == auth["request"] and data["parent"] == PARENT
         and data["node"] == NODE and data["holder"] == auth["prior_holder"]
         and data["independent_physical_clean"] is data["worker_returned"] is True
         and HEX.fullmatch(data["source_SHA256"]), "EXACT_INDEPENDENT_ROOT_RELEASE_PROOF")


def compute(args):
    work = run_dir(args.request)
    auth = scope(json.loads(read(work / "auth.json", args.auth_sha)), args.request)
    target, _, _ = staged(args.request, auth["stage_SHA256"])
    old, snapshot, _, owned = binder(target)
    role = args.mode.removeprefix("compute-")
    selected = json.loads(read(work / (role + "-intent.json")))
    need(selected["request"] == args.request and selected["auth_SHA256"] == args.auth_sha
         and auth["epoch"] <= time.time() < auth["deadline"], "EXACT_ROLE_INTENT")
    if role != "pre":
        auth["pre_SHA256"] = selected["pre_SHA256"]
    if role == "post":
        terminal = role_row(old, args.request, "gpu", terminal=True)
        need(terminal is not None and terminal[0] == selected["gpu_step"], "POST_REQUIRES_EXACT_GPU_TERMINAL")
        auth["prior_gpu_step"] = terminal[0]
    step = physical(target, auth, role, work, old, snapshot, owned)
    return gpu_compute(target, work, auth, step, owned) if role == "gpu" else 0


def wait_role(old, accounting, request, role):
    detail = accounting.wait_terminal(lambda: role_row(old, request, role), time.monotonic() + 18,
                                      interval=1, max_observations=12)
    row = detail["row"]
    need(detail["terminal_visible"] and row is not None and role_row(old, request, role, terminal=True) == row,
         "ORIGINAL_NUMBERED_TERMINAL_UNKNOWN")
    return row


def proof(work, request, role, row):
    raw = read(work / (role + ".json"), limit=64 << 10)
    result = json.loads(raw)
    need(result["request"] == request and result["role"] == role and result["step"] == row[0]
         and result["full_fd_inventory_performed"] is True and result["kfd_holder_count"] == 0
         and result["port_clear"] is True, "EXACT_ORIGINAL_PHYSICAL_RECEIPT")
    return result, sha256(raw).hexdigest()


def short(target, work, request, role, auth_sha, old, accounting):
    with (work / (role + ".log")).open("xb") as stream:
        child = subprocess.run(step_spec(target, work, request, role, auth_sha), stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=65, cwd="/")
    row = wait_role(old, accounting, request, role)
    need(child.returncode == 0 and row[2:4] == ["COMPLETED", "0:0"], "EXACT_CPU_PRE_OR_POST_COMPLETED")
    proof(work, request, role, row)
    put(work / (role + "-accounting.json"), wire(row))
    return row


BASE_PREDICATES = frozenset({
    "native_class_identity", "peft_config_absent", "parameter_tensors_698",
    "only_cuda0", "only_floating_bfloat16", "original_template", "eager_attention", "rope_initialized",
    "zero_lora_parameters", "finalizer_called_once", "finalizer_missing_keys_zero",
    "finalizer_unexpected_keys_zero", "finalizer_mismatched_keys_zero",
})


def base_predicates(diagnostic):
    predicates = diagnostic["predicates"]
    need(set(predicates) == BASE_PREDICATES
         and all(value is None or type(value) is bool for value in predicates.values()),
         "EXACT_NATIVE_BASE_PREDICATES")
    return predicates


def repair_counts(facts, complete):
    value = facts["cache_position_repair"]
    need(set(value) == {"qualification_SHA256", "arms"} and value["qualification_SHA256"] == CACHE_QUALIFICATION
         and set(value["arms"]) == {"base", "cp25"}, "EXACT_CACHE_REPAIR_QUALIFICATION")
    for counts in value["arms"].values():
        need(set(counts) == {"slots", "decisions", "none_positions", "supplied_positions"}
             and all(type(n) is int and n >= 0 for n in counts.values())
             and counts["slots"] <= 16 and (not complete or counts["slots"] == 16)
             and counts["slots"] <= counts["decisions"] <= counts["slots"] * 128
             and counts["decisions"] == counts["none_positions"] + counts["supplied_positions"],
             "EXACT_CACHE_REPAIR_ARM_COUNTS")
    return value["arms"]


def observe(request, old):
    work = run_dir(request)
    roles = {role: role_row(old, request, role) for role in ("pre", "gpu", "post")}
    result = dict(request=request, planned=32, accounting=roles,
                  gpu_intent=(work / "gpu-intent.json").is_file(), sealed=False, cleanup_verified=False, scoring_GO=False)
    if not (work / "auth.json").is_file():
        return result
    auth = scope(json.loads(read(work / "auth.json")), request)
    gpu = None
    if all(row is not None for row in roles.values()) and roles["post"][2:4] == ["COMPLETED", "0:0"]:
        try:
            pre, gpu, post = [role_row(old, request, role, terminal=True) for role in ("pre", "gpu", "post")]
            before, _ = proof(work, request, "pre", pre)
            after, _ = proof(work, request, "post", post)
            selected = json.loads(read(work / "post-intent.json"))
            need(selected["request"] == request and selected["gpu_step"] == gpu[0]
                 and before["pci"] == after["pci"] == auth["pci"] and before["boot_id"] == after["boot_id"] == auth["boot"]
                 and pre[8] <= gpu[7] <= gpu[8] <= post[7] and after["owned_processes_remaining"] == 0
                 and pre[2:4] == ["COMPLETED", "0:0"], "INDEPENDENT_MATCHED_PRE_GPU_POST")
            result["cleanup_verified"] = True
        except FileNotFoundError:
            result["cleanup_evidence"] = "ORIGINAL_PHYSICAL_RECEIPT_MISSING"
    folder = work / "eval/once"
    if not (folder / "result.json").is_file():
        return result
    seal_raw = read(folder / "result.json", limit=64 << 10)
    parsed = json.loads(seal_raw)
    ledger = read(folder / "predictions.jsonl", parsed["predictions_SHA256"], 512 << 10)
    rows = [json.loads(line) for line in ledger.splitlines()]
    need(len(ledger) == parsed["predictions_bytes"] and parsed["request"] == request and parsed["planned"] == 32
         and parsed["gold_opened"] is parsed["scoring_performed"] is False
         and [(row["arm"], row["slot"]) for row in rows] ==
         [(arm, i) for arm in ("base", "cp25") for i in range(16)]
         and all(set(row) == {"arm", "slot", "outcome", "text"}
                 and row["outcome"] in ("cap", "eos", "stop", "error", "missing")
                 and ((type(row["text"]) is str) if row["outcome"] in ("cap", "eos", "stop") else row["text"] is None)
                 for row in rows), "EXACT_COMPLETE_UNSCORED_LEDGER")
    need(parsed["counts"] == {arm: {key: sum(row["arm"] == arm and row["outcome"] == key for row in rows)
                                   for key in ("cap", "eos", "error", "missing", "stop")}
                                 for arm in ("base", "cp25")}, "EXACT_LEDGER_OUTCOME_COUNTS")
    counts = repair_counts(parsed["facts"], parsed.get("passed") is True)
    matched = all(counts[arm]["slots"] == 16 - sum(parsed["counts"][arm][key] for key in ("error", "missing"))
                  for arm in counts)
    need(parsed.get("passed") is not True or matched, "CACHE_REPAIR_AND_LEDGER_COUNTS_MATCH")
    result.update(sealed=True, seal_SHA256=sha256(seal_raw).hexdigest(), predictions_SHA256=parsed["predictions_SHA256"],
                  predictions_bytes=len(ledger), counts=parsed["counts"], cache_position_counts=counts,
                  cache_position_counts_match_ledger=matched)
    if roles["gpu"] is not None:
        need(roles["gpu"][0] == parsed["facts"]["step"], "SEALED_EXACT_GPU_STEP")
        failure = parsed["failure"]
        if failure is not None:
            failure_summary = {key: failure[key] for key in ("phase", "arm", "slot", "type", "detail_SHA256")}
            diagnostic = failure.get("base_diagnostic")
            if diagnostic is not None:
                failure_summary["base_predicates"] = base_predicates(diagnostic)
            result["worker_failure"] = failure_summary
    if not result["cleanup_verified"]:
        return result
    if parsed["passed"] is True:
        diagnostic = parsed["facts"]["base_diagnostic"]
        need(all(value is True for value in base_predicates(diagnostic).values())
             and diagnostic["parameter_tensors"] == 698 and diagnostic["lora_parameter_tensors"] == 0
             and diagnostic["checkpoint_finalizer"] ==
                 dict(calls=1, missing_keys=0, unexpected_keys=0, mismatched_keys=0),
             "EXACT_UNADAPTED_BASE_AND_ASSIGNMENT")
        result["base_checkpoint_assignment"] = diagnostic["checkpoint_finalizer"]
        result["base_parameter_tensors"] = diagnostic["parameter_tensors"]
    result["scoring_GO"] = (gpu[2:4] == ["COMPLETED", "0:0"] and parsed["passed"] is True
                            and parsed["facts"]["model_loads"] == 1 and parsed["facts"]["actual_generate_calls"] == 32
                            and parsed["facts"]["paired_equal_native_inputs"] == 16
                            and parsed["facts"]["adapter_tensors_verified"] == 504)
    return result


def start(args):
    raw = read(args.auth, args.auth_sha, 16 << 10)
    request = args.request
    auth = scope(json.loads(raw), request)
    need(os.geteuid() == UID and auth["epoch"] <= time.time() < auth["epoch"] + 55, "FRESH_ROOT_IDLE_GRANT")
    release = read(args.release, auth["prior_release_SHA256"], 16 << 10)
    release_document(auth, json.loads(release))
    target, _, _ = staged(request, auth["stage_SHA256"])
    old, _, accounting, _ = binder(target)
    work = run_dir(request)
    claimed, attempted, seen, row, failure = False, False, None, None, None
    phase = "admission"
    try:
        with old.held_lock():
            prior_holder(auth, old)
            no_occupancy(old)
            need(all(role_row(old, request, role) is None for role in ("pre", "gpu", "post"))
                 and time.time() < auth["epoch"] + 55, "FRESH_EXCLUSIVE_LABELS")
            work.mkdir(mode=0o700)
            claimed = True
            for name in ("owner", "eval"):
                (work / name).mkdir(mode=0o700)
            put(work / "auth.json", raw)
            put(work / "prior-release.json", release)
            phase = "pre"
            put(work / "pre-intent.json", wire(dict(request=request, auth_SHA256=args.auth_sha)))
            before = short(target, work, request, "pre", args.auth_sha, old, accounting)
            _, pre_sha = proof(work, request, "pre", before)
            phase = "gpu_admission"
            no_occupancy(old)
            need(time.time() < auth["epoch"] + 105 and role_row(old, request, "gpu") is None, "GPU_ADMISSION_EXPIRED")
            intent = dict(request=request, auth_SHA256=args.auth_sha, pre_SHA256=pre_sha)
            put(work / "gpu-intent.json", wire(intent))
            attempted = True
            phase = "gpu_visibility"
            with (work / "gpu.log").open("xb") as out:
                client = subprocess.Popen(step_spec(target, work, request, "gpu", args.auth_sha), cwd="/",
                                          stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
                until = time.monotonic() + 40
                while time.monotonic() < until and not (work / "gpu.json").is_file():
                    time.sleep(.25)
                known = role_row(old, request, "gpu")
                if known is not None:
                    seen = known[0]
                elif (work / "gpu.json").is_file():
                    evidence = json.loads(read(work / "gpu.json"))
                    need(evidence["request"] == request and evidence["role"] == "gpu"
                         and STEP.fullmatch(evidence["step"]), "EXACT_ACTIVE_BOUND_GPU_RECEIPT")
                    seen = evidence["step"]
                need(seen is not None, "UNKNOWN_GPU_LAUNCH_NEVER_REPLAY")
        phase = "gpu_terminal"
        client.wait(timeout=max(1, auth["deadline"] - time.time()))
        row = wait_role(old, accounting, request, "gpu")
        need(row[0] == seen, "EXACT_SAME_GPU_STEP_AT_TERMINAL")
        phase = "post"
        with old.held_lock():
            no_occupancy(old)
            need(role_row(old, request, "post") is None, "EXCLUSIVE_POST_LABEL")
            put(work / "post-intent.json", wire(dict(intent, gpu_step=row[0])))
            short(target, work, request, "post", args.auth_sha, old, accounting)
        phase = "independent_readback"
        report = observe(request, old)
    except BaseException as error:
        failure = dict(phase=phase, type=type(error).__name__, detail_SHA256=sha256(str(error).encode()).hexdigest())
        report = dict(request=request, planned=32, scoring_GO=False, known_gpu_step=seen, gpu_terminal=row)
    result = dict(report, claimed=claimed, gpu_launch_attempted=attempted, failure=failure, automatic_replay=False)
    if claimed:
        put(work / "controller-outcome.json", wire(result))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["scoring_GO"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("stage", "start", "terminal", "compute-pre", "compute-gpu", "compute-post"):
        cli = modes.add_parser(name)
        cli.add_argument("--request", required=True)
        if name == "stage":
            for key in ("native-source", "support", "reload-tool", "cache-proof"):
                cli.add_argument("--" + key, type=Path, required=True)
            cli.add_argument("--host-sha", required=True)
        elif name != "terminal":
            cli.add_argument("--auth-sha", required=True)
            if name == "start":
                cli.add_argument("--auth", type=Path, required=True)
                cli.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard")
    label(args.request)
    if args.mode == "stage":
        stage(args)
        return 0
    if args.mode == "start":
        return start(args)
    if args.mode == "terminal":
        target, _, _ = staged(args.request)
        old, _, _, _ = binder(target)
        print(json.dumps(observe(args.request, old), sort_keys=True, allow_nan=False), flush=True)
        return 0
    return compute(args)


if __name__ == "__main__":
    raise SystemExit(main())
