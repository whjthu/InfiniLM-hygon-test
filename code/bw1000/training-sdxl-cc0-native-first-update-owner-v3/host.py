"""One guarded n03 SDXL update with separate zero-GRES pre/post and sealed readback."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
PARENT, NODE, UID = "1194900", "f13r4n03", 154459
STEP = re.compile(PARENT + r"\.\d+")
HASH = re.compile(r"[a-f0-9]{64}")
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PIPE = RESULTS / "sdxl-official-pipeline18-verify-77973a00e2004b8cb79c38cce05b6eb9"
INPUT = RESULTS / "sdxl-cc0-first-update-imagefolder-525bd0bfd1e14123a8ebc50989d65447/input"
STOCK = ROOT / "incoming/SDXL-style18-apache-new-candidate-b3e855-v1/examples/text_to_image/train_text_to_image_lora_sdxl.py"
FRAME = RESULTS / "training-framework-bd24ec3-v1/framework/generation/src"
RUNTIME = RESULTS / "training-t06-sdxl-runtime-v1"
RUNTIME_PROBE = "01e28b94562367bff3fbb3fce25e563addb92df95b1592e45939e986f1f8d32e"
RUNTIME_STEP = "1194900.2666"
TRACKER_PROBE = "5b140c07143d8bf0adfaa97056f8a9b5d52f3e5214c6026b0db6d8d3981c1a7e"
TRACKER_STEP = "1194900.2667"
RUNTIME_TERMINAL = "bcf5d0fcbc9dc20682a0f63bd796ff865df7502feafe32e8c326b811d43617bb"
RUNTIME_ACCOUNTING_COLUMNS = "JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList,AllocCPUS,AllocTRES"
RUNTIME_IMPORT_PINS = {
    "diffusers": ("0.39.0", "27b79203d8e25aa693aefceed1a03838b51cfe6359317e16cb50e941a79e2ac9"),
    "transformers": ("5.12.1", "135ddabaeed9a78e2a40f5fd5ec948361b59650c96ce9222cba011826ee17632"),
}
PIPE_MANIFEST = "b0a2515ad9a6422d83991788f6af7749bc50812ad0b4e028e52b86350c519175"
PIPE_COMPUTE = "94889d4de4fa1cc1b2b0b29dd7a626ecf3dc9d1e9295cbe6e5db6efd1111806d"
PIPE_TERMINAL = "7f9b347ff57184c1d0529f23f7c566b0a04c8596d0c1cfd81dcfdf8d9c331ccd"
RUN = "15145c880bd9d3ce08c09607b8ed43a93fae7ddb9f3c73459a81e7c42bcf4c63"
WITNESS = "6305e21a7a85a11b6706b5fe17e7d9a6a852a68a49b7a5f4c8cf804706856568"
STOCK_PIN = "b3e85543eaeeb42edd67dca0ffbc54391e4d8fa33b44aa1522e24b31ffc6c7bd"
FRAME_PIN = "1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7"
COMMON = "2e930484905b4fc0f66ece94ab97f6a41cc9d29d623ea9ed8479b801ad02f6d2"
PILOT = "597fc5f98d31de70777f4df665c50f6ef19537be86b664d42d66872804e25fc6"
LEGACY = "ae52680a5916e28765f82fff72ac8cdd6374f667bf0849dcc00108854c6d34b0"
INPUT_PINS = {
    "painted_stool.png": "ebc9641bdc1c3b1db45821abb81585fa92ee65da97bbcef04bf227f47b82cd73",
    "painted_stool.txt": "c27d943853f15401eafa9efc17bda3b8c32e19da316e45496afd4fb30d30435b",
    "tool_chest.png": "df0806d6195e8b78ac45b5833356ae7ca656e6ce3bd2cce85434082d81cbf274",
    "tool_chest.txt": "0a2f6b50e1631cb5a5dc5ac47c6c34e694275288f6176f2a01b0bdcea41a4506",
}


def need(condition, code):
    if not condition:
        raise ValueError(code)


def label(request):
    need(re.fullmatch(r"[a-f0-9]{32}", request) and request[:12] not in {"c5eda9cfa6ad", "fd93b793bbbd"},
         "NEW_REQUEST_ID")
    return "sdxl-cc0-first1-v3-" + request[:12]


def paths(request):
    return ROOT / "incoming" / (label(request) + "-stage"), RESULTS / label(request)


def isolated(path, name, digest):
    raw = path.read_bytes()
    need(len(raw) <= 128 << 10 and sha256(raw).hexdigest() == digest and path.resolve(strict=True) == path,
         "PINNED_OPERATOR_DEPENDENCY")
    spec = spec_from_file_location(name, path)
    result = module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def common(target):
    base = isolated(target / "common.py", "qualified_single_card_common", COMMON)
    base.ROOT, base.RESULTS, base.PARENT, base.NODE, base.UID, base.STEP = ROOT, RESULTS, PARENT, NODE, UID, STEP
    base.label, base.step_spec = label, step_spec
    old, snap, account, owned = base.binder(target)
    old.ROOT, old.PARENT, old.NODE, old.UID, old.STEP = ROOT, PARENT, NODE, UID, STEP
    raw_account = old.account

    def scoped_account(job=PARENT, since=None):
        return raw_account(job=job, since=since)

    old.account = scoped_account
    snap.NODE, snap.STEP = NODE, STEP
    return base, old, snap, account, owned


def assets(base):
    pipeline = json.loads(base.read(PIPE / "source/pipeline.json", PIPE_MANIFEST))
    proof = json.loads(base.read(PIPE / "compute.json", PIPE_COMPUTE))
    terminal = json.loads(base.read(PIPE / "terminal.json", PIPE_TERMINAL))
    need(len(pipeline) == proof["exact_readonly_pipeline_files"] == 18
         and sum(pin["bytes"] for pin in pipeline.values()) == proof["full_verified_bytes"] == 13878864542
         and proof["step"] == terminal["accounting"][0][0] == "1103606.5007"
         and terminal["accounting"][0][2:4] == ["COMPLETED", "0:0"] and terminal["passed"] is True
         and proof["full_manifest_SHA256"] == PIPE_MANIFEST and proof["actual_optimizer_updates"] == 0,
         "EXACT_ORIGINAL_5007_ADMISSION")
    for path, digest in (("run.py", RUN), ("sdxl_cc0_witness.py", WITNESS)):
        base.read(PIPE / "source" / path, digest)
    base.read(STOCK, STOCK_PIN)
    base.read(FRAME / "gencuda/backends/sdxl_training.py", FRAME_PIN)
    need(INPUT.resolve(strict=True) == INPUT and {p.name for p in INPUT.iterdir()} == set(INPUT_PINS),
         "EXACT_5039_FOUR_SOURCE_INPUTS")
    for name, digest in INPUT_PINS.items():
        base.read(INPUT / name, digest, 1 << 20)


def runtime_import(base, path):
    raw = base.read(path, RUNTIME_PROBE, 48 << 10)
    observed = json.loads(raw)
    response = observed["response"]
    actual = response["actual"]
    need(observed["authority_verified"] is observed["remote_received"] is True
         and observed["request"] == response["request"] == "e7081657a38946ff939bedfe2b5326b7"
         and observed["rc"] == response["rc"] == response["srun_rc"] == actual["native_trainer_help_rc"] == 0
         and actual["step"] == RUNTIME_STEP and actual["actual_early_GPU_initialization"] is False
         and actual["known_errors"] == [] and all(
             actual["actual_imports"][name] == dict(version=version, provider_SHA256=pin, bound_training_runtime=True)
             for name, (version, pin) in RUNTIME_IMPORT_PINS.items()), "EXACT_N03_RUNTIME_IMPORT_PROOF")
    return raw


def live_runtime(base):
    need(RUNTIME.resolve(strict=True) == RUNTIME and (RUNTIME / "site").is_dir(), "EXACT_ORIGINAL_SD_XL_RUNTIME")
    for name, (_, pin) in RUNTIME_IMPORT_PINS.items():
        base.read(RUNTIME / "site" / name / "__init__.py", pin, 2 << 20)


def runtime_row(text, step):
    row = text.split("|")
    need(len(row) == 9 and row[:3] == [step, "COMPLETED", "0:0"]
         and row[6:] == [NODE, "2", "cpu=2,mem=8G,node=1"]
         and row[3].isdigit() and all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", x) for x in row[4:6])
         and row[4] <= row[5], "EXACT_N03_INDEPENDENT_RUNTIME_ACCOUNTING")


def tracker_probe(base, path):
    raw = base.read(path, TRACKER_PROBE, 48 << 10)
    observed = json.loads(raw)
    response = observed["response"]
    need(observed["authority_verified"] is observed["remote_received"] is True
         and observed["request"] == response["request"] == "67aab4de054c43bd9339800cfe39d9cc"
         and observed["rc"] == response["rc"] == 0 and response["step"] == TRACKER_STEP
         and response["accounting_columns"] == RUNTIME_ACCOUNTING_COLUMNS
         and response["actual"] == [dict(
             accelerate_tracking_source_SHA256="0b88474247d78f013bc2e3421ad336965d118bd36a5db61d5eaaa25bf931999a",
             accelerator_source_SHA256="47088e0ab3bf21eec97e16afa14595e1db511f6ead9ab85c4eaa5f6f66fe5e61",
             accelerator_version="1.14.0", constructor_rc=0, event_files=2, gpu_initialized=False,
             init_trackers_rc=0, tracker_warning=False, tracking_types=["TensorBoardTracker"])],
         "EXACT_REAL_ACCELERATE_TRACKER")
    row, = response["original_2666"]
    runtime_row(row, RUNTIME_STEP)
    return raw


def runtime_terminal(base, path):
    raw = base.read(path, RUNTIME_TERMINAL, 48 << 10)
    observed = json.loads(raw)
    response = observed["response"]
    need(observed["authority_verified"] is observed["remote_received"] is True
         and observed["request"] == response["request"] == "c29c531b64634a0b92b708f54e0f1978"
         and observed["rc"] == response["rc"] == response["accounting_rc"] == response["queue_rc"] == 0
         and response["accounting_columns"] == RUNTIME_ACCOUNTING_COLUMNS and response["exact_queue"] == {},
         "EXACT_INDEPENDENT_N03_RUNTIME_RECEIPT")
    for step, elapsed in ((RUNTIME_STEP, "34"), (TRACKER_STEP, "8")):
        row, = [value for value in response["accounting"] if value.split("|", 1)[0] == step]
        runtime_row(row, step)
        need(row.split("|")[3] == elapsed, "ORIGINAL_RUNTIME_ELAPSED")
    return raw


def stage(args):
    target, _ = paths(args.request)
    base = isolated(args.common, "qualified_single_card_stage", COMMON)
    support = {name: base.read(args.support / name, pin) for name, pin in base.SUPPORT.items()}
    extras = {"common.py": base.read(args.common, COMMON), "pilot.py": base.read(args.pilot, PILOT),
              "legacy.py": base.read(args.legacy, LEGACY)}
    names = ("host.py", "device.py")
    pins = json.loads(base.read(HERE / "SOURCE_FILES.json", args.manifest_sha))
    members = {name: base.read(HERE / name, pins[name]["SHA256"]) for name in names}
    need(all(len(data) == pins[name]["bytes"] for name, data in members.items()), "SOURCE_EXTENT")
    assets(base)
    live_runtime(base)
    proof = runtime_import(base, args.runtime_import_proof)
    tracker = tracker_probe(base, args.tracker_proof)
    terminal = runtime_terminal(base, args.runtime_terminal_proof)
    target.mkdir(mode=0o700)
    (target / "support").mkdir(mode=0o700)
    for name, data in support.items():
        base.put(target / "support" / name, data)
    for name, data in {**extras, **members}.items():
        base.put(target / name, data)
    base.put(target / "runtime-import-proof.json", proof)
    base.put(target / "tracker-proof.json", tracker)
    base.put(target / "runtime-terminal-proof.json", terminal)
    value = dict(request=args.request, manifest_SHA256=args.manifest_sha, GPU_GO=False,
                 pipeline_SHA256=PIPE_MANIFEST, compute5007_SHA256=PIPE_COMPUTE,
                 terminal5007_SHA256=PIPE_TERMINAL, native_input5039_members=4, runtime_import_SHA256=RUNTIME_PROBE,
                 runtime_tracker_SHA256=TRACKER_PROBE, runtime_terminal_SHA256=RUNTIME_TERMINAL,
                 installed={name: sha256(data).hexdigest() for name, data in members.items()})
    return dict(value, stage_SHA256=base.put(target / "stage.json", base.wire(value)))


def staged(base, target, request, digest=None):
    raw = base.read(target / "stage.json", digest)
    spec = json.loads(raw)
    need(spec["request"] == request and spec["GPU_GO"] is False and spec["pipeline_SHA256"] == PIPE_MANIFEST
         and spec["terminal5007_SHA256"] == PIPE_TERMINAL and spec["compute5007_SHA256"] == PIPE_COMPUTE
         and spec["runtime_import_SHA256"] == RUNTIME_PROBE and spec["runtime_tracker_SHA256"] == TRACKER_PROBE
         and spec["runtime_terminal_SHA256"] == RUNTIME_TERMINAL
         and set(spec["installed"]) == {"host.py", "device.py"}, "EXACT_STAGE")
    runtime_import(base, target / "runtime-import-proof.json")
    tracker_probe(base, target / "tracker-proof.json")
    runtime_terminal(base, target / "runtime-terminal-proof.json")
    live_runtime(base)
    for name, pin in spec["installed"].items():
        base.read(target / name, pin)


def scheduler_guard(target, old, own=None, providers=None):
    pilot, legacy = providers or (isolated(target / "pilot.py", "pinned_n03_pilot", PILOT),
                                  isolated(target / "legacy.py", "pinned_n03_history", LEGACY))
    preexisting = (legacy.OLD_LABEL, "TIMEOUT", legacy.ZERO_GRES)
    need((legacy.PARENT, legacy.NODE, legacy.OLD, legacy.ORIGINAL) == (PARENT, NODE, "1194900.189", "1194900.378"),
         "EXACT_PROTECTED_HISTORY")
    historical, = pilot.rows(legacy.ORIGINAL)
    need(historical[1:4] == [legacy.ORIGINAL_LABEL, "COMPLETED", "0:0"]
         and historical[5:] == [NODE, legacy.ZERO_GRES], "ORIGINAL_378_ZERO_GRES_TERMINAL")
    stale, = pilot.rows(legacy.OLD)
    need(stale[1] == legacy.OLD_LABEL and stale[2:4] in (["RUNNING", "0:0"], ["TIMEOUT", "0:0"], ["TIMEOUT", "0:1"])
         and stale[5:] == [NODE, legacy.ZERO_GRES], "EXACT_189_ACCOUNTING_CONFLICT")
    owner, = map(old.fields, old.query(["scontrol", "show", "job", PARENT, "--oneliner"]))
    reserved, = old.query(["squeue", "-h", "-j", PARENT, "-o", "%i|%T|%N"])
    jid, state, nodelist = reserved.split("|")
    need(owner["JobId"] == jid == PARENT and owner["JobState"] == state == "RUNNING"
         and owner["UserId"].endswith(f"({UID})") and "gres/dcu=320" in owner["AllocTRES"].split(",")
         and old.on_node(owner["NodeList"]) and old.on_node(nodelist), "EXACT_PARENT_DUAL_OWNER")
    local = {}
    for line in old.query(["scontrol", "show", "step", PARENT, "--oneliner"]):
        row = old.fields(line)
        step = row["StepId"]
        if step in (PARENT + ".extern", PARENT + ".batch"):
            continue
        need(STEP.fullmatch(step) and row.get("NodeList"), "EXACT_CONTROLLER_STEP_ROWS")
        if old.on_node(row["NodeList"]):
            need(row["NodeList"] == NODE and step not in local, "EXACT_NODE_SINGLETON_ROWS")
            local[step] = (row.get("Name", row.get("JobName")), row["State"], pilot.step_resources(row))
    need(local.pop(legacy.OLD, None) == preexisting, "PROTECTED_189_CONTROLLER_CHANGED")
    if own is not None:
        step, name = own
        row = local.pop(step, None)
        need(row is not None and row[:2] == (name, "RUNNING") and old.tres(row[2], 1, 8, 64000),
             "EXACT_NEW_ONE_DCU_STEP")
    need(not local, "OTHER_NUMBERED_CONTROLLER_STEP")
    for line in old.query(["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%j|%N"]):
        step, name, nodes = line.split("|")
        if step not in (PARENT + ".extern", PARENT + ".batch"):
            need(STEP.fullmatch(step) and (not old.on_node(nodes) or own == (step, name) and nodes == NODE),
                 "OTHER_NUMBERED_QUEUE_STEP")
    need(not old.query(["sstat", "-nP", "-j", legacy.OLD, "-o", "JobID,NTasks,MaxRSS"]),
         "PROTECTED_189_HAS_LIVE_STAT")


def scope(auth, request):
    need(set(auth) == {"request", "parent", "node", "stage_SHA256", "epoch", "deadline", "port", "boot", "pci",
         "root_GO", "ordinary_excluded", "fresh_physical_SHA256"} and auth["request"] == request
         and auth["parent"] == PARENT and auth["node"] == NODE and auth["root_GO"] is auth["ordinary_excluded"] is True
         and all(HASH.fullmatch(auth[key]) for key in ("stage_SHA256", "fresh_physical_SHA256"))
         and all(type(auth[key]) in (int, float) for key in ("epoch", "deadline"))
         and 300 < auth["deadline"] - auth["epoch"] <= 2250
         and type(auth["port"]) is int and 20000 <= auth["port"] <= 60999
         and re.fullmatch(r"[0-9a-f-]{36}", auth["boot"]) and len(auth["pci"]) == 8
         and sorted(set(auth["pci"])) == auth["pci"]
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in auth["pci"]),
         "EXACT_FRESH_ROOT_AUTHORITY")
    return auth


def step_spec(target, work, request, role, auth_sha):
    cpu, mem, mins = (8, 64000, 30) if role == "gpu" else (2, 2048, 1)
    return ["srun", "--exact", "--threads=1", "--immediate=5", "--jobid=" + PARENT, "--nodes=1", "--ntasks=1",
            "--nodelist=" + NODE, "--cpus-per-task=" + str(cpu), "--mem=" + str(mem) + "M", "--time=" + str(mins),
            "--gres=" + ("dcu:1" if role == "gpu" else "none"), "--job-name=" + label(request) + "-" + role,
            HOST_PYTHON, "-I", "-S", "-B", str(target / "host.py"), "compute-" + role,
            "--request", request, "--auth-sha", auth_sha]


def guest(target, work, scratch, request, pin):
    dirs = ((target, "/owner-code", "ro"), (PIPE / "source", "/scope", "ro"),
            (PIPE / "pipeline", "/official-sdxl", "ro"), (STOCK, "/stock-trainer.py", "ro"),
            (FRAME, "/framework", "ro"), (INPUT, "/two-cc0", "ro"), (RUNTIME, "/training-runtime", "ro"),
            (work / "owner", "/owner", "ro"), (work / "durable", "/durable", "rw"),
            (scratch, "/tmp", "rw"), (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"))
    command = [APPTAINER, "exec", "--rocm", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, dest, mode in dirs:
        command += ["--bind", f"{source.as_posix()}:{dest}:{mode}"]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; "
             "unset PYTHONPYCACHEPREFIX MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_DIR MIOPEN_LOCKFILE_PATH "
             "XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME; "
             "export PYTHONPATH=/training-runtime/site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MODELSCOPE_OFFLINE=1 "
             "HOME=/tmp HF_HOME=/tmp/hf OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1; "
             "export SDXL_SINGLE_DCU_OWNER_REQUEST=\"$1\"; shift; exec /usr/bin/python3 -B -u \"$@\"")
    return [*command, str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc",
            "-c", shell, "sdxl-owner", request, "/owner-code/device.py", pin, "run", "--request", request,
            "--pipeline", "/official-sdxl", "--pipeline-manifest", "/scope/pipeline.json", "--pipeline-manifest-sha",
            PIPE_MANIFEST, "--framework", "/framework", "--original-trainer", "/stock-trainer.py",
            "--input", "/two-cc0", "--work", "/durable/once"]


def compute(args, target, work, base, old, snap, owned):
    auth = scope(json.loads(base.read(work / "auth.json", args.auth_sha)), args.request)
    staged(base, target, args.request, auth["stage_SHA256"])
    role = args.mode.removeprefix("compute-")
    intent = json.loads(base.read(work / (role + "-intent.json")))
    need(intent["request"] == args.request and intent["auth_SHA256"] == args.auth_sha
         and auth["epoch"] <= time.time() < auth["deadline"], "EXACT_ROLE_INTENT")
    if role != "pre":
        auth["pre_SHA256"] = intent["pre_SHA256"]
    if role == "post":
        terminal = base.role_row(old, args.request, "gpu", terminal=True)
        need(terminal is not None and intent["gpu_step"] == terminal[0], "EXACT_GPU_TERMINAL_BEFORE_POST")
        auth["prior_gpu_step"] = terminal[0]
    step = base.physical(target, auth, role, work, old, snap, owned)
    if role != "gpu":
        return 0
    scratch = Path(tempfile.mkdtemp(prefix="sdxl-one-", dir="/tmp"))
    base.put(scratch / ".sdxl-owner-request", (args.request + "\n").encode())
    doc = dict(request=args.request, step=step, boot=auth["boot"], pci=auth["pci"])
    pin = base.put(work / "owner/DEVICE.json", base.wire(doc))
    owned.native_command = lambda argv, _: b"/owner-code/device.py" in argv and args.request.encode() in argv
    registry = {}
    child = rc = failure = cleanup = None
    def interrupt(signum, _):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        base.put(work / "gpu-worker-intent.json", base.wire(dict(doc, owner_SHA256=pin)))
        with (work / "native.log").open("xb") as out:
            child = subprocess.Popen(guest(target, work, scratch, args.request, pin), stdin=subprocess.DEVNULL,
                                     cwd="/", stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            pid = owned.snapshot().get(child.pid)
            if pid:
                registry[child.pid] = {key: pid[key] for key in ("PID", "PPID", "PGID", "startticks", "state")}
            deadline = min(auth["deadline"] - 30, time.time() + 1740)
            while rc is None:
                owned.refresh(registry, work)
                need(time.time() < deadline, "GPU_CHILD_DEADLINE")
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
            cleanup = owned.cleanup(registry, work, grace=6)
            if child is not None:
                child.wait(timeout=2)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failure = failure or type(error).__name__
        base.put(work / "owned-final.json", base.wire(dict(request=args.request, step=step, registry=registry,
                 cleanup=cleanup, child_rc=rc, error=failure, scratch_retained=True)))
    return 0 if rc == 0 and failure is None and cleanup["owned_processes_remaining"] == 0 else 2


def checkpoint_hash(path):
    need(path.resolve(strict=True) == path and path.is_file(), "CANONICAL_CHECKPOINT_OUTPUT")
    stamp0 = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        stamp = os.fstat(stream.fileno())
        need((stamp.st_dev, stamp.st_ino) == (stamp0.st_dev, stamp0.st_ino)
             and 0 < stamp.st_size <= 1 << 30, "BOUNDED_CHECKPOINT_FILE")
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
        end = os.fstat(stream.fileno())
    need((stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns) ==
         (end.st_size, end.st_mtime_ns, end.st_ctime_ns), "CHANGED_CHECKPOINT_OUTPUT")
    return digest.hexdigest()


def observe(request, work, base, old):
    roles = {role: base.role_row(old, request, role) for role in ("pre", "gpu", "post")}
    result = dict(request=request, planned_updates=1, accounting=roles, gpu_intent=(work / "gpu-intent.json").is_file(),
                  cleanup_verified=False, scientific_pass=False, actual_optimizer_updates=None,
                  original_540_qualification_changed=False, formal_domain_credit=0)
    if not (work / "auth.json").is_file():
        return result
    auth = scope(json.loads(base.read(work / "auth.json")), request)
    gpu = roles["gpu"]
    if all(roles.values()) and roles["post"][2:4] == ["COMPLETED", "0:0"]:
        pre, gpu, post = [base.role_row(old, request, role, terminal=True) for role in ("pre", "gpu", "post")]
        before, _ = base.proof(work, request, "pre", pre)
        after, _ = base.proof(work, request, "post", post)
        intended = json.loads(base.read(work / "post-intent.json"))
        need(before["boot_id"] == after["boot_id"] == auth["boot"] and before["pci"] == after["pci"] == auth["pci"]
             and after["owned_processes_remaining"] == 0 and pre[2:4] == ["COMPLETED", "0:0"]
             and intended["gpu_step"] == gpu[0] and pre[8] <= gpu[7] <= gpu[8] <= post[7],
             "INDEPENDENT_PHYSICAL_CLEAN_POST")
        result["cleanup_verified"] = True
    out = work / "durable/once"
    if not (out / "result.json").is_file():
        return result
    raw = base.read(out / "result.json")
    science = json.loads(raw)
    need(science["request"] == request and science["planned_updates"] == 1 and science["nominal_steps"] == 25,
         "EXACT_NATIVE_RESULT_SCOPE")
    result.update(scientific_result_SHA256=sha256(raw).hexdigest(),
                  actual_optimizer_updates=science["actual_optimizer_updates"], state=science["state"])
    if science["state"] != "NATIVE_FIRST_UPDATE_PASS":
        return result
    device = json.loads(base.read(work / "durable/device.json"))
    need(gpu is not None and device["request"] == request and device["step"] == gpu[0]
         and device["boot"] == auth["boot"] and device["pci"] in auth["pci"]
         and device["visible_devices"] == 1 and device["ordinal"] == 0, "EXACT_ONE_CARD_RESULT_BINDING")
    required = {"pytorch_lora_weights.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"}
    checkpoint = out / "native/checkpoint-1"
    need(set(science["checkpoint_SHA256"]) == required and all(
         checkpoint_hash(checkpoint / name) == digest for name, digest in science["checkpoint_SHA256"].items()),
         "INDEPENDENT_CHECKPOINT_READBACK")
    base.read(out / "native/before-forward.json", science["forward_SHA256"])
    first = json.loads(base.read(out / "native/first-adamw.json"))
    need(first == science["first"] and first["actual_optimizer_updates"] == science["actual_optimizer_updates"] == 1
         and first["applied_lr"] == 1e-4 and first["skipped"] is False and first["changed_tensors"] > 0
         and first["consumed_role"] in ("painted_stool", "tool_chest"), "INDEPENDENT_NATIVE_UPDATE_READBACK")
    result.update(actual_PCI=device["pci"], checkpoint_verified=True, consumed_role=first["consumed_role"])
    result["scientific_pass"] = result["cleanup_verified"] and gpu[2:4] == ["COMPLETED", "0:0"]
    return result


def start(args, target, work, base, old, account):
    raw = base.read(args.auth, args.auth_sha, 16 << 10)
    auth = scope(json.loads(raw), args.request)
    need(os.geteuid() == UID and auth["epoch"] <= time.time() < auth["epoch"] + 55, "FRESH_ROOT_AUTHORITY_WINDOW")
    staged(base, target, args.request, auth["stage_SHA256"])
    started = attempted = False
    seen = None
    phase = "shared_admission"
    try:
        with old.held_lock():
            scheduler_guard(target, old)
            need(all(base.role_row(old, args.request, role) is None for role in ("pre", "gpu", "post"))
                 and time.time() < auth["epoch"] + 55, "EXCLUSIVE_SLURM_LABELS")
            work.mkdir(mode=0o700)
            started = True
            for name in ("owner", "durable"):
                (work / name).mkdir(mode=0o700)
            base.put(work / "auth.json", raw)
            intent = dict(request=args.request, auth_SHA256=args.auth_sha)
            phase = "zero_gres_pre"
            base.put(work / "pre-intent.json", base.wire(intent))
            pre = base.short(target, work, args.request, "pre", args.auth_sha, old, account)
            _, pin = base.proof(work, args.request, "pre", pre)
            scheduler_guard(target, old)
            need(time.time() < auth["epoch"] + 105 and base.role_row(old, args.request, "gpu") is None,
                 "FRESH_ONE_DCU_ADMISSION")
            intent["pre_SHA256"] = pin
            base.put(work / "gpu-intent.json", base.wire(intent))
            attempted = True
            phase = "new_gpu_visibility"
            with (work / "gpu-srun.log").open("xb") as out:
                child = subprocess.Popen(step_spec(target, work, args.request, "gpu", args.auth_sha),
                                         cwd="/", stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
                until = time.monotonic() + 40
                while time.monotonic() < until and not (work / "gpu.json").is_file():
                    time.sleep(.25)
                if (work / "gpu.json").is_file():
                    observation = json.loads(base.read(work / "gpu.json"))
                    need(observation["request"] == args.request and observation["role"] == "gpu"
                         and STEP.fullmatch(observation["step"]), "UNIQUE_NUMBERED_GPU_OBSERVED")
                    seen = observation["step"]
                    if child.poll() is None:
                        scheduler_guard(target, old, (seen, label(args.request) + "-gpu"))
                else:
                    numbered = base.role_row(old, args.request, "gpu")
                    need(numbered is not None, "GPU_LAUNCH_UNKNOWN_NEVER_REPLAY")
                    seen = numbered[0]
        phase = "original_gpu_terminal"
        child.wait(timeout=max(1, auth["deadline"] - time.time()))
        gpu = base.wait_role(old, account, args.request, "gpu")
        need(gpu[0] == seen, "SAME_NUMBERED_GPU_TERMINAL")
        phase = "independent_zero_gres_post"
        with old.held_lock():
            scheduler_guard(target, old)
            need(base.role_row(old, args.request, "post") is None, "EXCLUSIVE_POST_LABEL")
            base.put(work / "post-intent.json", base.wire(dict(intent, gpu_step=gpu[0])))
            base.short(target, work, args.request, "post", args.auth_sha, old, account)
        phase = "terminal_readback"
        outcome = observe(args.request, work, base, old)
    except BaseException as error:
        outcome = dict(request=args.request, planned_updates=1, actual_optimizer_updates=None, scientific_pass=False,
                       known_numbered_gpu=seen, failure=dict(phase=phase, type=type(error).__name__,
                                                            detail_SHA256=sha256(str(error).encode()).hexdigest()))
    result = dict(outcome, launch_claimed=started, gpu_launch_attempted=attempted, auto_replay=False)
    if started:
        base.put(work / "controller-outcome.json", base.wire(result))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["scientific_pass"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("stage", "start", "terminal", "compute-pre", "compute-gpu", "compute-post"):
        command = sub.add_parser(name)
        command.add_argument("--request", required=True)
        if name == "stage":
            for key in ("common", "support", "pilot", "legacy", "runtime-import-proof", "tracker-proof",
                        "runtime-terminal-proof"):
                command.add_argument("--" + key, required=True, type=Path)
            command.add_argument("--manifest-sha", required=True)
        elif name != "terminal":
            command.add_argument("--auth-sha", required=True)
            if name == "start":
                command.add_argument("--auth", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard", LC_ALL="C")
    target, work = paths(args.request)
    if args.mode == "stage":
        print(json.dumps(stage(args), sort_keys=True))
        return 0
    base, old, snap, account, owned = common(target)
    if args.mode == "start":
        return start(args, target, work, base, old, account)
    if args.mode == "terminal":
        staged(base, target, args.request)
        print(json.dumps(observe(args.request, work, base, old), sort_keys=True))
        return 0
    return compute(args, target, work, base, old, snap, owned)


if __name__ == "__main__":
    raise SystemExit(main())
