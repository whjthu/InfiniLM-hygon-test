"""Run the reviewed Gemma VizWiz train-only input audit beside the original standby."""

from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-gemma-vizwiz-reviewed-train-native-CPU-v1"
PARENT, NODE = "1103606", "f13r2n09"
DATA_STAGE = RESULTS / "training-gemma-vizwiz-reviewed-train-native-v1"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
PROCESSOR = RESULTS / "training-gemma-processor-assets-v1/models/gemma-3-4b-it"
RUNTIME = RESULTS / "training-vlm-swift-offline-runtime-v1"
TRANSFORMERS = RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site"
CONTAINER = ROOT / "containers/dtk26-sandbox-r1"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
CPU_SECONDS, WORK_SECONDS = 300, 220
HIDDEN = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def source_module(data, name):
    scope = {"__name__": name}
    exec(compile(data, name, "exec"), scope)
    return scope


def command(work, epoch):
    binds = (
        (work / "source", "/audit-source", "ro"), (work / "evidence", "/audit-out", "rw"),
        (DATA_STAGE, DATA_STAGE.as_posix(), "ro"),
        (FRAMEWORK, "/input-framework", "ro"), (PROCESSOR, "/gemma-processor", "ro"),
        (RUNTIME, "/swift-runtime", "ro"), (TRANSFORMERS, "/minicpm-transformers", "ro"),
        (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
    )
    env = dict.fromkeys(HIDDEN, "-1")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH="/minicpm-transformers:/swift-runtime/site:/input-framework")
    deadline = epoch + WORK_SECONDS
    worker = ["python3", "-B", "-u", "/audit-source/inside.py"]
    remaining = shlex.quote(f"import time; print(int({deadline!r}-time.time()))")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export "
             + " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
             + f'; LEFT=$(python3 -B -c {remaining}); test "$LEFT" -gt 15; '
             + 'exec /usr/bin/timeout --kill-after=5s "$LEFT" ' + shlex.join(worker))
    argv = ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--nodelist=" + NODE, "--cpus-per-task=4",
            "--mem=8192M", "--gres=none", "--time=4",
            APPTAINER, "exec", "--contain", "--no-home", "--pwd", "/"]
    for host, private, mode in binds:
        argv += ["--bind", f"{host}:{private}:{mode}"]
    return argv + [str(CONTAINER), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def terminal_pass(row):
    if not row or len(row) != 7 or not re.fullmatch(re.escape(PARENT) + r"\.\d+", row[0]):
        return False
    return (row[1:4] == [LABEL, "COMPLETED", "0:0"] and row[4].isdecimal() and int(row[4]) <= 240
            and row[5] == NODE and set(row[6].split(",")) in (
                {"cpu=4", "mem=8G", "node=1"}, {"cpu=4", "mem=8192M", "node=1"}))


def worker_result(path, bounded_read, claim, epoch, step):
    raw, pin = bounded_read(path, 8192)
    item = json.loads(raw)
    counts = tuple(item[key] for key in ("checked", "failed", "not_started"))
    if (item["total_train_rows"] != claim["data_stage"]["train_rows"]
            or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != item["total_train_rows"]
            or item["epoch"] != epoch or item["deadline"] != epoch + WORK_SECONDS
            or item["actual_step"] != step or item["training_READY"] is not False or item["model_loaded"] is not False
            or item["optimizer_updates"] != 0 or item["gpu"] != 0 or item["device_init_attempts"] != 0
            or item["passed"] != (counts[1:] == (0, 0))):
        raise ValueError("native receipt scope differs")
    return dict(pin=pin, value=item)


def native_row_summary(path, bounded_read, worker, offer):
    raw, pin = bounded_read(path, 512 * 1024)
    reference, _ = bounded_read(DATA_STAGE / "data/train-swift.jsonl", 128 * 1024, offer["rows"])
    if pin["SHA256"] != worker["rows_SHA256"]:
        raise ValueError("native row evidence differs from the worker receipt")
    lines, observed = reference.splitlines(keepends=True), raw.splitlines()
    if len(lines) != offer["train_rows"] or len(observed) != len(lines):
        raise ValueError("native row evidence lost part of the original denominator")
    counts = dict(checked=0, failed=0, not_started=0)
    lengths, supervised = [], 0
    for index, (source, row) in enumerate(zip(lines, observed, strict=True), 1):
        item = json.loads(row)
        if (item["ordinal"] != index or item["source_line_SHA256"] != sha256(source).hexdigest()
                or item["state"].lower() not in counts):
            raise ValueError("native row identity or order differs")
        counts[item["state"].lower()] += 1
        if item["state"] != "CHECKED":
            continue
        n, m = item["tokens"], item["shifted_supervised_tokens"]
        if (type(n) is not int or type(m) is not int or not 0 < m < n <= 512
                or any(not re.fullmatch(r"[0-9a-f]{64}", item[key]["SHA256"]) for key in ("encoded", "collated"))):
            raise ValueError("invalid token or fingerprint evidence")
        lengths.append(n)
        supervised += m
    facts = dict(**counts, max_tokens=max(lengths, default=None), supervised_tokens=supervised)
    if any(worker[key] != value for key, value in facts.items()):
        raise ValueError("receipt and original native counts differ")
    return dict(**facts, minimum_tokens=min(lengths, default=None), pin=pin)


def execute(argv, stream, timeout):
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True, env=dict(os.environ, **dict.fromkeys(HIDDEN, "-1")))
    try:
        return child.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            try:
                return child.wait(timeout=5), True
            except subprocess.TimeoutExpired:
                if sig == signal.SIGKILL:
                    raise


def run(claim, files):
    raw_helper = b64decode(files["controller_helpers.py"], validate=True)
    expected = claim["payload_files"]["controller_helpers.py"]
    if len(raw_helper) != expected["bytes"] or sha256(raw_helper).hexdigest() != expected["SHA256"]:
        raise ValueError("qualified CPU controller helper differs")
    helpers = source_module(raw_helper, "cpu_controller")
    data = helpers["get_files"](claim["payload_files"], files)
    accounting = source_module(data.pop("slurm_accounting.py"), "slurm_accounting")
    target = source_module(data.pop("verify_gemma_vizwiz_target_stage.py"), "vizwiz_target_readback")
    data.pop("controller_helpers.py")
    scope = (claim["label"], claim["parent"], claim["node"], claim["cpu_cores"],
             claim["memory_MiB"], claim["gpu_cards"], claim["total_seconds"], claim["work_seconds"])
    if scope != (LABEL, PARENT, NODE, 4, 8192, 0, CPU_SECONDS, WORK_SECONDS):
        raise ValueError("published zero-GRES scope differs")
    epoch, start = time.time(), time.monotonic()
    wall_deadline = epoch + CPU_SECONDS

    def left(reserve=0):
        remaining = min(wall_deadline - time.time(), CPU_SECONDS - (time.monotonic() - start)) - reserve
        if remaining <= 0:
            raise TimeoutError("original CPU deadline passed")
        return remaining

    def ask(*argv):
        return subprocess.check_output(argv, text=True, timeout=min(9, left(4))).strip()

    def row_for_original():
        rows = [line.split("|") for line in ask("sacct", "-nP", "-j", PARENT, "-o",
                                                ",".join(helpers["ACCOUNTING"])).splitlines()]
        if any(len(row) != 7 for row in rows):
            raise ValueError("Slurm returned malformed accounting")
        matched = [row for row in rows if row[1] == LABEL]
        if len(matched) > 1:
            raise ValueError("multiple rows for the unique CPU step")
        return matched[0] if matched else None

    work = RESULTS / LABEL
    if work.exists() or work.is_symlink() or row_for_original() or epoch + CPU_SECONDS >= claim["latest_finish_epoch"]:
        raise ValueError("original CPU namespace already exists or misses the standby window")
    for path in (DATA_STAGE / "data", FRAMEWORK, PROCESSOR, RUNTIME / "site", TRANSFORMERS, CONTAINER):
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError("target mount absent or aliased")
    if not os.access(APPTAINER, os.X_OK):
        raise ValueError("target Apptainer missing")
    offer = claim["data_stage"]
    stage = source_module(data["stage_receiver.py"], "vizwiz_stage_scope")
    stage["check_claim"](offer)
    frozen_receipt, _ = target["read"](DATA_STAGE / "stage-outcome.json", 4096, claim["data_stage_outcome"])
    if json.loads(frozen_receipt) != dict(request=offer["request"], state="DATA_STAGED", role="train",
                                         completed=offer["train_rows"] + 2, train_rows=offer["train_rows"],
                                         payload_bytes=offer["payload_bytes"], slurm=0, gpu=0,
                                         optimizer_updates=0, training_READY=False):
        raise ValueError("original train-only staging receipt differs")
    before = helpers["verify_live"](ask, claim)
    if row_for_original() or left() < 245:
        raise ValueError("duplicate accounting row or insufficient CPU budget")
    os.umask(0o077)
    work.mkdir(mode=0o700)
    source, evidence = work / "source", work / "evidence"
    source.mkdir()
    evidence.mkdir()
    write = helpers["write_json"]
    write(work / "intent.json", dict(request=claim["request"], claim_SHA256=CLAIM_SHA256, epoch=epoch,
                                     deadline=wall_deadline, standby_before=before, replay_allowed=False))
    print(json.dumps(dict(request=claim["request"], event="REMOTE_INTENT_CREATED", epoch=epoch)), flush=True)
    failures, accounting_errors = {}, []
    rc, timed_out = None, False
    worker, after, row_summary = None, None, None

    def observe():
        try:
            return row_for_original()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            accounting_errors.append(type(error).__name__)
            return None

    try:
        for name, raw in data.items():
            with (source / name).open("xb") as stream:
                stream.write(raw)
        write(source / "CLAIM.json", dict(claim, epoch=epoch, worker_deadline=epoch + WORK_SECONDS))
        with (work / "CPU.log").open("xb") as stream:
            rc, timed_out = execute(command(work, epoch), stream, left(35))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        failures["step"] = type(error).__name__
    try:
        terminal = accounting["wait_terminal"](observe, time.monotonic() + min(17, left(11)),
                                               interval=1, max_observations=17)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        failures["accounting"] = type(error).__name__
        terminal = dict(terminal_visible=False, row=None, observations=[])
    try:
        after = helpers["verify_live"](ask, claim)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        failures["standby_after"] = type(error).__name__
    try:
        step = terminal["row"][0] if terminal["row"] else None
        worker = worker_result(evidence / "worker/result.json", target["read"], claim, epoch, step)
        row_summary = native_row_summary(evidence / "worker/native-train-rows-private.jsonl",
                                         target["read"], worker["value"], offer)
    except (KeyError, OSError, ValueError, TypeError) as error:
        failures["worker"] = type(error).__name__
    passed = bool(rc == 0 and not timed_out and not failures and terminal_pass(terminal["row"])
                  and before == after and worker and worker["value"]["passed"])
    report = dict(request=claim["request"], passed=passed, accounting_columns=helpers["ACCOUNTING"],
                  terminal=terminal, CPU_result=worker, native_row_summary=row_summary,
                  standby_before=before, standby_after=after,
                  failed_phases=failures, accounting_errors=accounting_errors, child_returncode=rc,
                  timed_out=timed_out, GPU_cards=0, optimizer_updates=0, training_READY=False)
    write(work / "outcome.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(run(CLAIM, FILES))
