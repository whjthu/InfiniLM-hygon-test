"""Run one five-row Gemma VizWiz processor audit without consuming any DCU."""

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
LABEL = "training-gemma-vizwiz-five-native-CPU-v1"
PARENT, NODE = "1103606", "f13r2n09"
DATA_STAGE = RESULTS / "training-gemma-vizwiz-five-native-input-audit-v1"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
PROCESSOR = RESULTS / "training-gemma-processor-assets-v1/models/gemma-3-4b-it"
RUNTIME = RESULTS / "training-vlm-swift-offline-runtime-v1"
TRANSFORMERS = RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site"
CONTAINER = ROOT / "containers/dtk26-sandbox-r1"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
CPU_SECONDS, WORK_SECONDS = 300, 220
HIDDEN = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
PLANNED = dict(planned_source_rows=6, omitted_by_supervision=1, planned_native_rows=5)


def source_module(data, name):
    scope = {"__name__": name}
    exec(compile(data, name, "exec"), scope)
    return scope


def command(work, epoch):
    binds = (
        (work / "source", "/audit-source", "ro"), (work / "evidence", "/audit-out", "rw"),
        (DATA_STAGE / "data", (DATA_STAGE / "data").as_posix(), "ro"),
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
    worker = [
        "python3", "-B", "-u", "/audit-source/audit_gemma_vizwiz_native.py",
        "--data", (DATA_STAGE / "data").as_posix(), "--processor", "/gemma-processor",
        "--processor-manifest", "/audit-source/MODEL_PROCESSOR_FILES.json",
        "--framework", "/input-framework", "--framework-manifest", "/audit-source/FRAMEWORK_FILES.json",
        "--output", "/audit-out/worker", "--parent", PARENT, "--node", NODE,
        "--epoch", str(epoch), "--deadline", str(deadline),
    ]
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


def worker_result(path, bounded_read, epoch):
    raw, pin = bounded_read(path, 8192)
    result = json.loads(raw)
    if (any(result.get(key) != count for key, count in PLANNED.items())
            or result.get("complete_native_rows", -1) + result.get("failed_native_rows", -1) != 5
            or result.get("epoch") != epoch or result.get("deadline") != epoch + WORK_SECONDS
            or result.get("training_READY") is not False or result.get("model_loaded") is not False
            or result.get("optimizer_updates") != 0
            or result.get("torch_lazy_device_initialization_attempts") != 0
            or bool(result.get("passed")) != (result.get("complete_native_rows") == 5)):
        raise ValueError("native processor receipt scope differs")
    return dict(pin=pin, value=result)


def native_row_summary(path, bounded_read, worker):
    raw, pin = bounded_read(path, 256 * 1024)
    reference, _ = bounded_read(DATA_STAGE / "data/train-swift.jsonl", 8192)
    if pin["SHA256"] != worker["private_rows_SHA256"]:
        raise ValueError("native row evidence differs from the worker receipt")
    lines, observed = reference.splitlines(keepends=True), raw.splitlines()
    if len(lines) != PLANNED["planned_native_rows"] or len(observed) != len(lines):
        raise ValueError("native row evidence lost part of the fixed denominator")
    tokens = []
    failed = 0
    for index, (source, row) in enumerate(zip(lines, observed, strict=True), 1):
        record = json.loads(row)
        if record["ordinal"] != index or record["source_line_SHA256"] != sha256(source).hexdigest():
            raise ValueError("native row identity or order differs")
        if record["state"] == "FAILED":
            failed += 1
            continue
        length, supervised = record["tokens"], record["shifted_supervised_tokens"]
        if (record["state"] != "CHECKED" or type(length) is not int or type(supervised) is not int
                or not 0 < supervised < length <= 512
                or any(not re.fullmatch(r"[0-9a-f]{64}", record[key]["SHA256"])
                       for key in ("encoded", "collated"))):
            raise ValueError("invalid native row token or fingerprint evidence")
        tokens.append(length)
    if failed != worker["failed_native_rows"] or len(tokens) != worker["complete_native_rows"]:
        raise ValueError("native row states differ from the worker receipt")
    return dict(checked=len(tokens), failed=failed, maximum_row_tokens=max(tokens, default=None), pin=pin)


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
    expected_receipt = target["verify_data"](DATA_STAGE, (DATA_STAGE / "data").as_posix())
    if expected_receipt["SHA256"] != claim["original_data_receipt_SHA256"]:
        raise ValueError("original data-stage receipt differs")
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
        worker = worker_result(evidence / "worker/result.json", target["read"], epoch)
        row_summary = native_row_summary(evidence / "worker/native-rows-private.jsonl",
                                         target["read"], worker["value"])
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
