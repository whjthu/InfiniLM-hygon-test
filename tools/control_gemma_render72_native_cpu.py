"""Run one tokenizer/collator-only Gemma input audit in its owned CPU Slurm slot."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import stat
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
TARGET = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-gemma-usgs-render72-native-input-audit-v1"
REQUEST = "8d6d573cc64b4ab3b082b17682043a11"
PARENT = "1103606"
NODE = "f13r1n12"
RESULTS = TARGET / "results"
RUN = RESULTS / LABEL
DATA = RESULTS / "training-gemma-usgs-render72-native-CPU-v1/data"
VIEW = RESULTS / "training-gemma-usgs-render72-native-CPU-v1/view"
PROCESSOR = RESULTS / "training-gemma-processor-assets-v1/models/gemma-3-4b-it"
RUNTIME = RESULTS / "training-vlm-swift-offline-runtime-v1"
TRANSFORMERS = RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site"
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
TERM_HELPER = "code/bw1000/training-qwen-native25-hip-v3/slurm_accounting.py"
ROSTER = "coordination/training/training80-active-node-task-map-v1.json"
MEMBERSHIP = "tools/read_training_slurm_membership.py"
PINS = {
    "tools/audit_gemma_training_native.py": "890b4ebcea30972f604ffaa1d4b7f5cca2b76e919722944123fa3368fd29b1ba",
    "tools/build_gemma_train_exposure.py": "c0c380c51ed4af09779fa270b4419ce511571137d95f1676b3dd9a4eaad715a7",
    "code/bw1000/training-gemma-native-inputs-HIP-CPU-v1/MODEL_PROCESSOR_FILES.json":
        "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b",
    "code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json":
        "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
    TERM_HELPER: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    MEMBERSHIP: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    ROSTER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
}
PLANNED = dict(unique_rows=72, encode_calls=144, collator_calls=72, optimizer_updates=0)
MASK = {k: "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
DATA_FILES = ("train-swift.jsonl", "data-binding-private.json", "train/media")
FRAMEWORK_MANIFEST = "FRAMEWORK_FILES.json"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path, expected, limit=150000):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "CPU source pin differs")
    return raw


def json_once(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def target_scope_path(raw):
    path = PurePosixPath(raw)
    require(path.is_absolute() and path.is_relative_to(PurePosixPath(RESULTS.as_posix()))
            and ".." not in path.parts and bool(re.fullmatch(r"/[a-zA-Z0-9_/.-]+", str(path))),
            "Framework must use an explicit immutable result directory")
    return path


def bind_map(framework):
    result = [
        (RUN / "src", "/audit-src", "ro"),
        (RUN, "/audit-run", "rw"),
        (framework, "/input-framework", "ro"),
        (PROCESSOR, "/gemma-processor", "ro"),
        (RUNTIME, "/swift-runtime", "ro"),
        (TRANSFORMERS, "/minicpm-transformers", "ro"),
        (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"),
    ]
    result.extend((DATA / name, (DATA / name).as_posix(), "ro") for name in DATA_FILES)
    result.append((VIEW / "train800-swift.jsonl", "/audit-view/train800-swift.jsonl", "ro"))
    return result


def build_command(framework, epoch, cpu_minutes):
    require(type(cpu_minutes) is int and 1 <= cpu_minutes <= 7, "CPU request exceeds original bound")
    exec_args = [
        "python3", "-B", "-u", "/audit-src/audit_gemma_training_native.py",
        "--prepared", DATA.as_posix(), "--view", "/audit-view", "--processor-root", "/gemma-processor",
        "--processor-manifest", "/audit-src/MODEL_PROCESSOR_FILES.json",
        "--framework-root", "/input-framework", "--framework-manifest", "/audit-src/" + FRAMEWORK_MANIFEST,
        "--output", "/audit-run/audit", "--epoch", str(epoch), "--deadline", str(epoch + 390),
    ]
    env = dict(MASK, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH="/minicpm-transformers:/swift-runtime/site:/input-framework")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export "
             + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
             + "; exec " + shlex.join(exec_args))
    cmd = [
        "srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
        "-N1", "-n1", "--nodelist=" + NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none",
        "--time=" + str(cpu_minutes), APPTAINER, "exec", "--contain", "--no-home", "--pwd", "/",
    ]
    for host, container, mode in bind_map(framework):
        cmd += ["--bind", f"{host.as_posix()}:{container}:{mode}"]
    cmd += [(TARGET / "containers/dtk26-sandbox-r1").as_posix(), "/bin/bash", "--noprofile", "--norc", "-c", shell]
    return cmd


def terminal_pass(row, allocated_dcu):
    if not row or len(row) != 7 or not re.fullmatch(re.escape(PARENT) + r"\.\d+", row[0]):
        return False
    try:
        devices = allocated_dcu(row[6])
        tres = dict(entry.split("=", 1) for entry in row[6].split(","))
    except ValueError:
        return False
    return (row[1:4] == [LABEL, "COMPLETED", "0:0"] and row[4].isdecimal() and int(row[4]) <= 420
            and row[5] == NODE and devices == 0 and tres.get("cpu") == "4"
            and tres.get("mem") in ("8G", "8192M") and tres.get("node") == "1")


def perform(framework, publication):
    require(os.name == "posix" and bool(re.fullmatch(r"[0-9a-f]{40}", publication)), "Target host and publication required")
    require(not RUN.exists() and not RUN.is_symlink(), "Original intent exists; reconcile without replay")
    require(framework.resolve(strict=True) == framework, "Framework root is aliased")
    sources = {relative: pinned(REPO / relative, pin) for relative, pin in PINS.items()}
    from read_training_slurm_membership import allocated_dcu, snapshot
    bound_wait = runpy.run_path(str(REPO / TERM_HELPER))["wait_terminal"]

    epoch = time.time()
    deadline = time.monotonic() + 480

    def query(*args):
        left = deadline - time.monotonic()
        require(left > 5, "Original CPU epoch exhausted")
        return subprocess.check_output(args, text=True, timeout=min(8, left - 2)).strip().splitlines()

    def accounting():
        rows = [line.split("|") for line in query("sacct", "-nP", "-j", PARENT, "-o", ",".join(COLUMNS))]
        require(all(len(row) == 7 for row in rows), "Malformed Slurm accounting")
        exact = [row for row in rows if row[1] == LABEL]
        require(len(exact) <= 1, "Original CPU step name already has multiple records")
        return exact[0] if exact else None

    old = accounting()
    require(old is None, "Original CPU step exists; reconcile without replay")
    ledger = json.loads(sources[ROSTER])
    roster = {PARENT: {name: row["allocations"][PARENT] for name, row in ledger["groups"].items()}}
    before = snapshot(roster, owner_uid=os.getuid(), run_query=query)["parents"][PARENT]
    require(before["owned_by_observer"] and before["ledger_exact"] and before["step_locations_exact"]
            and before["state"] == "RUNNING" and before["allocated_DCU"] == 320
            and NODE in roster[PARENT]["T02"]
            and all(NODE not in rec["member_nodes"] for rec in before["active_steps"].values()),
            "Frozen CPU node is unconfirmed or occupied")
    for host, _, _ in bind_map(framework)[2:]:
        mode = host.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Target bind missing or aliased")
    require(Path(APPTAINER).is_file() and os.access(APPTAINER, os.X_OK)
            and (TARGET / "containers/dtk26-sandbox-r1").is_dir(), "Pinned Apptainer or container unavailable")
    require(time.time() < epoch + 60, "Not enough original work budget after target admission")
    require(accounting() is None, "Original CPU step appeared after admission")
    os.umask(0o077)
    RUN.mkdir(mode=0o700)
    (RUN / "src").mkdir()
    for name, data in sources.items():
        if name not in (MEMBERSHIP, TERM_HELPER, ROSTER):
            with (RUN / "src" / Path(name).name).open("xb") as stream:
                stream.write(data)
    minutes = int(min(420, deadline - time.monotonic() - 38) // 60)
    cmd = build_command(framework, epoch, minutes)
    json_once(RUN / "intent.json", dict(request=REQUEST, publication=publication, parent=PARENT, node=NODE,
                                        epoch=epoch, worker_deadline=epoch + 390, whole_deadline=epoch + 480,
                                        source_SHA256=PINS, command=cmd, replay_allowed=False, **PLANNED))
    return_code = None
    failure = None
    result = None
    terminal = dict(terminal_visible=False, row=None, observations=[])
    accounting_errors = []

    def observe_terminal():
        try:
            return accounting()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            accounting_errors.append(type(error).__name__)
            return None

    try:
        with (RUN / "step.log").open("xb") as stream:
            return_code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=min(minutes * 60 + 4, deadline - time.monotonic() - 30),
                                         env=dict(os.environ, **MASK)).returncode
    except (OSError, subprocess.SubprocessError) as error:
        failure = type(error).__name__
    finally:
        try:
            terminal = bound_wait(observe_terminal, min(time.monotonic()+22, deadline-4), interval=1,
                                  max_observations=10)
            worker = RUN / "audit/result.json"
            if worker.is_file() and not worker.is_symlink() and worker.stat().st_size < 8192:
                raw = worker.read_bytes()
                observed = json.loads(raw)
                if (observed.get("passed") is True and observed.get("model_loaded") is False
                        and observed.get("optimizer_updates") == 0 and observed.get("epoch") == epoch
                        and observed.get("deadline") == epoch + 390
                        and all(observed.get(k) == v for k, v in PLANNED.items())):
                    result = dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failure = failure or type(error).__name__
        passed = bool(return_code == 0 and not failure and result and terminal_pass(terminal["row"], allocated_dcu))
        outcome = dict(request=REQUEST, terminal=terminal, accounting_columns=COLUMNS, passed=passed,
                       planned=PLANNED, worker_receipt=result, srun_return_code=return_code, failure_type=failure,
                       accounting_errors=accounting_errors,
                       target_training_admitted=False, optimizer_updates=0, granted_dcu=0)
        json_once(RUN / "outcome.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--publication", help="Verified publication SHA for executing the one-shot CPU request")
    args = parser.parse_args()
    framework = target_scope_path(args.framework_root)
    if args.publication is None:
        print(json.dumps(dict(mode="READ_ONLY_PLAN", case="T02.domain-usgs-native72-input-audit",
                              request=REQUEST, planned=PLANNED, target_invoked=False,
                              command=build_command(framework, 1700000000.0, 7))))
        return 0
    result = perform(Path(framework), args.publication)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
