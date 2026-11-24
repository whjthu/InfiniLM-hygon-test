"""Start the exact fixed-hour observer once, independently of the SSH connection."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INSTALL_REQUEST = "d32e877d95e0420c94b7f0ab209732ce"
INSTALL_PACKET_SHA = "6654863e64331f5c999c8707ab5f689549b23ff8783f6f0732917c4213f97023"
PRIOR = "training80-sync-after-rollover-12a31891d2544ae6a1fa7d13829642f1"
PINS = {
    "ledger.json": "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d",
    "pci.json": "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39",
}
SOURCES = {
    "run_training80_busy_long_host.py": "ef1262903a59046c2c57cb1b7a93d39147858551305b88d2f1d2a6f31676a099",
    "collect_training80_busy_pulses.py": "37f1e205a7eac52914978587534f41f9cd60fa6e71650e0ddd628ae80557efaa",
    "collect_training80_busy_long_window.py": "f3e47d96a52dcf7b4596707ffea671c4d8ff7c7511ab860c242dc8bf51c25e22",
    "check_training80_point_clock.py": "0997f63efe318fb310d1e77d21021ebd4dafe753e57670952d319a343263a79f",
    "sample_training80_busy.py": "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324",
    "training_dispatch_allocation.py": "e603bbbb1ef6dd539d667d547675022c2a5236fba2d7bf873dbff44950ed80b6",
    "verify_training80_busy_long_window.py": "d81057156631e0dbbfb7fb5b458b58ae6b957ed52eacd709459d186d764f04d8",
}
FIRST, LAST, HARD_END = ("2026-11-25T07:32:30+08:00", "2026-11-25T07:35:00+08:00", "2026-11-25T08:36:30+08:00")
REQUEST_RE = re.compile(r"[a-f0-9]{32}")
SHA_RE = re.compile(r"[a-f0-9]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path, *, sha=None, maximum=65_536):
    require(path.is_file() and path.resolve(strict=True) == path.absolute(), "EXACT_REGULAR_FILE")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (sha is None or sha256(raw).hexdigest() == sha), "EXACT_FILE_BYTES")
    return raw


def exclusive_write(path, value):
    with path.open("xb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def process_start_ticks(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = text[text.rindex(")") + 1:].split()
        return int(fields[19]) if fields[0] not in {"Z", "X"} else None
    except (OSError, ValueError, IndexError):
        return None


def paths(project, request):
    require(REQUEST_RE.fullmatch(request) and request != INSTALL_REQUEST, "EXACT_DISTINCT_RUN_REQUEST")
    incoming, results = project / "incoming", project / "results"
    for parent in (incoming, results):
        require(parent.is_dir() and parent.resolve(strict=True) == parent.absolute(), "EXACT_PROJECT_ROOT")
    return (incoming / ("training80-long-busy-host-" + INSTALL_REQUEST), incoming / PRIOR,
            incoming / ("training80-long-busy-launch-" + INSTALL_REQUEST), results / ("training80-long-busy-" + request))


def prepare(project, request, install_sha):
    require(SHA_RE.fullmatch(install_sha), "ACTUAL_INSTALL_RECEIPT_SHA_REQUIRED")
    stage, prior, launch, output = paths(project, request)
    require(not any(path.exists() or path.is_symlink() for path in (launch, output)), "REQUEST_ALREADY_USED")
    install = json.loads(read(stage / "INSTALL_RESULT.json", sha=install_sha, maximum=16_384))
    require(install["request"] == INSTALL_REQUEST and install["packet_SHA256"] == INSTALL_PACKET_SHA
            and install["state"] == "SHARED_HOST_CHECK_ONLY" and install["nodes"] == 80
            and install["cards"] == 640 and install["source_count"] == len(SOURCES)
            and install["new_Slurm_steps"] == install["new_GPU_steps"] == 0
            and install["source_SHA256"] == SOURCES, "EXACT_ORIGINAL_INSTALL_RECEIPT")
    for base, members in ((stage, SOURCES), (prior, PINS)):
        for name, digest in members.items():
            read(base / name, sha=digest)
    argv = [sys.executable, "-E", "-s", "-S", "-B", str(stage / "run_training80_busy_long_host.py"),
            "run", "--request", request, "--ledger", str(prior / "ledger.json"), "--pci-pins",
            str(prior / "pci.json"), "--output-dir", str(output)]
    return launch, output, argv


def launch_once(project, request, install_sha, *, clock=time.time, spawn=subprocess.Popen, ticks=process_start_ticks):
    launch, output, argv = prepare(project, request, install_sha)
    first, last = (datetime.fromisoformat(value).timestamp() for value in (FIRST, LAST))
    require(first <= clock() <= last, "OUTSIDE_ORIGINAL_CONTROLLER_WINDOW")
    launch.mkdir(mode=0o700)
    intent = {"request": request, "install_receipt_SHA256": install_sha, "observer_output": str(output),
              "argv_SHA256": sha256(encode(argv)).hexdigest(), "GPU_requested": 0, "replay_allowed": False,
              "hard_end_Beijing": HARD_END, "recorded_UTC": datetime.fromtimestamp(clock(), timezone.utc).isoformat()}
    exclusive_write(launch / "INTENT.json", intent)
    sync_directory(launch)
    sync_directory(launch.parent)
    result = {"request": request, "state": "CONTROLLER_START_UNKNOWN", "controller_PID": None,
              "controller_start_ticks": None, "Slurm_steps_confirmed": 0, "replay_allowed": False}
    try:
        with (launch / "controller.stdout.private").open("xb") as out, (launch / "controller.stderr.private").open("xb") as err:
            require(first <= clock() <= last, "ORIGINAL_CONTROLLER_WINDOW_EXPIRED_BEFORE_SPAWN")
            child = spawn(argv, cwd=launch, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            result.update(controller_PID=child.pid, controller_start_ticks=ticks(child.pid))
            if result["controller_start_ticks"] is not None and child.poll() is None:
                result["state"] = "CONTROLLER_STARTED"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = type(error).__name__
    exclusive_write(launch / "OUTCOME.json", result)
    sync_directory(launch)
    return result


def observe(project, request):
    _, _, launch, output = paths(project, request)
    members = (launch / "INTENT.json", launch / "OUTCOME.json", output / "host-intent.json",
               output / "ready.json", output / "host-outcome.json")
    result = {"request": request, "receipts": {}, "same_controller_live": None, "replay_allowed": False}
    for path in members:
        if not path.exists() and not path.is_symlink():
            continue
        raw = read(path)
        value = json.loads(raw)
        require(value["request"] == request, "OBSERVED_REQUEST_MISMATCH")
        result["receipts"][path.name] = {"SHA256": sha256(raw).hexdigest(), "bytes": len(raw)}
        if path.name == "OUTCOME.json" and value["controller_start_ticks"] is not None:
            result["same_controller_live"] = process_start_ticks(value["controller_PID"]) == value["controller_start_ticks"]
        elif path.name == "ready.json":
            result["actual_ready_steps"] = value["actual_READY_steps"]
            result["ready_nodes"] = value["ready_nodes"]
        elif path.name == "host-outcome.json":
            result["terminal_zero_GRES_verified"] = value["zero_GRES_terminal_accounting_verified"]
            result["full_hour_observer_C0"] = value["full_hour_observer_C0"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "launch", "observe"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--install-receipt-sha")
    args = parser.parse_args()
    require(os.name == "posix" and sys.flags.dont_write_bytecode, "POSIX_HOST_AND_NO_BYTECODE_REQUIRED")
    os.umask(0o077)
    if args.mode == "observe":
        result = observe(PROJECT, args.request)
    elif args.mode == "check":
        require(args.install_receipt_sha is not None, "ACTUAL_INSTALL_RECEIPT_SHA_REQUIRED")
        _, _, argv = prepare(PROJECT, args.request, args.install_receipt_sha)
        result = {"request": args.request, "argv_SHA256": sha256(encode(argv)).hexdigest(), "Slurm_calls": 0}
    else:
        require(args.install_receipt_sha is not None, "ACTUAL_INSTALL_RECEIPT_SHA_REQUIRED")
        result = launch_once(PROJECT, args.request, args.install_receipt_sha)
    print(json.dumps(result, sort_keys=True))
    return int(args.mode == "launch" and result["state"] != "CONTROLLER_STARTED")


if __name__ == "__main__":
    raise SystemExit(main())
