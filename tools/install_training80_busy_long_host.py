"""Verify and stage the fixed-hour observer on the shared HOST without submission."""

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PREVIOUS = ROOT / "incoming" / "training80-sync-after-rollover-12a31891d2544ae6a1fa7d13829642f1"
LEDGER = "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"
PCI = "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39"
SOURCES = {
    "run_training80_busy_long_host.py": "ef1262903a59046c2c57cb1b7a93d39147858551305b88d2f1d2a6f31676a099",
    "collect_training80_busy_pulses.py": "37f1e205a7eac52914978587534f41f9cd60fa6e71650e0ddd628ae80557efaa",
    "collect_training80_busy_long_window.py": "f3e47d96a52dcf7b4596707ffea671c4d8ff7c7511ab860c242dc8bf51c25e22",
    "check_training80_point_clock.py": "0997f63efe318fb310d1e77d21021ebd4dafe753e57670952d319a343263a79f",
    "sample_training80_busy.py": "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324",
    "training_dispatch_allocation.py": "e603bbbb1ef6dd539d667d547675022c2a5236fba2d7bf873dbff44950ed80b6",
    "verify_training80_busy_long_window.py": "d81057156631e0dbbfb7fb5b458b58ae6b957ed52eacd709459d186d764f04d8",
}
PACKET_CAP = 131_072
TIMEOUT = 12


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def pin_file(path, wanted, cap=65_536):
    require(path.is_file() and path.resolve(strict=True) == path.absolute(), "ORIGINAL_MEMBER_ALIAS")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and digest(raw) == wanted, "MEMBER_PIN")
    return {"bytes": len(raw), "SHA256": wanted}


def decode_packet(raw, wanted):
    require(0 < len(raw) <= PACKET_CAP and digest(raw) == wanted, "PACKET_IDENTITY")
    encoded = json.loads(raw)
    require(set(encoded) == set(SOURCES), "PACKET_SOURCE_CLOSURE")
    files = {name: base64.b64decode(payload, validate=True) for name, payload in encoded.items()}
    for name, payload in files.items():
        require(0 < len(payload) <= 32_768 and digest(payload) == SOURCES[name], "PUBLISHED_SOURCE")
        compile(payload, name, "exec")
    return files


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def check_report(report, request):
    require(report["request"] == request and report["parents"] == {"1103606": 40, "1194900": 40}
            and report["points"] == 1801 and report["planned_device_seconds"] == 2304000
            and report["GPU_requested"] == report["remote_calls"] == report["scheduler_calls"] == 0
            and report["hard_deadline_CST"] == "2026-11-25T08:36:30+08:00", "OBSERVER_CHECK_SCOPE")
    require(set(report["commands"]) == {"1103606", "1194900"}, "EXACT_PARENTS")
    for parent, argv in report["commands"].items():
        require(argv[0] == "srun" and {"--jobid=" + parent, "--nodes=40", "--ntasks=40", "--cpus-per-task=1",
                "--mem=256M", "--gres=none", "--time=01:01:00", "--overlap"} <= set(argv), "OBSERVER_CHECK_ARGV")


def install(stage, files, packet_sha, request, previous=PREVIOUS):
    ledger, pci = previous / "ledger.json", previous / "pci.json"
    prior = {"ledger": pin_file(ledger, LEDGER), "pci": pin_file(pci, PCI)}
    require(stage.parent.resolve(strict=True) == stage.parent.absolute(), "INSTALL_ROOT_ALIAS")
    stage.mkdir(mode=0o700)
    put(stage / "INSTALL_INTENT.json", encode({"request": request, "packet_SHA256": packet_sha,
                                             "GPU_requested": 0, "replay_allowed": False}))
    for name, data in files.items():
        put(stage / name, data)
        pin_file(stage / name, SOURCES[name])
    argv = [sys.executable, "-E", "-s", "-S", "-B", str(stage / "run_training80_busy_long_host.py"),
            "check", "--request", request, "--ledger", str(ledger), "--pci-pins", str(pci)]
    child = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT)
    require(child.returncode == 0 and 0 < len(child.stdout) <= 16_384, "OBSERVER_CHECK_EXIT")
    report = json.loads(child.stdout)
    check_report(report, request)
    result = {"state": "SHARED_HOST_CHECK_ONLY", "request": request, "packet_SHA256": packet_sha,
              "source_count": len(files), "source_SHA256": SOURCES, "existing_target_pins_reused": prior,
              "check_stdout_SHA256": digest(child.stdout), "check_stderr_SHA256": digest(child.stderr),
              "nodes": 80, "cards": 640, "new_Slurm_steps": 0, "new_GPU_steps": 0,
              "compute_node_validation": False, "continuous_60min_qualified": False}
    data = encode(result)
    put(stage / "INSTALL_RESULT.json", data)
    return {"result": result, "receipt_SHA256": digest(data), "receipt_bytes": len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--packet-sha256", required=True)
    args = parser.parse_args()
    require(sys.platform == "linux" and sys.flags.isolated and sys.flags.dont_write_bytecode
            and re.fullmatch("[a-f0-9]{32}", args.request), "ISOLATED_LINUX_REQUEST")
    files = decode_packet(sys.stdin.buffer.read(PACKET_CAP + 1), args.packet_sha256)
    os.umask(0o077)
    out = install(ROOT / "incoming" / ("training80-long-busy-host-" + args.request),
                  files, args.packet_sha256, args.request)
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
