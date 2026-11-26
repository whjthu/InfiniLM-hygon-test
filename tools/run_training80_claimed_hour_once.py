"""Launch one published, claim-bound zero-GRES 640-card measurement."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
INSTALL = PROJECT / "incoming/training80-long-busy-host-d32e877d95e0420c94b7f0ab209732ce"
ORIGINAL_PINS = PROJECT / "incoming/training80-sync-after-rollover-12a31891d2544ae6a1fa7d13829642f1"
BASE_FILE = "run_training80_busy_long_host.py"
BASE_SHA = "ef1262903a59046c2c57cb1b7a93d39147858551305b88d2f1d2a6f31676a099"
INSTALL_RECEIPT_SHA = "fdaf7cd6990707735315d9d1b22e9c389a10855531866f04e7a81eecadadeab6"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def read(path, cap, expected=None):
    require(path.resolve(strict=True) == path.absolute() and path.is_file(), "exact regular path required")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and (expected is None or sha256(raw).hexdigest() == expected), "byte identity differs")
    return raw


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def claim(raw):
    data = json.loads(raw)
    require(set(data) == {"request", "first_UTC", "last_UTC", "hard_end_UTC", "GPU_requested", "parents", "points"},
            "unexpected claim fields")
    require(re.fullmatch(r"[0-9a-f]{32}", data["request"]) and data["GPU_requested"] == 0
            and data["parents"] == {"1103606": 40, "1194900": 40} and data["points"] == 1801,
            "unexpected fleet scope")
    first, last, end = (datetime.fromisoformat(data[name]).timestamp()
                        for name in ("first_UTC", "last_UTC", "hard_end_UTC"))
    require(0 < last - first <= 1800 and end - last == 3690, "unexpected launch or cleanup budget")
    require([data[k] for k in ("first_UTC", "last_UTC", "hard_end_UTC")] == [stamp(x) for x in (first, last, end)],
            "use exact UTC dates")
    return data, first, last


def started_ticks(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw[raw.rindex(")") + 1:].split()
        return int(fields[19]) if fields[0] not in {"X", "Z"} else None
    except (ValueError, OSError, IndexError):
        return None


def verify_install():
    receipt = json.loads(read(INSTALL / "INSTALL_RESULT.json", 16384, INSTALL_RECEIPT_SHA))
    members = receipt["source_SHA256"]
    require(receipt["state"] == "SHARED_HOST_CHECK_ONLY" and receipt["cards"] == 640
            and len(members) == receipt["source_count"] == 7 and members[BASE_FILE] == BASE_SHA,
            "different installed observer")
    for name, sha in members.items():
        require(re.fullmatch(r"[a-z0-9_]+\.py", name), "invalid installed member name")
        read(INSTALL / name, 32768, sha)
    for field, name in (("ledger", "ledger.json"), ("pci", "pci.json")):
        read(ORIGINAL_PINS / name, 262144, receipt["existing_target_pins_reused"][field]["SHA256"])


def controller(data, output):
    source = INSTALL / BASE_FILE
    verify_install()
    sys.path.insert(0, str(INSTALL))
    spec = importlib.util.spec_from_file_location("pinned_hour_controller", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIRST, module.LAST, module.HARD_END = (data[k] for k in ("first_UTC", "last_UTC", "hard_end_UTC"))
    result = module.run(output, ORIGINAL_PINS / "ledger.json", ORIGINAL_PINS / "pci.json", data["request"])
    return int(not result["full_hour_observer_C0"])


def launch(data, first, last, here, original_sha, output):
    require(first <= time.time() <= last, "outside claimed start window")
    require(not output.exists() and not output.is_symlink(), "original output already exists")
    verify_install()
    for parent in data["parents"]:
        status = subprocess.run(["scontrol", "show", "step", parent, "--oneliner"], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=5, check=True)
        require(not any("Name=training80-long-busy-" in line and "State=RUNNING" in line
                        for line in status.stdout.splitlines()), "another full-hour observer is active")
    # The exclusive intent is the single-use boundary even if subprocess creation or observation fails.
    write(here / "LAUNCH_INTENT.json", {"request": data["request"], "claim_SHA256": original_sha,
                                      "start_UTC": stamp(time.time()), "GPU_requested": 0, "replay_allowed": False})
    require(first <= time.time() <= last, "claimed start window expired before spawn")
    argv = [sys.executable, "-I", "-S", "-B", str(here / "observer.py"), "child", original_sha]
    with (here / "controller.stdout.private").open("xb") as out, (here / "controller.stderr.private").open("xb") as err:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, cwd=here)
    ticks = started_ticks(child.pid)
    result = {"request": data["request"], "state": "CONTROLLER_STARTED" if ticks is not None else "CONTROLLER_UNKNOWN",
              "controller_PID": child.pid, "controller_start_ticks": ticks, "GPU_requested": 0, "replay_allowed": False}
    write(here / "LAUNCH_OUTCOME.json", result)
    print(json.dumps(result, sort_keys=True))


def observe(here, data, output):
    result = {"request": data["request"], "same_process_live": None, "receipts": {}}
    for path in (here / "LAUNCH_OUTCOME.json", output / "host-intent.json", output / "ready.json", output / "host-outcome.json"):
        if not path.exists():
            continue
        raw = read(path, 65536)
        value = json.loads(raw)
        require(value["request"] == data["request"], "cross-request observation")
        result["receipts"][path.name] = sha256(raw).hexdigest()
        if path.name == "LAUNCH_OUTCOME.json" and value["controller_start_ticks"] is not None:
            result["same_process_live"] = started_ticks(value["controller_PID"]) == value["controller_start_ticks"]
        elif path.name == "ready.json":
            result.update(ready_nodes=value["ready_nodes"], steps=value["actual_READY_steps"])
        elif path.name == "host-outcome.json":
            result["original_full_hour_C0"] = value["full_hour_observer_C0"]
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "child", "observe"))
    parser.add_argument("claim_SHA256")
    args = parser.parse_args()
    require(os.name == "posix" and sys.flags.isolated and sys.flags.dont_write_bytecode, "isolated POSIX launch required")
    os.umask(0o077)
    here = Path(__file__).resolve().parent
    data, first, last = claim(read(here / "claim.json", 4096, args.claim_SHA256))
    require(here == PROJECT / "incoming" / ("training80-claimed-hour-" + data["request"]), "incorrect installed scope")
    output = PROJECT / "results" / ("training80-long-busy-" + data["request"])
    if args.mode == "observe":
        observe(here, data, output)
    elif args.mode == "launch":
        launch(data, first, last, here, args.claim_SHA256, output)
    else:
        intent = json.loads(read(here / "LAUNCH_INTENT.json", 4096))
        require(intent["request"] == data["request"] and intent["claim_SHA256"] == args.claim_SHA256, "unbound child")
        return controller(data, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
