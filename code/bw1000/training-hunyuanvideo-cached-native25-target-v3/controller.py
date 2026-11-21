"""One fixed native cached T09 attempt; real accounting and independent post are mandatory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from commands import allowance, argv as phase_argv
from contract import (Budget, LABEL, NODE, PARENT, REQUEST, RUN, WHOLE, WORK, load,
                      need, save, validate_assets, validate_preflight, verify_package)
from review import COLUMNS, native, post
from slurm_accounting import is_terminal, wait_terminal
from flash_compat import prepare_overlay

LOG_LIMIT = 16 * 1024**2
RUN = Path(RUN)


def identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text.rsplit(") ", 1)[1].split()
    return {"PID": pid, "start_ticks": int(fields[19]), "PGID": int(fields[2]),
            "host": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def query(budget, args):
    left = budget.left() - 10
    need(left > 0, "Original accounting/query deadline exhausted")
    done = subprocess.run(args, capture_output=True, timeout=min(10, left), check=False)
    need(done.returncode == 0 and len(done.stdout) <= 2 * 1024**2 and len(done.stderr) <= 65536,
         "Bounded Slurm query failed")
    return done.stdout.decode("utf-8", errors="strict").splitlines()


def accounting_rows(lines, phase):
    parsed = []
    for line in lines:
        if not line.strip():
            continue
        row = line.split("|")
        need(len(row) == len(COLUMNS), "Malformed accounting row cannot be silently discarded")
        need(row[0] == PARENT or re.fullmatch(re.escape(PARENT) + r"\.(?:[0-9]+|batch|extern)", row[0]),
             "Accounting row outside original parent")
        parsed.append(row)
    need(sum(row[0] == PARENT for row in parsed) == 1, "Current accounting parent missing or ambiguous")
    rows = [row for row in parsed if row[1] == LABEL + "-" + phase]
    need(len(rows) <= 1, "Fixed scope label has multiple actual steps")
    if rows:
        need(re.fullmatch(re.escape(PARENT) + r"\.[0-9]+", rows[0][0]) and rows[0][5] in (NODE, "", "None assigned"),
             "Original phase placement differs")
    return rows[0] if rows else None


def accounting(budget, phase):
    return accounting_rows(query(budget, ["sacct", "-n", "-P", "-j", PARENT,
        "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"]), phase)


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def admit(budget):
    rows = query(budget, ["scontrol", "show", "job", PARENT, "--oneliner"])
    need(len(rows) == 1, "Parent allocation missing or ambiguous")
    job = fields(rows[0])
    need(job.get("JobId") == PARENT and job.get("JobState") == "RUNNING" and job.get("NodeList"),
         "Original parent allocation is not currently running")
    hosts = query(budget, ["scontrol", "show", "hostnames", job["NodeList"]])
    need(len(hosts) == len(set(hosts)) and NODE in hosts, "Requested node is not in original parent")
    seen = set()
    for line in query(budget, ["scontrol", "show", "step", PARENT, "--oneliner"]):
        if not line.strip():
            continue
        step = fields(line)
        handle = step.get("StepId", "")
        if handle in (PARENT + ".batch", PARENT + ".extern"):
            continue
        need(re.fullmatch(re.escape(PARENT) + r"\.[0-9]+", handle) and handle not in seen
             and step.get("NodeList"), "Active step inventory incomplete or ambiguous")
        seen.add(handle)
        members = query(budget, ["scontrol", "show", "hostnames", step["NodeList"]])
        need(members and len(members) == len(set(members)) and set(members) <= set(hosts),
             "Active step membership differs")
        need(NODE not in members, "Exact node already has an active owner; do not kill it")


def stop_owned(child, owner, budget):
    if child.poll() is not None:
        return
    need(identity(child.pid) == owner and owner["PGID"] == child.pid, "Owned srun identity changed")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=min(5, max(.05, budget.left() - 5)))
    except subprocess.TimeoutExpired:
        need(identity(child.pid) == owner, "Owned srun identity changed before KILL")
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=min(5, max(.05, budget.left())))


def phase(budget, name, origin, pf_sha, source_sha):
    seconds = allowance(budget, name)
    command = phase_argv(name, seconds, origin, pf_sha, source_sha)
    attempt = {"request_id": REQUEST, "phase": name, "argv": command,
               "actual_Slurm_seconds_limit": seconds, "client_seconds_limit": seconds + 5,
               "epoch": origin, "deadline": origin + WHOLE}
    save(RUN / (name + "-intent.json"), attempt)  # Persist before any Popen.
    receipt = {"phase": name, "srun_invoked": False, "actual_numbered_steps": None,
               "child_reaped": False, "passed": False, "row": None, "terminal_visible": False}
    child, owner = None, None
    logpath = RUN / (name + ".log")
    started = time.monotonic()
    try:
        with logpath.open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt["srun_invoked"] = True
            try:
                owner = identity(child.pid)
                save(RUN / (name + "-owner.json"), owner)
            except FileNotFoundError:
                need(child.poll() is not None, "Owned srun identity unavailable while live")
            while child.poll() is None:
                need(logpath.stat().st_size <= LOG_LIMIT, "Native log exceeded bounded private extent")
                if time.monotonic() - started >= seconds + 5 or budget.left() <= 25:
                    receipt["timed_out"] = True
                    break
                time.sleep(min(.2, max(.01, seconds + 5 - (time.monotonic() - started))))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["error_type"] = type(error).__name__
    finally:
        try:
            if child is not None:
                stop_owned(child, owner, budget)
                receipt["returncode"] = child.returncode
                receipt["child_reaped"] = child.poll() is not None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            receipt["cleanup_error_type"] = type(error).__name__
        try:
            # A returned client is not terminal evidence. Never relaunch on UNKNOWN.
            seconds_left = min(20, budget.left() - 10)
            need(seconds_left > 0, "No original terminal-accounting reserve remains")
            terminal = wait_terminal(lambda: accounting(budget, name), time.monotonic() + seconds_left,
                                     interval=1, max_observations=20)
            receipt.update(terminal)
            row = receipt["row"]
            receipt["actual_numbered_steps"] = 1 if row else None
            receipt["passed"] = (receipt["child_reaped"] and receipt.get("returncode") == 0
                and not receipt.get("timed_out", False) and not receipt.get("error_type") and row is not None
                and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            receipt["accounting_error_type"] = type(error).__name__
        if logpath.exists():
            info = logpath.stat()
            receipt["log"] = {"bytes": info.st_size,
                              "SHA256": hashlib.sha256(logpath.read_bytes()).hexdigest() if info.st_size <= LOG_LIMIT else None}
        save(RUN / (name + "-accounting.json"), receipt)
    return receipt


def run(args):
    budget = Budget(args.epoch)
    need(args.workspace == RUN and RUN.is_dir() and not RUN.is_symlink(), "Wrong or absent independent namespace")
    need(not any((RUN / name).exists() for name in ("intent.json", "outcome.json", "outputs")),
         "Original independent request already attempted; no replay")
    result = {"request_id": REQUEST, "parent": PARENT, "node": NODE, "epoch": args.epoch,
              "deadline": args.epoch + WHOLE, "source_manifest_SHA256": args.source_manifest_sha256,
              "passed": False, "execution_state": "FAILED", "release_permitted": False,
              "planned_updates": 25, "completed_updates": 0, "missing_updates": 25,
              "receipts": {}, "implicit_loader_GPU_use": "UNKNOWN", "Model_GO": False,
              "formal17_qualified": False, "original_numerics_qualified": False}
    try:
        sourcepin = verify_package(RUN / "src", args.source_manifest_sha256)
        preflight, pfpin = load(RUN / "preflight.json")
        need(pfpin["SHA256"] == args.preflight_sha256, "Preflight identity differs")
        validate_preflight(preflight, args.epoch)
        result["assets"] = validate_assets(preflight, execution_mount=False)
        for name in ("pre", "gpu", "post"):
            need(accounting(budget, name) is None, "Existing actual numbered step for fixed label")
        admit(budget)
        save(RUN / "intent.json", {"request_id": REQUEST, "epoch": args.epoch,
             "deadline": args.epoch + WHOLE, "source_manifest": sourcepin, "preflight": pfpin,
             "owner": identity(os.getpid()), "planned_updates": 25})
        for name in ("pre", "gpu-gate", "post", "locks"):
            (RUN / name).mkdir()
        result["compatibility_prepared"] = prepare_overlay()
        result["receipts"]["pre"] = phase(budget, "pre", args.epoch, args.preflight_sha256, args.source_manifest_sha256)
        if result["receipts"]["pre"]["passed"]:
            admit(budget)
            result["receipts"]["gpu"] = phase(budget, "gpu", args.epoch, args.preflight_sha256, args.source_manifest_sha256)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
    finally:
        receipts = result["receipts"]
        try:
            if "gpu" in receipts and receipts["gpu"]["terminal_visible"]:
                receipts["post"] = phase(budget, "post", args.epoch, args.preflight_sha256, args.source_manifest_sha256)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            result["post_error_type"] = type(error).__name__
        result["native"] = native(RUN / "outputs/hunyuanvideo-1.5/train" / LABEL)
        result["completed_updates"] = result["native"]["completed_updates"]
        result["missing_updates"] = 25 - result["completed_updates"]
        try:
            result["post"] = post(RUN, receipts)
            result["release_permitted"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["post"] = {"state": "UNKNOWN", "error_type": type(error).__name__}
        result["passed"] = (result["release_permitted"] and result["native"]["native_evidence_complete"]
                            and all(item["passed"] for item in receipts.values()))
        result["execution_state"] = "COMPLETED" if result["passed"] else "FAILED_OR_INCOMPLETE_NO_REPLAY"
        save(RUN / "outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("request_id", "passed", "execution_state", "planned_updates",
                                           "completed_updates", "missing_updates", "release_permitted")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
