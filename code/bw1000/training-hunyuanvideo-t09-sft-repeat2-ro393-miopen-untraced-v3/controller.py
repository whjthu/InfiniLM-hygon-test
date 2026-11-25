"""One fixed native cached T09 attempt; real accounting and independent post are mandatory."""
import argparse
from contextlib import ExitStack
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
from placement import bind_authority, live_views, shared_lock, step_tres
from repeat_clock import measured_native
from contract import ROOT, read
from framework_mount import verify_stage

LOG_LIMIT = 16 * 1024**2
ORIGINAL_OUTCOME_SHA='2cfb6717c52db028a5fd5290b7749a4201ac7db56b30372b4a8d73200b5b144d'
ORIGINAL_REQUEST='df2a0db37f494c0a9f149260d71cb5e6'
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
    return live_views(lambda args:query(budget,["env","TZ=UTC",*args]),os.getuid(),budget.origin)


def previous_terminal():
    prior=ROOT/"results"/"training-hunyuanvideo-cached-native25-target-v3"/"outcome.json"
    value,pin=load(prior,256<<10)
    need(pin["SHA256"]==ORIGINAL_OUTCOME_SHA and value["request_id"]==ORIGINAL_REQUEST
         and value["passed"] is True and value["release_permitted"] is True
         and {k:v["row"][0] for k,v in value["receipts"].items()}
         =={"pre":"1087656.793","gpu":"1087656.794","post":"1087656.798"},
         "T09_PRIOR_TERMINAL_AND_POST_READBACK")
    return pin


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


def phase(budget, name, origin, pf_sha, source_sha, release_lock=None):
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
            if release_lock is not None:
                try:
                    stop=min(time.monotonic()+10,time.monotonic()+max(0,budget.left()-180))
                    while time.monotonic()<stop:
                        row=accounting(budget,name)
                        if row and row[2] in ("RUNNING","COMPLETED","FAILED") and row[5]==NODE and step_tres(row[6],name):
                            save(RUN/"gpu-lock-handoff.json",dict(request_id=REQUEST,step=row[0],tres=row[6]))
                            receipt["gpu_lock_handoff_known"]=True
                            break
                        time.sleep(min(.2,max(0,stop-time.monotonic())))
                    if not receipt.get("gpu_lock_handoff_known"):
                        save(RUN/"gpu-lock-handoff-unknown.json",dict(request_id=REQUEST,
                             ordinary_exclusion_retained=True,replay_permitted=False))
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    save(RUN/"gpu-lock-handoff-unknown.json",dict(request_id=REQUEST,error_type=type(exc).__name__,
                         ordinary_exclusion_retained=True,replay_permitted=False))
                finally:
                    release_lock()
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
                and row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE and step_tres(row[6],name))
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
        result["original_794_terminal"]=previous_terminal()
        result["private_framework_stage"]=verify_stage()
        authority,authpin=load(RUN/"placement-authority.json",16384)
        route,routepin=load(RUN/"physical-route.json",16384)
        need(authpin["SHA256"]==args.authority_sha256 and routepin["SHA256"]==args.route_sha256,
             "T09_CURRENT_EXACT_PRIVATE_AUTHORITY_AND_ROUTE_PINS")
        route= dict(route,receipt_SHA256=routepin["SHA256"])
        bind_authority(authority,request=REQUEST,publication=args.publication,epoch=args.epoch,route=route)
        with ExitStack() as locks:
            locks.enter_context(shared_lock(Path(ROOT)))
            for name in ("pre", "gpu", "post"):
                need(accounting(budget, name) is None, "Existing actual numbered step for fixed label")
            admit(budget)
            bind_authority(authority,request=REQUEST,publication=args.publication,epoch=args.epoch,route=route)
            save(RUN / "intent.json", {"request_id": REQUEST, "epoch": args.epoch,
                 "deadline": args.epoch + WHOLE, "source_manifest": sourcepin, "preflight": pfpin,
                 "owner": identity(os.getpid()), "planned_updates": 25})
            for name in ("pre", "gpu-gate", "post", "locks"):
                (RUN / name).mkdir()
            result["compatibility_prepared"] = prepare_overlay()
            result["receipts"]["pre"] = phase(budget, "pre", args.epoch, args.preflight_sha256, args.source_manifest_sha256)
            if result["receipts"]["pre"]["passed"]:
                admit(budget)
                result["receipts"]["gpu"] = phase(budget, "gpu", args.epoch, args.preflight_sha256, args.source_manifest_sha256, locks.close)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result["error_type"] = type(error).__name__
    finally:
        receipts = result["receipts"]
        try:
            if "gpu" in receipts and receipts["gpu"]["terminal_visible"]:
                receipts["post"] = phase(budget, "post", args.epoch, args.preflight_sha256, args.source_manifest_sha256)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            result["post_error_type"] = type(error).__name__
        try:
            result["framework_mount_ingress"], result["framework_mount_ingress_pin"] = load(RUN / "framework-mount-ingress.json", 16384)
        except (OSError, ValueError, KeyError, TypeError):
            result["framework_mount_ingress"] = None
        result["native"] = native(RUN / "outputs/hunyuanvideo-1.5/train" / LABEL)
        result["completed_updates"] = result["native"]["completed_updates"]
        result["missing_updates"] = 25 - result["completed_updates"]
        try:
            result["timing"]=measured_native(RUN/"outputs/hunyuanvideo-1.5/train"/LABEL,read)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            result["timing"]={"state":"INCOMPLETE","type":type(exc).__name__}
        try:
            result["post"] = post(RUN, receipts)
            result["release_permitted"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["post"] = {"state": "UNKNOWN", "error_type": type(error).__name__}
        from production_scope import read_production_evidence
        gpu_row=receipts.get("gpu",{}).get("row")
        result["production_evidence"]=read_production_evidence(RUN,gpu_row[0] if gpu_row else None,args.epoch)
        if not result["production_evidence"]["passed"] and result["timing"].get("wall_seconds") is not None:
            result["timing"]["production_qualification"]="UNQUALIFIED_SCOPE_EVIDENCE"
        result["passed"] = (result["production_evidence"]["passed"] and result["release_permitted"] and result["native"]["native_evidence_complete"]
                            and result["timing"].get("wall_seconds") is not None
                            and all(item["passed"] for item in receipts.values()))
        result["execution_state"] = "COMPLETED" if result["passed"] else "FAILED_OR_INCOMPLETE_NO_REPLAY"
        save(RUN / "outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--authority-sha256",required=True)
    parser.add_argument("--route-sha256",required=True)
    parser.add_argument("--publication",required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("request_id", "passed", "execution_state", "planned_updates",
                                           "completed_updates", "missing_updates", "release_permitted")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
