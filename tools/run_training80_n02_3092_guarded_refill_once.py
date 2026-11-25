"""Renew f13r1n02 after the exact .3092 worker returns naturally."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import sqlite3
import subprocess
import time
from types import FunctionType


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
STEM = "training80-n02-3092-guarded-standby-renewal-v1"
CLAIM = REPO / "coordination/tasks" / f"{STEM}.json"
ORIGINAL = REPO / "tools/run_training80_n02_guarded_refill_once.py"
ORIGINAL_PIN = "49f637945d2850b81ed666cb6976f25ae57b233babbae021fc806ec3cbcf734c"
ORIGINAL_CLAIM_PIN = "0f2074c33f9fc84e889643bdaad57d67a4213f415e6f32ea49796a9d5f93ab5c"
NODE, PARENT, PREVIOUS, CLEARANCE = "f13r1n02", "1103606", "1103606.3092", "1103606.3091"
OLD_REQUEST = "6a96939821164a63bfb2e8a789453fff"
PREPARATION, BUDGET = 1795577922.086273, 1795578822.086273
GPU_TRES = "cpu=8,gres/dcu=8,mem=16G,node=1"
TERMINAL_AGE_SECONDS = 45
OBSERVE_RTT_SECONDS = 25


HOST_PREDECESSOR = r'''
def previous_file(folder, name):
    path = folder / name
    require(path.resolve(strict=True) == path, "old worker receipt path changed")
    with path.open("rb") as stream:
        raw = stream.read(4097)
    require(0 < len(raw) <= 4096, "old worker receipt size invalid")
    return json.loads(raw), sha256(raw).hexdigest()


def check_previous(row):
    require(row and row[:6] == [PREVIOUS, PREVIOUS_LABEL, "COMPLETED", "0:0", NODE, GPU_TRES]
            and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", row[6]),
            "old GPU actual C0/End not confirmed")
    if EXPECTED_TERMINAL is not None:
        require(row == EXPECTED_TERMINAL["accounting"], "old terminal identity differs from independent observed evidence")
    detail = [item for line in query("sacct", "-nP", "-j", PREVIOUS, "-o", "JobIDRaw,ElapsedRaw,End")
              if len(item := line.split("|")) == 3 and item[0] == PREVIOUS]
    require(len(detail) == 1 and detail[0][1].isdecimal() and int(detail[0][1]) >= OLD_SECONDS and detail[0][2] == row[6],
            "old accounting does not confirm natural worker budget")
    folder = RESULTS / ("training80-dcu-standby-" + OLD_REQUEST) / NODE
    (worker, worker_pin), (cgroup, cgroup_pin), (expected, expected_pin), (terminal, terminal_pin) = [
        previous_file(folder, name) for name in ("worker.json", "worker-cgroup.json", "expected.json", "terminal.json")]
    require([worker_pin, cgroup_pin, expected_pin] == OLD_WORKER_PINS
            and worker["node"] == cgroup["node"] == expected["node"] == terminal["node"] == NODE
            and worker["step"] == cgroup["step"] == terminal["step"] == PREVIOUS
            and worker["worker_pid"] == cgroup["worker_pid"] > 0 and cgroup["worker_start_ticks"] > 0
            and worker["synthetic"] is cgroup["same_UID"] is cgroup["exact_job_and_step_cgroup"] is True
            and expected["parent"] == PARENT and expected["request"] == OLD_REQUEST
            and expected["label"] == PREVIOUS_LABEL and expected["seconds"] == OLD_SECONDS
            and sorted(worker["pci"]) == sorted(expected["pci"]) == PCI and worker["epoch"] == OLD_START
            and terminal["state"] == "WORKER_RETURNED" and terminal["epoch"] >= OLD_START + OLD_SECONDS,
            "old exact worker did not return naturally")
    if EXPECTED_TERMINAL is not None:
        require(terminal_pin == EXPECTED_TERMINAL["worker_terminal_SHA256"], "old terminal bytes changed")
    return dict(accounting=row, worker_terminal_SHA256=terminal_pin, worker_pins=OLD_WORKER_PINS,
                worker_return_epoch=terminal["epoch"], elapsed_seconds=int(detail[0][1]))
'''

HOST_READ = r'''
try:
    row = account(PREVIOUS)
    if row and row[:6] == [PREVIOUS, PREVIOUS_LABEL, "RUNNING", "0:0", NODE, GPU_TRES] and row[6] == "Unknown":
        exact = [item for line in query("scontrol", "show", "step", "-o", PREVIOUS)
                 if (item := fields(line)).get("StepId") == PREVIOUS]
        require(len(exact) == 1 and (exact[0]["Name"], exact[0]["NodeList"], exact[0]["State"], exact[0]["TRES"])
                == (PREVIOUS_LABEL, NODE, "RUNNING", GPU_TRES), "old live controller identity differs")
        result = dict(state="OLD_STILL_RUNNING", accounting=row)
    else:
        result = dict(state="OLD_ACTUAL_TERMINAL_TARGET_EMPTY", **check_previous(row))
        scope()
        result["two_views_empty"] = True
except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
    result = dict(state="OLD_TERMINAL_OR_EMPTY_UNCONFIRMED", error_class=type(error).__name__,
                  detail_SHA256=sha256(str(error).encode()).hexdigest())
print(json.dumps(dict(result, read_id=READ_ID, predecessor=PREVIOUS, next_request=REQUEST,
                      new_Slurm_operations=0), sort_keys=True, allow_nan=False), flush=True)
'''


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def replace(text, anchor, replacement):
    require(text.count(anchor) == 1, "frozen guarded insertion point changed")
    return text.replace(anchor, replacement, 1)


def load():
    require(ORIGINAL.resolve(strict=True) == ORIGINAL and sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_PIN,
            "frozen n02 guard changed")
    original = runpy.run_path(str(ORIGINAL))
    raw = CLAIM.read_bytes()
    claim = json.loads(raw)
    prior_raw = original["pinned"](REPO / original["CLAIM"], ORIGINAL_CLAIM_PIN)
    prior_claim = json.loads(prior_raw)
    require((claim["scope"], claim["state"], claim["node"], claim["parent"], claim["previous_GPU"], claim["previous_probe"],
             claim["previous_request"], claim["preparation_epoch"], claim["natural_budget_epoch"], claim["max_invocations"],
             claim["replay_allowed"]) == (STEM, "PLANNING_NO_DISPATCH", NODE, PARENT, PREVIOUS, CLEARANCE, OLD_REQUEST,
                                          PREPARATION, BUDGET, 1, False)
            and prior_claim["request"] == OLD_REQUEST and re.fullmatch(r"[a-f0-9]{32}", claim["request"])
            and claim["request"] != OLD_REQUEST, "new exact n02 claim differs")
    proof = {}
    for name, spec in claim["proof"].items():
        path = REPO / spec["path"]
        require(path.resolve(strict=True) == path and path.is_relative_to(LOCAL), "invalid private prerequisite path")
        proof[name] = json.loads(original["pinned"](path, spec["SHA256"]))
    start = proof["startup"]["observed"]
    cross = proof["counter_cross_check"]
    prep = start["original_guard_outcome"][0]
    clear = prep["fresh_zero_GRES_accounting"]
    old_start, old_seconds = start["actual_worker_start_epoch"], start["worker_budget_seconds"]
    require(proof["startup"]["rc"] == 0 and start["request"] == cross["request"] == OLD_REQUEST
            and start["step"] == cross["step"] == PREVIOUS and start["node"] == cross["node"] == NODE
            and start["accounting"] == [PREVIOUS, f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}",
                                       "RUNNING", "0:0", NODE, GPU_TRES, "Unknown"]
            and cross["actual_verified_same_PCI_counter_growth"] == 8 and len(start["first"]) == 8
            and (old_start, old_seconds) == (claim["old_worker_start"], claim["old_worker_budget_s"])
            and old_seconds == cross["worker_budget_seconds"] == 10708 and old_start + old_seconds == BUDGET
            and start["next_preparation_epoch"] == cross["next_preparation_epoch"] == PREPARATION
            and prep["GPU_srun_attempted"] is True and prep["probe_step"] == CLEARANCE
            and clear[:6] == [CLEARANCE, f"training80-n02-fresh-{OLD_REQUEST[:12]}", "COMPLETED", "0:0", NODE,
                              "cpu=1,mem=256M,node=1"], "old worker startup or physical admission differs")
    pins = [start[field] for field in ("worker_SHA256", "worker_cgroup_SHA256", "expected_SHA256")]
    require(all(re.fullmatch(r"[a-f0-9]{64}", value) for value in pins)
            and pins[:2] == [cross["worker_SHA256"], cross["worker_cgroup_SHA256"]], "old worker identity differs")
    ordinary = proof["ordinary"]
    require(ordinary["planned_nodes"] == 58 and ordinary["independent_positive_growth_DCU"] == 464
            and ordinary["next_preparation_CST"] == claim["ordinary_first_preparation_CST"]
            and ordinary["next_budget_range_CST"][0] == claim["ordinary_first_budget_CST"], "ordinary overlap calendar differs")
    for path, sha in ((original["HELPER"], original["HELPER_SHA"]), (original["TRANSPORT"], original["TRANSPORT_SHA"])):
        original["pinned"](path, sha)
    return original, prior_claim, raw, claim, start, clear, pins


def render(material, publication=None, terminal=None, read_id=None):
    original, prior_claim, _, claim, start, clear, pins = material
    config = original["render"](dict(prior_claim, request=claim["request"]), publication)
    updates = dict(SCOPE=STEM, PREVIOUS=PREVIOUS, PREVIOUS_LABEL=start["accounting"][1],
                   CLEAR_STEP=clear[0], CLEAR_LABEL=clear[1], CLEAR_END=clear[6], GPU_TRES=GPU_TRES,
                   OLD_REQUEST=OLD_REQUEST, OLD_WORKER_PINS=pins, OLD_START=claim["old_worker_start"],
                   OLD_SECONDS=claim["old_worker_budget_s"], EXPECTED_TERMINAL=terminal, READ_ID=read_id)
    constants, remainder = config["launch"].split("\nimport fcntl", 1)
    bindings = {name: value for line in constants.splitlines() if line for name, _, value in [line.partition("=")]}
    del bindings["PREVIOUS_END"]
    bindings.update({key: repr(value) for key, value in updates.items()})
    prefix = "\n".join(f"{key}={value}" for key, value in bindings.items()) + "\n"
    host, guard = ("import fcntl" + remainder).split("\nimport base64, runpy, shlex, sys", 1)
    start_legacy = host.index("\ndef check_previous(row):")
    end_legacy = host.index("\ndef scope(", start_legacy)
    host = host[:start_legacy] + host[end_legacy:]
    empty = "({OBSERVER_STEP, own} if own else {OBSERVER_STEP})"
    host = replace(host, empty, "({own} if own else set())")
    host = prefix + host + HOST_PREDECESSOR
    guard = "\nimport base64, runpy, shlex, sys" + guard
    anchor = "            check_previous(account(PREVIOUS))"
    guard = replace(guard, anchor, "            outcome['previous_natural_terminal'] = check_previous(account(PREVIOUS))")
    config["launch"] = host + guard
    config["read_predecessor"] = host + HOST_READ
    observe = replace(config["observe"], f"SCOPE={original['STEM']!r}", f"SCOPE={STEM!r}")
    needle = '    [line] = query("scontrol", "show", "step", "-o", step)'
    config["observe"] = replace(observe, needle, '    [line] = [item for item in query("scontrol", "show", "step", "-o", step)\n'
                                '              if fields(item).get("StepId") == step]')
    for name in ("launch", "read_predecessor", "observe"):
        compile(config[name], f"<exact n02 {name}>", "exec")
    return config


def pinned_publication(original, revision):
    prior_raw = original["pinned"](REPO / original["CLAIM"], ORIGINAL_CLAIM_PIN)
    original["verify_publication"](revision, prior_raw)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    for path in (Path(__file__).resolve(), CLAIM):
        blob = subprocess.check_output([*git, "cat-file", "blob", revision + ":" + path.relative_to(REPO).as_posix()],
                                       stderr=subprocess.PIPE, cwd=REPO, timeout=8)
        require(blob == path.read_bytes(), "exact successor publication differs")


def authoritative_read(material, config, read_id):
    original, _, _, claim, *_ = material
    require(time.time() >= PREPARATION, "preparation window has not begun")
    path = read_path(read_id)
    require(not os.path.lexists(path), "use a unique read-only identifier")
    helper = runpy.run_path(str(original["HELPER"]))
    ssh = helper["ssh"](REPO.parent)
    command = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-"])
    before = time.monotonic()
    anchor = time.time() - before
    try:
        reply = subprocess.run([*ssh, command], input=config["read_predecessor"].encode(), capture_output=True,
                               timeout=OBSERVE_RTT_SECONDS)
        rc, stdout, stderr = reply.returncode, reply.stdout, reply.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    except OSError as error:
        rc, stdout, stderr = None, b"", type(error).__name__.encode()
    received = time.monotonic()
    result = None
    if rc == 0 and 0 < len(stdout) <= 8192 and 0 <= received - before <= OBSERVE_RTT_SECONDS:
        try:
            value = json.loads(stdout)
            if value["read_id"] == read_id and value["next_request"] == claim["request"] and value["predecessor"] == PREVIOUS:
                result = value
        except (ValueError, KeyError, TypeError):
            pass
    row = dict(received_monotonic=received, roundtrip_s=received-before, boot_epoch_anchor=anchor, rc=rc, result=result,
               stdout_SHA256=original["digest"](stdout), stderr_SHA256=original["digest"](stderr), replay_allowed=False)
    config["scope"]["save"](path, row)
    return dict(receipt=path.name, SHA256=original["digest"](path.read_bytes()), state=(result or {}).get("state", "UNCONFIRMED"),
                replay_allowed=False, GPU_invocations=0)


def read_path(read_id):
    require(re.fullmatch(r"[a-f0-9]{32}", read_id), "choose a unique read-only id")
    return LOCAL / f"{STEM}-predecessor-{read_id}.json"


def fresh_terminal(read_id, pin, original, claim, pins):
    proof = json.loads(original["pinned"](read_path(read_id), pin))
    now, epoch = time.monotonic(), time.time()
    require(epoch >= BUDGET and abs(epoch-now-proof["boot_epoch_anchor"]) < 2
            and 0 <= now-proof["received_monotonic"] <= TERMINAL_AGE_SECONDS
            and 0 <= proof["roundtrip_s"] <= OBSERVE_RTT_SECONDS and proof["rc"] == 0
            and proof["replay_allowed"] is False and proof["result"], "fresh exact local terminal transport required")
    body = proof["result"]
    require(body["read_id"] == read_id and body["predecessor"] == PREVIOUS and body["next_request"] == claim["request"]
            and body["new_Slurm_operations"] == 0 and body["state"] == "OLD_ACTUAL_TERMINAL_TARGET_EMPTY"
            and body["two_views_empty"] is True and body["worker_pins"] == pins
            and re.fullmatch(r"[a-f0-9]{64}", body["worker_terminal_SHA256"])
            and body["accounting"][:6] == [PREVIOUS, f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}",
                                            "COMPLETED", "0:0", NODE, GPU_TRES], "actual old C0 with both views empty required")
    return body


def original_reader(original):
    function = original["observe"]
    bound = dict(function.__globals__, STEM=STEM, INTENT=LOCAL / f"{STEM}-intent.json", TOKEN=LOCAL / f"{STEM}-send.json")
    return FunctionType(function.__code__, bound, "observe_successor")


def launch(material, config, revision, id_, pin):
    original, _, raw, claim, _, _, pins = material
    stem = LOCAL / STEM
    intent, token, reply = (stem.with_name(STEM + f"-{key}.json") for key in ("intent", "send", "reply"))
    require(not any(map(os.path.lexists, (intent, token, reply))), "new unique request already reserved; reconcile it")
    pinned_publication(original, revision)
    original["current_local_authority"](config)
    terminal = fresh_terminal(id_, pin, original, claim, pins)
    current = render(material, revision, terminal=terminal)
    helper = runpy.run_path(str(original["HELPER"]))
    sender = runpy.run_path(str(original["TRANSPORT"]))
    data = dict(request=claim["request"], publication=revision, terminal_read_id=id_, terminal_read_SHA256=pin,
                exact_program_SHA256=original["digest"](current["launch"].encode()), claim_SHA256=original["digest"](raw),
                GPU_requested=8, replay_allowed=False)
    current["scope"]["save"](intent, data)
    response = sender["send"](helper["ssh"](REPO.parent), request=claim["request"], program=current["launch"],
                              data=b"", send_token=token, authority_seconds=18, outcome_seconds=82)
    current["scope"]["save"](reply, dict(data, transport=response))
    return dict(receipt=reply.name, SHA256=original["digest"](reply.read_bytes()), request=claim["request"],
                guard_state=(response.get("response") or {}).get("state"), replay_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--check-publication")
    mode.add_argument("--observe-predecessor")
    mode.add_argument("--publication")
    mode.add_argument("--observe-original", action="store_true")
    parser.add_argument("--terminal-receipt", nargs=2, metavar=("READ_ID", "SHA256"))
    args = parser.parse_args()
    require(bool(args.publication) == bool(args.terminal_receipt), "publication requires the exact fresh terminal receipt")
    material = load()
    original, _, raw, claim, *_ = material
    config = render(material, args.publication or args.check_publication, read_id=args.observe_predecessor)
    if args.check or args.check_publication:
        if args.check_publication:
            pinned_publication(original, args.check_publication)
            original["current_local_authority"](config)
        result = dict(state="PLANNING_ONLY_NO_TERMINAL_OBSERVED", claim_SHA256=original["digest"](raw), GPU=0, Slurm=0, SCNet=0,
                      request=claim["request"], worker_seconds=config["worker_seconds"], preparation_epoch=PREPARATION,
                      natural_budget_epoch=BUDGET, read_SHA256=original["digest"](config["read_predecessor"].encode()),
                      launch_SHA256=original["digest"](config["launch"].encode()))
    elif args.observe_predecessor:
        result = authoritative_read(material, config, args.observe_predecessor)
    elif args.observe_original:
        reply = original_reader(original)(claim, config)
        observed = reply.get("observed") or {}
        result = dict(file=reply["file"], SHA256=reply["SHA256"], state=observed.get("state"), step=observed.get("step"),
                      verified_progress=observed.get("actual_DCU_progress"))
    else:
        result = launch(material, config, args.publication, *args.terminal_receipt)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(json.dumps(dict(error_class=type(error).__name__, detail_SHA256=sha256(str(error).encode()).hexdigest())))
        raise SystemExit(1)
