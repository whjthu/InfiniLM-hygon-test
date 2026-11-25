"""Run independent, fixed successors for five proven group 4 pre-bridge exclusions."""

import argparse
import ast
from base64 import b64encode
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import sqlite3
import subprocess
import sys
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
TOOLS = REPO / "tools"
STEM = "training80-prebridge-five-guarded-bf16-v1"
CLAIM = REPO / "coordination/tasks" / f"{STEM}.json"
N02 = TOOLS / "run_training80_n02_guarded_refill_once.py"
N02_PIN = "49f637945d2850b81ed666cb6976f25ae57b233babbae021fc806ec3cbcf734c"
JOURNAL = LOCAL / "training80-1738-ordinary-renewal-dispatch"
PARENT = "1103606"
VGGT = "f13r1n09"
FIVE = {
    "f13r1n10": ("1103606.2907", "1103606.3174"),
    "f13r1n11": ("1103606.2910", "1103606.3175"),
    "f13r1n17": ("1103606.2908", "1103606.3173"),
    "f13r2n03": ("1103606.2929", "1103606.3176"),
    "f13r2n10": ("1103606.2909", "1103606.3177"),
}
LEGACY = "edf616f6543e4dafa7ead6fe9394e695"
CALENDAR = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
PREP = "2c45034fe3374e45a6e253f3c2c5114e"
HOST_TRES = "cpu=1,mem=256M,node=1"
GPU_TRES = "cpu=8,gres/dcu=8,mem=16G,node=1"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return sha256(value).hexdigest()


def pin(path, expected=None, limit=262144):
    require(path.resolve(strict=True) == path, "noncanonical source or evidence")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (expected is None or digest(raw) == expected), "source or evidence bytes differ")
    return raw


def evidence(entry):
    path = REPO / entry["path"]
    require(path.is_relative_to(REPO) and ".." not in path.parts, "out-of-scope evidence")
    return pin(path, entry["SHA256"])


def clear_original_slot(node, old):
    require(all(not os.path.lexists(JOURNAL / f"{node}-{old}-{slot}.json") for slot in (1, 2)),
            "later legacy reservation exists")


def terminal(row, step, node, tres):
    return (type(row) is list and len(row) == 9 and row[0] == step and row[2:4] == ["COMPLETED", "0:0"]
            and row[5:7] == [node, tres] and row[8] not in {"", "Unknown", "N/A", "None"})


def originals(claim, original_claim, clearance):
    raw = {name: evidence(item) for name, item in claim["legacy_evidence"].items()}
    child, outcome, observed, classifier = (json.loads(raw[key]) for key in ("child", "outcome", "terminal", "classifier"))
    old_scope = set(FIVE) | {VGGT}
    require(child == dict(batch=4, calendar_SHA256=CALENDAR, nodes=outcome["terminal_ready_nodes"],
                          replay_allowed=False, wrapper_invocation=LEGACY) and set(child["nodes"]) == old_scope
            and len(child["nodes"]) == len(old_scope) and outcome["batch"] == observed["batch"] == 4
            and outcome["calendar_SHA256"] == CALENDAR and outcome["wrapper_invocation"] == LEGACY
            and outcome["refill_attempted"] is True and outcome["refill_rc"] == 255
            and outcome["publication_prechecked_before_terminal"] is True
            and outcome["transport_observation"] == "REFILL_PROCESS_RETURNED"
            and outcome["terminal_receipt_SHA256"] == digest(raw["terminal"])
            and observed["rc"] == 0, "exact legacy group 4 scope differs")
    prefix = classifier["groups"]["4"]
    require(prefix["observed_branch"] == "ADMISSION_PRECEDES_BRIDGE"
            and prefix["stdout_SHA256"] == outcome["stdout_SHA256"] == digest(raw["stdout"])
            and re.fullmatch(rb"Bounded admission did not complete 255 [0-9a-f]{64}\r?\n", raw["stdout"]),
            "actual pre-bridge return proof differs")
    launcher = pin(REPO.parent / "run-dcu-standby-cycle-20261123.ps1", clearance["RUNNER_PIN"]).decode("utf-8-sig")
    embedded = launcher.replace("\r\n", "\n").split("\n@'\n", 1)[1].split("\n'@ | & ", 1)[0]
    probe_exit = ('if a.returncode:\n print("Bounded admission did not complete",a.returncode,'
                  'sha256(a.stderr).hexdigest())\n sys.exit(a.returncode)')
    send = "q=subprocess.run(inner+[command],input=transport_body,capture_output=True,timeout=120)"
    require(embedded.count(probe_exit) == embedded.count(send) == 1 and embedded.index(probe_exit) < embedded.index(send),
            "original runner branch order differs")
    actual = observed["observed"]["targets"]
    by_node = {row["node"]: row for row in actual}
    require(len(actual) == len(by_node) == 6 and set(by_node) == old_scope, "incomplete legacy terminal roster")
    preparation = claim["independent_preparation"]
    device = json.loads(evidence(preparation["device"]))
    cleared = json.loads(evidence(preparation["terminal"]))
    observation = device["result"]
    require(device["request"] == cleared["request"] == observation["request_id"] == PREP
            and device["rc"] == 0 and device["replay_allowed"] is False and observation["complete"] is True
            and observation["device_probe_scope"] == "PRIORITY_ONLY" and observation["total_cards"] == 640
            and set(observation["requested_priority_nodes"]) == set(FIVE)
            and cleared["original_device_receipt_SHA256"] == preparation["device"]["SHA256"]
            and cleared["no_new_GPU"] is True and cleared["exclude_vggt_node"] == VGGT
            and set(cleared["actual_terminal_steps"]) == {scope[1] for scope in FIVE.values()},
            "independent five-node preparation differs")
    pci_ledger = json.loads(evidence(original_claim["pci_pins"]))
    rows = {}
    for node, (old, zero) in FIVE.items():
        spec = claim["targets"][node]
        marker = json.loads(pin(JOURNAL / f"{node}-{old}-0.json", spec["original_slot_zero_SHA256"]))
        old_row = by_node[node]["accounting"]
        clean_row = cleared["actual_terminal_steps"][zero]
        hardware = observation["nodes"][node]
        pci = pci_ledger[node]
        require((spec["previous"], spec["original_zero_GRES"]) == (old, zero)
                and (marker["node"], marker["predecessor"], marker["batch"], marker["wrapper_invocation"],
                     marker["calendar_SHA256"], marker["terminal_receipt_SHA256"], marker["replay_allowed"])
                == (node, old, 4, LEGACY, CALENDAR, digest(raw["terminal"]), False)
                and by_node[node]["step"] == old and by_node[node]["terminal_ready"] is True
                and terminal(old_row, old, node, GPU_TRES) and terminal(clean_row, zero, node, HOST_TRES)
                and hardware["parent"] == PARENT and hardware["membership_valid"] is True
                and hardware["device_probe_observed"] is True and hardware["device_probe_step"] == zero
                and hardware["active_steps"] == hardware["occupied_ports"] == []
                and hardware["ports_checked"] == [43843, 43844] and hardware["probe_error"] is None
                and hardware["kfd_process_count"] == hardware["own_device_fd_process_count"] == 0
                and hardware["PCIs"] == pci == sorted(set(pci)) and len(pci) == 8, "node-specific original evidence differs")
        rows[node] = (old_row, clean_row, pci)
    return rows


def prepared(claim):
    require((claim["scope"], claim["state"], claim["parent"], claim["legacy_group"], claim["legacy_wrapper"],
             claim["legacy_calendar_SHA256"], claim["max_invocations_per_node"], claim["GPU_per_selected_node"],
             claim["GPU_if_all_admitted"], claim["replay_allowed"], claim["excluded_vggt_node"])
            == (STEM, "PREPARED_ONLY_NO_LIVE_DISPATCH", PARENT, 4, LEGACY, CALENDAR, 1, 8, 40, False, VGGT)
            and set(claim["targets"]) == set(FIVE), "five-node claim scope differs")
    requests = {spec["request"] for spec in claim["targets"].values()}
    require(len(requests) == len(FIVE) and all(re.fullmatch(r"[0-9a-f]{32}", value) for value in requests),
            "request identifiers must be unique for each node")
    pin(N02, N02_PIN)
    source = runpy.run_path(str(N02))
    pin(source["OLD_SCOPE"], source["OLD_SCOPE_PIN"])
    old = runpy.run_path(str(source["OLD_SCOPE"]))
    original_claim = json.loads(pin(source["OLD_CLAIM"], source["OLD_CLAIM_PIN"]))
    rows = originals(claim, original_claim, old)
    pin(source["ADOPT_TOOL"], source["ADOPT_TOOL_SHA"])
    pin(source["ADOPTION"], source["ADOPTION_SHA"])
    adopt = runpy.run_path(str(source["ADOPT_TOOL"]))
    api, roster, _ = adopt["checked_plan"]()
    groups, _ = api["calendar_scope"](api["legacy"](), adopt["CALENDAR"], adopt["CALENDAR_SHA"])
    require(not set(FIVE) & {row["node"] for group in groups for row in group} and roster["excluded_nodes"] == 22,
            "one recovery target collides with the fixed ordinary roster")
    pins = source["SOURCE_PINS"]
    standby = {name: pin(source["STANDBY"] / name, expected) for name, expected in pins.items()}
    [function] = [value for value in ast.parse(standby["controller.py"]).body
                  if isinstance(value, ast.FunctionDef) and value.name == "node_seconds"]
    budget = {"sha256": sha256}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<frozen standby budget>", "exec"), budget)
    return source, old, original_claim, rows, adopt, roster, standby, budget["node_seconds"]


def render(node, claim, configuration, publication):
    source, old, original_claim, rows, _, _, standby, node_seconds = configuration
    old_row, clean_row, pci = rows[node]
    request = claim["targets"][node]["request"]
    seconds = node_seconds(node, 10800, request)
    require(10500 <= seconds <= 10800, "worker budget outside unchanged standby limits")
    fields = dict(REQUEST=request, PARENT=PARENT, NODE=node, SCOPE=f"{STEM}-{node}-{request}", PUBLICATION=publication,
                  PREVIOUS=old_row[0], PREVIOUS_LABEL=old_row[1], PREVIOUS_END=old_row[8], CLEAR_STEP=clean_row[0],
                  CLEAR_LABEL=clean_row[1], CLEAR_END=clean_row[8], LABEL=f"training80-g4-fresh-{request[:12]}",
                  PORTS=original_claim["ports"], PCI=pci, OBSERVER_STEP=original_claim["observer"]["step"],
                  OBSERVER_LABEL=original_claim["observer"]["label"], ALLOW_OBSERVER_OVERLAP=False,
                  PROBE_PINS=original_claim["staged_probe_SHA256"], PINS=source["SOURCE_PINS"],
                  SOURCES={name: b64encode(raw).decode() for name, raw in standby.items()},
                  NODE_PROBE=source["NODE_PROBE"], WORKER_CGROUP=source["WORKER_CGROUP"], WORK_SECONDS=seconds)
    prefix = "\n".join(f"{key}={value!r}" for key, value in fields.items()) + "\n"
    frozen = old["HOST"].split("\ndef main():", 1)[0]
    launch = prefix + frozen + source["HOST_SCOPE"]
    select = '    [line] = query("scontrol", "show", "step", "-o", step)'
    exact = ('    [line] = [value for value in query("scontrol", "show", "step", "-o", step)\n'
             '              if fields(value).get("StepId") == step]')
    require(source["OBSERVE"].count(select) == 1, "original exact step selection boundary changed")
    observe = prefix + frozen + source["OBSERVE"].replace(select, exact, 1)
    for program in (launch, observe, source["NODE_PROBE"], source["WORKER_CGROUP"]):
        compile(program, f"<frozen guard {node}>", "exec")
    return dict(request=request, worker_seconds=seconds, launch=launch, observe=observe)


def verify_publication(revision, configuration):
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "exact ordinary publication required")
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    subprocess.run([*git, "merge-base", "--is-ancestor", revision, "refs/remotes/origin/bw1000-experiments"],
                   cwd=REPO, capture_output=True, timeout=10, check=True)
    source = configuration[0]
    paths = [Path(__file__).resolve(), CLAIM, N02, source["OLD_SCOPE"], source["OLD_CLAIM"], source["TRANSPORT"],
             source["ADOPT_TOOL"], *(source["STANDBY"] / name for name in source["SOURCE_PINS"])]
    for path in paths:
        blob = subprocess.check_output([*git, "cat-file", "blob", revision + ":" + path.relative_to(REPO).as_posix()],
                                       stderr=subprocess.PIPE, cwd=REPO, timeout=10)
        require(blob == pin(path), "published guard or dependency bytes differ")


def live_authority(configuration):
    source, _, _, _, adopt, roster, _, _ = configuration
    api = runpy.run_path(str(adopt["COORDINATOR"]))
    api["calendar58"].require_adoption(source["ADOPTION"], source["ADOPTION_SHA"], adopt["WAVE"], adopt["LEDGER"], roster, LOCAL)
    adopt["ledger_identity"](adopt["LEDGER"])
    adopt["local_census"]()
    with closing(sqlite3.connect(adopt["LEDGER"].as_uri() + "?mode=ro&immutable=1", uri=True, timeout=5)) as db:
        return {node for node, in db.execute("SELECT DISTINCT node FROM targets")}


def local_holds():
    sys.path.insert(0, str(TOOLS))
    import report_training80_backlog as report

    holds = report.read_queue_holds(LOCAL / "tuning/queue.sqlite3")
    jobs = json.loads((REPO / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    known = set(report.allocation(report.TRAINING80_LEDGER.read_bytes())[2])
    known.update(node for job in jobs for node in job["placement"])
    original, errors, _ = report.read_original_holds(REPO, jobs, known, holds)
    blocked = {node for item in (*holds, *original.values()) for node in item["nodes"]}
    blocked.update(node for job in jobs if job["state"] == "ready" or job["id"] in errors for node in job["placement"])
    return blocked


def paths(node, request):
    base = f"{STEM}-{node}-{request}"
    return tuple(LOCAL / f"{base}-{kind}.json" for kind in ("intent", "send", "reply"))


def dispatch(node, program, publication, claim_pin, configuration, ledger_claimed):
    source, old, *_ = configuration
    intent_path, token, reply = paths(node, program["request"])
    require(not any(map(os.path.lexists, (intent_path, token, reply))), "fixed local request requires reconciliation")
    clear_original_slot(node, FIVE[node][0])
    require(node not in local_holds() and node not in ledger_claimed, "existing project work hold on target")
    record = dict(node=node, request=program["request"], publication=publication, claim_SHA256=claim_pin,
                  program_SHA256=digest(program["launch"].encode()), GPU_requested=8, replay_allowed=False)
    old["save"](intent_path, record)
    helper = runpy.run_path(str(source["HELPER"]))
    sender = runpy.run_path(str(source["TRANSPORT"]))
    response = sender["send"](helper["ssh"](REPO.parent), request=program["request"], program=program["launch"],
                              data=b"", send_token=token, authority_seconds=18, outcome_seconds=82)
    old["save"](reply, dict(record, transport=response))
    return dict(node=node, request=program["request"], receipt=reply.name, SHA256=digest(pin(reply)),
                state="RETURNED_RECONCILE_EXACT_NUMBERED_STEP")


def observe(node, program, configuration):
    source, old, *_ = configuration
    intent, token, _ = paths(node, program["request"])
    require(json.loads(pin(intent))["request"] == program["request"] and token.is_file(), "no original GPU request to reconcile")
    ssh = runpy.run_path(str(source["HELPER"]))["ssh"](REPO.parent)
    command = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-"])
    try:
        response = subprocess.run([*ssh, command], input=program["observe"].encode(), capture_output=True, timeout=48)
        rc, out, err = response.returncode, response.stdout, response.stderr
    except subprocess.TimeoutExpired as error:
        rc, out, err = None, error.stdout or b"", error.stderr or b""
    except OSError as error:
        rc, out, err = None, b"", type(error).__name__.encode()
    try:
        body = json.loads(out) if rc == 0 and 0 < len(out) <= 16384 else None
    except (UnicodeError, ValueError):
        body = None
    require(body is None or (isinstance(body, dict) and body["request"] == program["request"]), "foreign observation")
    receipt = LOCAL / f"{STEM}-{node}-independent-{uuid4().hex}.json"
    old["save"](receipt, dict(node=node, request=program["request"], rc=rc, observed=body,
                              stdout_SHA256=digest(out), stderr_SHA256=digest(err), replay_allowed=False))
    return dict(node=node, request=program["request"], receipt=receipt.name, SHA256=digest(pin(receipt)),
                state=(body or {}).get("state", "ORIGINAL_UNKNOWN_RECONCILE"), step=(body or {}).get("step"),
                growth_cards=(body or {}).get("actual_DCU_progress"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    mode.add_argument("--observe-original", action="store_true")
    parser.add_argument("--node", action="append", choices=sorted(FIVE))
    args = parser.parse_args()
    selected = args.node or (sorted(FIVE) if args.check else [])
    require(selected and len(selected) == len(set(selected)), "choose each exact node once")
    raw = pin(CLAIM)
    claim = json.loads(raw)
    configuration = prepared(claim)
    source = configuration[0]
    pin(source["HELPER"], source["HELPER_SHA"])
    pin(source["TRANSPORT"], source["TRANSPORT_SHA"])
    programs = {node: render(node, claim, configuration, args.publication) for node in selected}
    if args.check:
        print(json.dumps(dict(mode="LOCAL_PREPARATION_ONLY", claim_SHA256=digest(raw), Slurm=0, GPU=0, SCNet=0, Git=0,
                              nodes={node: dict(request=value["request"], worker_seconds=value["worker_seconds"],
                                                launch_SHA256=digest(value["launch"].encode()),
                                                observe_SHA256=digest(value["observe"].encode()))
                                     for node, value in programs.items()}), sort_keys=True))
        return
    if args.publication:
        verify_publication(args.publication, configuration)
        ledger_claimed = live_authority(configuration)
    failed = False
    for node, program in programs.items():
        try:
            result = (observe(node, program, configuration) if args.observe_original else
                      dispatch(node, program, args.publication, digest(raw), configuration, ledger_claimed))
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as error:
            failed = True
            state = "REJECTED_OR_ORIGINAL_RECONCILIATION_REQUIRED"
            result = dict(node=node, request=program["request"], state=state,
                          error_class=type(error).__name__, error_SHA256=digest(str(error).encode()), replay_allowed=False)
            config_old = configuration[1]
            failure = LOCAL / f"{STEM}-{node}-failure-{uuid4().hex}.json"
            config_old["save"](failure, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(json.dumps(dict(local_error=type(error).__name__, detail_SHA256=digest(str(error).encode()))))
        raise SystemExit(1)
