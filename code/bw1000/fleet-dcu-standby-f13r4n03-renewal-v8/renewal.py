"""Renew the f13r4n03 standby with the original shared-node safeguards."""

from hashlib import sha256
import json
import os
import re
from pathlib import Path
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1194900"
NODE = "f13r4n03"
PREVIOUS_STEP = "1194900.950"
PREVIOUS_REQUEST = "26e03fd2776344e4a1f81c6954a44b86"
EXPECTED_PREVIOUS_START = 1795531402.4709394
EXPECTED_PREVIOUS_DEADLINE = 1795541902.4709394
SUCCESSOR = "f82bc5efb0da4a9d82a25ea13481c26f"
PREVIOUS_LABEL = f"training80-dcu-standby-{PREVIOUS_REQUEST[:12]}-{NODE}"
EIGHT_DCU = "cpu=8,gres/dcu=8,mem=16G,node=1"
EXPECTED_SHARED_CLAIM = "caf0d443dcc985dfffe83a31d5476f53f7ffdfcc1890b46ee7b28f799bbe9ed8"
EXPECTED_SHARED_PROGRAM = "ae52680a5916e28765f82fff72ac8cdd6374f667bf0849dcc00108854c6d34b0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def verify_previous(rows, document, claim, prior_terminal_pin):
    previous = claim["previous_standby"]
    require((previous["request"], previous["step"], previous["label"], previous["worker_expected_end_epoch"])
            == (PREVIOUS_REQUEST, PREVIOUS_STEP, PREVIOUS_LABEL, EXPECTED_PREVIOUS_DEADLINE), "previous standby identity changed")
    records = rows(PREVIOUS_STEP)
    require(len(records) == 1, "previous standby has no unique accounting record")
    [row] = records
    require(row[:4] == [PREVIOUS_STEP, PREVIOUS_LABEL, "COMPLETED", "0:0"]
            and row[5:] == [NODE, EIGHT_DCU], "previous standby has not completed normally")
    worker_path = RESULTS / f"training80-dcu-standby-{PREVIOUS_REQUEST}" / NODE
    expected, expected_sha = document(worker_path / "expected.json")
    worker, worker_sha = document(worker_path / "worker.json")
    terminal, terminal_sha = document(worker_path / "terminal.json")
    require((expected["request"], expected["parent"], expected["node"], expected["label"], expected["seconds"])
            == (PREVIOUS_REQUEST, PARENT, NODE, PREVIOUS_LABEL, 10500) and expected["pci"] == claim["pci"],
            "previous standby device binding changed")
    require(expected_sha == previous["expected_SHA256"] and worker_sha == previous["worker_SHA256"]
            and worker["step"] == PREVIOUS_STEP and worker["node"] == NODE and worker["pci"] == claim["pci"]
            and worker["epoch"] == previous["worker_start_epoch"] == EXPECTED_PREVIOUS_START
            and worker["synthetic"] is True and worker["worker_pid"] == previous["worker_PID"],
            "preceding worker identity or previously pinned bytes differ")
    require((terminal["node"], terminal["step"], terminal["state"]) == (NODE, PREVIOUS_STEP, "WORKER_RETURNED"),
            "previous worker did not return normally")
    require(terminal_sha == prior_terminal_pin and type(terminal["epoch"]) in (int, float)
            and EXPECTED_PREVIOUS_DEADLINE - 1 <= terminal["epoch"] <= time.time() + 2,
            "worker terminal does not follow measured natural budget")
    return dict(accounting=row, worker_terminal_SHA256=terminal_sha, worker_expected_SHA256=expected_sha,
                worker_SHA256=worker_sha, worker_terminal_epoch=terminal["epoch"])


def combined_guard(primary, query, claim, controller, *, own=None):
    primary(claim, controller, own=own)
    known = set(claim["zero_GRES_controls"]) | set(claim["protected_zero_GRES_accounting_conflicts"])
    if own:
        known.add(own[0])
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"):
        step, specs = line.split("|")
        if step in {PARENT + ".batch", PARENT + ".extern"}:
            continue
        require(re.fullmatch(PARENT + r"\.[0-9]+", step) and specs not in {"", "(null)", "None", "N/A", "Unknown"},
                "unresolved SQUEUE step membership")
        members = ([specs] if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", specs)
                   else query("scontrol", "show", "hostnames", specs))
        require(members and len(members) == len(set(members)) and all(
            re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", member) for member in members),
            "unresolved SQUEUE host list")
        if NODE in members:
            require(members == [NODE] and step in known, "new or conflicting SQUEUE occupancy")


def run(claim, shared_program, shared_claim, sources, prior_terminal_pin):
    require(re.fullmatch(r"[0-9a-f]{64}", prior_terminal_pin)
            and claim["previous_standby"]["terminal_sha_observed_by_launcher_after_natural_completion"] is True,
            "actual predecessor terminal bytes must be bound at execution")
    require((claim["parent"], claim["node"], claim["GPU_cards"], claim["CPU_cores"], claim["memory_MiB"],
             claim["worker_seconds"], claim["slurm_seconds"])
            == (PARENT, NODE, 8, 8, 16384, 10500, 10800), "wrong node, hardware, or budget")
    require(claim["synthetic"] is True and claim["preemptible"] is True and claim["replay_allowed"] is False,
            "standby policy changed")
    require(digest(shared_program) == claim["shared_source_SHA256"] == EXPECTED_SHARED_PROGRAM
            and digest(shared_claim) == claim["shared_claim_SHA256"] == EXPECTED_SHARED_CLAIM,
            "qualified shared node source changed")
    old = json.loads(shared_claim)
    require(claim["pci"] == old["pci"] and claim["zero_GRES_controls"] == old["zero_GRES_controls"] == {}
            and claim["protected_zero_GRES_accounting_conflicts"] == old["protected_zero_GRES_accounting_conflicts"],
            "device pins or protected zero-GRES controls changed")
    request = claim["request"]
    require(request == SUCCESSOR and request not in ({PREVIOUS_REQUEST} | {v["request"] for v in old["stages"].values()}),
            "invalid or previously used renewal identity")
    os.umask(0o077)
    shared = {"__name__": "published_f13r4n03_shared_standby"}
    exec(compile(shared_program, "published-shared-standby.py", "exec"), shared)
    helper, controller = shared["bind"](old, sources)
    primary_guard = helper["guard"]

    def check_current(scope, namespace, *, own=None):
        combined_guard(primary_guard, helper["query"], scope, namespace, own=own)

    helper["guard"] = check_current
    previous = verify_previous(helper["rows"], shared["document"], claim, prior_terminal_pin)
    renewed = dict(old, stages=dict(old["stages"], standby=dict(old["stages"]["standby"], request=request)))
    print(json.dumps(dict(co_resident_renewal_gate=True, request=request, previous=previous, epoch=time.time()),
                     sort_keys=True), flush=True)
    shared["launch"](renewed, "standby", sources, helper, controller)
