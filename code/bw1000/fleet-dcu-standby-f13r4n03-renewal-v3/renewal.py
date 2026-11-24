"""Renew the f13r4n03 standby with the original shared-node safeguards."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time


RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PARENT = "1194900"
NODE = "f13r4n03"
PREVIOUS_STEP = "1194900.539"
PREVIOUS_REQUEST = "93e50c237efc4f7684e3f028ac1afa0e"
EXPECTED_PREVIOUS_DEADLINE = 1795486659.9822166
PREVIOUS_LABEL = f"training80-dcu-standby-{PREVIOUS_REQUEST[:12]}-{NODE}"
EIGHT_DCU = "cpu=8,gres/dcu=8,mem=16G,node=1"
EXPECTED_SHARED_CLAIM = "caf0d443dcc985dfffe83a31d5476f53f7ffdfcc1890b46ee7b28f799bbe9ed8"
EXPECTED_SHARED_PROGRAM = "ae52680a5916e28765f82fff72ac8cdd6374f667bf0849dcc00108854c6d34b0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def verify_previous(rows, document, claim):
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
    terminal, terminal_sha = document(worker_path / "terminal.json")
    require((expected["request"], expected["parent"], expected["node"], expected["label"], expected["seconds"])
            == (PREVIOUS_REQUEST, PARENT, NODE, PREVIOUS_LABEL, 10500) and expected["pci"] == claim["pci"],
            "previous standby device binding changed")
    require((expected_sha, terminal_sha) == (previous["expected_SHA256"], previous["terminal_SHA256"]),
            "preceding worker bytes differ from terminal evidence")
    require((terminal["node"], terminal["step"], terminal["state"]) == (NODE, PREVIOUS_STEP, "WORKER_RETURNED"),
            "previous worker did not return normally")
    return dict(accounting=row, worker_terminal_SHA256=terminal_sha, worker_expected_SHA256=expected_sha)


def run(claim, shared_program, shared_claim, sources):
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
    require(re.fullmatch(r"[a-f0-9]{32}", request) and request not in ({PREVIOUS_REQUEST} | {v["request"] for v in old["stages"].values()}),
            "invalid or previously used renewal identity")
    os.umask(0o077)
    shared = {"__name__": "published_f13r4n03_shared_standby"}
    exec(compile(shared_program, "published-shared-standby.py", "exec"), shared)
    helper, controller = shared["bind"](old, sources)
    previous = verify_previous(helper["rows"], shared["document"], claim)
    renewed = dict(old, stages=dict(old["stages"], standby=dict(old["stages"]["standby"], request=request)))
    print(json.dumps(dict(co_resident_renewal_gate=True, request=request, previous=previous, epoch=time.time()),
                     sort_keys=True), flush=True)
    shared["launch"](renewed, "standby", sources, helper, controller)
