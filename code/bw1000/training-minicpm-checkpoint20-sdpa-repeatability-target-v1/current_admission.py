"""Check the current T05 allocation with the published replacement ledger."""

from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


PARENT = "1103606"
NODE = "f13r1n19"
LEDGER_SHA = "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"
READER_SHA = "8d071a48cad3a36ae0f917a2b5c7714abecc749cca55c8a38d1942fb2cc679be"
HERE = Path(__file__).resolve().parent


def exact(path, pin):
    if path.resolve(strict=True) != path:
        raise ValueError("aliased current allocation source")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin:
        raise ValueError("current allocation source pin differs")
    return raw


def plan(ledger):
    if (ledger["state"] != "LIVE_REPLACEMENT_DCU640" or ledger["allocated_nodes"] != 80
            or ledger["allocated_DCU"] != 640 or {x["JobId"] for x in ledger["jobs"]} != {PARENT, "1194900"}):
        raise ValueError("wrong current replacement ledger identity")
    groups = {name: group["allocations"][PARENT] for name, group in ledger["groups"].items()}
    nodes = [node for group in groups.values() for node in group]
    if len(nodes) != 40 or len(set(nodes)) != 40 or NODE not in groups["T05"]:
        raise ValueError("exact T05 owner node required")
    return {PARENT: groups}


def check(host_uid, run_query=None):
    ledger = json.loads(exact(HERE / "current_ledger.json", LEDGER_SHA))
    source = HERE / "current_membership.py"
    module = ModuleType("mini_cpm_current_membership")
    module.__file__ = str(source)
    exec(compile(exact(source, READER_SHA), str(source), "exec"), vars(module))
    proof = module.snapshot(plan(ledger), owner_uid=host_uid, run_query=run_query)["parents"][PARENT]
    node = proof["coverage_by_group_and_node"]["T05"][NODE]
    if (proof["state"] != "RUNNING"
            or not all(proof[field] is True for field in ("ledger_exact", "owned_by_observer", "step_locations_exact"))
            or proof["nodes"] != 40 or proof["allocated_DCU"] != 320
            or node["state"] != "NO_NUMBERED_STEP" or node["numbered_step_assigned_DCU_bounds"] != [0, 0]
            or any(NODE in row["member_nodes"] for row in proof["active_steps"].values())):
        raise ValueError("T05 same-node fresh Slurm admission unavailable")
    return proof
