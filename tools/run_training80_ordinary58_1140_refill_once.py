"""Continue the verified ordinary 58 on the existing vNext coordinator and ledger."""

import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sqlite3
import subprocess
import sys
import time
from types import FunctionType
from uuid import NAMESPACE_URL, uuid5


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
WAVE = "training80-vnext-20261125-1140-ordinary58"
CLAIM = REPO / "coordination/tasks" / f"{WAVE}.json"
SUMMARY = LOCAL / "training80-0841-vnext-58-actual-terminal-and-two-point-verified-v1.json"
SUMMARY_PIN = "e6cf9395afee2fada97ba163766491bbaa28f27cc9e4cdc5e843dc8001eb395e"
PRIOR = REPO / "tools/run_training80_ordinary58_adoption_once.py"
PRIOR_PIN = "bd377144cf837ef34cd566b685abefd350a7c491e50cb67efe61c0e64e57ca65"
PRIOR_ADOPTION_PIN = "d9ea8816084974ba439f5fca0e96fc40daf159d9b0d892bb2c3305a80d283573"
AUTHORITY = LOCAL / f"{WAVE}-sole-adoption.json"
PARENTS = {"1103606": 24, "1194900": 34}
N02 = "f13r1n02"
CARD = re.compile(r"[a-f0-9]{4}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]")


HOST_VIEWS = r'''
import os,re
parent, = {item["parent"] for item in items}
targets = {item["node"] for item in items}
def query(*args):
    reply = subprocess.check_output(args,text=True,timeout=8)
    assert len(reply.encode()) <= 1048576
    return reply.splitlines()
def fields(line):
    return dict(token.split("=",1) for token in line.split() if "=" in token)
cache={}
node_pattern=r"f[0-9]{2}r[0-9]+n[0-9]{2}"
def expand(expression):
    if expression not in cache:
        values=[expression] if re.fullmatch(node_pattern,expression) else query("scontrol","show","hostnames",expression)
        assert values and len(set(values)) == len(values) and all(re.fullmatch(node_pattern,n) for n in values)
        cache[expression]=set(values)
    return cache[expression]
line, = query("scontrol","show","job","-o",parent)
owner = fields(line)
nodes = expand(owner["NodeList"])
assert owner["JobId"]==parent and owner["JobState"]=="RUNNING" and owner["UserId"].endswith(f"({os.getuid()})")
assert owner["NumNodes"]=="40" and len(nodes)==40 and "gres/dcu=320" in owner["AllocTRES"].split(",") and targets<=nodes
views = {node:{"scontrol":[],"squeue":[]} for node in targets}
def remember(view, step, expression):
    if step in (parent+".extern",parent+".batch"):
        return
    assert re.fullmatch(parent+r"\.[0-9]+",step)
    members=expand(expression)
    assert members<=nodes
    for node in members & targets:
        assert step not in views[node][view]
        views[node][view].append(step)
for line in query("scontrol","show","step","-o",parent):
    values=fields(line)
    remember("scontrol",values["StepId"],values["NodeList"])
for line in query("squeue","--steps","-h","-j",parent,"-o","%i|%N"):
    step,placement=line.split("|")
    remember("squeue",step,placement)
'''


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, sha=None, limit=262144):
    require(path.is_file() and path.resolve(strict=True) == path, "noncanonical file pin")
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    require(0 < len(raw) <= limit and (sha is None or digest(raw) == sha), "expected source or receipt bytes changed")
    return raw


def replace(text, anchor, replacement):
    require(text.count(anchor) == 1, "frozen ordinary observer insertion point changed")
    return text.replace(anchor, replacement, 1)


def strengthened_reader(legacy):
    raw = legacy["remote_reader"]()
    raw = replace(raw, "def read(path):", HOST_VIEWS + "\ndef read(path):")
    raw = replace(raw, "    with path.open(\"rb\") as handle:raw=handle.read(8193)",
                  "    assert path.resolve(strict=True)==path\n    with path.open(\"rb\") as handle:raw=handle.read(8193)")
    raw = replace(raw, '    ready=bool(fresh and fresh[2:4]==["COMPLETED","0:0"] and fresh[5]==x["node"]',
                  '''    placement=views[x["node"]]
    label=f"training80-dcu-standby-{x['request'][:12]}-{x['node']}"
    ready=bool(not placement["scontrol"] and not placement["squeue"] and fresh
               and fresh[1]==label and fresh[2:4]==["COMPLETED","0:0"] and fresh[5]==x["node"]
               and fresh[4].isdecimal() and int(fresh[4])>=x["worker_seconds"]
               and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}",fresh[8])''')
    raw = replace(raw, '               and expected["request"]==x["request"] and expected["seconds"]==x["worker_seconds"]',
                  '''               and expected["request"]==x["request"] and expected["seconds"]==x["worker_seconds"]
               and expected["node"]==x["node"] and expected["parent"]==x["parent"] and expected["label"]==label
               and sorted(expected["pci"])==x["pci"]''')
    raw = replace(raw, 'and worker["step"]==step\n', '''and worker["step"]==step
               and worker["node"]==x["node"] and worker["synthetic"] is True
               and worker["epoch"]==x["worker_start_epoch"] and sorted(worker["pci"])==x["pci"]
''')
    raw = replace(raw, "                worker_SHA256=wpin,expected_SHA256=epin,terminal_SHA256=tpin)",
                  "                worker_SHA256=wpin,expected_SHA256=epin,terminal_SHA256=tpin,\n"
                  "                scontrol_target_steps=sorted(placement['scontrol']),squeue_target_steps=sorted(placement['squeue']))")
    compile(raw, "<ordinary exact two-view return reader>", "exec")
    return raw


def rows_from_proof(summary, old_rows):
    proofs = {}
    for source in summary["sources"]:
        name = source["file"]
        require(re.fullmatch(r"training80-0841-vnext-independent-ack-progress-[a-f0-9]{32}\.json", name), "old original proof path differs")
        raw = json.loads(pin(LOCAL / name, source["SHA256"], 425984))
        require(raw["rc"] == 0 and raw["response"]["GPU_steps_created"] == 0, "old progress receipt is not read-only evidence")
        proofs[source["SHA256"]] = raw["response"]["nodes"]
    require(len(proofs) == 7, "expected seven earlier proof receipts")
    targets = []
    for row in summary["per_node"]:
        node, parent, step = (row[key] for key in ("node", "parent", "new_step"))
        old_group, old = old_rows[node]
        point = proofs[row["independent_proof_SHA256"]][node]
        start, seconds, deadline = (row[key] for key in ("worker_start_epoch", "worker_seconds", "next_budget_end_epoch"))
        require(row["group"] == old_group and (row["old_step"], parent) == (old["previous"], old["parent"])
                and re.fullmatch(parent + r"\.[0-9]+", step) and row["old_step"] != step
                and (point["parent"], point["previous"], point["request"]) == (parent, step, row["bridge"])
                and point["status"] == "RUNNING_WITH_ORIGINAL_BUDGET" and point["pair_state"] == "PASS"
                and point["worker_start_epoch"] == start and point["worker_seconds"] == seconds
                and point["planned_expiry_epoch"] == deadline == start + seconds
                and row["next_preparation_epoch"] == point["prepare_epoch"] == deadline - 900
                and point["worker_PID"] == row["worker_pid"] > 0 and type(seconds) is int and 10500 <= seconds <= 10800
                and point["worker_SHA256"] == row["worker_SHA256"] and point["expected_SHA256"] == row["expected_SHA256"]
                and point["growth_DCU"] == row["matched_cards"] == 8, "latest predecessor identity differs")
        first, second, deltas = (point[key] for key in ("first", "second", "count_deltas"))
        require(len(first) == len(second) == len(deltas) == 8 and first.keys() == second.keys() == deltas.keys()
                and all(CARD.fullmatch(pci) and start <= a["epoch"] < second[pci]["epoch"]
                        and second[pci]["calls"] - a["calls"] == deltas[pci] > 0 for pci, a in first.items()),
                "latest same-PCI progress evidence differs")
        targets.append(dict(node=node, parent=parent, previous=step, request=row["bridge"], pci=sorted(first),
                            previous_worker_pid=row["worker_pid"], worker_SHA256=row["worker_SHA256"],
                            expected_SHA256=row["expected_SHA256"], worker_start_epoch=start, worker_seconds=seconds,
                            planned_end_epoch=deadline, prepare_epoch=deadline - 900))
    require(len(targets) == len({row["previous"] for row in targets}) == 58
            and {row["node"] for row in targets} == set(old_rows) and N02 not in old_rows
            and Counter(item["parent"] for item in targets) == PARENTS, "58 ordinary nodes or original fences changed")
    return targets


def plan():
    pin(PRIOR, PRIOR_PIN)
    prior = runpy.run_path(str(PRIOR))
    old, old_scope, old_plan = prior["checked_plan"]()
    legacy = old["legacy"]()
    old_groups, _ = old["calendar_scope"](legacy, prior["CALENDAR"], prior["CALENDAR_SHA"])
    old_rows = old["calendar58"].identities(old_groups)
    require(N02 not in old_rows and not set(old_rows) & legacy["FENCES"] and old_scope["excluded_nodes"] == 22,
            "old allocation and specialist fences changed")
    summary = json.loads(pin(SUMMARY, SUMMARY_PIN))
    raw = pin(CLAIM)
    claim = json.loads(raw)
    groups = old["calendar58"].batches(rows_from_proof(summary, old_rows))
    ids = [[dict(node=r["node"], parent=r["parent"], previous=r["previous"], worker_budget_epoch=r["planned_end_epoch"])
            for r in group] for group in groups]
    roster = digest(old["encode"](ids))
    requests = [uuid5(NAMESPACE_URL, f"{WAVE}:{roster}:{n}").hex for n in range(1, len(groups) + 1)]
    require((claim["scope"], claim["state"], claim["calendar_SHA256"], claim["new_group_roster_SHA256"], claim["replay_allowed"])
            == (WAVE, "PREPARED_NO_TERMINAL_OBSERVATION_OR_DISPATCH", SUMMARY_PIN, roster, False)
            and len(groups) == 16 and len(set(requests)) == 16 and claim["request_ids"] == requests
            and claim["first_preparation_Beijing"] == summary["next_preparation_CST"]
            and claim["worker_budget_Beijing"] == summary["next_budget_range_CST"]
            and not set(requests) & {row["request"] for row in old_plan["groups"]}, "frozen new 16-group calendar differs")
    reader = strengthened_reader(legacy)
    observe = legacy["observe"]
    legacy["observe"] = FunctionType(observe.__code__, dict(observe.__globals__, WAVE=WAVE, remote_reader=lambda: reader))
    return prior, old, old_scope, summary, claim, digest(raw), legacy, groups


def custody(material):
    prior, core, old_scope, summary, _, _, _, _ = material
    prior_path, ledger = prior["ADOPTION"], prior["LEDGER"]
    core["calendar58"].require_adoption(prior_path, PRIOR_ADOPTION_PIN, prior["WAVE"], ledger, old_scope, LOCAL)
    actual = prior["ledger_identity"](ledger)
    require(actual["wave_claims"] == 16, "prior 16 claims need reconciliation")
    with closing(sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True, timeout=5)) as database:
        database.execute("PRAGMA query_only=ON")
        database.execute("BEGIN")
        known_targets = {item["group"]: set() for item in summary["attempts"]}
        for row in summary["per_node"]:
            known_targets[row["group"]].add((row["parent"], row["node"], row["old_step"]))
        for item in summary["attempts"]:
            request = item["ledger_request"]
            expected = (prior["WAVE"], item["group"], prior["CALENDAR_SHA"], item["terminal_SHA256"], item["cards"] // 8)
            read = database.execute("SELECT wave,group_id,calendar_sha,observation_sha,roster_count FROM claims WHERE request=?",
                                    (request,)).fetchone()
            replay = database.execute("SELECT state,detail FROM events WHERE request=? ORDER BY sequence", (request,)).fetchall()
            owned = set(database.execute("SELECT parent,node,predecessor FROM targets WHERE request=?", (request,)))
            require(read == expected and [state for state, _ in replay] == ["CLAIMED", "BRIDGE_INTENT", "BRIDGE_ACKNOWLEDGED"]
                    and owned == known_targets[item["group"]]
                    and all(json.loads(raw)["bridge_request"] == item["bridge"] for _, raw in replay[1:]),
                    "prior exact acknowledged ledger step differs")
        prospective, = database.execute("SELECT count(*) FROM claims WHERE wave=?", (WAVE,)).fetchone()
    return ledger, actual, prospective


def authority_fields(material, ledger, identity):
    return dict(schema="training80.ordinary.1140-same-ledger-adoption-v1", wave=WAVE, operator="/root",
                ledger_absolute=ledger.as_posix(), ledger_device=identity["device"], ledger_inode=identity["inode"],
                prior_adoption_SHA256=PRIOR_ADOPTION_PIN, prior_actual_summary_SHA256=SUMMARY_PIN,
                claim_SHA256=material[5], group_roster_SHA256=material[4]["new_group_roster_SHA256"],
                prior_owner_released=True, sole_current_coordinator_authorized=True, replay_allowed=False)


def publication(revision, material, publisher):
    require(re.fullmatch(r"[a-f0-9]{40}", revision), "exact published source commit required")
    git = publisher["git"]
    git("merge-base", "--is-ancestor", revision, "refs/remotes/origin/bw1000-experiments")
    paths = (Path(__file__).resolve(), CLAIM, PRIOR, material[0]["COORDINATOR"], material[0]["MODULE"] / "calendar58.py")
    for path in paths:
        require(git("cat-file", "blob", revision + ":" + path.relative_to(REPO).as_posix()) == pin(path),
                "published successor code differs")


def check_authority(material, ledger, identity, pin_value):
    require(pin_value and re.fullmatch(r"[a-f0-9]{64}", pin_value), "exact root adoption receipt required")
    document = json.loads(pin(AUTHORITY, pin_value, 8192))
    required = authority_fields(material, ledger, identity)
    require(set(document) == set(required) | {"publication", "issued_epoch"} and all(document[key] == value for key, value in required.items())
            and 0 < document["issued_epoch"] <= time.time(), "exact adoption is absent or points to a different ledger")
    return document


def group_status(core, ledger, request):
    # No write access for operator status queries.
    base = core["Ledger"](ledger)
    base.connect = lambda: sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True, timeout=5)
    return base.status(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "check-ledger", "adopt", "observe", "dispatch", "status"))
    parser.add_argument("--publication")
    parser.add_argument("--authority-sha")
    parser.add_argument("--group", type=int)
    args = parser.parse_args()
    material = plan()
    prior, core, _, _, claim, claim_pin, legacy, groups = material
    stamp = core["calendar58"].stamp
    if args.mode == "check":
        _, publisher = legacy["modules"]()
        core["frozen_action"](publisher)
        result = dict(state="PLANNING_ONLY_NO_TERMINAL_OBSERVATION", claim_SHA256=claim_pin,
                      groups=[dict(group=i, request=claim["request_ids"][i-1], parent=g[0]["parent"], nodes=len(g),
                                   preparation=stamp(g[0]["prepare_epoch"]), first=stamp(g[0]["planned_end_epoch"]),
                                   last=stamp(g[-1]["planned_end_epoch"])) for i,g in enumerate(groups,1)],
                      first_group=[(r["node"], r["previous"]) for r in groups[0]], GPU=0, Slurm=0, SCNet=0, Git=0)
    else:
        ledger, identity, prospective = custody(material)
        if args.mode == "check-ledger":
            result = dict(state="EXISTING_LEDGER_ONLY", prior_acks=identity["wave_claims"], same_ledger=ledger == prior["LEDGER"],
                          new_wave_claims=prospective, new_adoption=AUTHORITY.exists(), GPU=0, Slurm=0, SCNet=0, Git=0)
        elif args.mode == "adopt":
            require(args.publication and not any((prospective, os.path.lexists(AUTHORITY))), "existing new scope requires reconciliation")
            _, publisher = legacy["modules"]()
            publication(args.publication, material, publisher)
            prior["local_census"]()
            receipt = dict(authority_fields(material, ledger, identity), publication=args.publication, issued_epoch=time.time())
            pin_value = prior["exclusive_save"](AUTHORITY, receipt)
            result = dict(state="SAME_EXISTING_LEDGER_ADOPTED", authority_SHA256=pin_value, ledger_initialized=False, GPU=0, Slurm=0, SCNet=0)
        else:
            require(type(args.group) is int and 1 <= args.group <= len(groups), "exact group required")
            rows, request = groups[args.group-1], claim["request_ids"][args.group-1]
            if args.mode == "status":
                result = group_status(core, ledger, request)
            else:
                adopted = check_authority(material, ledger, identity, args.authority_sha)
                base, publisher = legacy["modules"]()
                if args.mode == "observe":
                    require(time.time() >= rows[0]["prepare_epoch"], "preparation window has not begun")
                    path, doc, released = legacy["observe"](args.group, rows, base)
                    result = dict(state="PREPARATION_OBSERVATION_ONLY", receipt=path.name, SHA256=digest(pin(path)), request=request,
                                  rc=doc["rc"], observed=[(x["node"], x["accounting"][2] if x["accounting"] else None,
                                       x["scontrol_target_steps"], x["squeue_target_steps"]) for x in (doc["observed"] or {}).get("targets", [])],
                                  actual_released=[r["node"] for r in released], GPU=0, new_Slurm_steps=0)
                else:
                    action = core["frozen_action"](publisher)
                    def certify():
                        state = publisher["precheck"]()
                        publication(state["head"], material, publisher)
                        publisher["git"]("merge-base", "--is-ancestor", adopted["publication"], state["head"])
                        return state
                    result = core["dispatch_group"](core["Ledger"](ledger), args.group, rows, request, WAVE, SUMMARY_PIN, legacy,
                        base, dict(publisher, precheck=certify), action, require_all=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(json.dumps(dict(state="REFUSED_OR_RECONCILE_ORIGINAL", error_class=type(error).__name__,
                              detail_SHA256=digest(str(error).encode()), replay_allowed=False)))
        raise SystemExit(1)
