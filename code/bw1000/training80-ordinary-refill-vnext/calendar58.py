"""Verify and freeze the 58 proved successors for the next ordinary refill wave."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re


VERIFIED_FILE = "training80-0544-ordinary-58-node-verified-standby-v1.json"
VERIFIED_SHA = "56da79542a9bb1181385e66f68ac9523f32c323b3cd31cb73a530283a329c83f"
WAVE = "training80-vnext-20261125-0841-ordinary58"
MANIFEST_FILE = "training80-0544-ordinary-58-verified-source-manifest-v1.json"
MANIFEST_SHA = "7eaf3eb6245278d776fbad5e2c51af5c272ece6a755588268c95879817d50e54"
HISTORICAL_SHA = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
FLEET_SHA = "7db18fd3bceca5175e1d9795c696069182ac2b742f849b0e2e82cda8bd3fe8a4"
UNKNOWN_SHA = "20c0add6f35c536ac334a761d39dd3f3f3f3848f30d3f10641ac02ff0653b819"
PARENTS = {"1103606": 24, "1194900": 34}
FIRST, LAST = "2026-11-25T08:56:03.651+08:00", "2026-11-25T09:24:39.022+08:00"
PREP = "2026-11-25T08:41:03.651+08:00"
MISSING = {
    "admission_process_prebridge_exit_rc255_original_journal_preserved": ("PRE_BRIDGE", 7),
    "bridge_transport_unknown_original_journal_preserved": ("BRIDGE_UNKNOWN", 3),
    "not_sent_terminal_outside_first_envelope_and_whole_group_repeat_rejected": ("APPROVAL_EXCLUDED", 2),
}
CST = timezone(timedelta(hours=8))
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(raw):
    return sha256(raw).hexdigest()


def encoded(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path, expected, maximum=450000):
    need(path.is_file() and path.resolve(strict=True) == path.absolute(), "EXACT_PRIVATE_CALENDAR_PATH")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    need(0 < len(raw) <= maximum and pin(raw) == expected, "EXACT_PRIVATE_CALENDAR_BYTES")
    return json.loads(raw)


def named(local, name):
    need(Path(name).name == name and "/" not in name and "\\" not in name and name.endswith(".json"), "EXACT_PROOF_FILENAME")
    return local / name


def stamp(epoch):
    return datetime.fromtimestamp(epoch, CST).isoformat(timespec="milliseconds")


def is_pin(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


def identities(groups):
    return {row["node"]: (group, row) for group, current in enumerate(groups, 1) for row in current}


def classify(verified, manifest, historical, known, fences):
    classifications = {}
    def declare(nodes, label, count):
        need(len(nodes) == len(set(nodes)) == count and not set(nodes) & classifications.keys(),
             "EXACT_DISJOINT_FLEET_CLASSES")
        classifications.update(dict.fromkeys(nodes, label))

    declare([row["node"] for row in verified["per_node"]], "VERIFIED_ORDINARY_SUCCESSOR", 58)
    missing = manifest["missing_classification"]
    need(missing.keys() == MISSING.keys(), "FROZEN_MISSING_CLASS_SCHEMA")
    for key, (label, count) in MISSING.items():
        declare(missing[key], label, count)
    need(set(classifications) == set(known) and len(known) == 70
         and {node for node, label in classifications.items() if label != "VERIFIED_ORDINARY_SUCCESSOR"}
         == set(verified["current_unadmitted_nodes"]), "HISTORICAL_70_ORDINARY_DENOMINATOR")
    declare(historical["existing_held_fences"], "EXISTING_PROTECTION", 5)
    declare(historical["special_scopes_excluded"], "SINGLE_NODE_SCOPE", 5)
    need(set(verified["held_fences"]) == fences == set(classifications) - set(known)
         and len(classifications) == 80, "EXACT_80_NODE_FLEET_DENOMINATOR")
    return classifications


def progress(verified, manifest, local):
    scheduled = {row["file"]: row["sha"] for row in manifest["proofs"]}
    need(len(scheduled) == len(manifest["proofs"]) == len(verified["proofs"]) == 7
         and scheduled == {row["file"]: row["SHA256"] for row in verified["proofs"]}, "SEVEN_PINNED_PROGRESS_SOURCES")
    by_pin = {}
    for row in verified["proofs"]:
        raw = read(named(local, row["file"]), row["SHA256"])
        response = raw["response"]
        need(raw["rc"] == 0 and response["GPU_steps_created"] == 0
             and stamp(response["observed_epoch"]) == row["observed_CST"], "EXACT_NON_MUTATING_PROGRESS_SOURCE")
        by_pin[row["SHA256"]] = response
    return by_pin


def verify_points(point, observed, first_epoch):
    first, second, deltas = (point[key] for key in ("first", "second", "count_deltas"))
    need(point["pair_state"] == "PASS" and point["growth_DCU"] == len(first) == 8
         and first.keys() == second.keys() == deltas.keys(), "EXACT_SAME_EIGHT_PCIS")
    for pci, old in first.items():
        new = second[pci]
        need(PCI.fullmatch(pci) and is_pin(old["SHA256"]) and is_pin(new["SHA256"])
             and type(old["calls"]) is type(new["calls"]) is type(deltas[pci]) is int
             and new["calls"] - old["calls"] == deltas[pci] > 0
             and first_epoch <= old["epoch"] < new["epoch"] <= observed,
             "EXACT_ORIGINAL_TWO_POINT_DCU_PROGRESS")


def target_rows(verified, manifest, by_pin, known, local):
    ack = {entry["wrapper_invocation"]: entry for entry in manifest["acks"]}
    attempts = {entry["wrapper"]: entry for entry in verified["wrapper_attempts"]}
    need(len(ack) == len(attempts) == 13 and ack.keys() == attempts.keys(), "EXACT_ORIGINAL_WRAPPER_ACK_COUNT")
    counts = Counter(row["wrapper"] for row in verified["per_node"])
    terminal_rows = {}
    for wrapper, attempt in attempts.items():
        terminal = read(named(local, attempt["terminal_file"]), attempt["terminal_SHA256"], 65536)
        need(terminal["batch"] == attempt["group"] and terminal["rc"] == 0
             and terminal["observed"]["plans_only"] is True, "PINNED_ORIGINAL_TERMINAL_RECEIPT")
        before = terminal["observed"]["targets"]
        terminal_rows[wrapper] = {row["node"]: row for row in before}
        need(len(terminal_rows[wrapper]) == len(before), "NO_AMBIGUOUS_ORIGINAL_TERMINAL_ROWS")
    targets = []
    for row in verified["per_node"]:
        node, parent = row["node"], row["parent"]
        group, previous = known[node]
        original, attempt = ack[row["wrapper"]], attempts[row["wrapper"]]
        response = by_pin[row["proof_SHA256"]]
        point = response["nodes"][node]
        terminal = terminal_rows[row["wrapper"]][node]
        accounting = terminal["accounting"]
        new = row["new_step"]
        need(group == row["group"] == attempt["group"] and previous["parent"] == parent == point["parent"]
             and previous["previous"] == row["previous"] and new != previous["previous"]
             and terminal["step"] == row["previous"] and terminal["request"] == previous["request"]
             and terminal["terminal_ready"] is True and terminal["terminal_SHA256"] == row["previous_terminal_SHA256"]
             and len(accounting) == 9 and accounting[0] == row["previous"] and accounting[2:4] == ["COMPLETED", "0:0"]
             and accounting[5] == node and accounting[8] == row["previous_End"]
             and re.fullmatch(parent + r"\.[1-9][0-9]*", new) and row["independently_running"] is True
             and row["same_PCI_advancing_DCU"] == point["growth_DCU"] == 8
             and is_pin(row["previous_terminal_SHA256"])
             and original["bridge_ack"]["request"] == attempt["request"] == row["request"] == point["request"]
             and original["bridge_ack"]["requested_cards"] == 8 * attempt["node_count"] == 8 * counts[row["wrapper"]]
             and original["terminal_receipt"] == attempt["terminal_file"],
             "EXACT_PREDECESSOR_PARENT_REQUEST_AND_ACK")
        start, duration, budget, prepare = (row[key] for key in (
            "new_worker_start_epoch", "new_worker_seconds", "new_budget_epoch", "new_preparation_epoch"))
        pid, worker, expected = (row[key] for key in ("new_worker_PID", "new_worker_SHA256", "new_expected_SHA256"))
        need(point["status"] == "RUNNING_WITH_ORIGINAL_BUDGET" and point["previous"] == new
             and type(pid) is int and pid > 0 and pid == point["worker_PID"]
             and is_pin(worker) and worker == point["worker_SHA256"]
             and is_pin(expected) and expected == point["expected_SHA256"]
             and type(duration) is int and duration > 0 and duration == point["worker_seconds"]
             and start == point["worker_start_epoch"] and budget == point["planned_expiry_epoch"] == start + duration
             and prepare == point["prepare_epoch"] == budget - 900
             and datetime.fromisoformat(row["previous_End"]).replace(tzinfo=CST).timestamp() <= start <= response["observed_epoch"],
             "EXACT_ORIGINAL_WORKER_AND_BUDGET")
        verify_points(point, response["observed_epoch"], start)
        targets.append(dict(node=node, parent=parent, previous=new, request=row["request"],
                            worker_PID=pid, previous_worker_pid=pid, worker_SHA256=worker, expected_SHA256=expected,
                            worker_start_epoch=start, worker_seconds=duration,
                            planned_end_epoch=budget, prepare_epoch=prepare,
                            planned_end_CST=stamp(budget), prepare_CST=stamp(prepare)))
    need(len(targets) == len({row["previous"] for row in targets}) == 58
         and not ({row["previous"] for row in targets} & {row["previous"] for _, row in known.values()})
         and Counter(r["parent"] for r in targets) == PARENTS
         and min(row["planned_end_CST"] for row in targets) == FIRST and max(row["planned_end_CST"] for row in targets) == LAST
         and min(row["prepare_CST"] for row in targets) == PREP, "EXACT_58_NEXT_SUCCESSOR_IDENTITIES")
    return targets


def batches(targets):
    groups = []
    for parent in PARENTS:
        current = []
        for row in sorted((r for r in targets if r["parent"] == parent),
                          key=lambda r: (r["planned_end_epoch"], r["node"])):
            if current and (len(current) == 8 or row["planned_end_epoch"] - current[0]["planned_end_epoch"] > 150):
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)
    return sorted(groups, key=lambda group: (group[0]["planned_end_epoch"], group[0]["parent"]))


def validate(path, wanted, original):
    local = original["LOCAL"]
    need(path == local / VERIFIED_FILE and wanted == VERIFIED_SHA and original["CALENDAR_SHA"] == HISTORICAL_SHA,
         "EXACT_58_CALENDAR_ARGUMENT")
    verified = read(path, wanted)
    manifest = read(local / MANIFEST_FILE, MANIFEST_SHA)
    historical = read(original["CALENDAR"], HISTORICAL_SHA)
    need(verified["schema"] == "training80-0544-ordinary-verified-standby" and verified["calendar_SHA256"] == HISTORICAL_SHA
         and verified["source_manifest_SHA256"] == MANIFEST_SHA and verified["preemptible"] is True
         and verified["synthetic_BF16"] is True and historical["node_count"] == 70 and historical["DCU_count"] == 560,
         "FROZEN_CALENDAR_LINEAGE")
    extra = manifest["additional_evidence"]
    need(extra["post_wave_fleet_SHA256"] == FLEET_SHA and extra["unknown_original_reconciliation_SHA256"] == UNKNOWN_SHA,
         "FROZEN_ORIGINAL_FLEET_AND_UNKNOWN_EVIDENCE")
    fleet = read(named(local, extra["post_wave_fleet_file"]), FLEET_SHA)
    unknown = read(named(local, extra["unknown_original_reconciliation_file"]), UNKNOWN_SHA)
    need(fleet["rc"] == 0 and fleet["GPU_steps_created"] == 0 and fleet["observed"]["distinct_nodes"] == 80
         and fleet["observed"]["allocated_DCU"] == 640, "FROZEN_ORIGINAL_FLEET_80_640")
    known = identities(original["groups"]())
    classified = classify(verified, manifest, historical, known, original["FENCES"])
    attempts = unknown["observed"]["targets"]
    need(unknown["rc"] == 0 and unknown["replay_allowed"] is False and unknown["observed"]["GPU_steps_created"] == 0
         and set(unknown["original_scope"]) == set(attempts) and len(attempts) == 2
         and all(item["replay_allowed"] is False and item["execution_absence_proven"] is False for item in attempts.values())
         and {node for item in attempts.values() for node in item["worker_SHA256"]}
         == {node for node, state in classified.items() if state == "BRIDGE_UNKNOWN"}, "UNKNOWN_NODES_REMAIN_WITH_ORIGINALS")
    targets = target_rows(verified, manifest, progress(verified, manifest, local), known, local)
    groups = batches(targets)
    need(sum(map(len, groups)) == 58 and all(1 <= len(group) <= 8 for group in groups), "EXACT_NEXT_GROUP_COVERAGE")
    group_sha = pin(encoded(groups))
    return groups, {
        "state": "OFFLINE_58_SUCCESSORS_AND_22_EXCLUDED_FROZEN",
        "wave": WAVE,
        "calendar_SHA256": wanted, "source_manifest_SHA256": MANIFEST_SHA, "historical_70_SHA256": HISTORICAL_SHA,
        "same_PCI_two_point_cards": 464, "parent_candidates": PARENTS, "whole_pool_nodes": 80, "whole_pool_DCU": 640,
        "ordinary_candidates": 58, "excluded_nodes": 22, "net_new_nodes": 0, "net_new_DCU": 0,
        "classification": dict(sorted(Counter(classified.values()).items())),
        "classification_SHA256": pin(encoded(classified)), "group_roster_SHA256": group_sha,
        "budget_Beijing": [FIRST, LAST], "first_preparation_Beijing": PREP, "remote_calls": 0,
        "groups": [dict(group=i, parent=g[0]["parent"], nodes=len(g), from_Beijing=g[0]["planned_end_CST"],
                        through_Beijing=g[-1]["planned_end_CST"]) for i, g in enumerate(groups, 1)],
    }


def require_adoption(path, sha, wave, ledger_path, scope, local):
    need(path is not None and path.is_absolute() and path.is_relative_to(local) and is_pin(sha),
         "EXPLICIT_SOLE_COORDINATOR_ADOPTION_REQUIRED")
    document = read(path, sha, 16384)
    need(ledger_path.is_file() and ledger_path.resolve(strict=True) == ledger_path, "SAME_EXISTING_VNEXT_LEDGER_REQUIRED")
    file = ledger_path.stat()
    need(document == {
        "schema": "training80.ordinary.vnext-sole-adoption.v1", "wave": wave,
        "calendar_SHA256": VERIFIED_SHA, "group_roster_SHA256": scope["group_roster_SHA256"],
        "ledger_absolute": ledger_path.as_posix(), "ledger_device": file.st_dev, "ledger_inode": file.st_ino,
        "previous_ordinary_controller_released": True, "sole_current_coordinator_authorized": True,
    }, "ADOPTION_MUST_BIND_SAME_EXISTING_LEDGER_AND_CALENDAR")
