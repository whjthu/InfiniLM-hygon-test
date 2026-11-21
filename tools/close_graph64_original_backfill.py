"""Close only separately proven shards of the original Graph64 backfill."""

import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from prepare_training80_vggt_direct_finish import bounded
from report_training80_backlog import QUEUE_DB, REPO, TRAINING80_LEDGER, fleet_view, hex_value, need
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


REQUEST = "811eb13381e146e72bea160af7b61037"
PACKAGE = Path("code/bw1000/fleet-bf16-projection-graph64-backfill-v1")
DIRECTORY = Path("local-only") / ("O01-resident-BF16-graph64-fleet-v1-" + REQUEST + "-original")
PLAN_SHA = "7dafcd7567f1290b022c3a0ec07242eea5375860c3615c9035e0cc0a95afe08b"
CLAIM_SHA = "95dfc7f2d70fc3cd13e1f72e190fe132b318d8ed61509faea9e5df6cf4a3af71"
SCOPE_SHA = "6b6838dc7c46d8fb291d193c5e561cb695018f52e0745bcfd451dc966c1ae288"
CONTROLLER_SHA = "22265016b13fe32a5c98e7fb83da68f52fd1604d4035335a7dd00bc9f0056e13"
PUBLICATION = "97df23406d0c2a6d405b6c108265b9d148480456"
CASE = "O01.resident-bf16-graph64-op-backed-backfill"
ACTIVE = {"UNKNOWN", "SUBMITTED", "DISPATCHING"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
DENOMINATOR = dict(planned_nodes=75, planned_cards=600, planned_arms=2400, planned_busy_card_samples=9600)
HOST = re.compile(r"[af][0-9]{2}r[0-9]n[0-9]{2}")
HOST_GROUP = re.compile(r"([af][0-9]{2}r[0-9]n)\[([0-9,-]+)\]")


def sha(raw):
    return sha256(raw).hexdigest()


def private_bytes(path, limit=2**21):
    path = path.absolute()
    need(path.resolve(strict=True) == path and path.is_file(), "expected an original unaliased receipt file")
    return bounded(path, limit)


def original_inputs(repo):
    """Read the pinned plan and only the original per-shard markers it names."""
    directory = repo / DIRECTORY
    plan_raw = private_bytes(directory / "original-plan.json", 65536)
    need(sha(plan_raw) == PLAN_SHA, "original Graph64 plan bytes differ")
    plan = json.loads(plan_raw)
    scope_file = repo / PACKAGE / "scope.py"
    scope_raw = bounded(scope_file, 32768)
    need(sha(scope_raw) == SCOPE_SHA
         and sha(bounded(repo / PACKAGE / "controller.py", 32768)) == CONTROLLER_SHA,
         "original Graph64 scope or controller source differs")
    scope = {"__name__": "original_graph64_scope"}
    exec(compile(scope_raw, str(scope_file), "exec"), scope)
    ledger = repo / TRAINING80_LEDGER.relative_to(REPO)
    owners = scope["ledger"](bounded(ledger, 65536))
    shards = scope["validate_plan"](plan, owners)
    need(plan["publication"] == PUBLICATION and len(shards) == 31 and len(plan["selected"]) == 75,
         "original Graph64 plan changed its 31-shard or 75-node scope")
    handles_raw = private_bytes(directory / "original-shards.json", 196608)
    handles = json.loads(handles_raw)
    need(set(handles) <= set(shards), "original shard index has foreign labels")
    markers = {}
    for label in shards.keys() & handles.keys():
        try:
            markers[label] = private_bytes(directory / "shards" / (label + ".json"), 16384)
        except (OSError, ValueError):
            continue
    return plan, handles, markers, sha(handles_raw), ledger


def terminal_receipt(raw, *, monotonic=time.monotonic, wall=time.time):
    """Record the authenticated controller --reconcile response as it arrives, before requesting the CPU post."""
    verdict = json.loads(raw)
    need(verdict["request"] == REQUEST and verdict["replay_allowed"] is False,
         "response is not the original Graph64 reconcile")
    received = monotonic()
    return dict(request=REQUEST, received_monotonic=received, boot_epoch_anchor=wall() - received,
                raw_reconcile_SHA256=sha(raw))


def hosts(expression):
    """Expand only the numbered Slurm host format used by the two original parents."""
    need(isinstance(expression, str) and 0 < len(expression) <= 2048, "invalid accounting NodeList")
    parts = re.findall(r"[^,\[\]]+(?:\[[^\[\]]+\])?", expression)
    need(",".join(parts) == expression, "malformed accounting NodeList")
    nodes = []
    for part in parts:
        if HOST.fullmatch(part):
            nodes.append(part)
            continue
        group = HOST_GROUP.fullmatch(part)
        need(group is not None, "unsupported accounting NodeList")
        prefix, tail = group.groups()
        for token in tail.split(","):
            interval = re.fullmatch(r"([0-9]{2})(?:-([0-9]{2}))?", token)
            need(interval is not None, "invalid accounting host interval")
            first, last = interval.groups()
            lower, upper = int(first), int(last or first)
            need(lower <= upper and len(nodes) + upper - lower + 1 <= 80, "oversized accounting host interval")
            nodes.extend(f"{prefix}{number:02}" for number in range(lower, upper + 1))
    need(len(nodes) == len(set(nodes)) <= 80, "duplicate accounting hosts")
    return set(nodes)


def claimed_shard(connection, plan, label, unit, original, marker):
    need(json.loads(marker) == original, "per-shard intent differs from the original index")
    packet, handle = original["packet"], original["handle"]
    expected_run = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
    placement = {unit["parent"]: dict.fromkeys(unit["nodes"], 8)}
    expected = dict(candidate=label, request=unit["request"], case=CASE, kind="operator",
                    campaign_request=REQUEST, publication_commit=PUBLICATION, placement_by_parent=placement,
                    workload_SHA256=CLAIM_SHA, snapshot_payload_SHA256=plan["before_snapshot_SHA256"],
                    launcher=(PACKAGE / "controller.py").as_posix(), invocation={"kind": "pinned_finite_code"},
                    replay_allowed=False)
    handoff = digest({key: value for key, value in packet.items() if key != "handoff_SHA256"})
    intent = (DIRECTORY / "shards" / (label + ".json")).as_posix()
    expected_handle = dict(request_id=unit["request"], intent=intent, parent=unit["parent"], nodes=unit["nodes"],
                           placement_by_parent=placement, direct_kind="operator", handoff_SHA256=handoff)
    need(original["run"] == expected_run and handle == expected_handle
         and packet["handoff_SHA256"] == handoff and all(packet[key] == value for key, value in expected.items())
         and hex_value(packet["snapshot_request"], 32) and hex_value(packet["snapshot_raw_SHA256"], 64),
         "original shard handoff identity differs")
    row = connection.execute("SELECT stage,state,binding,nodes,handle,result FROM runs WHERE id=?", (expected_run,)).fetchone()
    need(row is not None and row[0] == DIRECT_STAGE and json.loads(row[2]) == packet
         and json.loads(row[3]) == unit["nodes"] and json.loads(row[4]) == handle,
         "shared queue claim differs from the original shard markers")
    return row[1], handle, packet, json.loads(row[5]) if row[5] else None


def terminal_row(item, label, unit):
    row = item["accounting"]
    need(isinstance(row, list) and len(row) == 7 and all(isinstance(part, str) for part in row),
         "terminal accounting has no original seven-column row")
    need(item["terminal"] is True and item["resources_match"] is True and item["independent_post_required"] is True
         and item["attempted"] is True and hex_value(item["attempt_SHA256"], 64)
         and all(item[key] == value for key, value in unit.items())
         and item["planned_cards"] == 8 * len(unit["nodes"])
         and re.fullmatch(unit["parent"] + r"\.[0-9]+", row[0]) and row[1] == label
         and row[2].split() and row[2].split()[0].rstrip("+") == item["state"] and item["state"] in TERMINAL
         and re.fullmatch(r"[0-9]+:[0-9]+", row[3]) and row[4].isdecimal()
         and hosts(row[5]) == set(unit["nodes"]), "terminal accounting differs from the original shard")
    fields = [part.split("=", 1) for part in row[6].split(",")]
    need(all(len(part) == 2 for part in fields), "malformed terminal AllocTRES")
    resources = dict(fields)
    count = len(unit["nodes"])
    need(len(resources) == len(fields) and resources["node"] == str(count) and resources["cpu"] == str(12 * count)
         and resources["gres/dcu"] == str(8 * count) and resources["mem"] in {f"{64 * count}G", f"{65536 * count}M"}
         and {key for key in resources if key.startswith("gres/")} == {"gres/dcu"},
         "terminal Slurm GPU resources differ from the frozen shard")
    return row


def post_view(snapshot, post_raw, capture, ledger, *, now, boot_anchor):
    membership, nodes, fresh = fleet_view(snapshot, now, boot_anchor, ledger)
    receipt = snapshot["local_receipt"]
    received = capture["received_monotonic"]
    anchored = capture["boot_epoch_anchor"]
    need(fresh and snapshot["snapshot_kind"] == "training80-dispatch-v1"
         and hex_value(receipt["request_id"], 32) and receipt["request_id"] != REQUEST
         and all(type(value) in (int, float) and math.isfinite(value) for value in (received, anchored))
         and abs(anchored - boot_anchor) < 2
         and 0 <= received <= receipt["received_monotonic"] - receipt["roundtrip_s"]
         and receipt["raw_response_SHA256"] == sha(post_raw)
         and json.loads(post_raw) == {key: value for key, value in snapshot.items() if key != "local_receipt"},
         "post is stale, altered, or began before terminal receipt")
    return membership, nodes, receipt


def post_steps(snapshot, view, plan, unit, packet, original_step):
    membership, nodes, receipt = view
    parent = unit["parent"]
    observed_parent = membership["parents"][parent]
    need(receipt["request_id"] != packet["snapshot_request"]
         and receipt["raw_response_SHA256"] != packet["snapshot_raw_SHA256"]
         and receipt["payload_SHA256"] != packet["snapshot_payload_SHA256"]
         and observed_parent["state"] == "RUNNING" and observed_parent["owned_by_observer"] is True
         and observed_parent["step_locations_exact"] is True
         and original_step not in observed_parent["active_steps"],
         "post does not independently verify the unchanged owned parent")
    post = {}
    for name in unit["nodes"]:
        probe = snapshot["nodes"][name]
        before = plan["selected"][name]
        step = probe["device_probe_step"]
        need(nodes[name]["admission"] == "clear" and nodes[name]["membership"] == "NO_NUMBERED_STEP"
             and nodes[name]["numbered_step_assigned_DCU_bounds"] == [0, 0]
             and (probe["parent"], probe["zone"]) == (parent, unit["zone"])
             and set(probe["PCIs"]) == set(before["PCI"]) and probe["probe_error"] is None and probe["active_steps"] == []
             and all(type(probe[key]) is int and probe[key] == 0
                     for key in ("kfd_process_count", "own_device_fd_process_count"))
             and isinstance(step, str) and re.fullmatch(parent + r"\.[0-9]+", step) and step not in
             {original_step, before["before_CPU_step"]} and step not in observed_parent["active_steps"],
             f"no independent clear PCI/KFD/PID snapshot for {name}")
        batches = [batch for batch in snapshot["probes"][parent]["batches"] if name in batch.get("received_nodes", [])]
        need(len(batches) == 1 and name in batches[0]["requested_nodes"]
             and batches[0]["state"] == "RETURNED" and type(batches[0]["returncode"]) is int
             and batches[0]["returncode"] == 0 and hex_value(batches[0]["request_id"], 32)
             and hex_value(batches[0]["stdout_SHA256"], 64)
             and all(snapshot["nodes"][member]["parent"] == parent
                     and snapshot["nodes"][member]["device_probe_step"] == step for member in batches[0]["received_nodes"]),
             f"no successful zero-GRES collector batch for {name}")
        post[name] = dict(step=step, batch_request=batches[0]["request_id"], batch_stdout_SHA256=batches[0]["stdout_SHA256"])
    return post


def counters(verdict, unit, item, row):
    maxima = {"valid_cards": 8, "completed_arms": 32, "actual_busy_card_samples": 128}
    counts = dict.fromkeys(maxima, 0)
    audited = [verdict["nodes"].get(node, {}) for node in unit["nodes"]]
    for node in audited:
        for key, maximum in maxima.items():
            number = node.get(key, 0)
            need(type(number) is int and 0 <= number <= maximum, "malformed worker audit counters")
            counts[key] += number
    passed = item["device_qualified"]
    need(type(passed) is bool and (not passed or (
        row[2:4] == ["COMPLETED", "0:0"] and 0 < int(row[4]) <= 140 and all(
            node.get("passed") is True and node.get("original_step") == row[0]
            and all(node[key] == maximum for key, maximum in maxima.items()) for node in audited))),
        "device qualification contradicts the terminal worker evidence")
    return passed, counts


def close(connection, originals, reconcile_raw, capture_raw, snapshot_raw, post_raw, *,
          now=None, boot_anchor=None, queue=None):
    """Return the full original denominator. If supplied, queue is the existing Queue on this connection."""
    plan, handles, markers, handles_sha, ledger = originals
    verdict, capture, snapshot = map(json.loads, (reconcile_raw, capture_raw, snapshot_raw))
    shards = plan["shards"]
    need(verdict["request"] == capture["request"] == REQUEST and capture["raw_reconcile_SHA256"] == sha(reconcile_raw)
         and verdict["replay_allowed"] is False and all(verdict[key] == value for key, value in DENOMINATOR.items())
         and set(verdict["shards"]) <= set(shards) and set(verdict["nodes"]) <= set(plan["selected"]),
         "reconcile is not the original authenticated 31-shard result")
    reported_steps = Counter(item["accounting"][0] for item in verdict["shards"].values()
                             if isinstance(item, dict) and isinstance(item.get("accounting"), list)
                             and item["accounting"] and isinstance(item["accounting"][0], str))
    now = time.monotonic() if now is None else now
    boot_anchor = time.time() - now if boot_anchor is None else boot_anchor
    try:
        view = post_view(snapshot, post_raw, capture, ledger, now=now, boot_anchor=boot_anchor)
    except (ValueError, KeyError, TypeError, OSError) as error:
        view, post_error = None, str(error)
    rows = {}
    for label, unit in shards.items():
        total = 8 * len(unit["nodes"])
        status = rows[label] = dict(parent=unit["parent"], nodes=unit["nodes"], planned_cards=total)
        try:
            state, handle, packet, prior = claimed_shard(connection, plan, label, unit, handles[label], markers[label])
            status.update(run=handles[label]["run"], request=handle["request_id"])
            if state == "TERMINAL":
                need(prior["request_id"] == handle["request_id"] and prior["intent"] == handle["intent"]
                     and prior["placement_by_parent"] == handle["placement_by_parent"] and prior["post_clean"] is True
                     and prior["planned"] == total and prior["terminal"] is True and prior["qualification"] == "operator"
                     and prior["evidence_SHA256"] == digest(prior["evidence"]),
                     "existing terminal does not bind the original shard")
                status.update(state="ALREADY_CLOSED", result=prior)
                continue
            need(state in ACTIVE, "shared queue shard is not active")
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="HOLD_CLAIM_UNVERIFIED", reason=str(error))
            continue
        item = verdict["shards"].get(label)
        if not isinstance(item, dict) or item.get("terminal") is not True or not item.get("accounting"):
            status.update(state="HOLD_TERMINAL_UNKNOWN", reported_state=item.get("state") if isinstance(item, dict) else None)
            continue
        try:
            row = terminal_row(item, label, unit)
            need(reported_steps[row[0]] == 1, "numbered GPU step claimed by several shards")
            passed, counts = counters(verdict, unit, item, row)
            status.update(accounting=row, **counts)
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="HOLD_TERMINAL_UNVERIFIED", reason=str(error))
            continue
        try:
            need(view is not None, post_error if view is None else "")
            probes = post_steps(snapshot, view, plan, unit, packet, row[0])
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="HOLD_POST_UNVERIFIED", reason=str(error))
            continue
        evidence = dict(campaign_request=REQUEST, shard=label, local_plan_SHA256=PLAN_SHA,
                        controller_SHA256=CONTROLLER_SHA,
                        local_shard_index_SHA256=handles_sha, local_shard_intent_SHA256=sha(markers[label]),
                        target_shard_attempt_SHA256=item["attempt_SHA256"], terminal_row_SHA256=digest(row),
                        reconcile_SHA256=sha(reconcile_raw), reconcile_capture_SHA256=sha(capture_raw),
                        post_snapshot_SHA256=sha(snapshot_raw), raw_post_SHA256=sha(post_raw), post_steps=probes)
        completed = counts["valid_cards"]
        remainder = total - completed
        canceled = remainder if item["state"] == "CANCELLED" else 0
        failed = remainder if item["state"] not in {"COMPLETED", "CANCELLED"} or (
            item["state"] == "COMPLETED" and row[3] != "0:0") else 0
        result = dict(request_id=handle["request_id"], intent=handle["intent"], parent=unit["parent"],
                      placement_by_parent=handle["placement_by_parent"], qualification="operator", terminal=True,
                      post_clean=True, passed=passed, keeper_passed=None, promote=False, planned=total,
                      completed=completed, failed=failed, canceled=canceled, rejected=0,
                      missing=remainder-failed-canceled, evidence_SHA256=digest(evidence), evidence=evidence,
                      gpu_step=row[0], gpu_accounting=row, **counts)
        status.update(state="READY_TO_CLOSE", result=result)
        if queue is not None:
            try:
                expires = view[2]["received_monotonic"] + SNAPSHOT_MAX_AGE_S - view[2]["roundtrip_s"]
                need(time.monotonic() <= expires, "post expired before queue finish")
                queue.finish(status["run"], result)
                status["state"] = "CLOSED"
            except (ValueError, sqlite3.Error) as error:
                status.update(state="HOLD_FINISH_ERROR", reason=str(error))
    closed = {"ALREADY_CLOSED", "CLOSED"}
    measured = {key: sum(row.get(key, row.get("result", {}).get(key, 0)) for row in rows.values())
                for key in ("valid_cards", "completed_arms", "actual_busy_card_samples")}
    return dict(request=REQUEST, planned_shards=len(shards), **DENOMINATOR, shards=rows,
                verified_worker_counters=measured,
                closed_shards=sum(row["state"] in closed for row in rows.values()),
                closed_cards=sum(row["planned_cards"] for row in rows.values() if row["state"] in closed),
                ready_to_close_shards=sum(row["state"] == "READY_TO_CLOSE" for row in rows.values()),
                held_shards=sum(row["state"].startswith("HOLD_") for row in rows.values()),
                held_cards=sum(row["planned_cards"] for row in rows.values() if row["state"].startswith("HOLD_")),
                replay_allowed=False, new_gpu_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Persist terminal_receipt(raw) at authenticated reconcile response arrival; then run the existing "
        "CPU collector and retain both its raw response and once-stamped snapshot. Default is read-only."))
    for name in ("reconcile", "reconcile-receipt", "post-snapshot", "post-response"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="finish only verified original shards in the shared queue")
    args = parser.parse_args()
    originals = original_inputs(REPO)
    inputs = [private_bytes(path) for path in (args.reconcile, args.reconcile_receipt, args.post_snapshot, args.post_response)]
    database = QUEUE_DB.resolve(strict=True)
    if args.apply:
        queue = Queue(database)
        with closing(queue.db):
            result = close(queue.db, originals, *inputs, queue=queue)
    else:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            result = close(connection, originals, *inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
