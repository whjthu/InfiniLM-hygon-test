"""Retire individually clear Graph64 shards after the pinned original pre-exec failure."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

import close_graph64_original_backfill as graph
from report_training80_backlog import QUEUE_DB, REPO, fleet_view, hex_value, need
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, canonical, digest


ABSENCE = "recovered-authenticated-presrun-observation.json"
ABSENCE_SHA = "477a3d4316c4a42e377a0132dd36ba98286e400988ba2164da8a1f36a0dabb1c"
TRANSPORT_SHA = "23fc63346d040eba80689611317cd236b7ede717c4db0a231fde39638ff446b2"
STDERR_SHA = "8fe165def39d37115f7b1fc631e2deeff2445be8007ad723eea3bcbcf64f0def"
OBSERVED_EPOCH = 1795296654.5551798
EVENT = "graph64_original_preexec_retired"


def original_failure(plan, absence_raw, transport_raw, stderr_raw, stdout_raw):
    need((graph.sha(absence_raw), graph.sha(transport_raw), graph.sha(stderr_raw)) ==
         (ABSENCE_SHA, TRANSPORT_SHA, STDERR_SHA), "original absence, transport or banner receipt bytes differ")
    need(stdout_raw == b"" and
         any(line.strip() == b"Connection timed out during banner exchange" for line in stderr_raw.splitlines()),
         "initial OpenSSH transport did not prove the exact original banner failure")
    transport = json.loads(transport_raw)
    expected = dict(request=graph.REQUEST, planned_nodes=75, planned_cards=600, planned_shards=31,
                    transport_attempted=True, remote_outcome="UNKNOWN", transport_returncode=255,
                    terminal_verified=False, replay_allowed=False)
    need(all(type(transport[key]) is type(value) and transport[key] == value for key, value in expected.items()),
         "original transport scope differs")
    absence = json.loads(absence_raw)
    need(absence == dict(epoch=OBSERVED_EPOCH, stage_plan=False, result_intent=False, stage_dir=False,
                         result_dir=False, steps=[]) and
         all(absence[key] is False for key in ("stage_plan", "result_intent", "stage_dir", "result_dir"))
         and plan["campaign_epoch"] + 180 < absence["epoch"],
         "authenticated original namespace or accounting absence after controller expiry is unproven")
    return dict(campaign_request=graph.REQUEST, original_plan_SHA256=graph.PLAN_SHA, absence_SHA256=ABSENCE_SHA,
                transport_SHA256=TRANSPORT_SHA, initial_stderr_SHA256=STDERR_SHA,
                initial_stdout_SHA256=graph.sha(stdout_raw), observed_epoch=absence["epoch"],
                original_controller_deadline_epoch=plan["campaign_epoch"] + 180)


def read_failure(repo):
    folder = repo / graph.DIRECTORY
    stdout = folder / "stdout.private"
    need(stdout.resolve(strict=True) == stdout and stdout.is_file(), "original private stdout is unavailable")
    with stdout.open("rb") as source:
        stdout_raw = source.read(1)
    return (*[graph.private_bytes(folder / name, 16384) for name in (ABSENCE, "transport.json", "stderr.private")],
            stdout_raw)


def fresh_post(snapshot, raw, ledger, now, anchor):
    view = fleet_view(snapshot, now, anchor, ledger)
    membership, nodes, fresh = view
    receipt = snapshot["local_receipt"]
    need(fresh and snapshot["snapshot_kind"] == "training80-dispatch-v1"
         and hex_value(receipt["request_id"], 32) and receipt["request_id"] != graph.REQUEST
         and receipt["raw_response_SHA256"] == graph.sha(raw)
         and json.loads(raw) == {key: value for key, value in snapshot.items() if key != "local_receipt"}
         and membership["observed_epoch"] == snapshot["observed_epoch"] >= OBSERVED_EPOCH,
         "clearance is stale, altered, or predates the authenticated absence")
    return membership, nodes, receipt


def verify_already_retired(db, label, unit, indexed, marker):
    run = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
    row = db.execute("SELECT stage,state,binding,nodes,handle,result FROM runs WHERE id=?", (run,)).fetchone()
    if row is None or row[1] != "RETIRED_PRELAUNCH":
        return False
    need(indexed == json.loads(marker) and indexed["run"] == run and row[0] == DIRECT_STAGE and json.loads(row[3]) == []
         and json.loads(row[2]) == indexed["packet"] and json.loads(row[4]) == indexed["handle"],
         "retired reservation differs from the original shard")
    result = json.loads(row[5])
    evidence = result["evidence"]
    event = db.execute("SELECT data FROM events WHERE run=? AND kind=?", (run, EVENT)).fetchall()
    need(len(event) == 1 and json.loads(event[0][0]) == result and result["request_id"] == unit["request"]
         and result["prelaunch"] is True and result["measurement_terminal"] is False and result["replay_allowed"] is False
         and result["evidence_SHA256"] == digest(evidence) and evidence["shard"] == label
         and evidence["absence_SHA256"] == ABSENCE_SHA and evidence["transport_SHA256"] == TRANSPORT_SHA
         and evidence["initial_stderr_SHA256"] == STDERR_SHA, "retired evidence is not this original pre-exec transition")
    return True


def apply_retirement(queue, plan, label, unit, indexed, marker, result, expires):
    """Commit only this campaign's original operator reservation in one queue transaction."""
    db = queue.db
    db.execute("BEGIN IMMEDIATE")
    try:
        if verify_already_retired(db, label, unit, indexed, marker):
            db.commit()
            return "ALREADY_RETIRED"
        state, handle, _, _ = graph.claimed_shard(db, plan, label, unit, indexed, marker)
        run = indexed["run"]
        need(state == "UNKNOWN" and queue.reservation(run)["dispatch_ticket"] == handle["handoff_SHA256"]
             and time.monotonic() <= expires, "original UNKNOWN scope changed or clearance expired before retirement")
        db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",nodes="[]",result=? WHERE id=?',
                   (canonical(result).decode(), run))
        queue.event(run, EVENT, result)
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return "RETIRED_PRELAUNCH"


def retire(db, originals, failure, snapshot_raw, post_raw, *, now=None, anchor=None, queue=None):
    plan, handles, markers, handles_sha, ledger = originals
    absence = original_failure(plan, *failure)
    snapshot = json.loads(snapshot_raw)
    now = time.monotonic() if now is None else now
    anchor = time.time() - now if anchor is None else anchor
    try:
        post = fresh_post(snapshot, post_raw, ledger, now, anchor)
        post_error = ""
    except (OSError, ValueError, KeyError, TypeError) as error:
        post, post_error = None, str(error)
    rows = {}
    for label, unit in plan["shards"].items():
        status = rows[label] = dict(request=unit["request"], parent=unit["parent"], nodes=unit["nodes"],
                                    planned_cards=8 * len(unit["nodes"]))
        try:
            indexed, marker = handles[label], markers[label]
            if verify_already_retired(db, label, unit, indexed, marker):
                status.update(state="ALREADY_RETIRED", run=indexed["run"])
                continue
            state, handle, packet, _ = graph.claimed_shard(db, plan, label, unit, indexed, marker)
            need(state == "UNKNOWN", "shared queue is not in the original UNKNOWN state")
            status["run"] = indexed["run"]
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="HOLD_ORIGINAL_UNVERIFIED", reason=str(error))
            continue
        try:
            need(post is not None, post_error)
            probes = graph.post_steps(snapshot, post, plan, unit, packet, original_step=None)
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="HOLD_CLEARANCE_UNVERIFIED", reason=str(error))
            continue
        evidence = dict(absence, shard=label, original_handoff_SHA256=handle["handoff_SHA256"],
                        original_index_SHA256=handles_sha, original_shard_intent_SHA256=graph.sha(marker),
                        post_snapshot_SHA256=graph.sha(snapshot_raw), raw_post_SHA256=graph.sha(post_raw), post_steps=probes)
        result = dict(request_id=handle["request_id"], intent=handle["intent"], parent=handle["parent"],
                      placement_by_parent=handle["placement_by_parent"], prelaunch=True, measurement_terminal=False,
                      transport_attempted=True, remote_execution_attempted=False, replay_allowed=False,
                      evidence_SHA256=digest(evidence), evidence=evidence)
        status.update(state="READY_TO_RETIRE", evidence_SHA256=result["evidence_SHA256"])
        if queue is not None:
            try:
                receipt = post[2]
                expires = receipt["received_monotonic"] + SNAPSHOT_MAX_AGE_S - receipt["roundtrip_s"]
                status["state"] = apply_retirement(queue, plan, label, unit, indexed, marker, result, expires)
            except (ValueError, KeyError, TypeError, sqlite3.Error) as error:
                status.update(state="HOLD_APPLY_ERROR", reason=str(error))
    retired = {"ALREADY_RETIRED", "RETIRED_PRELAUNCH"}
    return dict(request=graph.REQUEST, planned_shards=len(rows), **graph.DENOMINATOR, shards=rows,
                retired_shards=sum(item["state"] in retired for item in rows.values()),
                retired_cards=sum(item["planned_cards"] for item in rows.values() if item["state"] in retired),
                ready_to_retire_shards=sum(item["state"] == "READY_TO_RETIRE" for item in rows.values()),
                held_shards=sum(item["state"].startswith("HOLD_") for item in rows.values()),
                actual_graph_GPU_terminal_shards=0, actual_graph_GPU_steps=0, replay_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use the controller's pinned original absence evidence and the existing CPU collector's new raw response "
        "and once-stamped snapshot. Default is read-only."))
    parser.add_argument("--post-snapshot", type=Path, required=True)
    parser.add_argument("--post-response", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="record actual prelaunch retirements in the shared queue")
    args = parser.parse_args()
    originals = graph.original_inputs(REPO)
    failure = read_failure(REPO)
    post = [graph.private_bytes(path) for path in (args.post_snapshot, args.post_response)]
    database = QUEUE_DB.resolve(strict=True)
    if args.apply:
        queue = Queue(database)
        with closing(queue.db):
            summary = retire(queue.db, originals, failure, *post, queue=queue)
    else:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            summary = retire(db, originals, failure, *post)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
