"""Independently retire the exact GLM fresh-KV v1 reservation rejected before srun."""

from contextlib import closing
from hashlib import sha256
import argparse
import json
from pathlib import Path
import re
import sqlite3
import time

from report_training80_backlog import QUEUE_DB, REPO, TRAINING80_LEDGER, fleet_view
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, canonical, digest


REQUEST = "b9a7f0b78c566911ed40004fc906ce44"
RUN = "a15a917ab631c11244951af1a0cf528f7becdc4086251ce9bc886fdef71437ec"
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v1"
PARENT, NODE, ZONE = "1103606", "f12r2n13", "O01"
PARENTS = ["1087656", PARENT]
PLACEMENT = {PARENT: {NODE: 0}}
INTENT = f"local-only/{LABEL}-original-dispatch/intent.json"
HOST_CLAIM = "005a011e94cbe1fa5f6f11a735d1deb1afd9ee88656632656b48e091c576a5f9"
EVENT = "glm474_fresh_kv_original_presrun_rejected"
INTENT_SHA = "6e7b626b98983da3fa34cd976e0994e14fa7122c17d395226c787e6f524e7c05"
TRANSPORT_SHA = "3209131f22adf33ec6e36712388c06cdb3c4f96b88c400074dd8f6ad7890c6dd"
STDERR_SHA = "fc532517f5caa8ba7323ecc163ff415fdf2bd1d4e900f1a9bda5533559b9c649"
ABSENCE_SHA = "1ed18fa46c459731f590fe29a35923015c49e286f549cb8606b424411f44a58c"
COLUMNS = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return sha256(raw).hexdigest()


def read(path, limit=2 * 1024 * 1024):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(len(raw) <= limit, "input proof exceeds its byte bound")
    return raw


def exact_failure(intent_raw, transport_raw, stdout, stderr):
    need((sha(intent_raw), sha(transport_raw), sha(stderr), stdout) ==
         (INTENT_SHA, TRANSPORT_SHA, STDERR_SHA, b""), "original dispatch evidence bytes differ")
    intent, transport = json.loads(intent_raw), json.loads(transport_raw)
    need(intent["request"] == transport["request"] == REQUEST and intent["queue_run"] == transport["queue_run"] == RUN
         and intent["local_intent"] == transport["local_intent"] == INTENT
         and transport["transport_returncode"] == 1 and transport["transport_error"] is None
         and transport["terminal_verified"] is False and transport["replay_allowed"] is False,
         "original dispatch identity or return code differs")
    traceback = stderr.decode("utf-8").splitlines()
    need(any("line 77, in claim_identity" in line for line in traceback)
         and any(line.startswith("FileNotFoundError: [Errno 2]") and
                 line.endswith("/fresh-kv-pack-CPU-v1/LOCAL_RESULT.json'") for line in traceback),
         "original failure did not stop in the pre-srun package gate")
    return intent


def exact_absence(raw, pinned_sha):
    need(pinned_sha == ABSENCE_SHA and sha(raw) == pinned_sha,
         "independent authenticated namespace/accounting receipt bytes differ")
    row = json.loads(raw)
    fields = dict(request=REQUEST, label=LABEL, parents=PARENTS, accounting_columns=COLUMNS,
                  complete_accounting=True, matching_rows=[], controller_exists=False, candidate_exists=False,
                  original_missing_source_leaf="LOCAL_RESULT.json", original_missing_source_present=False,
                  stage_source_manifest_SHA256="333551bc79cbd2e56e18274737b00c57c0a918cc00700b167a61b9a316ac7850")
    streams = row["exact_accounting_stream_SHA256"]
    need(all(type(row.get(key)) is type(value) and row[key] == value for key, value in fields.items())
         and streams.keys() == set(PARENTS) and all(re.fullmatch(r"[0-9a-f]{64}", pin) for pin in streams.values())
         and type(row.get("observed_epoch")) in (int, float) and 0 < row["observed_epoch"] < float("inf"),
         "no independent evidence of absent controller, candidate and matching Slurm rows")
    return row


def clean_after(absence, snapshot, source_raw, now, anchor, ledger):
    membership, nodes, fresh = fleet_view(snapshot, now, anchor, ledger)
    receipt = snapshot["local_receipt"]
    node, row = nodes[NODE], snapshot["nodes"][NODE]
    pcis = row["PCIs"]
    need(fresh and snapshot["snapshot_kind"] == "training80-dispatch-v1"
         and sha(source_raw) == receipt["raw_response_SHA256"]
         and json.loads(source_raw) == {key: value for key, value in snapshot.items() if key != "local_receipt"}
         and re.fullmatch(r"[0-9a-f]{32}", receipt["request_id"]) and receipt["request_id"] != REQUEST
         and membership["observed_epoch"] == snapshot["observed_epoch"] >= absence["observed_epoch"]
         and node["admission"] == "clear" and node["parent"] == row["parent"] == PARENT and node["zone"] == row["zone"] == ZONE
         and row["probe_error"] is None and len(set(pcis)) == len(pcis) == 8 and all(PCI.fullmatch(pci) for pci in pcis)
         and all(type(row.get(key)) is int and row[key] == 0
                 for key in ("kfd_process_count", "own_device_fd_process_count"))
         and re.fullmatch(PARENT + r"\.\d+", row["device_probe_step"]), "fresh independent O01 cleanup is missing")
    expires = receipt["received_monotonic"] + SNAPSHOT_MAX_AGE_S - receipt["roundtrip_s"]
    need(now <= expires, "original fresh clearance expired")
    return dict(raw_SHA256=receipt["raw_response_SHA256"], payload_SHA256=receipt["payload_SHA256"],
                probe_step=row["device_probe_step"], observed_epoch=snapshot["observed_epoch"], PCI=sorted(pcis)), expires


def row_state(db, intent):
    row = db.execute("SELECT stage,state,handle,nodes,binding,result FROM runs WHERE id=?", (RUN,)).fetchone()
    need(row is not None and row[0] == DIRECT_STAGE
         and row[1] in ("DISPATCHING", "RETIRED_PRELAUNCH")
         and json.loads(row[4]) == intent, "original legacy reservation differs")
    handle = dict(request_id=REQUEST, intent=INTENT, nodes=[NODE], direct_kind="validation",
                  parent=PARENT, placement_by_parent=PLACEMENT)
    need(json.loads(row[2]) == handle, "unexpected handoff or changed original owner")
    events = [(kind, json.loads(data)) for kind, data in
              db.execute("SELECT kind,data FROM events WHERE run=? ORDER BY seq", (RUN,))]
    expected = [("reserved", dict(parent=PARENT, nodes=[NODE], claim_SHA256=HOST_CLAIM)),
                ("dispatch_claimed", dict(handle=handle))]
    if row[1] == "DISPATCHING":
        need(json.loads(row[3]) == [NODE] and row[5] is None and events == expected,
             "the original pre-srun claim has other execution evidence")
        return None
    result = json.loads(row[5])
    need(json.loads(row[3]) == [] and events == [*expected, (EVENT, result)]
         and result["request_id"] == REQUEST and result["retired_prelaunch"] is True
         and result["evidence_SHA256"] == digest(result["evidence"]),
         "existing retirement does not bind the original evidence and event")
    return result


def retire(queue, intent, result, expires, *, clock=time.monotonic):
    """Apply one original transition without upgrading it to a Slurm terminal."""
    db = queue.db
    db.execute("BEGIN IMMEDIATE")
    try:
        prior = row_state(db, intent)
        if prior is not None:
            need(prior == result, "original reservation is already retired with different evidence")
            db.commit()
            return "ALREADY_RETIRED"
        need(clock() <= expires, "original cleanup expired before atomic retirement")
        db.execute('UPDATE runs SET state="RETIRED_PRELAUNCH",nodes="[]",result=? WHERE id=?',
                   (canonical(result).decode(), RUN))
        queue.event(RUN, EVENT, result)
        db.commit()
        return "RETIRED_PRELAUNCH"
    except BaseException:
        db.rollback()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--absence", type=Path, required=True)
    ap.add_argument("--absence-sha256", required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--snapshot-raw", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    folder = (REPO / INTENT).parent
    intent = exact_failure(*(read(folder / name, 8192) for name in
                             ("intent.json", "transport.json", "stdout.private", "stderr.private")))
    absence = exact_absence(read(args.absence, 16384), args.absence_sha256)
    now = time.monotonic()
    post, expires = clean_after(absence, json.loads(read(args.snapshot)), read(args.snapshot_raw),
                                now, time.time() - now, TRAINING80_LEDGER)
    evidence = dict(absence_SHA256=args.absence_sha256, intent_SHA256=INTENT_SHA,
                    transport_SHA256=TRANSPORT_SHA, stderr_SHA256=STDERR_SHA, post=post)
    result = dict(request_id=REQUEST, intent=INTENT, retired_prelaunch=True, failure_phase="PRE_SRUN_SOURCE_INDEX",
                  measurement_terminal=False, replay_allowed=False, launch_attempted=False, transport_attempted=True,
                  matching_accounting_rows=0, placement_by_parent=PLACEMENT, evidence=evidence,
                  evidence_SHA256=digest(evidence))
    if args.apply:
        queue = Queue(QUEUE_DB)
        with closing(queue.db):
            state = retire(queue, intent, result, expires)
    else:
        with closing(sqlite3.connect(QUEUE_DB.as_uri() + "?mode=ro", uri=True)) as db:
            prior = row_state(db, intent)
        state = "ALREADY_RETIRED" if prior == result else "READY_TO_RETIRE"
        need(prior is None or prior == result, "original retirement already recorded with different evidence")
    print(json.dumps(dict(run=RUN, request=REQUEST, state=state, result=result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
