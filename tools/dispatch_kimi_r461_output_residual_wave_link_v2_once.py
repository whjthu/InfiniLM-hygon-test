"""Reserve and hand off the audited Kimi wave link through the existing transport."""

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import PurePosixPath
import time

from dispatch_glm474_fresh_kv_cpu_once import PROJECT, write_once
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, REPO, hex_value, need, pin_file, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


PACKAGE = "code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-wave-component-link-CPU-v2"
TASK = "coordination/tasks/kimi-r461-output-residual-wave-component-link-cpu-v2.json"
ENTRY = PACKAGE + "/host_once.py"
MANIFEST = PACKAGE + "/SOURCE_FILES.json"
DISPATCHER = "tools/dispatch_kimi_r461_output_residual_wave_link_v2_once.py"
IDENT = "kimi-r461-output-residual-wave-component-link-cpu-v2"
CASE = "I01.kimi-r461-output-residual-wave-component-link-v2"
PARENT, NODE, ZONE = "1103606", "f12r2n13", "O01"
LABEL = "Kimi461-output-residual-wave-component-link-CPU-v2"
SCORE_PIN = "4f961a007757591d3c918f84e6f07753d6b1fbd6210a0f6f139093a21e810ee7"
REQUEST = sha256(f"{LABEL}:{PARENT}:{NODE}:{SCORE_PIN}".encode()).hexdigest()[:32]
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
INTENT = f"local-only/{LABEL}-original-dispatch/intent.json"
TARGET = PROJECT / "results" / (LABEL + "-controller")


def frozen_inputs(repo):
    raw = (repo / TASK).read_bytes()
    task = json.loads(raw)
    scope = dict(request=REQUEST, label=LABEL, placement=PLACEMENT, case=CASE, backlog_id=IDENT,
                 CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False, host_TUs=1, device_TUs=0,
                 score_receipt_SHA256=SCORE_PIN, score_compile_step="1103606.637", score_compile_post="1103606.639",
                 wall_seconds=300)
    need(all(type(task[key]) is type(value) and task[key] == value for key, value in scope.items()),
         "original Kimi link scope differs")
    manifest = dict(path=MANIFEST, SHA256=task["source_manifest_SHA256"])
    members = json.loads(pin_file(repo, manifest, ("code/bw1000/",)))["files_SHA256"]
    need(ENTRY in members and len(members) == 8, "missing frozen Kimi link controller members")
    for name, pin in members.items():
        pin_file(repo, dict(path=name, SHA256=pin), ("code/bw1000/", "tools/", "coordination/training/", "results/bw1000/"))
    return dict(path=TASK, bytes=len(raw), SHA256=sha256(raw).hexdigest()), {TASK, MANIFEST, DISPATCHER, *members}


def backlog_identity(task):
    return dict(id=IDENT, case=CASE, kind="validation", state="blocked", cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, attempt_receipts=[INTENT], evidence=task)


def original_argv(remote_repo, claim_sha):
    root = PurePosixPath(remote_repo)
    need(root.is_relative_to(PROJECT) and root.as_posix() == remote_repo and ".." not in root.parts
         and all(part.isascii() and all(ch.isalnum() or ch in "-_." for ch in part) for part in root.parts[1:]),
         "expected a canonical target checkout within the project")
    return ("/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-u", str(root / ENTRY),
            "--claim-sha256", claim_sha)


def claim(queue, repo, backlog, board, task, record, snapshot):
    intent = repo / INTENT
    local = repo / "local-only"
    need(local.resolve(strict=True) == local and intent.resolve().is_relative_to(local),
         "Kimi link intent escaped local-only")
    db = queue.db
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    db.execute("BEGIN IMMEDIATE")
    try:
        need(backlog.read_bytes() == board, "shared backlog changed after frozen scope verification")
        rows = [row for row in json.loads(board)["items"] if row["id"] == IDENT]
        need(len(rows) == 1 and all(rows[0].get(key) == value for key, value in backlog_identity(task).items()),
             "register the exact zero-card Kimi link in the shared backlog")
        need(db.execute("SELECT 1 FROM runs WHERE id=?", (run,)).fetchone() is None
             and not intent.parent.exists() and not intent.parent.is_symlink(),
             "original Kimi link already claimed; reconcile it")
        report = plan(snapshot, json.loads(board), repo=repo, holds=queue_holds(db))
        zone = report["admission_by_zone_and_parent"][ZONE][PARENT]
        need(report["fresh_device_receipt"] is True and NODE in zone.get("clear", []),
             "fresh full-device-clear O01 node needed for the zero-card link")
        receipt = snapshot["local_receipt"]
        packet = dict(record, case=CASE, kind="validation", request=REQUEST, workload_SHA256=task["SHA256"],
                      placement_by_parent={PARENT: {NODE: 0}}, invocation=None, launcher=ENTRY,
                      snapshot_request=receipt["request_id"], snapshot_payload_SHA256=receipt["payload_SHA256"],
                      snapshot_raw_SHA256=receipt["raw_response_SHA256"], boot_epoch_anchor=receipt["boot_epoch_anchor"],
                      expires_local_monotonic=receipt["received_monotonic"] + SNAPSHOT_MAX_AGE_S - receipt["roundtrip_s"])
        packet["handoff_SHA256"] = digest(packet)
        claim = reserve_handoff(queue, run, packet, INTENT)
        need(time.monotonic() <= packet["expires_local_monotonic"], "fresh O01 receipt expired before claim")
        intent.parent.mkdir()
        write_once(intent, dict(packet, queue_run=run))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return run, claim["handle"], intent


def dispatch(publication, remote_repo, snapshot, transport, *, repo=REPO, backlog=BACKLOG, database=QUEUE_DB,
             check_publication=published):
    """Use the approved argv transport supplied by the root coordinator."""
    need(hex_value(publication, 40) and callable(transport), "published revision and existing transport required")
    repo, backlog, database = (path.resolve(strict=True) for path in (repo, backlog, database))
    need(backlog == repo / BACKLOG.relative_to(REPO) and database == repo / QUEUE_DB.relative_to(REPO),
         "expected the original shared backlog and queue")
    task, members = frozen_inputs(repo)
    board = backlog.read_bytes()
    check_publication(repo, publication, sorted(members | {backlog.relative_to(repo).as_posix()}))
    argv = original_argv(remote_repo, task["SHA256"])
    cap = 320
    record = dict(label=LABEL, publication=publication, placement=PLACEMENT, claim_SHA256=task["SHA256"],
                  argv=argv, argv_SHA256=digest(argv), local_intent=INTENT, target_intent=str(TARGET / "intent.json"),
                  target_host_result=str(TARGET / "accounting-result.json"), remote_outcome="UNKNOWN",
                  replay_allowed=False, counts_toward_training_reserve=False, transport_timeout_seconds=cap,
                  claimed_UTC=datetime.now(timezone.utc).isoformat())
    queue = Queue(database)
    with closing(queue.db):
        run, handle, intent = claim(queue, repo, backlog, board, task, record, snapshot)
        result = dict(request=REQUEST, local_intent=INTENT, queue_run=run, remote_outcome="UNKNOWN",
                      terminal_verified=False, replay_allowed=False, transport_returncode=None)
        try:
            with intent.with_name("stdout.private").open("xb") as stdout, intent.with_name("stderr.private").open("xb") as stderr:
                rc = transport(argv, stdout, stderr, cap).returncode
            need(type(rc) is int, "expected a finite transport exit code")
            result["transport_returncode"] = rc
        except Exception as error:
            result["transport_error"] = type(error).__name__
        write_once(intent.with_name("transport.json"), result)
        queue.submission(run, handle, unknown=True)
    return result


if __name__ == "__main__":
    task, _ = frozen_inputs(REPO)
    print(json.dumps(backlog_identity(task), indent=2))
