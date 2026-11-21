"""Reserve Kimi r461's original zero-GRES score request before its finite transport."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import time

from dispatch_glm474_fresh_kv_cpu_once import PROJECT, write_once
from prepare_training80_vggt_admission import published
from report_training80_backlog import (
    BACKLOG, QUEUE_DB, REPO, hex_value, need, pin_file, queue_holds, read_original_holds,
)
from tuning_queue import DIRECT_STAGE, Queue, canonical, digest
from verify_training80_manifest import verify_package_members


PACKAGE = "code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-wave-host-compile-v1"
WORKER = "code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-wave-compile-CPU-v1"
HOST = dict(path=PACKAGE + "/host_once.py", SHA256="ad70c04a2279a2eebf7fd51e07b714b890bb430d259b60aa114ca55c2d557a08")
CLAIM = dict(path=PACKAGE + "/CLAIM.json", SHA256="2893775796879edffbc468be1115cbdcdb8b6bc760f50dbcea6d68c284d4b6e1")
MANIFEST = dict(path=PACKAGE + "/SOURCE_FILES.json",
                SHA256="192dce1d3a5a56f369d0f536105f254dccf1f0e07e8cf4649087ef670c776a00")
WORKER_MANIFEST = dict(path=WORKER + "/SOURCE_FILES.json",
                       SHA256="e55041e1f3b5b48c776ce0a3983fce550559dbdc0daa4a50e4dacd4319b13efe")
RENDERED_SHA = "d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6"
TASK = dict(path="coordination/tasks/kimi-r461-output-residual-wave-host-compile-v1.json", bytes=1491,
            SHA256="126dc0bd23d5d6f64ee21a065a775be84d05ac638a0a42291546d3e59e1327e7")
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
SHARED = {
    LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    "tools/read_training_slurm_membership.py": "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py":
        "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
LABEL = "Kimi461-output-residual-wave-score-gfx936-CPU-v1"
REQUEST = "e5ce43bacc53bc64c75a74005e5e249c"
PARENT, NODE, ZONE = "1103606", "f12r2n11", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
IDENT = "kimi-r461-output-residual-wave-score-gfx936-host-compile-v1"
CASE = "I01.kimi-r461-output-residual-wave-score-compile"
INTENT = f"local-only/{LABEL}-original-dispatch/intent.json"
TARGET = PROJECT / "results" / (LABEL + "-controller")
DISPATCHER = "tools/dispatch_kimi_r461_output_residual_wave_once.py"


def backlog_identity():
    return dict(id=IDENT, case=CASE, kind="validation", state="blocked", cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, attempt_receipts=[INTENT], evidence=TASK)


def frozen_inputs(repo):
    members = {DISPATCHER, TASK["path"]}
    for manifest, required in ((MANIFEST, (HOST, CLAIM)), (WORKER_MANIFEST, ())):
        verify_package_members(repo, manifest, required_files=required)
        path = repo / manifest["path"]
        entries = json.loads(path.read_bytes())["files_SHA256"]
        members.update((path.parent / name).relative_to(repo).as_posix() for name in entries)
        members.add(manifest["path"])
    claim = json.loads(pin_file(repo, CLAIM, ("code/bw1000/",)))
    raw = pin_file(repo, TASK, ("coordination/tasks/",))
    task = json.loads(raw)
    need(len(raw) == TASK["bytes"] and task["request"] == REQUEST and task["case"] == CASE
         and task["backlog_id"] == IDENT and task["placement_candidate"] == PLACEMENT
         and task["source_manifest"]["SHA256"] == MANIFEST["SHA256"] and task["claim"]["SHA256"] == CLAIM["SHA256"]
         and claim["request"] == REQUEST and claim["label"] == LABEL and claim["placement"] == PLACEMENT
         and claim["wall_seconds"] == 300 and claim["planned_device_TUs"] == 2
         and claim["rendered_manifest_SHA256"] == RENDERED_SHA
         and claim["frozen_package_manifest_SHA256"] == WORKER_MANIFEST["SHA256"]
         and claim["GPU_GO"] is claim["Model_GO"] is claim["replay_allowed"] is False,
         "original Kimi score CPU scope differs")
    plan = json.loads((repo / WORKER / "BUILD_PLAN.json").read_bytes())
    for name, sha in {**plan["repo_source_SHA256"], **SHARED}.items():
        pin_file(repo, dict(path=name, SHA256=sha), ("code/bw1000/", "tools/", "results/bw1000/", "coordination/training/"))
        members.add(name)
    return members


def original_argv(remote_repository):
    root = PurePosixPath(remote_repository)
    need(root.is_relative_to(PROJECT) and root.as_posix() == remote_repository and ".." not in root.parts
         and all(part.isascii() and all(ch.isalnum() or ch in "-_." for ch in part) for part in root.parts[1:]),
         "expected a canonical target checkout within the project")
    return ("/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-u", str(root / HOST["path"]),
            "--claim-sha256", CLAIM["SHA256"])


def claim(queue, repo, backlog, approved_board, record):
    intent = repo / INTENT
    local = repo / "local-only"
    need(local.resolve(strict=True) == local and intent.resolve().is_relative_to(local),
         "compile intent escaped the canonical local-only directory")
    group = json.loads((repo / LEDGER).read_bytes())["groups"]
    need(NODE in group[ZONE]["allocations"][PARENT], "original Kimi node is outside approved O01")
    nodes = {node for entry in group.values() for allocation in entry["allocations"].values() for node in allocation}
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    spec = dict(case=CASE, kind="validation", CPU=4, DCU=0, scope_SHA256=CLAIM["SHA256"])
    candidate = digest(dict(spec, request=REQUEST))
    ticket = digest(dict(request=REQUEST, parent=PARENT, node=NODE, source_manifest_SHA256=MANIFEST["SHA256"]))
    handle = dict(request_id=REQUEST, intent=INTENT, nodes=[NODE], parent=PARENT, direct_kind="validation",
                  placement_by_parent={PARENT: {NODE: 0}}, handoff_SHA256=ticket)
    db = queue.db
    db.execute("BEGIN IMMEDIATE")
    try:
        raw = backlog.read_bytes()
        need(raw == approved_board, "shared backlog changed after the published scope check")
        jobs = json.loads(raw)["items"]
        rows = [row for row in jobs if row["id"] == IDENT]
        need(len(rows) == 1 and all(rows[0].get(key) == value for key, value in backlog_identity().items()),
             "register the exact Kimi original-intent row in the shared backlog")
        need(db.execute("SELECT 1 FROM runs WHERE id=?", (run,)).fetchone() is None
             and not intent.parent.exists() and not intent.parent.is_symlink(),
             "original Kimi request already claimed; reconcile it")
        original, _, _ = read_original_holds(repo, jobs, nodes)
        need(not any(NODE in row["nodes"] for row in (*queue_holds(db), *original.values())),
             "another original request holds the Kimi O01 node")
        intent.parent.mkdir()
        binding = dict(record, queue_run=run)
        write_once(intent, binding)
        db.execute("INSERT INTO candidates(id,spec) VALUES(?,?)", (candidate, canonical(spec).decode()))
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)", (run, candidate, DIRECT_STAGE, "DISPATCHING",
                   canonical(binding).decode(), canonical([NODE]).decode(), canonical(handle).decode(), None, time.time()))
        queue.event(run, "reserved", dict(parent=PARENT, nodes=[NODE], dispatch_ticket=ticket))
        queue.event(run, "dispatch_claimed", dict(handle=handle))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return run, handle, intent


def dispatch(publication, remote_repository, transport, *, repo=REPO, backlog=BACKLOG, database=QUEUE_DB,
             check_publication=published):
    """The coordinator injects its authorized finite argv transport. No SSH service is started."""
    need(hex_value(publication, 40) and callable(transport), "published revision and existing finite transport required")
    repo = repo.resolve(strict=True)
    backlog = backlog.resolve(strict=True)
    database = database.resolve(strict=True)
    need(backlog == repo / BACKLOG.relative_to(REPO), "expected the original shared backlog")
    need(database == repo / QUEUE_DB.relative_to(REPO), "expected the original shared planning database")
    argv = original_argv(remote_repository)
    members = frozen_inputs(repo)
    approved_board = backlog.read_bytes()
    check_publication(repo, publication, sorted(members | {backlog.relative_to(repo).as_posix()}))
    cap = 320
    record = dict(request=REQUEST, label=LABEL, publication=publication, placement=PLACEMENT, kind="validation",
                  host_entry_SHA256=HOST["SHA256"], claim_SHA256=CLAIM["SHA256"],
                  target_manifest_SHA256=RENDERED_SHA, target_intent=str(TARGET / "intent.json"),
                  target_host_result=str(TARGET / "accounting-result.json"), argv=argv, argv_SHA256=digest(argv),
                  local_intent=INTENT, claimed_UTC=datetime.now(timezone.utc).isoformat(), transport_timeout_seconds=cap,
                  remote_outcome="UNKNOWN", replay_allowed=False, counts_toward_training_reserve=False)
    queue = Queue(database)
    with closing(queue.db):
        run, handle, intent = claim(queue, repo, backlog, approved_board, record)
        result = dict(request=REQUEST, local_intent=INTENT, queue_run=run, remote_outcome="UNKNOWN",
                      transport_returncode=None, terminal_verified=False, replay_allowed=False)
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
    print(json.dumps(backlog_identity(), ensure_ascii=False, indent=2, allow_nan=False))
