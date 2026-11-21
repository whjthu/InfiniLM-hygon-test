"""Hold the shared O01 slot before invoking the frozen Deep inverse RoPE controller once."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import time

from dispatch_glm474_fresh_kv_cpu_once import PROJECT, write_once
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, REPO, hex_value, need, pin_file, queue_holds, read_original_holds
from tuning_queue import DIRECT_STAGE, Queue, canonical, digest
from verify_training80_manifest import verify_package_members


SOURCE = "code/bw1000/Deep-C4-causal-consumer-r483"
MANIFEST = dict(path=SOURCE + "/inverse-output-rope-CPU-v1/SOURCE_FILES.json", bytes=759,
                SHA256="fb0edd19695838a6df632e88ae10351569362de28ed11067848bb909b3aa7465")
AUDIT = dict(path=SOURCE + "/inverse-output-rope-terminal-audit-v1/SOURCE_FILES.json", bytes=390,
             SHA256="9bb82cf742fa643ccc94697862550074cc167a67ccf5ac734675426aee63ce1d")
CLAIM = dict(path="coordination/tasks/deep-inverse-output-rope-gfx936-host-compile-v1.json", bytes=2052,
             SHA256="96d5da415b35a3f2b615667103ebdc897bdd56a11cf27fb3315ac038388a536b")
HOST = "tools/control_deep_inverse_rope_compile.py"
MEMBER = "tools/read_training_slurm_membership.py"
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
HOST_FILES = {
    HOST: "05e1e3258fab050902650bd9ffc0e71ff9a99badf3dd4af7d6cb0e2ad1980489",
    MEMBER: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py":
        "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
LABEL = "Deep-inverse-output-rope-gfx936-CPU-v1"
REQUEST = "ed7f8bd81af545252986bd3d25810f73"
PARENT, NODE, ZONE = "1103606", "f12r2n12", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
IDENT = "deep-inverse-output-rope-gfx936-host-compile-v1"
CASE = "O01.deep-inverse-output-rope-compile"
INTENT = f"local-only/{LABEL}-original-dispatch/intent.json"
TARGET = PROJECT / "results" / (LABEL + "-controller")
DISPATCHER = "tools/dispatch_deep_inverse_rope_compile_once.py"


def backlog_identity():
    return dict(id=IDENT, case=CASE, kind="validation", state="blocked", cpu_prerequisite=True, cpu_request=REQUEST,
                placement={NODE: 0}, attempt_receipts=[INTENT], evidence=CLAIM)


def scope(repo):
    raw = pin_file(repo, CLAIM, ("coordination/tasks/",))
    claim = json.loads(raw)
    need(len(raw) == CLAIM["bytes"] and claim["request"] == REQUEST and claim["label"] == LABEL
         and claim["placement"] == PLACEMENT and claim["wall_seconds"] == 300 and claim["planned_device_translation_units"] == 1
         and claim["controller_result"] == str(TARGET.relative_to(PROJECT) / "accounting-result.json")
         and claim["host_files"] == HOST_FILES and claim["package_manifest"] == MANIFEST["path"]
         and claim["package_manifest_SHA256"] == MANIFEST["SHA256"] and claim["CPU_GO"] is True
         and claim["GPU_GO"] is claim["Model_GO"] is claim["replay_allowed"] is False, "original Deep RoPE claim differs")
    for path, pin in HOST_FILES.items():
        pin_file(repo, dict(path=path, SHA256=pin), ("tools/", "code/bw1000/", "coordination/training/"))
    names = {DISPATCHER, CLAIM["path"], *HOST_FILES}
    for manifest in (MANIFEST, AUDIT):
        verify_package_members(repo, manifest)
        path = repo / manifest["path"]
        names.add(manifest["path"])
        names.update((path.parent / name).relative_to(repo).as_posix()
                     for name in json.loads(path.read_bytes())["files_SHA256"])
    return claim, names


def original_argv(remote_repository):
    root = PurePosixPath(remote_repository)
    need(root.is_relative_to(PROJECT) and root.as_posix() == remote_repository and ".." not in root.parts
         and all(part.isascii() and all(ch.isalnum() or ch in "-_." for ch in part) for part in root.parts[1:]),
         "expected a canonical target checkout within the project")
    # Isolated Python drops the script directory. Check the sibling before making it importable.
    bootstrap = (
        "import hashlib,runpy,sys\n"
        "from pathlib import Path\n"
        "entry,sibling=map(Path,sys.argv[1:3])\n"
        "for path,pin in zip((entry,sibling),sys.argv[3:5],strict=True):\n"
        "    if path.resolve(strict=True)!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:\n"
        "        raise ValueError('frozen Deep controller or membership helper differs')\n"
        "sys.path.append(str(entry.parent))\n"
        "sys.argv=[str(entry),'--claim-sha256',sys.argv[5]]\n"
        "runpy.run_path(str(entry),run_name='__main__')\n"
    )
    return ("/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-u", "-c", bootstrap,
            str(root / HOST), str(root / MEMBER), HOST_FILES[HOST], HOST_FILES[MEMBER], CLAIM["SHA256"])


def claim_local(queue, repo, board, approved_board, record):
    local = repo / "local-only"
    intent = repo / INTENT
    need(local.resolve(strict=True) == local and intent.resolve().is_relative_to(local),
         "original intent must stay in canonical local-only")
    groups = json.loads((repo / LEDGER).read_bytes())["groups"]
    need(NODE in groups[ZONE]["allocations"][PARENT], "original Deep node is outside O01")
    nodes = {node for group in groups.values() for allocation in group["allocations"].values() for node in allocation}
    run = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
    spec = dict(case=CASE, kind="validation", CPU=4, DCU=0, scope_SHA256=CLAIM["SHA256"])
    candidate = digest(dict(spec, request=REQUEST))
    ticket = digest(dict(request=REQUEST, parent=PARENT, node=NODE, source_manifest_SHA256=MANIFEST["SHA256"]))
    handle = dict(request_id=REQUEST, intent=INTENT, parent=PARENT, nodes=[NODE], direct_kind="validation",
                  placement_by_parent={PARENT: {NODE: 0}}, handoff_SHA256=ticket)
    db = queue.db
    db.execute("BEGIN IMMEDIATE")
    try:
        raw = board.read_bytes()
        need(raw == approved_board, "shared backlog changed after the publication check")
        jobs = json.loads(raw)["items"]
        rows = [row for row in jobs if row["id"] == IDENT]
        need(len(rows) == 1 and all(rows[0].get(key) == value for key, value in backlog_identity().items()),
             "register the exact Deep original-intent row in the shared backlog")
        scopes = (intent.parent, local / LABEL, local / (LABEL + "-controller"), local / (LABEL + "-intent.json"))
        prior = db.execute("SELECT 1 FROM runs WHERE id=? OR instr(coalesce(handle,''),?)>0 OR instr(binding,?)>0",
                           (run, REQUEST, REQUEST)).fetchone()
        need(prior is None and not any(path.exists() or path.is_symlink() for path in scopes),
             "original Deep request already has a local scope; reconcile it")
        original, _, _ = read_original_holds(repo, jobs, nodes)
        need(not any(NODE in row["nodes"] for row in (*queue_holds(db), *original.values())),
             "another original request holds the Deep O01 node")
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
    """The coordinator supplies the existing finite argv callback."""
    need(hex_value(publication, 40) and callable(transport), "published revision and finite transport required")
    repo, board, database = (path.resolve(strict=True) for path in (repo, backlog, database))
    need(board == repo / BACKLOG.relative_to(REPO) and database == repo / QUEUE_DB.relative_to(REPO),
         "expected the shared planning database and backlog")
    argv = original_argv(remote_repository)
    frozen, names = scope(repo)
    approved_board = board.read_bytes()
    check_publication(repo, publication, sorted(names | {board.relative_to(repo).as_posix()}))
    cap = 320
    record = dict(request=REQUEST, label=LABEL, publication=publication, placement=PLACEMENT, kind="validation",
                  claim_SHA256=CLAIM["SHA256"], host_entry_SHA256=HOST_FILES[HOST],
                  source_manifest_SHA256=MANIFEST["SHA256"], terminal_reader_manifest_SHA256=AUDIT["SHA256"],
                  target_intent=str(TARGET / "intent.json"), target_controller=str(PROJECT / frozen["controller_result"]),
                  local_intent=INTENT, argv=argv, argv_SHA256=digest(argv), claimed_UTC=datetime.now(timezone.utc).isoformat(),
                  transport_timeout_seconds=cap, remote_outcome="UNKNOWN", replay_allowed=False,
                  counts_toward_training_reserve=False)
    queue = Queue(database)
    with closing(queue.db):
        run, handle, intent = claim_local(queue, repo, board, approved_board, record)
        result = dict(request=REQUEST, queue_run=run, local_intent=INTENT, remote_outcome="UNKNOWN",
                      transport_returncode=None, terminal_verified=False, replay_allowed=False)
        try:
            with intent.with_name("stdout.private").open("xb") as stdout, intent.with_name("stderr.private").open("xb") as stderr:
                rc = transport(argv, stdout, stderr, cap).returncode
            need(type(rc) is int, "finite transport did not return an exit code")
            result["transport_returncode"] = rc
        except Exception as error:
            result["transport_error"] = type(error).__name__
        write_once(intent.with_name("transport.json"), result)
        queue.submission(run, handle, unknown=True)
    return result


if __name__ == "__main__":
    print(json.dumps(backlog_identity(), indent=2, allow_nan=False))
