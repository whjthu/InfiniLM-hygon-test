"""Reserve disjoint idle shards and send one bounded weight-layout comparison."""
import base64
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time

from layout_scope import KNOWN_FOREIGN_KFD, LABEL, REQUEST, owners, shards


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE.relative_to(REPO)
PRIOR = BASE.parent / "fleet-lora-isolated-backfill-v1"
RUNTIME = ("CLAIM.json", "dispatch.py", "layout_scope.py", "worker.py")
COPIED = ("admission.py", "ledger.json", "node_sample.py", "qualified_sampler.py")
REFERENCE = Path("results/bw1000/2026-11-22/fleet608-qwen-projection-sync-depth-terminal.json")
DIRECTORY = Path("local-only") / f"{LABEL}-original-{REQUEST}"
CASE = "O01.qwen-resident-projection-weight-layout"

sys.path.insert(0, str(REPO / "tools"))
from prepare_training80_dispatch_handoff import reserve_handoff
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, TRAINING80_LEDGER, fleet_view, plan, queue_holds
from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest


def need(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, pin):
    raw = path.read_bytes()
    need((len(raw), sha256(raw).hexdigest()) == (pin["bytes"], pin["SHA256"]),
         "Pinned source changed: " + path.name)
    return raw


def source(repo):
    manifest = json.loads((repo / BASE / "SOURCE_FILES.json").read_bytes())
    index = manifest["files"]
    dependencies = json.loads((repo / PRIOR / "SOURCE_FILES.json").read_bytes())["files"]
    need(manifest["request"] == REQUEST and manifest["label"] == LABEL
         and set(dependencies) >= set(COPIED), "Frozen request or dependencies differ")
    local = {name: exact(repo / BASE / name, index[name]) for name in RUNTIME}
    reused = {name: exact(repo / PRIOR / name, index[name]) for name in COPIED}
    need(all(dependencies[name] == index[name] for name in COPIED), "Reused bytes are not the original runtime")
    return {**local, **reused}


def make_plan(snapshot, schedule, local_now, epoch, ledgers, requested=None):
    _, admission, fresh = fleet_view(snapshot, local_now, epoch-local_now, TRAINING80_LEDGER)
    need(fresh, "No authenticated fresh fleet receipt")
    protected = set(schedule["queue_held_nodes"]) | KNOWN_FOREIGN_KFD
    for hold in schedule["original_launch_holds"].values():
        protected.update(hold["nodes"])
    for row in schedule["rows"]:
        if row["dispatch_state"] == "proposed_for_root" or (
                row["kind"] == "training" and row["dispatch_state"] == "awaiting_original_queue_release"):
            protected.update(row["placement"])
    targets = set(ledgers) if requested is None else set(requested)
    need(targets <= ledgers.keys(), "Requested node outside the two owned allocations")
    selected = {}
    for node in sorted(targets-protected):
        facts = admission[node]
        if facts["admission"] != "clear":
            continue
        row = snapshot["nodes"][node]
        if (facts["parent"] == ledgers[node][0] and len(set(row["PCIs"])) == 8 and not row["active_steps"]
                and row.get("kfd_process_count") == 0 and row.get("own_device_fd_process_count") == 0):
            selected[node] = row["PCIs"]
    return dict(request=REQUEST, epoch=epoch, PCI=selected, defer=sorted(targets-selected.keys()),
                groups=shards(selected, ledgers),
                authenticated_snapshot_payload_SHA256=snapshot["local_receipt"]["payload_SHA256"])


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def reserve_wave(queue, campaign, publication, receipt, cutoff, claim_pin, directory):
    handles = {}
    for label, unit in campaign["groups"].items():
        packet = dict(candidate=label, case=CASE, kind="operator", request=unit["request"],
                      placement_by_parent={unit["parent"]:dict.fromkeys(unit["nodes"],8)},
                      publication_commit=publication, workload_SHA256=claim_pin, campaign_request=REQUEST,
                      snapshot_payload_SHA256=receipt["payload_SHA256"], snapshot_raw_SHA256=receipt["raw_response_SHA256"],
                      snapshot_request=receipt["request_id"], expires_local_monotonic=cutoff,
                      launcher=(BASE/"dispatch.py").as_posix(), invocation={"kind":"pinned_finite_code"},
                      replay_allowed=False)
        packet["handoff_SHA256"] = digest(packet)
        ident = digest(dict(stage=DIRECT_STAGE, request=unit["request"]))
        hold = reserve_handoff(queue, ident, packet, (directory/"shards"/(label+".json")).as_posix())
        handles[label] = dict(run=ident, handle=hold["handle"], packet=packet)
    return handles


def ship(publication, snapshot, transport, *, requested=None, repo=REPO, check_publication=published):
    files = source(repo)
    claim = json.loads(files["CLAIM.json"])
    need(re.fullmatch(r"[0-9a-f]{32}", REQUEST) and claim["request"] == REQUEST
         and claim["operator_GO"] is True and claim["training_GO"] is False,
         "Operative one-shot operator claim required")
    exact(repo / REFERENCE, claim["shape_source"])
    paths = [*(BASE/name for name in (*RUNTIME,"launch.py","receive.py","SOURCE_FILES.json")),
             *(PRIOR/name for name in (*COPIED,"SOURCE_FILES.json")), REFERENCE]
    check_publication(repo, publication, [path.as_posix() for path in paths])
    ledgers = owners(json.loads(files["ledger.json"]))
    receipt = snapshot["local_receipt"]
    cutoff = receipt["received_monotonic"] + SNAPSHOT_MAX_AGE_S - receipt["roundtrip_s"]
    directory = repo / DIRECTORY
    receive = (repo/BASE/"receive.py").read_bytes()
    queue = Queue(repo / QUEUE_DB.relative_to(REPO))
    with closing(queue.db):
        db = queue.db
        db.execute("BEGIN IMMEDIATE")
        try:
            current, wall = time.monotonic(), time.time()
            need(current < cutoff and not directory.exists() and not directory.is_symlink(), "Stale or original exists")
            schedule = plan(snapshot, json.loads((repo/BACKLOG.relative_to(REPO)).read_bytes()), repo=repo,
                            holds=queue_holds(db), now=current, boot_anchor=wall-current)
            campaign = make_plan(snapshot, schedule, current, wall, ledgers, requested)
            payload = dict(publication=publication, plan=campaign, files={
                name:[sha256(raw).hexdigest(),base64.b64encode(raw).decode("ascii")] for name,raw in files.items()})
            body = ("PAYLOAD=" + repr(payload) + "\n").encode("ascii") + receive
            need(len(body) <= 131072 and time.monotonic() < cutoff, "Stale or oversized one-shot payload")
            handles = reserve_wave(queue,campaign,publication,receipt,cutoff,sha256(files["CLAIM.json"]).hexdigest(),DIRECTORY)
            directory.mkdir(mode=0o700)
            (directory/"shards").mkdir(mode=0o700)
            for label, hold in handles.items():
                write_once(directory/"shards"/(label+".json"), hold)
            write_once(directory/"intent.json", dict(plan=campaign,publication=publication,body_SHA256=sha256(body).hexdigest(),
                                                      replay_allowed=False))
            write_once(directory/"original-shards.json", handles)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        result = dict(request=REQUEST, planned_nodes=len(campaign["PCI"]), planned_cards=8*len(campaign["PCI"]),
                      planned_shards=len(handles), transport_attempted=False, remote_outcome="NOT_ATTEMPTED")
        marked = set()
        try:
            with (directory/"stdout.private").open("xb") as out, (directory/"stderr.private").open("xb") as err:
                need(time.monotonic() < cutoff, "Fresh admission expired")
                for label, hold in handles.items():
                    queue.submission(hold["run"],hold["handle"],unknown=True)
                    marked.add(label)
                need(time.monotonic() < cutoff and 0 <= time.time()-campaign["epoch"] < 20,
                     "Original request expired before transport")
                result.update(transport_attempted=True,remote_outcome="UNKNOWN")
                result["transport_rc"] = transport(body,out,err,255).returncode
        except Exception as exc:
            result["transport_error"] = type(exc).__name__
        record = directory/"transport.json"
        write_once(record, result)
        if not result["transport_attempted"]:
            record_sha = sha256(record.read_bytes()).hexdigest()
            for label, hold in handles.items():
                if label not in marked:
                    queue.retire_unattempted_direct(hold["run"],hold["handle"],record_sha)
    return result
