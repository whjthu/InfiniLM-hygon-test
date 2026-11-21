"""Dispatch the one published checkpoint-20 continuation through the shared queue."""

import base64
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import recovery_binding as binding
from render import BASE, runtime


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = "1103606"
NODE = "f13r1n10"
CASE = "T01.domain-usgs-from-original-checkpoint20"
INTENT = f"local-only/{binding.LABEL}-original-dispatch/intent.json"
TIMEOUT = 2400


def package():
    files = runtime(HERE)
    manifest = (HERE / "SOURCE_FILES.json").read_bytes()
    members = {name: {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()} for name, raw in files.items()}
    binding.require(json.loads(manifest) == {"files": members}, "rendered runtime differs from the published manifest")
    return files, manifest


def body(publication, files, manifest, epoch):
    payload = dict(publication=publication, request=binding.REQUEST, epoch=epoch, deadline=epoch + TIMEOUT,
                   files={name: base64.b64encode(raw).decode()
                          for name, raw in dict(files, **{"SOURCE_FILES.json": manifest}).items()},
                   source_manifest_SHA256=sha256(manifest).hexdigest(),
                   claim_SHA256=sha256(files["CLAIM.json"]).hexdigest())
    return ("PAYLOAD=" + repr(payload) + "\n").encode("ascii") + files["receive.py"]


def admit(snapshot, ledger, fleet_view):
    current = time.monotonic()
    _, devices, fresh = fleet_view(snapshot, current, time.time() - current, ledger)
    row = snapshot["nodes"][NODE]
    device = devices[NODE]
    binding.require(fresh and device["admission"] == "clear" and device["parent"] == PARENT
                    and row["zone"] == "T01" and row["kfd_process_count"] == 0
                    and row["own_device_fd_process_count"] == 0 and len(set(row["PCIs"])) == 8,
                    "original T01 node lacks current PCI, KFD and process clearance")
    binding.require(binding.PORT in (row.get("ports_checked") or [])
                    and binding.PORT not in row["occupied_ports"], "original rendezvous port was not cleared")
    return snapshot["local_receipt"]


def dispatch(publication, snapshot, preflight, transport):
    from prepare_training80_dispatch_handoff import reserve_handoff
    from prepare_training80_vggt_admission import published
    from report_training80_backlog import BACKLOG, QUEUE_DB, TRAINING80_LEDGER, fleet_view, queue_holds, read_original_holds
    from tuning_queue import DIRECT_STAGE, SNAPSHOT_MAX_AGE_S, Queue, digest

    files, manifest = package()
    approved = BACKLOG.read_bytes()
    original = json.loads((BASE / "SOURCE_FILES.json").read_bytes())["files"]
    sources = {*[path.relative_to(REPO).as_posix() for path in HERE.glob("*.py")],
               *[(HERE / name).relative_to(REPO).as_posix() for name in ("CLAIM.json", "SOURCE_FILES.json")],
               *[(BASE / name).relative_to(REPO).as_posix() for name in original],
               (BASE / "SOURCE_FILES.json").relative_to(REPO).as_posix(),
               BACKLOG.relative_to(REPO).as_posix()}
    published(REPO, publication, sorted(sources))
    proof = json.loads(preflight)
    claim = json.loads(files["CLAIM.json"])
    binding.require(sha256(preflight).hexdigest() == claim["recovery_preflight_SHA256"]
                    and proof["request"] == binding.REQUEST and proof["read_only"] is True
                    and proof["original_fresh_directory_absence"] is True
                    and proof["only_planned_steps"] == claim["steps"],
                    "actual original recovery asset preflight differs")

    path = REPO / INTENT
    queue = Queue(QUEUE_DB)
    with closing(queue.db) as db:
        # The shared transaction and unique intent are acquired before transport.
        run = digest(dict(stage=DIRECT_STAGE, request=binding.REQUEST))
        db.execute("BEGIN IMMEDIATE")
        try:
            binding.require(BACKLOG.read_bytes() == approved, "published backlog changed before dispatch")
            retained = queue_holds(db)
            old, _, _ = read_original_holds(REPO, json.loads(approved)["items"], snapshot["nodes"], retained)
            binding.require(not path.parent.exists() and not path.parent.is_symlink()
                            and not any(NODE in x["nodes"] for x in (*retained, *old.values()))
                            and db.execute("SELECT id FROM runs WHERE id=?", (run,)).fetchone() is None,
                            "the original node or request is already held")
            device = admit(snapshot, TRAINING80_LEDGER, fleet_view)
            epoch = time.time()
            request = body(publication, files, manifest, epoch)
            record = dict(case=CASE, kind="training", request=binding.REQUEST,
                          workload_SHA256=sha256(manifest).hexdigest(),
                          placement_by_parent={PARENT: {NODE: 8}},
                          snapshot_payload_SHA256=device["payload_SHA256"],
                          invocation={"kind": "existing_authenticated_transport"},
                          launcher={"path": Path(__file__).relative_to(REPO).as_posix()},
                          expires_local_monotonic=device["received_monotonic"] + SNAPSHOT_MAX_AGE_S - device["roundtrip_s"],
                          no_replay=True, preflight_SHA256=sha256(preflight).hexdigest(),
                          source_manifest_SHA256=sha256(manifest).hexdigest(), publication_commit=publication)
            record["handoff_SHA256"] = digest(record)
            handle = reserve_handoff(queue, run, record, INTENT)["handle"]
            path.parent.mkdir(mode=0o700)
            with path.open("x", encoding="utf-8") as out:
                json.dump(dict(record, epoch=epoch, body_SHA256=sha256(request).hexdigest(), queue_run=run), out)
                out.flush()
                os.fsync(out.fileno())
            db.commit()
        except BaseException:
            db.rollback()
            raise
        result = dict(request=binding.REQUEST, queue_run=run, transport_attempted=False,
                      remote_outcome="NOT_ATTEMPTED", replay_allowed=False)
        retire = False
        try:
            with path.with_name("stdout.private").open("xb") as stdout, path.with_name("stderr.private").open("xb") as stderr:
                try:
                    binding.require(admit(snapshot, TRAINING80_LEDGER, fleet_view) == device, "device receipt changed")
                except ValueError:
                    retire = True
                    raise
                queue.submission(run, handle, unknown=True)
                result.update(transport_attempted=True, remote_outcome="UNKNOWN")
                result["transport_rc"] = transport(request, stdout, stderr, epoch + TIMEOUT - time.time() - 8).returncode
        except Exception as error:
            result["error_type"] = type(error).__name__
        receipt = path.with_name("transport.json")
        with receipt.open("x", encoding="utf-8") as out:
            json.dump(result, out)
            out.flush()
            os.fsync(out.fileno())
        if retire:
            queue.retire_unattempted_direct(run, handle, sha256(receipt.read_bytes()).hexdigest())
    return result
