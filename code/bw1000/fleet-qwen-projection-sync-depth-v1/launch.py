"""Build the fixed backfill payload from an authenticated current fleet snapshot."""

import base64
import hashlib
import json
import os
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIOR = HERE.parent / "fleet-lora-isolated-backfill-v1"
PRIOR_FILES = ("admission.py", "ledger.json", "node_sample.py", "qualified_sampler.py")
FILES = ("CLAIM.json", "dispatch.py", "worker.py")
REQUEST = "92b334901da041b3a15342d7cca88639"


def exact(path, pin):
    raw = path.read_bytes()
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["SHA256"]:
        raise ValueError(f"published bundle file changed: {path.name}")
    return raw


def source():
    prior_index = json.loads((PRIOR / "SOURCE_FILES.json").read_bytes())["files"]
    index = json.loads((HERE / "SOURCE_FILES.json").read_bytes())["files"]
    if set(index) != set(FILES) | set(PRIOR_FILES):
        raise ValueError("backfill runtime source set changed")
    files = {name: exact(HERE / name, index[name]) for name in FILES}
    files.update({name: exact(PRIOR / name, prior_index[name]) for name in PRIOR_FILES})
    for name, raw in files.items():
        if len(raw) != index[name]["bytes"] or hashlib.sha256(raw).hexdigest() != index[name]["SHA256"]:
            raise ValueError("backfill and reusable runtime index differ")
    return files


def pick(snapshot, halted, current, epoch, ledger, fleet_view):
    """Choose whole clean nodes from both allocations, retaining every existing hold."""
    _, admission, fresh = fleet_view(snapshot, current, epoch - current, ledger)
    if not fresh:
        raise ValueError("no authenticated fresh fleet receipt")
    protected = set(halted)
    selected = {}
    for node, facts in admission.items():
        if node in protected or facts["admission"] != "clear":
            continue
        row = snapshot["nodes"][node]
        if (len(set(row["PCIs"])) != 8 or row['active_steps'] or
                row.get('kfd_process_count') != 0 or row.get('own_device_fd_process_count') != 0):
            protected.add(node)
            continue
        selected[node] = row["PCIs"]
    if not selected:
        raise ValueError("no whole clear nodes for this eight-card workload")
    return dict(request=REQUEST, epoch=epoch, PCI=selected, defer=sorted(protected),
                authenticated_snapshot_payload_SHA256=snapshot["local_receipt"]["payload_SHA256"])


def write_once(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())


def protected_nodes(schedule):
    """Keep both proposed work and the clean half of a training handover clear."""
    protected = set(schedule['queue_held_nodes'])
    for hold in schedule['original_launch_holds'].values():
        protected.update(hold['nodes'])
    for row in schedule['rows']:
        if (row['dispatch_state'] == 'proposed_for_root' or
                (row['kind'] == 'training' and row['dispatch_state'] == 'awaiting_original_queue_release')):
            protected.update(row['placement'])
    return protected


def ship(publication, snapshot, transport):
    """Use the existing authenticated client. Its exit status is transport only."""
    from prepare_training80_vggt_admission import published
    from report_training80_backlog import (BACKLOG, QUEUE_DB, TRAINING80_LEDGER, fleet_view,
                                           plan as schedule_backlog, read_queue_holds)
    files = source()
    local = REPO / f"local-only/fleet-qwen-projection-sync-depth-v1-original-{REQUEST}"
    sources = [*(HERE / name for name in (*FILES, "launch.py", "receive.py", "SOURCE_FILES.json")),
               *(PRIOR / name for name in (*PRIOR_FILES, "SOURCE_FILES.json"))]
    published(REPO, publication, [p.relative_to(REPO).as_posix() for p in sources])
    def select():
        backlog = json.loads(BACKLOG.read_bytes())
        holds = read_queue_holds(QUEUE_DB)
        current, wall = time.monotonic(), time.time()
        schedule = schedule_backlog(snapshot, backlog, repo=REPO, holds=holds,
                                    now=current, boot_anchor=wall - current)
        return pick(snapshot, protected_nodes(schedule), current, wall, TRAINING80_LEDGER, fleet_view)

    plan = select()
    script = (HERE / "receive.py").read_bytes()
    payload = dict(publication=publication, plan=plan, files={
        name: [hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode()]
        for name, raw in sorted(files.items())})
    code = ("PAYLOAD=" + repr(payload) + "\n").encode("ascii") + script
    local.mkdir(mode=0o700, exist_ok=False)
    write_once(local / "intent.json", dict(plan=plan, publication=publication,
                                           source_sha256=hashlib.sha256(code).hexdigest(), replay_allowed=False))
    result = dict(request=REQUEST, transport_attempted=False, transport_rc=None,
                  selected=len(plan["PCI"]), known_terminal=False)
    try:
        with (local / "stdout.private").open("xb") as out, (local / "stderr.private").open("xb") as err:
            latest = select()
            if any(latest['PCI'].get(node) != pci for node, pci in plan['PCI'].items()):
                raise ValueError('a selected node is no longer available to backfill')
            result['transport_attempted'] = True
            result["transport_rc"] = transport(code, out, err, 255).returncode
    except Exception as exc:
        result["transport_error"] = type(exc).__name__
    write_once(local / "transport.json", result)
    return result
