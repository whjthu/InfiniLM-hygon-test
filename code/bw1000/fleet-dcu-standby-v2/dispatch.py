"""Build a standby dispatch from one fresh, authenticated device receipt."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))
from prepare_training80_vggt_admission import published
from report_training80_backlog import BACKLOG, QUEUE_DB, fleet_view, plan, read_queue_holds
from training_dispatch_allocation import REPLACEMENT_LEDGER
from tuning_queue import SNAPSHOT_MAX_AGE_S


def build(receipt, publication, request, seconds, *, now=None, wall=None):
    if (not re.fullmatch(r"[a-f0-9]{32}", request) or not re.fullmatch(r"[a-f0-9]{40}", publication)
            or seconds not in (120, 10800)):
        raise ValueError("Invalid standby request identity, publication or budget")
    now = time.monotonic() if now is None else now
    wall = time.time() if wall is None else wall
    holds = read_queue_holds(QUEUE_DB)
    backlog = json.loads(BACKLOG.read_bytes())
    report = plan(receipt, backlog, holds=holds, now=now, boot_anchor=wall - now)
    _, facts, fresh = fleet_view(receipt, now, wall - now, REPLACEMENT_LEDGER)
    if not fresh:
        raise ValueError("A new independent device admission is required")
    protected = set(report["queue_held_nodes"])
    protected.update(node for row in report["original_launch_holds"].values() for node in row["nodes"])
    protected.update(node for row in report["rows"] if row["dispatch_state"] == "proposed_for_root"
                     and row["kind"] == "training" for node in row["placement"])
    eligible = {
        node: {"parent": row["parent"], "pci": row["PCIs"]}
        for node, row in receipt["nodes"].items()
        if node not in protected and facts[node]["admission"] == "clear" and row["active_steps"] == []
    }
    if seconds == 120:
        selected = {}
        for parent in ("1103606", "1194900"):
            options = sorted((node for node, row in eligible.items() if row["parent"] == parent),
                             key=lambda node: (facts[node]["zone"] != "O01", node))
            if options:
                selected[options[0]] = eligible[options[0]]
    else:
        selected = eligible
    if not selected:
        raise ValueError("No standby node is currently admitted")
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    worker = (HERE / "worker.py").read_bytes()
    controller = (HERE / "controller.py").read_bytes()
    if sha256(worker).hexdigest() != claim["worker_SHA256"] or sha256(controller).hexdigest() != claim["controller_SHA256"]:
        raise ValueError("Frozen standby source differs from its claim")
    original = receipt["local_receipt"]
    data = dict(request=request, publication=publication, seconds=seconds, nodes=selected, worker=worker,
                claim=claim, admission_SHA256=original["payload_SHA256"])
    body = ("INPUT=" + repr(data) + "\n").encode() + controller
    intent = dict(request=request, publication=publication, receipt_request=original["request_id"],
                  payload_SHA256=sha256(body).hexdigest(), requested_cards=8 * len(selected),
                  nodes=sorted(selected), protected_nodes=sorted(protected), replay_allowed=False,
                  synthetic=True, ready_training=report["outstanding_training_ready"])
    cutoff = original["received_monotonic"] + SNAPSHOT_MAX_AGE_S - original["roundtrip_s"]
    return body, intent, cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("publication")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    outputs = args.destination.resolve()
    if not outputs.is_relative_to(REPO / "local-only"):
        raise ValueError("Standby transport artifacts must stay in local-only")
    published(REPO, args.publication, [
        (HERE / name).relative_to(REPO).as_posix() for name in ("CLAIM.json", "worker.py", "controller.py", "dispatch.py")
    ])
    body, intent, cutoff = build(json.loads(args.receipt.read_bytes()), args.publication, uuid4().hex,
                                 120 if args.pilot else 10800)
    if not time.monotonic() < cutoff:
        raise ValueError("Device admission expired before dispatch artifacts were created")
    outputs.mkdir(mode=0o700)
    with (outputs / "entry.py").open("xb") as stream:
        stream.write(body)
    with (outputs / "intent.json").open("x", encoding="utf8") as stream:
        json.dump(dict(intent, latest_send_monotonic=cutoff), stream, sort_keys=True)
    print(json.dumps(intent, sort_keys=True))


if __name__ == "__main__":
    main()
