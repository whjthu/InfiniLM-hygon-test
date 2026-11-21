"""Prepare one private Kimi score finish packet. The shared queue is read-only here."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time

from dispatch_glm474_fresh_kv_cpu_once import write_once
from dispatch_kimi_r461_output_residual_wave_once import (
    CLAIM, HOST, INTENT, LABEL, MANIFEST, NODE, PARENT, PLACEMENT, PROJECT, RENDERED_SHA, REQUEST, TARGET, ZONE,
    original_argv,
)
from prepare_training80_vggt_direct_finish import bounded
from read_training_slurm_membership import allocated_dcu, completed_numbered_step
from read_vggt_tum_bidirectional30_terminal import TERMINAL
from report_training80_backlog import QUEUE_DB, REPO, TRAINING80_LEDGER, fleet_view, hex_value, need
from tuning_queue import DIRECT_STAGE, digest


DIRECT_RUN = digest(dict(stage=DIRECT_STAGE, request=REQUEST))
ACTIVE = {"DISPATCHING", "SUBMITTED", "UNKNOWN"}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
ARMS = ("full-tree", "wave32")
PLACEMENT_BY_PARENT = {PARENT: {NODE: 0}}
TARGET_CONTAINER = PurePosixPath("/refactor") / TARGET.relative_to(PROJECT)
LINK_ACCOUNTING = TARGET_CONTAINER / "original-terminal-reconcile.json"
FINISH_NAME = "finish.json"


def claimed_run(database):
    uri = database.resolve(strict=True).as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        row = connection.execute("SELECT stage,state,binding,handle FROM runs WHERE id=?", (DIRECT_RUN,)).fetchone()
    need(row is not None and row[0] == DIRECT_STAGE and row[1] in ACTIVE, "original Kimi direct claim is not active")
    return row[1], json.loads(row[2]), json.loads(row[3])


def original_scope(claim, local_raw, target_raw):
    state, binding, handle = claim
    local, target = map(json.loads, (local_raw, target_raw))
    ticket = digest(dict(request=REQUEST, parent=PARENT, node=NODE, source_manifest_SHA256=MANIFEST["SHA256"]))
    expected_handle = dict(request_id=REQUEST, intent=INTENT, nodes=[NODE], parent=PARENT, direct_kind="validation",
                           placement_by_parent=PLACEMENT_BY_PARENT, handoff_SHA256=ticket)
    need(state in ACTIVE and binding == local
         and all(handle.get(key) == value for key, value in expected_handle.items()),
         "original Kimi reservation and intent differ")
    argv = local["argv"]
    need(isinstance(argv, list) and len(argv) == 8 and isinstance(argv[5], str)
         and argv[5].endswith("/" + HOST["path"]), "original Kimi launch argv differs")
    root = argv[5].removesuffix("/" + HOST["path"])
    need(tuple(argv) == original_argv(root) and local["argv_SHA256"] == digest(argv)
         and hex_value(local["publication"], 40) and local["request"] == REQUEST and local["queue_run"] == DIRECT_RUN
         and local["label"] == LABEL and local["kind"] == "validation" and local["placement"] == PLACEMENT
         and local["local_intent"] == INTENT and local["target_intent"] == str(TARGET / "intent.json")
         and local["target_host_result"] == str(TARGET / "accounting-result.json")
         and local["target_manifest_SHA256"] == RENDERED_SHA and local["claim_SHA256"] == CLAIM["SHA256"]
         and local["host_entry_SHA256"] == HOST["SHA256"] and local["replay_allowed"] is False,
         "original Kimi local intent does not bind the frozen target")
    epoch = target["epoch"]
    need(type(epoch) is float and math.isfinite(epoch) and epoch > 0 and target["deadline"] == epoch + 300
         and target["request"] == REQUEST and target["label"] == LABEL and target["placement"] == PLACEMENT
         and target["claim_SHA256"] == CLAIM["SHA256"] and target["source_manifest_SHA256"] == RENDERED_SHA
         and target["replay_allowed"] is False, "original Kimi target epoch or scope differs")
    return handle


def zero_gres(tres):
    resources = dict(field.split("=", 1) for field in tres.split(","))
    need(allocated_dcu(tres) == 0 and all(value == "0" for key, value in resources.items()
                                          if key.startswith("gres/")), "accounting does not prove a zero-GRES step")
    return resources


def terminal_row(verdict):
    row = verdict.get("row")
    need(verdict.get("request") == REQUEST and verdict.get("independent_observation") is True
         and verdict.get("state") in {"COMPILE_VERIFIED", "TERMINAL_UNVERIFIED"}
         and verdict.get("columns") == COLUMNS and isinstance(row, list) and len(row) == len(COLUMNS)
         and all(isinstance(part, str) for part in row) and verdict.get("observations") == [row]
         and re.fullmatch(PARENT + r"\.[0-9]+", row[0]) and row[1] == LABEL and row[5] == NODE
         and row[2].split() and row[2].split()[0] in TERMINAL and re.fullmatch(r"[0-9]+:[0-9]+", row[3])
         and row[4].isdecimal() and int(row[4]) <= 300 and verdict.get("terminal_row_SHA256") == digest(row),
         "original Kimi --reconcile has no independently verified terminal accounting row")
    resources = zero_gres(row[6])
    need(resources.get("cpu") == "4" and resources.get("node") == "1"
         and resources.get("mem") in {"8G", "8192M"}, "original Kimi terminal did not prove the zero-GRES scope")
    return row


def terminal_receipt(raw, *, monotonic=time.monotonic, wall=time.time):
    """Call when the authenticated --reconcile response arrives, before requesting post."""
    terminal_row(json.loads(raw))
    received = monotonic()
    return dict(request=REQUEST, received_monotonic=received, boot_epoch_anchor=wall() - received,
                raw_reconcile_SHA256=sha256(raw).hexdigest())


def verify_post(snapshot, accounting, capture, original_step, *, now, boot_anchor):
    _, nodes, fresh = fleet_view(snapshot, now, boot_anchor, TRAINING80_LEDGER)
    need(fresh and snapshot.get("snapshot_kind") == "training80-dispatch-v1", "post evidence is not fresh")
    membership = snapshot["membership"]
    receipt = snapshot["local_receipt"]
    started = receipt["received_monotonic"] - receipt["roundtrip_s"]
    terminal_received = capture["received_monotonic"]
    terminal_anchor = capture["boot_epoch_anchor"]
    need(all(type(value) in (float, int) and math.isfinite(value) for value in (terminal_received, terminal_anchor))
         and abs(terminal_anchor - boot_anchor) < 2 and 0 <= terminal_received <= started
         and hex_value(snapshot.get("request_id"), 32) and snapshot["request_id"] != REQUEST,
         "post evidence is stale, a different local clock, or predates reconcile receipt")
    probe = snapshot["nodes"][NODE]
    probe_step = probe["device_probe_step"]
    need(nodes[NODE]["admission"] == "clear" and probe["zone"] == ZONE and probe["parent"] == PARENT
         and all(type(probe[key]) is int and probe[key] == 0
                 for key in ("kfd_process_count", "own_device_fd_process_count"))
         and isinstance(probe_step, str) and re.fullmatch(PARENT + r"\.[0-9]+", probe_step)
         and probe_step != original_step and all(
             not ({original_step, probe_step} & parent["active_steps"].keys())
             for parent in membership["parents"].values()),
         "original Kimi scope still has a Slurm step or a device holder, or no independent probe")
    batches = [batch for batch in snapshot["probes"][PARENT]["batches"]
               if batch["received_nodes"] == [NODE] and batch["requested_nodes"] == [NODE]]
    need(len(batches) == 1 and batches[0]["state"] == "RETURNED" and batches[0]["returncode"] == 0
         and hex_value(batches[0]["request_id"], 32) and hex_value(batches[0]["stdout_SHA256"], 64),
         "post probe was not the independent one-node collector step")
    expected_label = "training80-admit-" + batches[0]["request_id"]
    row = completed_numbered_step(accounting["columns"], accounting["row"], probe_step, scope="Kimi post probe")
    need(accounting["columns"] == COLUMNS and row["JobName"] == expected_label and row["NodeList"] == NODE
         and zero_gres(row["AllocTRES"]).get("node") == "1",
         "post probe accounting does not prove a completed zero-GRES step on the Kimi node")
    return dict(post_step=probe_step, post_request=snapshot["request_id"], raw_post_SHA256=receipt["raw_response_SHA256"],
                post_payload_SHA256=receipt["payload_SHA256"])


def prepare(claim, local_raw, target_raw, reconcile_raw, capture_raw, snapshot_raw, post_accounting_raw, *,
            now, boot_anchor):
    """Validate original private receipts and build one direct Queue.finish-compatible payload."""
    handle = original_scope(claim, local_raw, target_raw)
    verdict, capture = map(json.loads, (reconcile_raw, capture_raw))
    row = terminal_row(verdict)
    target_pin = sha256(target_raw).hexdigest()
    reconcile_pin = sha256(reconcile_raw).hexdigest()
    need(verdict["intent_SHA256"] == target_pin and capture["request"] == REQUEST
         and capture["raw_reconcile_SHA256"] == reconcile_pin, "reconcile bytes do not bind the original target intent")
    post = verify_post(json.loads(snapshot_raw), json.loads(post_accounting_raw), capture, row[0],
                       now=now, boot_anchor=boot_anchor)
    passed = verdict["state"] == "COMPILE_VERIFIED"
    link = None
    if passed:
        arms = verdict["arms"]
        need(row[2:4] == ["COMPLETED", "0:0"] and set(arms) == set(ARMS)
             and all(hex_value(arms[arm][key], 64) for arm in ARMS for key in
                     ("object_SHA256", "fresh_M_SHA256", "actual_MD_SHA256", "symbols_stdout_SHA256"))
             and arms[ARMS[0]]["object_SHA256"] != arms[ARMS[1]]["object_SHA256"]
             and hex_value(verdict["score_receipt_SHA256"], 64), "qualified score pair or receipt pin differs")
        link = dict(generated_source=str(TARGET_CONTAINER / "generated"),
                    score_receipt_sha256=verdict["score_receipt_SHA256"], score_accounting=str(LINK_ACCOUNTING))
    evidence = dict(request=REQUEST, host_entry_SHA256=HOST["SHA256"], target_manifest_SHA256=RENDERED_SHA,
                    local_intent_SHA256=sha256(local_raw).hexdigest(), target_intent_SHA256=target_pin,
                    original_reconcile_SHA256=reconcile_pin, terminal_row_SHA256=verdict["terminal_row_SHA256"],
                    local_terminal_capture_SHA256=sha256(capture_raw).hexdigest(),
                    post_accounting_SHA256=sha256(post_accounting_raw).hexdigest(),
                    local_post_snapshot_SHA256=sha256(snapshot_raw).hexdigest(), **post)
    return dict(request_id=REQUEST, intent=INTENT, parent=PARENT, placement_by_parent=PLACEMENT_BY_PARENT,
                qualification="validation", terminal=True, post_clean=True, passed=passed, keeper_passed=None,
                planned=2, completed=2 if passed else 0, failed=0, rejected=0, canceled=0, missing=0 if passed else 2,
                actual_compile_step=row[0], actual_compile_state=row[2], actual_compile_exit_code=row[3],
                original_compile_audit_state=verdict["state"], promote=False, evidence_SHA256=digest(evidence),
                evidence=evidence, handoff_SHA256=handle["handoff_SHA256"], link_kwargs=link,
                link_score_accounting_SHA256=reconcile_pin if passed else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Save terminal_receipt() when the original --reconcile response arrives. Then use the existing single-node "
        "priority collector and save its received snapshot plus the probe's actual zero-GRES accounting columns/row. "
        "After a passing finish packet, the linker requires the exact --reconcile bytes at link_kwargs.score_accounting."
    ))
    for field in ("target-intent", "reconcile", "terminal-receipt", "post-snapshot", "post-accounting"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    intent = REPO / INTENT
    need(intent.resolve(strict=True) == intent, "Kimi intent escaped its canonical private directory")
    paths = (intent, args.target_intent, args.reconcile, args.terminal_receipt, args.post_snapshot, args.post_accounting)
    inputs = [bounded(path, 2**21 if path == args.post_snapshot else 65536) for path in paths]
    now = time.monotonic()
    packet = prepare(claimed_run(QUEUE_DB), *inputs, now=now, boot_anchor=time.time() - now)
    output = intent.with_name(FINISH_NAME)
    write_once(output, packet)
    print(json.dumps(dict(request=REQUEST, queue_run=DIRECT_RUN, finish=output.as_posix(),
                          finish_SHA256=sha256(bounded(output)).hexdigest(), queue_modified=False)))


if __name__ == "__main__":
    main()
