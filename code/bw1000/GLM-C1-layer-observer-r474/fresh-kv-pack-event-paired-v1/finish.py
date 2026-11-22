"""Read the original device trace and emit the queue finish packet once."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import host_once as H


def sha(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def safe_json(path, limit=131072):
    raw = H.bounded(path, limit)
    return json.loads(raw), raw


def accounting_rows(raw):
    rows = [line.split("|") for line in raw.decode().splitlines() if line.strip()]
    expect(rows and all(len(row) == len(H.COLUMNS) for row in rows), "unqualified independent accounting columns")
    return rows


def exact_row(rows, step):
    same = [row for row in rows if row[0] == step]
    expect(len(same) == 1 and H.STEP.fullmatch(step), "missing or ambiguous numbered Slurm step")
    return same[0]


def independent_post(snapshot, after, observed_rows, gpu_step, post_step, upstream_admission_sha, tres, now, wall):
    receipt = snapshot["local_receipt"]
    this = snapshot["nodes"][H.NODE]
    received, rtt, boot = (receipt[key] for key in ("received_monotonic", "roundtrip_s", "boot_epoch_anchor"))
    expect(all(type(value) in {int, float} and math.isfinite(value) for value in (received, rtt, boot, now, wall))
           and rtt >= 0 and 0 <= now - received and now - received + rtt <= 60
           and abs(wall - now - boot) < 2, "independent post receipt is stale or has another boot anchor")
    payload = {key: value for key, value in snapshot.items() if key != "local_receipt"}
    expect(receipt["request_id"] == snapshot["request_id"] and sha(encoded(payload)) == receipt["payload_SHA256"]
           and re.fullmatch(r"[0-9a-f]{64}", receipt["raw_response_SHA256"])
           and receipt["payload_SHA256"] != upstream_admission_sha
           and snapshot["snapshot_kind"] == "training80-dispatch-v1" and snapshot["complete"] is True,
           "independent post payload identity differs")
    expect(this["parent"] == H.PARENT and this["zone"] == "O01" and this["active_steps"] == []
           and sorted(this["PCIs"]) == sorted(after["pci"])
           and all(this[key] is True for key in ("device_probe_observed", "membership_valid", "KFD_clean", "PID_clean"))
           and all(type(this[key]) is int and this[key] == 0 for key in ("kfd_process_count", "own_device_fd_process_count"))
           and this.get("probe_error") is None, "independent physical-device inventory is not clean")
    actual = exact_row(observed_rows, this["device_probe_step"])
    number = lambda step: int(step.split(".")[1])
    expect(number(actual[0]) > max(number(gpu_step), number(post_step)) and actual[2:4] == ["COMPLETED", "0:0"]
           and actual[5] == H.NODE and tres.allocated_dcu(actual[6]) == 0, "independent post has no later zero-GRES terminal")
    return {"row": actual, "payload_SHA256": receipt["payload_SHA256"], "raw_SHA256": receipt["raw_response_SHA256"]}


def audit(run, accounting_raw, post_snapshot, handle, probe, waiter, tres, timing, result, *,
          now=time.monotonic, wall=time.time):
    placement = {H.PARENT: {H.NODE: 1}}
    expect(handle["request_id"] == H.REQUEST and handle["direct_kind"] == "operator"
           and handle["placement_by_parent"] == placement and isinstance(handle["intent"], str) and handle["intent"],
           "queue handle is not the original GLM component request")
    intent, intent_raw = safe_json(run / "intent.json", 16384)
    epoch = intent["epoch"]
    expect(intent["request"] == H.REQUEST and intent["label"] == H.LABEL and intent["parent"] == H.PARENT
           and intent["node"] == H.NODE and intent["archive_SHA256"] == probe.ARCHIVE_SHA
           and re.fullmatch(r"[0-9a-f]{64}", intent["upstream_admission_SHA256"])
           and type(epoch) is float and math.isfinite(epoch) and epoch > 0 and intent["deadline"] == epoch + H.WHOLE
           and intent["work_deadline"] == epoch + H.WORK and intent["replay_allowed"] is False,
           "single original GPU epoch or request differs")
    intent_sha = sha(intent_raw)
    record, record_raw = safe_json(run / "outcome.json")
    expect(record["request"] == H.REQUEST and record["intent_SHA256"] == intent_sha
           and record["denominator"] == timing.PLAN, "recorded controller request, intent or denominator differs")
    rows = accounting_rows(accounting_raw)
    recognized = H.select_rows(accounting_raw)
    for phase in H.PHASES:
        expect(phase in recognized and record["phases"][phase]["row"] == recognized[phase]
               and len([row for row in rows if row[0] == recognized[phase][0]]) == 1
               and waiter.is_terminal(recognized[phase])
               and recognized[phase][5] == H.NODE and tres.allocated_dcu(recognized[phase][6]) == H.PHASES[phase][0]
               and f"cpu={H.PHASES[phase][1]}" in recognized[phase][6].split(","),
               "original controller and fresh terminal accounting differ")
    pre, gpu, post = (recognized[phase] for phase in ("pre", "gpu", "post"))
    expect(pre[2:4] == post[2:4] == ["COMPLETED", "0:0"]
           and int(pre[0].split(".")[1]) < int(gpu[0].split(".")[1]) < int(post[0].split(".")[1]),
           "original before/GPU/post sequence is unqualified")
    before_raw, before = H.read_snapshot(run / "before.json", pre, probe)
    after_raw, after = H.read_snapshot(run / "post.log", post, probe)
    expect(sha(before_raw) == intent["pre_admission_SHA256"] and sha(after_raw) == record["post_snapshot_SHA256"]
           and before["boot_id"] == after["boot_id"] and before["pci"] == after["pci"]
           and before["kfd_holder_count"] == after["kfd_holder_count"] == 0,
           "original same-boot host inventory differs")
    context = SimpleNamespace(bounded=H.bounded, own_identity=H.own_identity, need=H.need)
    trace = result.inspect_probe(run / "probe.jsonl", gpu, intent_sha, before, probe, timing, context)
    expect(trace["worker"] is not None and trace["worker"] == after["worker_identity"]
           and after["worker_state"] in {"OWNER_ABSENT", "PID_REUSED", "OWNER_ZOMBIE"}
           and trace.get("trace_SHA256") == record["probe"].get("trace_SHA256")
           and isinstance(trace.get("trace_SHA256"), str),
           "original worker identity, trace or cleanup unqualified")
    independent = independent_post(post_snapshot, after, rows, gpu[0], post[0],
                                   intent["upstream_admission_SHA256"], tres, now(), wall())
    completed = len(trace["passed"])
    failed = int(trace["failed_arm"] is not None)
    planned = len(timing.PLAN["arms"])
    evidence = dict(request=H.REQUEST, actual_accounting_SHA256=sha(accounting_raw),
                    controller_outcome_SHA256=sha(record_raw), before_SHA256=sha(before_raw), post_SHA256=sha(after_raw),
                    trace_SHA256=trace["trace_SHA256"], independent_post=independent, actual_gpu_accounting=gpu,
                    component_only=True, Model_GO=False)
    passed = trace["trace_pass"] and gpu[2:4] == ["COMPLETED", "0:0"] and completed == planned
    evidence["pairs_by_tokens"] = trace.get("pairs_by_tokens") if passed else None
    packet = dict(request_id=H.REQUEST, intent=handle["intent"], parent=H.PARENT, placement_by_parent=placement,
                  qualification="operator", terminal=True, post_clean=True, planned=planned, completed=completed,
                  failed=failed, rejected=0, canceled=0, missing=planned-completed-failed,
                  passed=passed, keeper_passed=None, promote=False, evidence_SHA256=sha(encoded(evidence)))
    return {"finish": packet, "evidence": evidence, "independent_trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "actual-accounting", "independent-post", "handle", "out", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    manifest, raw = safe_json(args.manifest, 16384)
    expect(sha(raw) == args.manifest_sha256 and manifest["request"] == H.REQUEST,
           "published component manifest differs")
    for name in ("finish.py", "host_once.py", "probe.py", "timing.py", "result.py"):
        source = (HERE / name).read_bytes()
        binding = manifest["files"][name]
        expect(len(source) == binding["bytes"] and sha(source) == binding["SHA256"], "published source differs: " + name)
    probe, waiter, tres = H.verify_stage(REPO)
    timing = H.load(H.TIMING, H.pin(REPO, H.TIMING, H.PINS[H.TIMING]))
    result = H.load(H.RESULT_READER, H.pin(REPO, H.RESULT_READER, H.PINS[H.RESULT_READER]))
    snapshot, _ = safe_json(args.independent_post, 2 * 1024**2)
    handle, _ = safe_json(args.handle, 16384)
    output = audit(args.run, H.bounded(args.actual_accounting, 2 * 1024**2), snapshot, handle, probe, waiter, tres, timing, result)
    H.save(args.out, output)
    print(json.dumps(output["finish"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
