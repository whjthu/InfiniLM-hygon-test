"""Independent accounting, resource release and original Dev24 pose-event recomputation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import runpy

from authority import (ARM, BASE, HEAD_FILE_SHA, HEAD_MODULE_SHA, INPUT_SHA, LEGACY_INDEX, READER, RESULTS,
                       DEV_DIR, PROJECT, SHA, encode, js, need, read, sha)


RESOURCES = {"pre": (0, 2, 2048, 60), "gpu": (1, 4, 65536, 1200), "post": (0, 2, 2048, 60)}
MEMORY = {2048: {"2G", "2048M"}, 65536: {"64G", "65536M"}, 256: {"256M"}}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
ROUNDOFF_PIN = "4a0fa432a61734ac21c102c274a1028217d6a212e9ba8b19acfd83c054f43d97"
PLANNED = {"planned_windows": 24, "input_pairs": 22, "expected_predictions": 44, "scoring_slots": 48,
           "original_missing_windows": 2, "original_failed_scoring_slots": 4,
           "optimizer_updates": 0, "formal_CO3D": False, "promotion": False}


def epoch(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def resources(value, gpu, cpus, memory):
    bits = value.split(",")
    tref = dict(part.split("=", 1) for part in bits)
    need(len(bits) == len(tref) == 3 + gpu and tref.get("cpu") == str(cpus)
         and tref.get("mem") in MEMORY[memory] and tref.get("node") == "1", "INDEPENDENT_EXACT_CPU_MEMORY_NODE")
    if gpu:
        need(tref.pop("gres/dcu", None) == "1", "INDEPENDENT_EXACT_ONE_DCU")
    need(tref.keys() == {"cpu", "mem", "node"}, "NO_OTHER_GRES")


def timeline(intent, outcome, raw_rows):
    request = intent["request"]
    begin, deadline = intent["epoch"], intent["deadline"]
    need(re.fullmatch(r"[a-f0-9]{32}", request) and re.fullmatch(r"[a-f0-9]{40}", intent["publication"])
         and outcome["request"] == request and outcome["publication"] == intent["publication"]
         and intent["max_seconds"] == 1500 and deadline == begin + 1800, "EXACT_ONE_EPOCH_AND_PUBLICATION")
    label = "vg-1318-dev24-" + request
    need(set(raw_rows) <= RESOURCES.keys() and set(outcome["stages"]) <= RESOURCES.keys(), "EXACT_PHASE_SET")
    intervals = {}
    prior_end = int(begin)
    steps = []
    for phase in RESOURCES:
        outcome_row = outcome["stages"].get(phase, {}).get("row")
        own = raw_rows.get(phase)
        if own is None:
            need(outcome_row is None, "MISSING_INDEPENDENT_ACTUAL_PHASE")
            continue
        need(len(own) == 9 and own[:7] == outcome_row and own[1] == label + "-" + phase
             and own[5] == intent["node"] and re.fullmatch(intent["parent"] + r"\.[0-9]+", own[0])
             and own[2].split()[0].rstrip("+") in TERMINAL and re.fullmatch(r"[0-9]+:[0-9]+", own[3]),
             "ACTUAL_NUMERIC_PHASE_TERMINATION_AND_SCOPE")
        gpu, cpu, mem, max_seconds = RESOURCES[phase]
        resources(own[6], gpu, cpu, mem)
        start, end = epoch(own[7]), epoch(own[8])
        need(prior_end <= start <= end <= int(begin + 1500) + 1
             and own[4].isdecimal() and 0 <= int(own[4]) <= max_seconds
             and abs(end - start - int(own[4])) <= 1, "EXACT_PHASE_EPOCH_ELAPSED_AND_CHRONOLOGY")
        prior_end = end
        intervals[phase] = start, end
        steps.append(int(own[0].split(".")[-1]))
    need(steps == sorted(set(steps)), "ACTUAL_PHASE_NUMBER_ORDER")
    return intervals


def verify_post(before, after, assignment, provider, probe, post_row, gpu_row, intent):
    parent, node = intent["parent"], intent["node"]
    current = probe["nodes"][node]
    handle = current["device_probe_step"]
    need(re.fullmatch(parent + r"\.[0-9]+", handle) and int(handle.split(".")[-1]) > int(post_row[0].split(".")[-1])
         and current["parent"] == parent and current["device_probe_observed"] is True
         and current["KFD_clean"] is current["PID_clean"] is True and current["active_steps"] == []
         and probe["observed_epoch"] <= intent["deadline"]
         and 0 <= probe["observed_epoch"] - epoch(post_row[8]) <= 300, "FRESH_INDEPENDENT_POST_PCI_KFD_AND_PID")
    need(before["step"] != after["step"] == post_row[0] and before["node"] == after["node"] == node
         and before["full_fd_inventory_performed"] is after["full_fd_inventory_performed"] is True
         and before["kfd_holder_count"] == after["kfd_holder_count"] == 0
         and before["boot_id"] == after["boot_id"] and len(before["pci"]) == len(set(before["pci"])) == 8
         and before["pci"] == after["pci"] and set(before["pci"]) == set(current["PCIs"]), "EXACT_SAME_BOOT_8_PCI_AND_CLEAN_FDS")
    if assignment is not None and provider is not None:
        need(assignment["step"] == gpu_row[0] and assignment["node"] == node and assignment["pci"] in before["pci"]
             and assignment["kfd"]["pci"] == assignment["pci"] and type(assignment["kfd"]["gpu_id"]) is int
             and assignment["kfd"]["gpu_id"] > 0 and SHA.fullmatch(assignment["kfd"]["properties_SHA256"])
             and provider["path"].startswith("/") and provider["bytes"] > 0 and SHA.fullmatch(provider["SHA256"]),
             "ACTUAL_ONE_ASSIGNED_PCI_KFD_AND_HIP_SO")
    return handle


def observed_progress(path):
    if not path.is_file():
        return {"observed_prediction_events": 0, "progress_SHA256": None}
    raw = read(path, 2 * 1024**2)
    lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
    events = [v for v in lines if "arm" in v and v["arm"] in ("base", ARM)]
    keys = {(r["arm"], r["sequence"], r["window"]) for r in events}
    need(len(keys) == len(events) <= 44, "PREDICTIONS_CANNOT_EXCEED_OR_DUPLICATE_FROZEN_SLOTS")
    return {"observed_prediction_events": len(events), "progress_SHA256": sha(raw)}


def consume(root, independent_actual, independent_post, verified_roundoff_file):
    intent = js(root / "score-intent.json")
    outcome = js(root / "score-outcome.json")
    label = "vg-1318-dev24-" + intent["request"]
    need(root == RESULTS / label, "EXACT_ORIGINAL_PAIRED_OUTPUT_NAMESPACE")
    intervals = timeline(intent, outcome, independent_actual)
    installed = PROJECT / "incoming" / label
    stage = js(root / "stage-complete.json", intent["stage_SHA256"])
    index = js(installed / LEGACY_INDEX, intent["manifest_SHA256"])
    heads = js(root / "head-authority.json", stage["head_authority_SHA256"])
    need(stage["head_authority_SHA256"] == intent["head_authority_SHA256"] and set(heads) == {ARM}
         and heads[ARM]["file_SHA256"] == HEAD_FILE_SHA and heads[ARM]["module_SHA256"] == HEAD_MODULE_SHA,
         "EXACT_ORIGINAL_1318_HEAD_NOT_OLD_498")
    path = root / "score/progress-private.jsonl"
    out = dict(PLANNED, request=intent["request"], accounting=independent_actual, independent_post_clean=False,
               qualified_predictions=0, **observed_progress(path))
    if "gpu" not in intervals:
        return dict(out, state="NO_INDEPENDENT_GPU_TERMINAL")
    if "post" not in intervals or independent_post is None:
        return dict(out, state="GPU_TERMINAL_INDEPENDENT_POST_PENDING")
    need("pre" in intervals, "ACTUAL_GPU_WITHOUT_PREDECESSOR_PRE")
    before, after = js(root / "before.json"), js(root / "after.json")
    need(before["step"] == independent_actual["pre"][0], "EXACT_COMPUTE_PRE_READBACK_STEP")
    assignment, provider = (js(path) if path.is_file() else None
                            for path in (root / "assignment.json", root / "hip-provider.json"))
    step = verify_post(before, after, assignment, provider, independent_post, independent_actual["post"],
                       independent_actual["gpu"], intent)
    attested = assignment is not None and provider is not None
    out.update(independent_post_clean=True, post_step=step, runtime_PCI_KFD_SO_attested=attested)
    if not attested:
        return dict(out, state="EXACT_TERMINAL_AND_CLEAN_POST_WITH_RUNTIME_IDENTITY_MISSING")
    if (set(intervals) != RESOURCES.keys() or outcome["completed"] is not True or outcome["clean_post"] is not True
            or any(row[2:4] != ["COMPLETED", "0:0"] for row in independent_actual.values())):
        return dict(out, state="EXACT_TERMINAL_AND_POST_WITH_INCOMPLETE_SCORING")
    read(verified_roundoff_file, 65536, ROUNDOFF_PIN)
    audit = runpy.run_path(str(verified_roundoff_file))["audit"]
    independent = audit(installed / READER, index[READER], RESULTS / DEV_DIR / "inputs/inputs-private.json",
                        INPUT_SHA, intent["head_authority_SHA256"], path)
    need(independent["assessed_windows"] == 24 and independent["input_pairs"] == 22
         and independent["actual_prediction_events"] == 44 and independent["optimizer_updates"] == 0,
         "FORMAL_FIXED_DEVELOPMENT_DENOMINATORS")
    base, head = [independent["AUC"]["all_windows"][key] for key in ("base", ARM)]
    threshold = (head["development_AUC_at_30"] >= base["development_AUC_at_30"]
                 and head["development_AUC_at_15"] >= .95 * base["development_AUC_at_15"]
                 and head["failed_pairs"] <= base["failed_pairs"])
    return dict(out, state="INDEPENDENT_1318_DEV24_RECOMPUTED", qualified_predictions=44,
                original_quality_threshold_met=threshold, audit=independent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage", "accounting", "independent-post", "roundoff", "output"):
        parser.add_argument("--" + flag, type=Path, required=flag != "independent-post")
    a = parser.parse_args()
    result = consume(a.stage, js(a.accounting), js(a.independent_post, cap=2**21) if a.independent_post else None, a.roundoff)
    with a.output.open("xb") as out:
        out.write(encode(result))


if __name__ == "__main__":
    main()
