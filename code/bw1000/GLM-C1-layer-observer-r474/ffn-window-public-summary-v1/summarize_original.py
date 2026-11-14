
"""Summarize the original GLM110 terminal audit without copying private artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

RUN = "4d4f0c4b2cd344ae92c81ea40c06aecf"
REQUEST = "29dee83872aa4843afa6d1e46a843ed0"
CLAIM = "fcbe791b159652abb9f04bcb78cf1a80032ce94d"
READER = "375e5e764a62dda57a885a707a83b327e843c00121f1aff1e7676e9e0fda80fe"
GPU = "1103606.110"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
WINDOWS = ("ffn_envelope", "gate_up_prefix", "activation", "down_to_fence", "down_fence",
           "tp_copy_to_reduce", "tp_allreduce_call", "output_fence")
CLOCKS = ("owner_host_s", "gpu_from_first_s")


def require(condition, error):
    if not condition:
        raise ValueError(error)


def accounting_row(row, phase):
    if row is None and phase != "gate":
        return None
    job, label, state, exit_code, elapsed, nodes = row
    expected_label = f"GLM474-ffn-{RUN}-socket-{phase}"
    require(re.fullmatch(r"1103606\.\d+", job) and label == expected_label
            and state.split()[0].rstrip("+") in TERMINAL and re.fullmatch(r"\d+:\d+", exit_code)
            and re.fullmatch(r"\d+", elapsed)
            and nodes in {"f13r1n[05,07]", "f13r1n05,f13r1n07"},
            "unexpected original Slurm accounting")
    require(phase != "gate" or job == GPU, "wrong original GPU step")
    return dict(job=job, state=state, exit_code=exit_code, elapsed_seconds=int(elapsed))


def read_clocks(group):
    windows = {}
    require(set(group) == set(WINDOWS), "unexpected FFN window inventory")
    for window in WINDOWS:
        windows[window] = {}
        require(set(group[window]) == set(CLOCKS), "unexpected FFN clock scope")
        for clock in CLOCKS:
            row = group[window][clock]
            low, mid, high = (row[key] for key in ("minimum", "median", "maximum"))
            neg = row["negative_samples"]
            floor = 0 if clock == "owner_host_s" else -2e-6
            require(all(type(val) in (int, float) and math.isfinite(val) for val in (low, mid, high))
                    and floor <= low <= mid <= high and type(neg) is int and neg >= 0
                    and (neg > 0) == (low < 0), "invalid FFN clock summary")
            windows[window][clock] = dict(minimum=low, median=mid, maximum=high, negative_samples=neg)
    return windows


def summarize(raw, digest):
    require(hashlib.sha256(raw).hexdigest() == digest, "FFN private audit SHA mismatch")
    report = json.loads(raw)
    term = report["terminal"]
    require((report["terminal_reader_SHA256"], term["run_id"], term["original_request_id"],
             term["claim_commit"]) == (READER, RUN, REQUEST, CLAIM), "unexpected FFN audit identity")
    require(term["planned_owner_ranks"] == 16 and term["planned_requests_per_rank"] == 33
            and type(term["launcher_rc"]) is int, "unexpected original request denominator")
    flags = {key: term[key] for key in ("valid_ffn_evidence", "independent_post_clean", "full_ffn_coverage")}
    require(all(type(flag) is bool for flag in flags.values()), "invalid audit flags")
    steps = {phase: accounting_row(term["actual_accounting"][phase], phase)
             for phase in ("pre", "gate", "post")}
    if flags["independent_post_clean"]:
        require(steps["post"] and steps["post"]["state"] == "COMPLETED"
                and steps["post"]["exit_code"] == "0:0", "cleanup flag contradicts accounting")
    if flags["valid_ffn_evidence"]:
        require(flags["independent_post_clean"] and term["launcher_rc"] == 0
                and steps["gate"]["state"] == "COMPLETED" and steps["gate"]["exit_code"] == "0:0",
                "FFN qualification contradicts original lifecycle")
    timing = report["timings"]
    groups = []
    observed = None
    if flags["valid_ffn_evidence"]:
        observed = term["ffn_observation"]["observed_frames"]
        require(type(observed) is int and 0 <= observed <= 48
                and flags["full_ffn_coverage"] == (observed == 48), "invalid FFN frame coverage")
    else:
        require(not flags["full_ffn_coverage"], "invalid FFN evidence cannot qualify coverage")
    if timing is not None:
        require(observed is not None and timing["actual_GPU"] == GPU and timing["run_id"] == RUN
                and timing["planned_frames"] == 48 and timing["observed_frames"] == observed
                and timing["full_coverage"] is flags["full_ffn_coverage"], "timing provenance differs")
        seen = set()
        totals = {"full": 0, "shared": 0}
        for row in timing["group_summaries"]:
            kind, layer = row["kind"], row["layer"]
            count, ranks, frames = row["observations"], row["ranks"], row["frame_ordinals"]
            require(kind in totals and type(layer) is int and 0 <= layer < 78
                    and (kind, layer) not in seen and type(count) is int and type(ranks) is int
                    and 0 < ranks <= min(count, 16) and isinstance(frames, list)
                    and frames == sorted(set(frames)) and all(type(n) is int and 0 <= n < 3 for n in frames)
                    and 0 < count <= min(observed, ranks * len(frames)), "invalid FFN group coverage")
            seen.add((kind, layer))
            totals[kind] += count
            windows = read_clocks(row["windows_s"])
            require(all(v["negative_samples"] <= count for clocks in windows.values() for v in clocks.values()),
                    "invalid negative sample count")
            groups.append(dict(kind=kind, layer=layer, observations=count, ranks=ranks,
                               frame_ordinals=frames, windows_s=windows))
        require(totals == {"full": observed, "shared": observed}, "FFN group totals differ from audit")
    return dict(original_GPU=GPU, claim_commit=CLAIM, private_audit_SHA256=digest,
                scope="Three autoregressive frames at most, two detailed layers per rank. Host and same-stream event envelopes can overlap.",
                accounting=steps, launcher_rc=term["launcher_rc"], **flags,
                planned_requests=33, planned_rank_frames=48, observed_rank_frames=observed,
                timing_available=timing is not None, group_summaries=groups,
                quality_qualified=False, service_speed_qualified=False, fleet_busy_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_audit", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("public_summary", type=Path)
    args = parser.parse_args()
    with args.private_audit.open("rb") as stream:
        raw = stream.read(2**20 + 1)
    require(len(raw) <= 2**20, "FFN audit exceeds expected size")
    report = summarize(raw, args.sha256)
    with args.public_summary.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
