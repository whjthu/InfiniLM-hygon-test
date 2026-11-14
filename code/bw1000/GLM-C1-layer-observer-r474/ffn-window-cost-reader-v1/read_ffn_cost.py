"""Summarize FFN clocks for original owner files accepted by the terminal reader."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

TERMINAL_READER_SHA256 = "375e5e764a62dda57a885a707a83b327e843c00121f1aff1e7676e9e0fda80fe"
WINDOWS = {
    "ffn_envelope": ("FFN.begin", "FFN.end"),
    "gate_up_prefix": ("FFN.begin", "FFN.local.gate_up.return"),
    "activation": ("FFN.local.gate_up.return", "FFN.local.activation.end"),
    "down_to_fence": ("FFN.local.down.begin", "FFN.local.down.before_original_fence"),
    "down_fence": ("FFN.local.down.before_original_fence", "FFN.local.down.after_original_fence"),
    "tp_copy_to_reduce": ("FFN.tp.copy.begin", "FFN.tp.allreduce.begin"),
    "tp_allreduce_call": ("FFN.tp.allreduce.begin", "FFN.tp.allreduce.return"),
    "output_fence": ("FFN.out.before_original_fence", "FFN.out.after_original_fence"),
}
CLOCKS = ("owner_host_s", "gpu_from_first_s")


def summarize(owners):
    rows = []
    for owner in owners:
        rank = owner["global_rank"]
        for index, frame in enumerate(owner["requests"][0]["observer_report"]["samples"]):
            for kind, field in (("full", "full_detail_layer"), ("shared", "shared_detail_layer")):
                layer = frame[field]
                prefix = f"detail.layer.{layer}."
                marks = {mark["name"][len(prefix):]: mark for mark in frame["marks"]
                         if mark["name"].startswith(prefix)}
                windows = {}
                for name, (begin, end) in WINDOWS.items():
                    deltas = [marks[end][clock] - marks[begin][clock] for clock in CLOCKS]
                    if deltas[0] < 0 or deltas[1] < -2e-6:
                        raise ValueError(f"negative duration in {name}, rank {rank}, frame {index}")
                    windows[name] = dict(zip(CLOCKS, deltas))
                rows.append(dict(rank=rank, frame=index, layer=layer, kind=kind, windows_s=windows))
    groups = []
    for kind in ("full", "shared"):
        for layer in sorted({row["layer"] for row in rows if row["kind"] == kind}):
            selected = [row for row in rows if row["kind"] == kind and row["layer"] == layer]
            groups.append(dict(kind=kind, layer=layer, observations=len(selected),
                               ranks=len({row["rank"] for row in selected}),
                               frame_ordinals=sorted({row["frame"] for row in selected}),
                               windows_s={
                                   name: {clock: summary([r["windows_s"][name][clock] for r in selected])
                                          for clock in CLOCKS} for name in WINDOWS
                               }))
    return rows, groups


def summary(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values),
                negative_samples=sum(value < 0 for value in values))


def extract(terminal, root):
    if (terminal["valid_ffn_evidence"] is not True or terminal["independent_post_clean"] is not True
            or terminal["launcher_rc"] != 0):
        raise ValueError("original FFN evidence has not passed terminal audit")
    run = terminal["run_id"]
    if len(run) != 32 or any(ch not in "0123456789abcdef" for ch in run):
        raise ValueError("invalid original run identifier")
    label = f"GLM474-ffn-{run}-socket"
    gate = terminal["actual_accounting"]["gate"]
    if gate[1:4] != [label + "-gate", "COMPLETED", "0:0"]:
        raise ValueError("original GPU step did not complete")
    owners = []
    for rank in range(16):
        name = f"{label}/owner-{rank:02d}.json"
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != terminal["raw_artifact_SHA256"][name]:
            raise ValueError(f"terminal owner artifact changed at rank {rank}")
        owner = json.loads(raw)
        if (owner["global_rank"] != rank or owner["actual_GPU"] != gate[0]
                or owner["passed"] is not True):
            raise ValueError(f"terminal owner identity mismatch at rank {rank}")
        owners.append(owner)
    rows, groups = summarize(owners)
    expected_frames = terminal["ffn_observation"]["observed_frames"]
    if len(rows) != expected_frames * 2 or not 0 <= expected_frames <= 48:
        raise ValueError("terminal frame inventory mismatch")
    return dict(actual_GPU=gate[0], run_id=run, planned_frames=48,
                observed_frames=expected_frames, full_coverage=terminal["full_ffn_coverage"],
                clock_scope="Host elapsed and same-stream event envelopes. Windows overlap; stream durations include enqueue gaps and dependencies.",
                excluded_claims=["exclusive kernel time", "speedup", "service decode", "fleet busy"],
                group_summaries=groups, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("terminal_reader", type=Path)
    parser.add_argument("run")
    parser.add_argument("claim_sha256")
    parser.add_argument("request")
    parser.add_argument("commit")
    parser.add_argument("private_result", type=Path)
    args = parser.parse_args()
    reader = args.terminal_reader.resolve(strict=True)
    if hashlib.sha256(reader.read_bytes()).hexdigest() != TERMINAL_READER_SHA256:
        raise ValueError("unexpected terminal reader bytes")
    sys.path.insert(0, str(reader.parent))
    import read_terminal as pinned_reader
    if Path(pinned_reader.__file__).resolve() != reader:
        raise ValueError("terminal reader import resolved elsewhere")
    terminal = pinned_reader.read_terminal(args.run, args.claim_sha256, args.request, args.commit)
    report = dict(terminal_reader_SHA256=TERMINAL_READER_SHA256, terminal=terminal, timings=None)
    if terminal["valid_ffn_evidence"]:
        try:
            report["timings"] = extract(terminal, pinned_reader.HOST / "results")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            report["timing_error"] = type(exc).__name__ + ": " + str(exc)
    with args.private_result.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["timings"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
