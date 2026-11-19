"""Bound how much the observed GLM layers could affect their existing marker windows."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics as stats

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
PIN = ("08fec67b994adfe9660a5c26d73ba0b026ceb70e0fdeec62079bbc18562a3528", 3520737)
RUN = "5f162445c8e04a399b29a8b47a555d0a"
REQUEST = "719b98d46277485897647639ecbea9c5"
SOURCE = ROOT / "GLM474-original188-saved-frame-cost-v1/cost-private.json"
TARGET_SECONDS = 1/15


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def merge(intervals):
    """Return a disjoint union, allowing nested windows and adjacency."""
    combined = []
    for start, end in sorted(intervals):
        need(math.isfinite(start) and math.isfinite(end) and 0 <= start <= end, "INVALID_INTERVAL")
        if combined and start <= combined[-1][1]:
            combined[-1] = (combined[-1][0], max(combined[-1][1], end))
        else:
            combined.append((start, end))
    return combined


def union_s(intervals):
    return sum(end-start for start, end in merge(intervals))


def region(row, clock, span, identity):
    need(row["clock_identity"] == identity, "MIXED_CLOCK")
    start, end, elapsed = (row[clock+"_"+k+"_s"] for k in ("start", "end", "envelope"))
    if elapsed is None:
        return None
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (start,end,elapsed))
         and -2e-6 <= start <= end <= span+2e-6 and abs(end-start-elapsed) <= 1e-8, "REGION_OUT_OF_FRAME")
    return max(0, start), min(span, end)


def describe(values):
    return dict(min=min(values), median=stats.median(values), max=max(values), occurrences=len(values))


def derive(data):
    context = data["context"]
    need(data["derived_complete"] and not data["errors"] and data["observed_rank_frames"] == 48
         and data["actual_AR_steps"] == 3 and context["original_run"] == RUN
         and context["request_id"] == REQUEST, "ORIGINAL_DERIVATION_ID")
    owners = data["owner_results"]
    need([o["rank"] for o in owners] == list(range(16)) and all(len(o["frames"]) == 3 for o in owners), "OWNER_COUNT")
    outcomes = {}
    for clock in ("host", "event"):
        rows = []
        by_step = {i: [] for i in range(3)}
        by_layer = {i: [] for i in range(78)}
        kinds = {}
        missing = []
        for owner in owners:
            rank = owner["rank"]
            for step, frame in enumerate(owner["frames"]):
                identity = [rank, step, frame["sequence"]]
                need(frame["global_rank"] == rank and frame["AR_index"] == step and
                     set(frame["units"]) == {"owner_host_s", "gpu_from_first_s"}
                     and set(frame["units"].values()) == {"seconds"}, "ORIGINAL_FRAME_ID_OR_UNITS")
                regions = frame["regions"]
                span_rows = [row for row in regions if row["name"] == "first_to_last_observed_marker"]
                need(len(span_rows) == 1, "FRAME_SPAN_COUNT")
                span = span_rows[0][clock+"_envelope_s"]
                need(type(span) in (int,float) and math.isfinite(span) and span>0, "FRAME_SPAN")
                sections = {i: [] for i in range(78)}
                details = []
                for row in regions:
                    if row["name"] == "decoder_layer":
                        sections[row["layer"]].append(row)
                    if row["name"] in ("attention.history.selected_value", "attention.output.forward", "FFN.enclosing"):
                        details.append(row)
                need(all(len(v) == 1 for v in sections.values()), "LAYER_MEMBERSHIP")
                layer_windows = {i:region(sections[i][0], clock, span, identity) for i in sections}
                for i, window in layer_windows.items():
                    kind = sections[i][0]["observed_kind"]
                    need(kind in ("full", "shared") and kinds.setdefault(i,kind) == kind, "LAYER_KIND")
                    if window is not None:
                        by_layer[i].append(window[1]-window[0])
                if any(q is None for q in layer_windows.values()):
                    missing.append([rank,step])
                    continue
                all_layers_s = union_s(layer_windows.values())
                two_layers_s = union_s([layer_windows[i] for i in (0,3)])
                full = union_s([v for i,v in layer_windows.items() if kinds[i]=="full"])
                shared = union_s([v for i,v in layer_windows.items() if kinds[i]=="shared"])
                selected = {}
                for label in ("attention.history.selected_value","attention.output.forward","FFN.enclosing"):
                    d = [x for x in details if x["name"]==label]
                    need(len(d)==2 and {x["layer"] for x in d}=={0,3}, "DETAIL_LAYER_MEMBERSHIP")
                    vals = [region(x,clock,span,identity) for x in d]
                    selected[label] = union_s(vals) if all(v is not None for v in vals) else None
                row = dict(span_s=span, all_decoder_layers_s=all_layers_s, full_layer_union_s=full,
                           shared_layer_union_s=shared, outside_decoder_union_s=span-all_layers_s,
                           two_observed_layers_union_s=two_layers_s, two_layer_fraction=two_layers_s/span,
                           optimistic_residual_if_both_complete_layers_removed_s=span-two_layers_s,
                           selected_detail=selected)
                by_step[step].append(row)
                rows.append(row)
        need(len(rows)+len(missing)==48 and rows, "EVIDENCE_DENOMINATOR")
        heads = sorted([dict(layer=i,kind=kinds[i],seconds=describe(v)) for i,v in by_layer.items() if v],
                       key=lambda item: (-item["seconds"]["median"],item["layer"]))[:8]
        outcomes[clock] = dict(complete_frames=len(rows), missing_frames=missing, top_layer_envelopes=heads,
             full_layer_count=sum(k=="full" for k in kinds.values()), shared_layer_count=sum(k=="shared" for k in kinds.values()),
             frame_summaries={k:describe([r[k] for r in rows]) for k in rows[0] if k!="selected_detail"},
             detail_summaries={k:describe([r["selected_detail"][k] for r in rows if r["selected_detail"][k] is not None])
                               for k in rows[0]["selected_detail"] if any(r["selected_detail"][k] is not None for r in rows)},
             step_durations=[dict(AR_index=i,rank_count=len(rs),max_rank_observed_span_s=max(r["span_s"] for r in rs),
                        max_rank_residual_after_hypothetical_removal_s=max(r["optimistic_residual_if_both_complete_layers_removed_s"] for r in rs))
                            for i,rs in by_step.items() if rs],
             residuals_at_or_under_15_token_interval=sum(r["optimistic_residual_if_both_complete_layers_removed_s"]<=TARGET_SECONDS for r in rows))
    return dict(original_GPU_step="1103606.188", actual_AR_steps=3, expected_rank_frames=48, clocks=outcomes,
                threshold_seconds_per_increment=TARGET_SECONDS, current_busy_measured=False, service_qualified=False,
                interpretation="Unions use one rank and one clock at a time. Hypothetical wholesale layer removal changes that marked interval only. These enclosing clocks do not identify exclusive device cost or predict a measured service rate.")


def read_once():
    with os.fdopen(os.open(SOURCE,os.O_RDONLY|os.O_NOFOLLOW),"rb") as src:
        start = os.fstat(src.fileno())
        need(stat.S_ISREG(start.st_mode) and start.st_size == PIN[1], "PRIVATE_DERIVATION_SIZE")
        raw = src.read(PIN[1]+1)
        finish = os.fstat(src.fileno())
    need((start.st_ino,start.st_size,start.st_mtime_ns)==(finish.st_ino,finish.st_size,finish.st_mtime_ns)
         and len(raw)==PIN[1] and hashlib.sha256(raw).hexdigest()==PIN[0], "PRIVATE_DERIVATION_PIN")
    out=derive(json.loads(raw))
    out.update(input_sha256=PIN[0], input_bytes=len(raw), raw_owner_files_read=0, new_GPU_calls=0)
    return out


if __name__=="__main__":
    print(json.dumps(read_once(), allow_nan=False))
