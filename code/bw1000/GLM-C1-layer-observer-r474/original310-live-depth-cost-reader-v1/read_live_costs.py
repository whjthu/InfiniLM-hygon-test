"""Read the saved in-flight model diagnostic without admitting a new Slurm step."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = "12d5bc02ca0a49a89db5c1628c64160b"
REQUEST = "5d9c77b0b9004989941fd73eb09b03fe"
COMMIT = "9fe9496f286ea293cc55c843c8ba68c92b49c458"
NATIVE = "ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5"
STEP = "1103606.310"
LABEL = f"GLM474-depth-observer-model-{RUN}-socket"
ROOT_OWNER = ROOT / LABEL
DISPATCH = ROOT / f"GLM474-depth-observer-model-{RUN}-dispatch"
SOURCE_PINS = json.loads("{\"attention_evidence.py\":\"4255460b2eae0ed879a4975188a0463a2d0e4444c7ef781e37fd7d8d99db98e1\",\"ffn_evidence.py\":\"1987ef55c03cf3b3517b12e397947578849d9934870310369f057824408d5739\",\"depth_binding.py\":\"70fc7ce827ff590502b39f46eacccf28cd3f1aca1658d499c85f242456dfd74b\",\"LAYER_BINDING.json\":\"09c5d53240624ab082dfacb022209641de505bcf75c76341214404557fa13355\",\"GPU_RESOURCE_CLAIM.json\":\"95f5b7a8e1c759e04cb2bcc873cab800eb7fd70772aa25814c254939ddd55569\",\"INPUT_BINDING.json\":\"81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e\"}")
WINDOWS = (
    "history.selected_value", "prepare.forward_latent", "fresh.expand_cached",
    "gather.key", "gather.value_conditional", "attention.selected",
    "output.forward", "output.original_stage_finish_or_stream_wait",
)
ROUTED = (
    "router_and_entry", "routed_gateup", "routed_activation", "routed_downquant",
    "routed_down", "shared_experts_and_combine", "FFN_allreduce", "FFN_original_output_fence",
)


def need(test, reason):
    if not test:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= limit, "original byte budget")
        raw = stream.read(before.st_size)
        after = os.fstat(stream.fileno())
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    need(len(raw) == before.st_size and identity(before) == identity(after), "changed open file")
    return raw


def summarize(numbers):
    if not numbers:
        return None
    need(all(math.isfinite(x) and x >= 0 for x in numbers), "invalid duration")
    return dict(samples=len(numbers), min_ms=1000*min(numbers),
                median_ms=1000*statistics.median(numbers), max_ms=1000*max(numbers))


def accumulate(mapping, label, host_s, stream_s):
    host, stream = mapping.setdefault(label, ([], []))
    host.append(host_s)
    if stream_s is not None:
        stream.append(stream_s)


def run():
    sys.dont_write_bytecode = True
    source = DISPATCH / "src"
    for name, expected in SOURCE_PINS.items():
        need(sha(read(source / name, 512*1024)) == expected, "original reader source binding")
    sys.path.insert(0, str(source))
    from attention_evidence import attention_windows
    from depth_binding import INDEXER_TYPES, PAIRS
    from ffn_evidence import check_frame, elapsed, indexer_windows, routed_windows

    original = json.loads(read(DISPATCH / "request.json", 4096))
    need(original["run_id"] == RUN and original["request_id"] == REQUEST
         and original["claim_commit"] == COMMIT and original["claim_SHA256"] == SOURCE_PINS["GPU_RESOURCE_CLAIM.json"],
         "original live launch identity")
    columns = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%130"
    p = subprocess.run(["sacct", "-n", "-P", "-j", "1103606", "--format="+columns],
                       capture_output=True, text=True, timeout=8, check=True)
    rows = [line.split("|") for line in p.stdout.splitlines() if line.split("|")[:2] == [STEP,LABEL+"-gate"]]
    need(len(rows) == 1 and rows[0][2] == "RUNNING" and "gres/dcu=16" in rows[0][-1].split(","),
         "current original GPU step accounting")
    files = {}
    selected = {}
    caps = []
    envelope = {"CAPI decode": [], "observer marker span host": [], "observer marker span stream": [],
                "full layer subtotal host": [], "shared layer subtotal host": [],
                "full layer subtotal stream": [], "shared layer subtotal stream": []}
    event_counts = []
    per_frame_rank_caps = []
    diagnostic_pin = json.loads(read(source / "INPUT_BINDING.json", 512*1024))["private_plan"]["requests"][0]
    output_hash = None
    for rank in range(16):
        name = f"owner-{rank:02d}.json"
        raw = read(ROOT_OWNER / name, 4*1024**2)
        files[name] = sha(raw)
        owner = json.loads(raw)
        need(owner["global_rank"] == rank and owner["device"] == rank % 8 and owner["actual_GPU"] == STEP
             and owner["Native_SHA256"] == NATIVE and owner["Model_READY"] and owner["weights"]
             and owner["loaded_parameters"] == 117060 and not owner.get("error"),
             "original weight and rank binding")
        row = owner["requests"][1]
        need(all(row[k] == diagnostic_pin[k] for k in ("ordinal","case_id","prompt_tokens","prompt_IDs_SHA256","body_SHA256"))
             and row["arm"] == "ON" and row["diagnostic"] and row["completed"] and row["reset_after"]
             and row["planned_max_new_tokens"] == 4096, "original diagnostic binding")
        ids = row["generated_token_ids"]
        need(bool(ids) and sha(json.dumps(ids,separators=(",",":")).encode()) == row["generated_token_ids_SHA256"],
             "original generated output hash")
        if output_hash is None:
            output_hash = row["generated_token_ids_SHA256"]
        need(row["generated_token_ids_SHA256"] == output_hash, "original all-rank diagnostic agreement")
        owner_obs, obs = owner.get("observer_diagnostic"), row["observer_report"]
        need(owner_obs == obs and obs["global_tp_rank"] == rank and obs["total_frames_used"] == len(obs["samples"]) == 3
             and obs["frame_cap"] == 3 and obs["events_allocated"] == 512 and not obs["pending_next_decode"],
             "original diagnostic frame coverage")
        for i, frame in enumerate(obs["samples"]):
            check_frame(frame, row, rank, i)
            event_counts.append(frame["event_count"])
            decode = row["decode_CAPI_s"][i]
            envelope["CAPI decode"].append(decode)
            marks = {x["name"]: x for x in frame["marks"] if not ".attnd1." in x["name"]}
            span = elapsed(frame["marks"][0],frame["marks"][-1])
            envelope["observer marker span host"].append(span["host_s"])
            if span["enclosing_stream_s"] is not None:
                envelope["observer marker span stream"].append(span["enclosing_stream_s"])
            partial = {"full": [0.0,0.0], "shared": [0.0,0.0]}
            for layer, k in enumerate(INDEXER_TYPES):
                x = elapsed(marks[f"layer.{layer}.{k}.begin"],marks[f"layer.{layer}.end"])
                partial[k][0] += x["host_s"]
                need(x["enclosing_stream_s"] is not None, "invalid layer window for summation")
                partial[k][1] += x["enclosing_stream_s"]
            need(sum(x[0] for x in partial.values()) <= span["host_s"] and span["host_s"] <= decode+2e-6,
                 "layer subtotals outside owner call")
            for kind, (host, stream) in partial.items():
                envelope[f"{kind} layer subtotal host"].append(host)
                envelope[f"{kind} layer subtotal stream"].append(stream)
            cap = dict(remove_full=max(0,decode-partial["full"][0]),
                       remove_shared=max(0,decode-partial["shared"][0]))
            caps.append(cap)
            per_frame_rank_caps.append((rank,i+1,decode))
            for layer in attention_windows(frame,rank):
                layer_index = layer["layer"]
                dest = selected.setdefault(layer_index,{})
                need(layer_index in PAIRS[i], "detail outside depth binding")
                outer = elapsed(marks[f"layer.{layer_index}.{INDEXER_TYPES[layer_index]}.begin"],
                                marks[f"layer.{layer_index}.end"])
                accumulate(dest,"layer",outer["host_s"],outer["enclosing_stream_s"])
                for entry in layer["windows"]:
                    if entry["name"] in WINDOWS:
                        accumulate(dest,"attention."+entry["name"],entry["host_envelope_seconds"],entry["stream_envelope_seconds"])
                for key, entry in routed_windows(frame,layer_index).items():
                    if key in ROUTED:
                        accumulate(dest,"ffn."+key,entry["host_s"],entry["enclosing_stream_s"])
                if INDEXER_TYPES[layer_index] == "full":
                    for key,entry in indexer_windows(frame).items():
                        if key in ("features","scored_selector") and entry is not None:
                            accumulate(dest,"indexer."+key,entry["host_s"],entry["enclosing_stream_s"])
    need(len(per_frame_rank_caps) == 48 and len(selected) == 6, "complete saved rank/layer denominator")
    print(json.dumps(dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), accounting_columns=columns.split(","),
        actual_accounting=rows[0], source_commit=COMMIT, request_id=REQUEST,
        ranks=16, rank_frames=48, layers_per_frame=78, detailed_layer_rank_frames=96,
        event_count_range=[min(event_counts),max(event_counts)], diagnostic_output_SHA256=output_hash,
        owner_SHA256=files, frozen_reader_pins=SOURCE_PINS,
        scope="Read-only in-flight diagnostic, fixed original ON arm and first three true AR tokens after the full prompt",
        timings={k:summarize(v) for k,v in envelope.items()},
        budget=dict(strict_ms_per_token=1000/15,
                    required_median_capi_reduction_ms=summarize(envelope["CAPI decode"])["median_ms"]-1000/15,
                    residual_if_entire_full_group_host_removed=summarize([x["remove_full"] for x in caps]),
                    residual_if_entire_shared_group_host_removed=summarize([x["remove_shared"] for x in caps]),
                    illustrative_arithmetic_only=True),
        detailed_layers={str(layer):dict(indexer=INDEXER_TYPES[layer],windows={
            name:dict(host=summarize(host),stream=summarize(stream)) for name,(host,stream) in costs.items()})
            for layer,costs in sorted(selected.items())},
        exclusive_GPU_time=False,formal_model_or_throughput_qualified=False),allow_nan=False))


if __name__ == "__main__":
    try:
        run()
    except (OSError, TypeError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps(dict(reader_failed=True,kind=type(exc).__name__,message=str(exc)[:140])))
