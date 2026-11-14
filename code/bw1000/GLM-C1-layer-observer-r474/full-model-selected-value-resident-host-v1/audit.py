"""Independently read private rank receipts after terminal accounting."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from selected_trial import phases, prompt_fragments, schedule
from shared import D, NODES, require
from verify_transport import INFINICCL_SHA256, RCCL_SHA256, network_channels

HEX = re.compile(r"[0-9a-f]{64}")
FLOOR = 8 * 1024**3


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(32 * 1024**2 + 1)
    require(len(raw) <= 32 * 1024**2, "rank receipt exceeds original bounds")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def audit_request(request, item, pin, previous, eos, special):
    require(all(request[k] == pin[k] for k in
                ("ordinal", "case_id", "suite", "prompt_tokens", "prompt_IDs_SHA256", "body_SHA256"))
            and all(request[k] == item[k] for k in ("arm", "pair", "warmup", "owner_cores"))
            and request["schedule_index"] == item["index"]
            and request["completed"] is True and request["reset_before"] is True
            and request["reset_after"] is True and request["planned_max_new_tokens"] == 4096,
            "request identity or completion differs")
    require(request["forced_singleton_prefill_tail"] is item["warmup"]
            and request["runtime_prefill_chunk"] == 512
            and request["original_request_prefill_window"] == 4096, "prefill protocol changed")
    shape = [(offset, len(chunk)) for offset, chunk in
             prompt_fragments(range(pin["prompt_tokens"]), item["warmup"])]
    fragments = request["prefill_fragment_plan"]
    require([(r["offset"], r["tokens"]) for r in fragments] == shape
            and all(HEX.fullmatch(r["IDs_SHA256"]) for r in fragments), "prefill fragment plan changed")
    ids = request["generated_token_ids"]
    natural = bool(ids and ids[-1] in eos)
    times = request["sampled_ns"]
    entered = request["request_enter_ns"]
    require(1 <= len(ids) <= 4096 and all(type(t) is int and 0 <= t < 154880 for t in ids)
            and not any(t in eos for t in ids[:-1])
            and request["natural_EOS"] is natural and request["cap_hit"] is (not natural and len(ids) == 4096)
            and (natural or len(ids) == 4096)
            and request["generated_token_ids_SHA256"] == digest_ids(ids)
            and len(times) == len(ids) and type(entered) is int and times[0] > entered
            and all(type(t) is int for t in times)
            and all(a < b for a, b in zip(times, times[1:])), "generated order, EOS or clock differs")
    decode_calls = len(ids) - 1
    for key, count in (("prefill_CAPI_s", len(shape)), ("decode_CAPI_s", decode_calls)):
        values = request[key]
        require(len(values) == count and all(type(t) in (float, int) and math.isfinite(t)
                                             and t > 0 for t in values), "forward timing differs")
    require(request["past_after"] == pin["prompt_tokens"] + decode_calls
            and request["Native_forward_end"] - request["Native_forward_begin"] == len(shape) + decode_calls,
            "Model forward or context count differs")
    selected = request["selected_value"]
    begin, prefill, end = (selected[k] for k in ("begin", "after_prefill", "end"))
    require(begin == previous == prefill and all(
        isinstance(c, list) and len(c) == 3 and all(type(n) is int for n in c)
        and c[0] == 1 and 0 <= c[2] <= c[1] for c in (begin, end)),
        "prefill changed counters or request counters lost continuity")
    packed, cached = end[1] - begin[1], end[2] - begin[2]
    require(0 <= cached <= packed <= decode_calls and (item["arm"] == "on" or packed == 0)
            and selected["packed_calls"] == packed and selected["cached_dispatch_calls"] == cached
            and selected["packed_path_observed"] is (packed > 0), "selected continuation counters differ")
    visible = [when for token, when in zip(ids, times) if token not in eos and token not in special]
    decode = (len(visible)-1)*1e9/(visible[-1]-visible[0]) if len(visible) > 1 else None
    ttft = (visible[0]-entered)/1e9 if visible else None
    require(request["backend_decode_tok_s"] == decode and request["backend_sampled_TTFT_s"] == ttft
            and request["visible_nonEOS_nonspecial"] == len(visible)
            and request["whole_backend_s"] == (times[-1]-entered)/1e9, "backend clock differs")
    return end, dict(arm=item["arm"], ordinal=pin["ordinal"], length=len(ids),
                     output_SHA256=request["generated_token_ids_SHA256"], natural_EOS=natural,
                     cap_hit=request["cap_hit"], visible=len(visible), decode_tok_s=decode, ttft_s=ttft,
                     packed_calls=packed, cached_calls=cached)


def verify_owners(out, manifest, step, gate, native_sha):
    binding = json.loads((D / "INPUT_BINDING.json").read_bytes())
    pins = binding["private_plan"]["requests"]
    eos, special = set(binding["eos_token_ids"]), set(binding["tokenizer"]["special_ids"])
    hashes, barriers, uids, root, logs = {}, None, None, None, []
    coverage = [0] * 8
    per_rank_counters = {}
    for node_index, node in enumerate(NODES):
        worker_path = out / f"worker-{node_index}.json"
        worker, hashes[worker_path.name] = read_json(worker_path)
        identity = worker["identity"]
        require(worker["node"] == node and worker["completed"] is True and worker["passed"] is True
                and worker["actual_GPU"] == step and identity["arm"] == "socket"
                and identity["Native_SHA256"] == native_sha and identity["run_manifest_SHA256"] == manifest
                and not worker.get("forced_TERM") and not worker.get("forced_KILL")
                and [v["phase"] for v in worker["rounds"]] == phases(), "leader completion differs")
        digests = [r["digest"] for r in worker["rounds"]]
        require(barriers in (None, digests) and all(HEX.fullmatch(x) for x in digests), "leader barriers differ")
        barriers = digests
        seen_pci = set()
        for device, owned in enumerate(worker["owners"]):
            rank = node_index * 8 + device
            path = out / f"owner-{rank:02d}.json"
            owner, hashes[path.name] = read_json(path)
            require(owned["rank"] == rank == owner["global_rank"] and owner["device"] == device
                    and owner["PID"] == owned["PID"] and owner["startticks"] == owned["startticks"]
                    and owner["node"] == node and owner["actual_GPU"] == step
                    and owner["Native_SHA256"] == native_sha and owner["GPU_manifest_SHA256"] == manifest
                    and owner["completed"] is True and owner["passed"] is True and owner["normal_close"] is True
                    and owner["Runtime_weak_expired_CABI_assertion"] and owner["resets"] == 16
                    and owner["all78layers_loaded"] is True and owner["loaded_parameters"] == owner["checkpoint_key_count"] == 117060
                    and owner["selected_after_load"] == [1, 0, 0] and owner["planned_requests"] == schedule()
                    and len(owner["requests"]) == 8 and owner["driver_memory"]["min_observed"] >= FLOOR
                    and not owner["driver_memory"]["errors"] and all(r["free"] >= FLOOR for r in owner["memory_boundaries"]),
                    "owner identity, load, memory or shutdown differs")
            ids = owner["UID_SHA256"]
            require(owner["transport"] == "socket" and len(ids) == 2 and ids[0] != ids[1]
                    and all(HEX.fullmatch(x) for x in ids) and uids in (None, ids), "communicator IDs differ")
            uids = ids
            actual = {name: provider["SHA256"] for name, provider in owner["transport_providers"].items()}
            require(actual == dict(infiniccl=INFINICCL_SHA256, rccl=RCCL_SHA256), "Socket providers differ")
            require(all(owner["CAPI_bindings"][k]["SHA256"] == native_sha for k in ("selected_forward","selected_read")),
                    "selected C ABI providers differ")
            previous = [1, 0, 0]
            calls = 0
            rows = []
            for index, (request, item) in enumerate(zip(owner["requests"], schedule())):
                previous, row = audit_request(request, item, pins[item["case_index"]], previous, eos, special)
                require(request["Native_forward_begin"] == calls, "rank forward counter skipped")
                calls = request["Native_forward_end"]
                coverage[index] += bool(row["packed_calls"])
                row["driver_min_observed_bytes"] = owner["driver_memory"]["min_observed"]
                rows.append(row)
            require(owner["Native_forward_attempts"] == owner["Native_forward_completed"] == calls
                    and owner["control_calls"] == sum(r["length"] for r in rows), "owner totals differ")
            per_rank_counters[str(rank)] = [[r["packed_calls"], r["cached_calls"]] for r in rows]
            if rank == 0:
                root = rows
                prefill = [r["final_prefill_logits_SHA256"] for r in owner["requests"][:2]]
                require(all(HEX.fullmatch(x) for x in prefill), "missing diagnostic prefill logit hashes")
                require(owner["singleton_prefill_logits_match"] is (prefill[0] == prefill[1]),
                        "diagnostic prefill comparison differs")
                for row, request in zip(root, owner["requests"]):
                    score = request["score"]
                    row["score"] = {k: score[k] for k in ("correct", "parse_failure", "answer_matches_gold")}
                    require(all(type(v) is bool for v in row["score"].values()), "invalid score flags")
            else:
                require([r["output_SHA256"] for r in rows] == [r["output_SHA256"] for r in root],
                        "rank output hashes disagree")
            seen_pci.add(owner["physical_PCI"])
            with (out / f"owner-{rank:02d}.log").open("rb") as log:
                logs.append(log.read(1024**2).decode(errors="replace"))
        require(len(worker["owners"]) == 8 and seen_pci == set(gate[node]["pci_to_kfd"]), "leased PCI differs")
    pairs = []
    for index in (2, 4, 6):
        off, on = sorted(root[index:index+2], key=lambda r: r["arm"])
        same = off["output_SHA256"] == on["output_SHA256"]
        active = coverage[index + int(root[index]["arm"] == "off")] == 16
        ratio = on["decode_tok_s"]/off["decode_tok_s"] if same and active and off["decode_tok_s"] and on["decode_tok_s"] else None
        pairs.append(dict(ordinal=off["ordinal"], output_match=same, packed_on_all_ranks=active, ratio=ratio))
    valid = [p["ratio"] for p in pairs if p["ratio"] is not None]
    return dict(planned_requests=8, planned_measured_pairs=3, root_requests=root, per_rank_counters=per_rank_counters,
                rank_receipt_SHA256=hashes, packed_ranks_by_request=coverage, singleton_prefill_logit_SHA256=prefill,
                singleton_prefill_exact=prefill[0] == prefill[1], paired=pairs, comparable_pairs=len(valid),
                median_ratio=statistics.median(valid) if len(valid)==3 else None,
                network=network_channels("\n".join(logs), "socket"), online_SLO_qualified=False,
                broad_quality_qualified=False)
