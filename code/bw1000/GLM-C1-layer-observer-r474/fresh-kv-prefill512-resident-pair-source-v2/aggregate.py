"""Independently verify all 16 primary records from the prefill pair."""

import re

from bindings import LOAD_KEYS, NATIVE_SHA, PROMPT_SHA
from run_pair import ARMS, CHUNKS, FLOOR, LAYERS, PCI_PATTERN, PLAN, counts, driver_summary, need


def positive(value):
    return type(value) is int and value > 0


def verify_owner(item, rank, pci):
    need(item["rank"] == rank and item["PCI"] == item["expected_PCI"] == pci
         and item["Native_SHA256"] == NATIVE_SHA and item["full_input_SHA256"] == PROMPT_SHA
         and item["plans"] == PLAN and item["state"] == "CLOSED" and item["normal_close"] is True
         and item["attempted_forwards"] == item["successful_forwards"] == 6 and item["reset_calls"] == 3
         and item["device_input_audits"] == 3
         and item["Model_GO"] is item["model_speed_win"] is False,
         "owner did not complete both full prompts and normal close")
    load = item["load_progress"]
    driver = driver_summary(lambda: item["driver_memory"])
    need(item["context_creations"] == item["checkpoint_loads"] == load["context_creations"]
         == load["context_create_attempts"] == load["checkpoint_loads"] == load["checkpoint_load_attempts"] == 1
         and item["parameters"] == load["checkpoint_key_count"] == load["loaded_parameters"] == LOAD_KEYS
         and load["actual_layers"] == list(range(LAYERS)) and load["all78layers_loaded"] is True,
         "full checkpoint and single load evidence differs")
    need([arm["arm"] for arm in item["arms"]] == list(ARMS), "full arm order differs")
    memories = [item["final_memory"], item["memory_after_close"]]
    all_fragments = []
    for arm in item["arms"]:
        memories.extend((arm["memory_before"], arm["memory_after"]))
        need([(f["past"], f["tokens"]) for f in arm["fragments"]] == list(CHUNKS), "full prompt fragment schedule differs")
        times = []
        for f in arm["fragments"]:
            opt_in = arm["arm"] == "ON" and f["past"] == 0
            need(f["opt_in"] is opt_in and f["state"] == "RETURNED" and positive(f["forward_ns"])
                 and re.fullmatch(r"[0-9a-f]{64}", f["IDs_SHA256"]), "completed fragment evidence differs")
            if opt_in:
                need(counts(f["observed_route_counts"]) == f["route_counts"], "actual route counters differ")
            else:
                need("route_counts" not in f, "unexpected optimization on unchanged fragment")
            if rank == 0:
                need(re.fullmatch(r"[0-9a-f]{64}", f["root_BF16_logits_SHA256"]), "complete rank0 output digest missing")
            else:
                need("root_BF16_logits_SHA256" not in f, "non-root rank claimed root output")
            times.append(f["forward_ns"])
            memories.append(f["memory_after"])
        need(arm["first_forward_ns"] == times[0] and arm["sum_three_forward_ns"] == sum(times)
             and arm["host_scope_ns"] >= sum(times), "first, full CABI and host clocks disagree")
        all_fragments.append(arm["fragments"])
    need(len({m["total"] for m in memories}) == 1 and
         all(type(m["free"]) is int and type(m["total"]) is int and FLOOR <= m["free"] <= m["total"]
             for m in memories), "original 8GiB memory evidence differs")
    off, on = all_fragments
    need(all(a["IDs_SHA256"] == b["IDs_SHA256"] for a,b in zip(off,on)), "fragment input changed between arms")
    routes = counts(on[0]["route_counts"])
    equal = None
    if rank == 0:
        checks = []
        for a,b in zip(off,on):
            same = a["root_BF16_logits_SHA256"] == b["root_BF16_logits_SHA256"]
            need(b["bitwise_equal_to_OFF"] is same, "rank0 byte comparison and output hashes disagree")
            checks.append(same)
        equal = sum(checks)
    need(item["all_local_layers_packed"] is (routes["packed"] == LAYERS)
         and item["root_logits_equal"] is (equal == 3 if rank == 0 else None), "owner outcome differs")
    return {"rank": rank, "route_counts": routes, "rank0_bitwise_fragments": equal,
            "minimum_boundary_free": min(m["free"] for m in memories),
            "minimum_sampled_free": driver["min_observed"],
            "fragment_input_SHA256": [fragment["IDs_SHA256"] for fragment in off],
            "arms": [{k: arm[k] for k in ("arm","first_forward_ns","sum_three_forward_ns","host_scope_ns")}
                     for arm in item["arms"]]}


def summarize(records, admitted_pci):
    """Return bounded results and keep absent, rejected, and fallback owners in the planned denominator."""
    ranks = set(range(16))
    need(set(admitted_pci) == ranks and all(re.fullmatch(PCI_PATTERN, pci) for pci in admitted_pci.values())
         and all(len({admitted_pci[rank] for rank in node}) == 8 for node in (range(8),range(8,16))),
         "expected eight fresh physical PCI functions per node")
    need(set(records) <= ranks, "unexpected rank in primary results")
    accepted, rejected = [], []
    for rank in range(16):
        if rank not in records:
            rejected.append({"rank": rank, "reason": "MISSING"})
            continue
        try:
            accepted.append(verify_owner(records[rank], rank, admitted_pci[rank]))
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"rank": rank, "reason": str(error)[:256]})
    packed = sum(item["route_counts"]["packed"] for item in accepted)
    equal = next((item["rank0_bitwise_fragments"] for item in accepted if item["rank"] == 0), None)
    input_agrees = len({tuple(item["fragment_input_SHA256"]) for item in accepted}) == 1
    return dict(owners_planned=16, forwards_planned=96, route_attempts_planned=16 * LAYERS,
                rank0_bitwise_fragments_planned=3, owners_verified=len(accepted), rank0_bitwise_fragments=equal,
                route_layers_packed=packed, rejected=rejected, owners=accepted, verified_inputs_agree=input_agrees,
                prefill_pair_evidence_pass=(len(accepted) == 16 and packed == 16 * LAYERS and equal == 3 and input_agrees),
                Model_GO=False, model_speed_win=False)
