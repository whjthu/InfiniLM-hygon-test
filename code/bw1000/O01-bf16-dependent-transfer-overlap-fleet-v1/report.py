"""Independently audit paired depth, exact logical work, and host boundary receipts."""

from hashlib import sha256
import json
import math
import re
import statistics

from contract import CAPTURE_DEPTHS, HOST_COUNTERS, LANES, NANO, ORDER, PHASES, PROBES, need, validate_arm
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS


MEMORY_PHASES = ("memory_before","memory_after_fixture","memory_after_capture")


HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
METRICS = ("groups", "graph_replays", "stream_fences", "transfer_calls", "d2h_payload_bytes", "logical_pairs", "positive_pairs", "negative_pairs", "projection_calls", "consumer_calls",
           "useful_rows", "input_d2d_bytes", "wall_ns", "compute_stream_ms", "copy_stream_ms", *HOST_COUNTERS)


def records(root, name, pins, *, lines=False):
    path = root / name
    need(path.resolve(strict=True) == path and path.is_file(), "worker receipt path differs")
    with path.open("rb") as stream:
        raw = stream.read(131073)
    need(0 < len(raw) <= 131072, "worker receipt size differs")
    pins[name] = sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)


def timely_gate(gate, cutoff):
    arrivals = gate["arrived_epoch_ns"]
    return (gate["first_failure"] is None and gate["expected_cards"] == 8
            and set(arrivals) == set("01234567") and gate["cutoff_epoch_ns"] == cutoff
            and all(type(value) is int and value<cutoff for value in arrivals.values())
            and gate["released_epoch_ns"] == max(arrivals.values()))


def node_setup(intent):
    fields = ("process_enter", "setup_enter", "torch_import_begin", "torch_import_end", "hip_ready")
    ticks = [intent[name+"_epoch_ns"] for name in fields]
    opening = intent["setup_open_epoch_ns"]
    need(all(type(value) is int for value in [opening, *ticks])
         and ticks == sorted(ticks) and opening <= ticks[1], "owner initialization stages are invalid")
    return dict(setup_open_epoch_ns=opening, **dict(zip(fields, ticks, strict=True)))


def card_setup(saved, node_setup_marks):
    fields = ("owner_start", "projection_ready", "pair_capture_begin", "pair_capture_proved")
    ticks = [saved["setup_marks"][name+"_epoch_ns"] for name in fields]
    need(all(type(value) is int for value in ticks) and ticks == sorted(ticks)
         and ticks[0] >= node_setup_marks["hip_ready"], "card initialization stages are invalid")
    return dict(zip(fields, ticks, strict=True))


def compare(root):
    audit = dict(valid_cards=0, completed_arms=0, actual_busy_card_samples=0, passed=False,
                 cooperative_stopped=False, raw_SHA256={}, failures=[], per_arm=[], card_setup={}, common_window_epoch_ns=None, common_sample_observations=[])
    pins = audit["raw_SHA256"]
    try:
        intent = records(root, "intent.json", pins)
        terminal = records(root, "terminal.json", pins)
        need(all(intent[key] == terminal[key] for key in intent) and intent["planned_cards"] == 8
             and intent["planned_arms"] == 8*len(ORDER) and intent["planned_busy_card_samples"] == 8*len(ORDER)*len(PROBES)
             and len(intent["selected_PCI"]) == len(set(intent["selected_PCI"])) == 8,
             "worker terminal changed the original identity or denominator")
        audit["original_step"] = intent["step"]
        audit["cohort"] = intent["cohort"]
        audit["setup"] = node_setup(intent)
        audit["cooperative_stopped"] = terminal["cooperative_stop"] is True
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit["failures"].append(str(error)[:120])
        return audit
    addresses, grid = intent["selected_PCI"], intent["grid"]
    if len(grid) != len(ORDER) or tuple(item[0] for item in grid) != ORDER:
        audit["failures"].append("worker arm sequence differs")
        return audit
    card_arms, valid, pairs = {}, set(), {}
    totals = [dict(mode=mode, planned_cards=8, complete_cards=0, measured_partial_cards=0, complete_groups_by_quarter=[0]*4,
                   measured_card_samples=0, **dict.fromkeys(METRICS, 0)) for mode, _, _ in grid]
    for card, pci in enumerate(addresses):
        arms = card_arms[pci] = {}
        try:
            saved = records(root, f"card-{card}.json", pins)
            need(saved["card"] == card and saved["PCI"] == pci and saved["provider_SHA256"] == HIP_SHA
                 and saved["preparation"]["graph_changed_input_across_replays_cpu_proof"] is True,
                 "owner, HIP provider or changed-input graph proof differs")
            prep = saved["preparation"]
            transfer_sha = prep["original_transfer_SHA256"]
            need(isinstance(transfer_sha,str) and re.fullmatch(r"[0-9a-f]{64}",transfer_sha),
                 "pinned original transfer payload proof differs")
            stages = prep["diagnostic_stage_ms_by_lane"]
            need(type(stages) is list and len(stages)==LANES
                 and all(set(lane)==set(PHASES) and all(type(ms) in (int,float) and math.isfinite(ms) and ms > 0
                                                      for ms in lane.values()) for lane in stages)
                 and set(prep["by_depth"])==set(CAPTURE_DEPTHS)
                 and all(set(times)=={"capture_ns","first_replay_ns"}
                         and all(type(pairs) is list and len(pairs)==2 and all(type(ns) is int and ns>0 for ns in pairs)
                                 for pairs in times.values()) for times in prep["by_depth"].values()),
                 "original per-stage, capture or first-validation timing differs")
            memory = {phase: saved[phase] for phase in MEMORY_PHASES}
            need(all(type(info["free_bytes"]) is int and type(info["total_bytes"]) is int
                     and 8*1024**3 <= info["free_bytes"] <= info["total_bytes"] for info in memory.values())
                 and len({info["total_bytes"] for info in memory.values()}) == 1,
                 "qualification memory driver boundaries differ")
            readiness = records(root, f"setup-card-{card}.json", pins)
            need(all(readiness[key] == expected for key, expected in (
                ("request", intent["request"]), ("step", intent["step"]), ("node", intent["node"]),
                ("card", card), ("preparation", prep), ("setup_marks", saved["setup_marks"]), ("memory", memory))),
                "durable setup markers differ from card owner")
            audit["card_setup"][pci] = dict(card_setup(saved, audit["setup"]), preparation=prep,
                                                 boundary_memory=memory)
            for index, arm in enumerate(saved["arms"]):
                if "state" not in arm:
                    break
                independent = records(root, f"card-{card}-arm-{index}.json", pins)
                need(independent == arm and index < len(grid), "owner arm differs from durable receipt")
                state = validate_arm(arm,index,grid[index])
                need(not arm["groups"] or arm["transfer_SHA256"]==transfer_sha,
                     "arm transfer differs from the original source buffer")
                arms[index] = arm
                totals[index]["complete_cards"] += state == "COMPLETE"
                totals[index]["measured_partial_cards"] += state == "COOPERATIVE_STOPPED" and arm["groups"] > 0
                for key in METRICS:
                    totals[index][key] += arm[key]
                totals[index]["complete_groups_by_quarter"] = [a+b for a,b in zip(
                    totals[index]["complete_groups_by_quarter"], arm["complete_groups_by_quarter"], strict=True)]
            if (len(arms) == len(ORDER) and all(arm["passed"] for arm in arms.values())
                    and saved["normal_release"] is True and saved["error"] is None):
                valid.add(pci)
                rate = lambda index: arms[index]["useful_rows"]*NANO/arms[index]["wall_ns"]
                pairs[pci] = [rate(1)/rate(0), rate(2)/rate(3)]
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit["failures"].append(f"card-{card}: {str(error)[:100]}")

    audit.update(valid_cards=len(valid), completed_arms=sum(row["complete_cards"] for row in totals),
                 per_card_overlap_over_serial_pairs=pairs, per_arm=totals)
    by_arm = [[] for _ in grid]
    common_samples = []
    try:
        seen = set()
        for sample in records(root, "busy.jsonl", pins, lines=True):
            index = sample["arm"]
            need(type(index) is int and 0 <= index < len(grid), "unknown driver sample arm")
            mode, start, _ = grid[index]
            instant = sample["planned_epoch_ns"]
            key = (index, instant)
            need(sample["mode"] == mode and key not in seen and instant in {start+offset*NANO for offset in PROBES},
                 "duplicate or off-grid driver sample")
            seen.add(key)
            values = sample["busy_percent"]
            if values is None:
                continue
            begin, end = sample["observed_begin_epoch_ns"], sample["observed_end_epoch_ns"]
            need(set(values) == set(addresses) and all(type(value) is int and 0 <= value <= 100 for value in values.values())
                 and type(begin) is type(end) is int and instant <= begin <= end and end-begin <= 150_000_000,
                 "unsupported driver sample")
            accepted = []
            for pci,value in values.items():
                arm = card_arms[pci].get(index)
                if arm and arm["groups"] > 0 and arm["actual_start_epoch_ns"] <= begin <= end <= arm["actual_end_epoch_ns"]:
                    accepted.append(value)
            by_arm[index].extend(accepted)
            if index == len(ORDER)-1 and len(accepted) == 8:
                common_samples.append(dict(planned_epoch_ns=instant,opened_epoch_ns=begin,closed_epoch_ns=end,
                                           cards=8,sum_busy_percent=sum(accepted)))
    except (OSError, KeyError, TypeError, ValueError) as error:
        audit["failures"].append("driver: "+str(error)[:120])
    for total, values in zip(totals, by_arm, strict=True):
        total.update(measured_card_samples=len(values), busy_mean=statistics.mean(values) if values else None)
    audit["actual_busy_card_samples"] = sum(len(values) for values in by_arm)
    audit["passed"] = (not audit["failures"] and not audit["cooperative_stopped"] and len(valid) == 8
                       and audit["completed_arms"] == terminal["completed_arms"] == 8*len(ORDER)
                       and terminal["valid_cards"] == 8 and audit["actual_busy_card_samples"] == 8*len(ORDER)*len(PROBES)
                       and timely_gate(terminal["setup_gate"], grid[0][1]-SETUP_LEAD_NS)
                       and len(terminal["arm_gates"]) == len(ORDER)
                       and all(timely_gate(gate, window[1]-ARM_LEAD_NS)
                               for gate, window in zip(terminal["arm_gates"], grid, strict=True)))
    if audit["passed"]:
        common = [card_arms[pci][len(ORDER)-1] for pci in addresses]
        start = max(arm["actual_start_epoch_ns"] for arm in common)
        end = min(arm["actual_end_epoch_ns"] for arm in common)
        if start < end:
            audit["common_window_epoch_ns"] = [start,end]
            audit["common_sample_observations"] = common_samples
    return audit
