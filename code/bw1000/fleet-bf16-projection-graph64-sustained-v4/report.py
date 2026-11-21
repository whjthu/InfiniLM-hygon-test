"""Independently audit the original sustained Graph64 worker receipts."""

from hashlib import sha256
import json
import math
import statistics

from contract import NANO, ORDER, PROBES, need, validate_arm
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS


HIP_SHA = "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
METRICS = ("groups", "gemm_calls", "useful_rows", "physical_rows", "padded_rows")


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


def compare(root):
    audit = dict(valid_cards=0, completed_arms=0, actual_busy_card_samples=0, passed=False,
                 cooperative_stopped=False, raw_SHA256={}, failures=[], per_arm=[])
    pins = audit["raw_SHA256"]
    try:
        intent = records(root, "intent.json", pins)
        terminal = records(root, "terminal.json", pins)
        need(all(intent[key] == terminal[key] for key in intent) and intent["planned_cards"] == 8
             and intent["planned_arms"] == 32 and intent["planned_busy_card_samples"] == 8*4*len(PROBES)
             and len(intent["selected_PCI"]) == len(set(intent["selected_PCI"])) == 8,
             "worker terminal changed the original identity or denominator")
        audit["original_step"] = intent["step"]
        audit["cooperative_stopped"] = terminal["cooperative_stop"] is True
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit["failures"].append(str(error)[:120])
        return audit
    addresses, grid = intent["selected_PCI"], intent["grid"]
    need(len(grid) == 4 and tuple(item[0] for item in grid) == ORDER, "worker arm sequence differs")
    card_arms, valid, pairs = {}, set(), {}
    totals = [dict(mode=mode, planned_cards=8, complete_cards=0, measured_partial_cards=0, complete_groups_by_quarter=[0]*4,
                   measured_card_samples=0, **dict.fromkeys(METRICS, 0)) for mode, _, _ in grid]
    for card, pci in enumerate(addresses):
        arms = card_arms[pci] = {}
        try:
            saved = records(root, f"card-{card}.json", pins)
            need(saved["card"] == card and saved["PCI"] == pci and saved["provider_SHA256"] == HIP_SHA
                 and set(saved["capture_ns"]) == set(ORDER)
                 and all(type(ns) is int and ns > 0 for ns in saved["capture_ns"].values()),
                 "owner, HIP provider or two graph captures differ")
            for index, arm in enumerate(saved["arms"]):
                if "state" not in arm:
                    break
                independent = records(root, f"card-{card}-arm-{index}.json", pins)
                need(independent == arm and index < len(grid), "owner arm differs from durable receipt")
                state = validate_arm(arm, index, grid[index])
                arms[index] = arm
                totals[index]["complete_cards"] += state == "COMPLETE"
                totals[index]["measured_partial_cards"] += state == "COOPERATIVE_STOPPED" and arm["groups"] > 0
                for key in METRICS:
                    totals[index][key] += arm[key]
                totals[index]["complete_groups_by_quarter"] = [a+b for a,b in zip(
                    totals[index]["complete_groups_by_quarter"], arm["complete_groups_by_quarter"], strict=True)]
            if (len(arms) == 4 and all(arm["passed"] for arm in arms.values())
                    and saved["normal_release"] is True and saved["error"] is None):
                valid.add(pci)
                rate = lambda index: arms[index]["useful_rows"]*NANO/arms[index]["wall_ns"]
                pairs[pci] = [rate(1)/rate(0), rate(2)/rate(3)]
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit["failures"].append(f"card-{card}: {str(error)[:100]}")

    audit.update(valid_cards=len(valid), completed_arms=sum(row["complete_cards"] for row in totals),
                 per_card_padded_over_native_pairs=pairs, per_arm=totals)
    by_arm = [[] for _ in grid]
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
            for pci, value in values.items():
                arm = card_arms[pci].get(index)
                if arm and arm["groups"] > 0 and arm["actual_start_epoch_ns"] <= begin <= end <= arm["actual_end_epoch_ns"]:
                    by_arm[index].append(value)
    except (OSError, KeyError, TypeError, ValueError) as error:
        audit["failures"].append("driver: "+str(error)[:120])
    for total, values in zip(totals, by_arm, strict=True):
        total.update(measured_card_samples=len(values), busy_mean=statistics.mean(values) if values else None)
    audit["actual_busy_card_samples"] = sum(len(values) for values in by_arm)
    audit["passed"] = (not audit["failures"] and not audit["cooperative_stopped"] and len(valid) == 8
                       and audit["completed_arms"] == terminal["completed_arms"] == 32
                       and terminal["valid_cards"] == 8 and audit["actual_busy_card_samples"] == 8*4*len(PROBES)
                       and timely_gate(terminal["setup_gate"], grid[0][1]-SETUP_LEAD_NS)
                       and len(terminal["arm_gates"]) == 4
                       and all(timely_gate(gate, window[1]-ARM_LEAD_NS)
                               for gate, window in zip(terminal["arm_gates"], grid, strict=True)))
    return audit
