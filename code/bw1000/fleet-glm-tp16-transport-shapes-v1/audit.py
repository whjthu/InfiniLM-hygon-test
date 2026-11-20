"""Consume exact independently attested per-rank and scheduler results."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from collective import PAYLOAD_ELEMENTS, SAMPLE_STEPS

PLUGIN_SHA256 = "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"

MODES=("socket","shca","shca","socket")
NETWORK={"socket":"Socket","shca":"IBext_v8"}


def need(condition,reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    raw=path.read_bytes()
    need(0<len(raw)<4*1024**2,"RECEIPT_BOUND")
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def verify(root,pairs,request):
    terminal,pin=read(root/"dispatch-terminal.json")
    before=terminal["before"]; after=terminal["after"]
    need(terminal["request"]==request,"TERMINAL_REQUEST")
    need(after is not None and before["accounting"][2:4]==after["accounting"][2:4]==["COMPLETED","0:0"],
         "INDEPENDENT_TERMINAL_CLEANUP")
    pin_before={r["node"]:r["pci"] for r in before["nodes"]}
    pin_after={r["node"]:r["pci"] for r in after["nodes"]}
    nodes=[n for pair in pairs for n in pair]
    need(pin_before==pin_after and set(pin_before)==set(nodes)
         and all(r["holders"]==0 for r in before["nodes"]+after["nodes"]),"INDEPENDENT_PHYSICAL_CLEANUP")
    results=[]; failed=[]
    for pair in pairs:
        try:
            row=verify_pair(root,pair,terminal["arms"],pin_before)
            results.append(row)
        except (OSError,ValueError,KeyError,TypeError) as error:
            reason=str(error) if isinstance(error,ValueError) and re.fullmatch("[A-Z_0-9]+",str(error)) else type(error).__name__
            failed.append(dict(nodes=pair,reason=reason))
    return dict(planned_pairs=len(pairs),complete_pairs=len(results),planned_dcu=len(nodes)*8,
                accounting=terminal["arms"],independent_pre=before["accounting"],
                independent_post=after["accounting"],terminal_SHA256=pin,pairs=results,failures=failed,
                controller_failure=terminal["failure"])


def verify_pair(root, pair, arms, pci):
    arm_medians = {str(elements): [] for elements in PAYLOAD_ELEMENTS}
    arm_busy = {str(elements): [] for elements in PAYLOAD_ELEMENTS}
    receipts = {}
    expected_shapes = set(arm_medians)
    for i, mode in enumerate(MODES):
        step = arms[str(i)]["row"][0]
        measurements = {shape: [] for shape in expected_shapes}
        busy = {shape: [] for shape in expected_shapes}
        expected_marker = NETWORK[mode].encode()
        expected_plugin = [PLUGIN_SHA256] if mode == "shca" else []
        for side, node in enumerate(pair):
            directory = root / f"arm-{i}" / node
            outcome, digest = read(directory / "terminal.json")
            receipts[f"{i}/{node}/terminal"] = digest
            need(outcome["step"] == step and outcome["mode"] == mode and outcome["cohort"] == pair
                 and not outcome["error"] and not outcome["post_kfd_holders"]
                 and outcome["exits"] == {str(j): 0 for j in range(8)}, "NODE_ARM_RECEIPT")
            observed = []
            for card in range(8):
                rank, digest = read(directory / f"card-{card}.json")
                receipts[f"{i}/{node}/{card}"] = digest
                need(rank["step"] == step and rank["card"] == card and rank["node"] == node and not rank["error"],
                     "ACTUAL_CARD_RECEIPT")
                observed.append(rank["pci"])
                result, = rank["results"]
                need(result["logical_rank"] == side * 8 + card and result["world_size"] == 16
                     and result["network_policy"] == NETWORK[mode] and result["plugin_sha256"] == expected_plugin
                     and set(result["sizes"]) == expected_shapes, "MATCHED_PAYLOAD_AND_WORLD_SIZE")
                for shape, data in result["sizes"].items():
                    need(data["payload_bytes"] == int(shape) * 4 and data["verified_elements"] == 2 * int(shape),
                         "PAYLOAD_AND_NUMERIC_DENOMINATOR")
                    times = data["rank_max_seconds"]
                    need(len(times) == 96 and all(math.isfinite(x) and x > 0 for x in times), "TIMED_SERIES")
                    measurements[shape].append(times)
                    samples = data["rank_busy_after_collective"]
                    need(len(samples) == len(SAMPLE_STEPS)
                         and [s[0] for s in samples] == list(SAMPLE_STEPS)
                         and all(math.isfinite(s[1]) and s[1] > 0 and type(s[2]) is int and 0 <= s[2] <= 100
                                 for s in samples)
                         and all(a[1] < b[1] for a, b in zip(samples, samples[1:])), "DEVICE_SAMPLE_POINTS")
                    busy[shape].extend(s[2] for s in samples)
                log = directory / f"card-{card}.log"
                need(0 < log.stat().st_size <= 512 * 1024, "LOG_BOUND")
                raw = log.read_bytes()
                need(set(re.findall(rb"Using network (Socket|IBext_v8)\b", raw)) == {expected_marker}
                     and set(re.findall(rb"via NET/(Socket|IBext(?:_v8)?)(?:/|\b)", raw)) == {expected_marker},
                     "ACTUAL_DATA_CHANNEL")
                receipts[f"{i}/{node}/{card}-log"] = hashlib.sha256(raw).hexdigest()
            need(sorted(observed) == pci[node] and len(set(observed)) == 8, "EIGHT_PHYSICAL_DEVICES")
        for shape, rows in measurements.items():
            need(len(rows) == 16 and all(row == rows[0] for row in rows), "SLOWEST_RANK_AGREEMENT")
            arm_medians[shape].append(statistics.median(rows[0]) * 1000)
            arm_busy[shape].append(dict(samples=len(busy[shape]), mean_percent=statistics.mean(busy[shape])))
    shapes = {}
    for shape, medians in arm_medians.items():
        socket = statistics.median(medians[0:4:3])
        shca = statistics.median(medians[1:3])
        shapes[shape] = dict(arms_median_ms=medians, socket_ms=socket, shca_ms=shca, socket_over_shca=socket/shca,
                             rank_local_busy_after_collective=arm_busy[shape])
    return dict(nodes=pair, shapes=shapes,
                receipt_index_SHA256=hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest())
