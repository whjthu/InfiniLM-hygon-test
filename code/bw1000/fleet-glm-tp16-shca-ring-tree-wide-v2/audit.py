"""Consume exact independently attested per-rank and scheduler results."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from collective import PAYLOAD_ELEMENTS, SAMPLE_STEPS

PLUGIN_SHA256 = "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"

MODES = ("ring", "tree", "tree", "ring")
NETWORK = {mode: "IBext_v8" for mode in MODES}
ALGORITHM = {"ring": "Ring", "tree": "Tree"}


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


CHANNEL = re.compile(
    rb"Channel (\d+)/(\d+)\s*:\s*(\d+)\[[0-9a-f]+\] -> "
    rb"(\d+)\[[0-9a-f]+\](?: \[(send|receive)\])? via ([\w./-]+)"
    rb" comm(?: 0x[0-9a-f]+ nRanks (\d+)\b|"
    rb"(?: 0x[0-9a-f]+| )?I\d{4} [\d:.]+ \d+ ProcessGroupNCCL\.cpp:\d+\])"
)


def channel_edges(raw, logical_rank, expected):
    announced = re.findall(rb"Using network ([\w-]+)\b", raw)
    need(set(announced) == {expected}, "RCCL_SELECTED_NETWORK")
    evidence = {}
    node_local = False
    entries = CHANNEL.findall(raw)
    need(len(re.findall(rb"Channel [^\r\n]+ via NET/", raw)) ==
         sum(route.startswith(b"NET/") for _, _, _, _, _, route, _ in entries),
         "UNPARSED_NETWORK_CHANNEL")
    for channel, connection, source, target, role, route, ranks in entries:
        channel, connection, source, target = map(int, (channel, connection, source, target))
        need(ranks in (b"", b"16") and 0 <= source < 16 and 0 <= target < 16 and source != target, "CHANNEL_WORLD")
        if source // 8 == target // 8:
            need(not route.startswith(b"NET/"), "UNEXPECTED_NETWORK_FOR_LOCAL_CHANNEL")
            if ranks == b"16" and logical_rank in (source, target):
                node_local = True
            continue
        need(route.startswith(b"NET/" + expected + b"/") and
             ((role == b"send" and logical_rank == source) or
              (role == b"receive" and logical_rank == target)), "CROSS_NODE_CHANNEL_IDENTITY")
        evidence.setdefault((channel, connection, source, target), set()).add((role, bool(ranks)))
    need(bool(evidence) or node_local, "NO_CHANNEL_EVIDENCE_FOR_RANK")
    return evidence


def verify_edges(arm_edges):
    need({(s // 8, t // 8) for _, _, s, t in arm_edges} == {(0, 1), (1, 0)} and
         all({role for role, _ in roles} == {b"send", b"receive"} and any(full for _, full in roles)
             for roles in arm_edges.values()), "CROSS_NODE_SEND_RECEIVE_PAIR")
    return {f"{a}->{b}": sum(s // 8 == a and t // 8 == b for _, _, s, t in arm_edges)
            for a, b in ((0, 1), (1, 0))}


def verify_pair(root, pair, arms, pci):
    arm_medians = {str(elements): [] for elements in PAYLOAD_ELEMENTS}
    arm_busy = {str(elements): [] for elements in PAYLOAD_ELEMENTS}
    receipts = {}
    channel_summary = []
    algorithm_receipts = []
    expected_shapes = set(arm_medians)
    for i, mode in enumerate(MODES):
        step = arms[str(i)]["row"][0]
        measurements = {shape: [] for shape in expected_shapes}
        busy = {shape: [] for shape in expected_shapes}
        expected_marker = NETWORK[mode].encode()
        arm_edges = {}
        expected_plugin = [PLUGIN_SHA256]
        acknowledged = 0
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
                     and result["algorithm_policy"] == ALGORITHM[mode]
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
                log = directory / f"nccl-{card}.log"
                need(0 < log.stat().st_size <= 512 * 1024, "LOG_BOUND")
                raw = log.read_bytes()
                reported = set(re.findall(rb"NCCL_ALGO set by environment to (\w+)\b", raw))
                need(not reported or reported == {ALGORITHM[mode].encode()}, "RCCL_ALGORITHM_SELECTION_CONFLICT")
                acknowledged += bool(reported)
                for key, roles in channel_edges(raw, side * 8 + card, expected_marker).items():
                    arm_edges.setdefault(key, set()).update(roles)
                receipts[f"{i}/{node}/{card}-log"] = hashlib.sha256(raw).hexdigest()
            need(sorted(observed) == pci[node] and len(set(observed)) == 8, "EIGHT_PHYSICAL_DEVICES")
        need(acknowledged == 16, "RCCL_ALGORITHM_ANNOUNCEMENT_COVERAGE")
        channel_summary.append(verify_edges(arm_edges))
        algorithm_receipts.append(acknowledged)
        for shape, rows in measurements.items():
            need(len(rows) == 16 and all(row == rows[0] for row in rows), "SLOWEST_RANK_AGREEMENT")
            arm_medians[shape].append(statistics.median(rows[0]) * 1000)
            arm_busy[shape].append(dict(samples=len(busy[shape]), mean_percent=statistics.mean(busy[shape])))
    shapes = {}
    for shape, medians in arm_medians.items():
        ring = statistics.median(medians[0:4:3])
        tree = statistics.median(medians[1:3])
        shapes[shape] = dict(arms_median_ms=medians, ring_ms=ring, tree_ms=tree, ring_over_tree=ring/tree,
                             rank_local_busy_after_collective=arm_busy[shape])
    return dict(nodes=pair, shapes=shapes, directional_channel_edges_by_arm=channel_summary,
                algorithm_env_announcements_by_arm=algorithm_receipts,
                receipt_index_SHA256=hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest())
