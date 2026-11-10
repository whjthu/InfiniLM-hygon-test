"""Reduce validated sparse R454 packets, with matched-prefix observer cost controls."""

import json
import math
import statistics

from pilot_inputs import require, sha

INDICES = (8, 87)
PAST = (1249, 1328)


def median_spans(spans):
    return {name: statistics.median(values) for name, values in spans.items()}


def summarize(requests, root):
    require(len(requests) == 3, "wrong diagnostic request denominator")
    arms = [row["i03_sparse_observation"] for row in requests]
    require([row["armed"] for row in arms] == [False, True, False], "wrong observer schedule")
    tables = []
    for arm in arms:
        rows = arm["windows"]
        lookup = {row["decode_index"]: row for row in rows}
        require(len(lookup) == len(rows) and set(lookup) <= set(INDICES)
                and arm["missing_samples"] == [sample for sample in INDICES if sample not in lookup],
                "wrong sparse sample inventory")
        tables.append(lookup)
    output, paths = [], []
    for index, past in zip(INDICES, PAST):
        row = {"index": index, "past": past}
        observations = [table.get(index) for table in tables]
        observed = [item for item in observations if item is not None]
        require(all(item["past"] == past and math.isfinite(item["host_wrapper_s_including_observer"])
                    and item["host_wrapper_s_including_observer"] > 0 for item in observed),
                "wrong decode position or elapsed time")
        row["arm_presence"] = [item is not None for item in observations]
        row["matched_generated_prefix"] = len(observed) == 3 and len({
            item["generated_prefix_SHA256"] for item in observed}) == 1
        armed = observations[1]
        if armed is None:
            output.append(row)
            continue
        profile = armed["profile"]
        require(profile["native_window_closed"] is True and profile["owner_event_validation_passed"] is True
                and profile["inner_event_validation_passed"] is True
                and profile["native_returncode"] == 0 and profile["geometry"]["past"] == [past],
                "unverified observer record")
        stem = f"request-1-decode-{index}"
        names = [stem + "-" + suffix + ".json" for suffix in ("owner", "inner", "metadata")]
        for suffix, filename in zip(("owner", "inner"), names):
            packet = profile["scalar_packets"][suffix]
            raw = (root / filename).read_bytes()
            require(len(raw) == packet["bytes"] and sha(raw) == packet["SHA256"], "scalar packet differs")
        metadata = json.loads((root / names[2]).read_bytes())
        require(metadata["geometry"] == profile["geometry"]
                and metadata["packets"] == profile["scalar_packets"]
                and metadata["native_returncode"] == 0, "profile packet metadata differs")
        paths.extend(names)
        inner = json.loads((root / names[1]).read_bytes())
        expected = {item["layer"]: item["kind"] for item in inner["rows"] if item["device"] == 0}
        kinds = list(expected.values())
        require(set(expected) == set(range(43)) and [kinds.count(k) for k in (0, 1, 2)] == [2, 20, 21],
                "wrong compressed-cache layer counts")
        # Only named cache-compression/index intervals, summed across layers on each rank.
        values = {ratio: {metric: [0.] * 8 for metric in ("stream_ms", "host_wall_ms", "thread_cpu_ms")}
                  for ratio in ("0", "128", "4")}
        for item in inner["rows"]:
            require(expected[item["layer"]] == item["kind"], "cache kind differs across owners")
            ratio = {0:"0",1:"128",2:"4"}[item["kind"]]
            for metric in values[ratio]:
                values[ratio][metric][item["device"]] += item["segments"][1][metric]
        row["cache_compression_index_by_ratio_median_owner_ms"] = {
            ratio: median_spans(metrics) for ratio, metrics in values.items()}
        row["owner_median_stream_ms"] = median_spans(profile["old_perstage_perdevice_ms"])
        row["attention_inner_median_stream_ms"] = median_spans(profile["inner_persegment_perdevice_ms"])
        row["armed_policy_forward_s"] = profile["native_forward_wall_s"]
        row["observer_total_s"] = profile["observer_total_wall_s"]
        if row["matched_generated_prefix"]:
            off = [observations[i]["host_wrapper_s_including_observer"] for i in (0, 2)]
            row["off_policy_forward_s"] = off
            row["armed_forward_over_off_mean"] = row["armed_policy_forward_s"] / statistics.mean(off)
        else:
            row["armed_forward_over_off_mean"] = None
        output.append(row)
    return {
        "planned_samples": len(INDICES),
        "qualified_samples": sum("armed_policy_forward_s" in row for row in output),
        "matched_samples": sum(row["matched_generated_prefix"] for row in output),
        "samples": output,
        "scope": "Named stream and host enclosures; overlapping parent/child stages and parallel owners must not be added. No isolated kernel, CCL or GPU busy attribution.",
    }, paths
