"""Independently verify the paired fresh K/V worker's complete trace."""

from hashlib import sha256
import json
import math
from statistics import median


def number(value):
    return type(value) in {int, float} and math.isfinite(value)


def memory(info, floor):
    free, total = info["free_bytes"], info["total_bytes"]
    if type(free) is not int or type(total) is not int or not floor <= free <= total:
        raise ValueError("actual free memory or total capacity differs")
    return total


def audit_arm(row, identity, expected, total, timing):
    inputs, outputs = expected
    tokens = identity[0]
    if not (row["state"] == "ARM_PASS" and row["id"] == identity
            and type(row["warm_submitted"]) is int and row["warm_submitted"] == timing.WARM
            and type(row["timed_submitted"]) is int and row["timed_submitted"] == timing.TIMED[tokens]
            and row["input_SHA256"] == inputs and row["warm_output_SHA256"] == row["timed_output_SHA256"] == outputs
            and row["guards"] == "bitwise" and memory(row["before"], timing.FLOOR) == total
            and memory(row["after"], timing.FLOOR) == total):
        raise ValueError("BF16 result, redzones, memory, or complete arm count differs")
    gpu = row["gpu_event_ms"]
    submit, wait, elapsed, query = (row[name] for name in
                                    ("host_submit_ns", "host_wait_ns", "host_elapsed_ns", "event_query_ns"))
    if not (number(gpu) and gpu > 0 and all(type(value) is int for value in (submit, wait, elapsed, query))
            and submit > 0 and wait >= 0 and elapsed == submit + wait and query >= 0):
        raise ValueError("device event or host interval is incomplete")
    return dict(id=identity, gpu_event_ms=gpu, host_elapsed_ns=elapsed, host_submit_ns=submit, host_wait_ns=wait,
                gpu_event_ns_per_call=gpu * 10**6 / timing.TIMED[tokens], timed_submitted=timing.TIMED[tokens],
                free_bytes_before=row["before"]["free_bytes"], free_bytes_after=row["after"]["free_bytes"])


def summaries(arms, timing):
    index = {tuple(arm["id"]): arm for arm in arms}
    result = {}
    for tokens in timing.SHAPES:
        paired, unmatched = [], []
        for pair in range(len(timing.ORDERS)):
            first, second = (index.get((tokens, pair, tile)) for tile in timing.TILES)
            if first is None or second is None:
                continue
            before = [first["free_bytes_before"], second["free_bytes_before"]]
            after = [first["free_bytes_after"], second["free_bytes_after"]]
            if before[0] != before[1] or after[0] != after[1]:
                unmatched.append(dict(pair=pair, free_bytes_before=before, free_bytes_after=after))
                continue
            device = first["gpu_event_ms"] / second["gpu_event_ms"]
            host = first["host_elapsed_ns"] / second["host_elapsed_ns"]
            if not all(number(value) and value > 0 for value in (device, host)):
                raise ValueError("invalid measured pair ratio")
            paired.append(dict(pair=pair, device_throughput_128_over_64=device, host_throughput_128_over_64=host))
        result[str(tokens)] = dict(planned_pairs=len(timing.ORDERS), measured_pairs=len(paired), pairs=paired,
                                   unmatched_available_memory=unmatched,
                                   median_device_rate_128_over_64=median(p["device_throughput_128_over_64"] for p in paired)
                                   if paired else None,
                                   median_host_rate_128_over_64=median(p["host_throughput_128_over_64"] for p in paired)
                                   if paired else None)
    return result


def inspect_probe(path, row, intent_sha, before, probe, timing, host):
    result = dict(worker=None, expected=probe.PLANNED, passed=[], failed_arm=None, trace_pass=False,
                  component_only=True, Model_GO=False)
    if not path.is_file():
        return result
    records, offset = [], 0
    try:
        raw = host.bounded(path, 131072)
        result["trace_SHA256"] = sha256(raw).hexdigest()
        records = raw.splitlines(keepends=True)

        def read(index):
            host.need(index < len(records) and records[index].endswith(b"\n"), "original trace is incomplete")
            return json.loads(records[index])

        owner = host.own_identity(read(0), row, intent_sha, probe)
        host.need(owner["boot_id"] == before["boot_id"], "original worker boot differs from admit")
        result["worker"] = owner
        offset = 1
        bound = read(offset)
        if bound.get("state") == "FAIL":
            host.need(len(records) == 2 and type(bound.get("error")) is str and type(bound.get("type")) is str,
                      "ill-formed pre-binding failure")
            result["worker_failure"] = {key: bound[key][:1024] for key in ("type", "error")}
            return result
        host.need(bound["state"] == "BOUND", "original HIP binding receipt missing")
        dev, providers = bound["physical_device"], bound["providers"]
        host.need(providers["count"] == 71 and bound["hip_SHA256"] == probe.HIP_SHA
                  and providers["canonical_HIP"] == str(probe.HIP_CANONICAL)
                  and providers["canonical_HIP_bytes"] == probe.HIP_BYTES
                  and dev["pci"] in before["pci"] and dev["visible_devices"] == 1
                  and dev["kfd"]["pci"] == dev["pci"] and dev["kfd"]["gpu_id"] > 0,
                  "HIP binding, provider closure or physical PCI differs")
        result["physical_device"] = dev
        device_total = memory(bound["before"], timing.FLOOR)
        expected = {tokens: timing.fixed_hashes(tokens) for tokens in timing.SHAPES}
        offset += 1
        for identity in timing.PLAN["arms"]:
            tok = identity[0]
            host.need(read(offset) == dict(state="ARM_START", id=identity, planned_warm=timing.WARM,
                                          planned_timed=timing.TIMED[tok]), "original arm order/count differs")
            result["last_started"] = identity
            offset += 1
            arm = read(offset)
            if arm["state"] == "ARM_FAIL":
                host.need(arm["id"] == identity and type(arm["type"]) is str and type(arm["error"]) is str
                          and type(arm["warm_submitted"]) is int and type(arm["timed_submitted"]) is int
                          and 0 <= arm["warm_submitted"] <= timing.WARM and 0 <= arm["timed_submitted"] <= timing.TIMED[tok],
                          "failed arm identity or partial count differs")
                result["failed_arm"] = {key: arm[key] for key in (
                    "id", "warm_submitted", "timed_submitted", "type", "error")}
                host.need(len(records) == offset+2 and read(offset+1)["state"] == "FAIL", "failed arm terminal differs")
                return result
            result["passed"].append(audit_arm(arm, identity, expected[tok], device_total, timing))
            offset += 1
        finish = read(offset)
        host.need(len(records) == offset+1 and finish["state"] == "PASS" and finish["teardown"] == "normal"
                  and all(memory(finish[key], timing.FLOOR) == device_total for key in ("during", "after")),
                  "normal cleanup receipt or memory differs")
        result["pairs_by_tokens"] = summaries(result["passed"], timing)
        result["trace_pass"] = all(item["measured_pairs"] == item["planned_pairs"] for item in result["pairs_by_tokens"].values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:320]}"
    return result
