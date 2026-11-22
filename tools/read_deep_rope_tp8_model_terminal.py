"""Audit the three original DeepSeek TP8 steps with only a bounded summary leaving SCNet."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil, isclose, isfinite
from pathlib import Path
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
            "deep-rope-real-model-tp8-diagnostic-v2")
REQUEST = "36e74fadf2c84d7b83b55c063842638e"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
CASES = {
    "f13r2n01": ("legacy", "1103606.1375", "1103606.1377", "1103606.1379"),
    "f13r2n02": ("h1-pack32", "1103606.1373", "1103606.1376", "1103606.1380"),
    "f13r2n03": ("h1-pack64", "1103606.1374", "1103606.1378", "1103606.1381"),
}
KEYS = 67569
KEY_DIGEST = "0f75a01821e63110339f40f44ff00681fc3e4e37d0c0a7bacdd1557d0ffe70c6"
FLOOR = 16 * 1024**3


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def read_file(path, limit=65536):
    need(path.resolve(strict=True) == path and path.is_file(), "Missing or aliased original receipt")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    need(0 < len(raw) <= limit, "Original receipt exceeds expected extent")
    return raw


def file_pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def read_json(path):
    return json.loads(read_file(path))


def clock_value(value):
    return type(value) in (int, float) and isfinite(value)


def memory_after(values):
    need(len(values) == 8 and all(type(free) is int and type(total) is int
                                 and FLOOR <= free <= total for free, total in values),
         "Driver sample below the 16 GiB floor or incomplete")
    return min(row[0] for row in values)


def request_metrics(saved):
    ids, ticks = saved["generated_ids"], saved["token_timestamps"]
    eos, cap = saved["EOS"], saved["cap"]
    need(1 <= len(ids) <= 4096 and len(ids) == len(ticks) and
         all(type(token) is int and 0 <= token < 129280 for token in ids), "Invalid complete output denominator")
    need(type(eos) is bool and type(cap) is bool and eos != cap and
         ((eos and ids[-1] == 1 and ids.count(1) == 1) or
          (cap and len(ids) == 4096 and 1 not in ids)), "Invalid EOS or capped-output boundary")
    need(all(map(clock_value, ticks)) and all(a < b for a, b in zip(ticks, ticks[1:])),
         "Invalid emitter time order")
    visible_ticks = ticks[:-1] if eos else ticks
    intervals = [after - before for before, after in zip(visible_ticks, visible_ticks[1:])]
    rate = len(intervals) / (visible_ticks[-1] - visible_ticks[0]) if intervals else None
    recorded = saved["owner_decode_tokens_s"]
    need((rate is None and recorded is None) or
         (rate is not None and clock_value(recorded) and isclose(rate, recorded, rel_tol=1e-12)),
         "Recorded owner rate does not recompute")
    ttft = saved["owner_first_emit_s"]
    need((not visible_ticks and ttft is None) or
         (bool(visible_ticks) and clock_value(ttft) and ttft > 0), "Invalid owner first emit time")
    expected = []
    position = 0
    for tokens in (1024,) * 8 + (297,):
        expected.append(dict(kind="prefill", past=position, tokens=tokens, total=position+tokens, emitted=False))
        position += tokens
    expected[-1]["emitted"] = True
    if len(ids) > 1:
        expected.append(dict(kind="decode_summary", calls=len(ids)-1, first_past=position,
                             last_total=position+len(ids)-1))
    need(saved["fragments"] == expected and saved["rows"] == 1 and saved["context"] == 32768,
         "Prompt or autoregressive forward coverage differs")
    count = len(ids) + 8
    checked = saved["native_forward_counters"]
    need(checked["HC_dispatch_checked_forwards"] == checked["layout_dispatch_checked_forwards"] == count
         and sum(checked["tail_SUM_dispatch_checks"].values()) == count, "Unexpected native forward counter span")
    sorted_intervals = sorted(intervals)
    return {
        "input_tokens": position, "prefill_forward_calls": 9, "decode_forward_calls": len(ids)-1,
        "generated_tokens_with_EOS": len(ids), "visible_output_tokens": len(visible_ticks),
        "EOS": eos, "cap": cap, "owner_TTFT_seconds": ttft, "owner_decode_tokens_s": rate,
        "owner_ITL_ms_nearest_rank": {
            f"p{percent}": sorted_intervals[ceil(len(intervals) * percent / 100)-1] * 1000 if intervals else None
            for percent in (50, 95, 99)
        }, "output_IDs_SHA256": sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
    }


def audit_node(node, case, controller, actual):
    arm, before_id, gpu_id, post_id = case
    original = ROOT / node
    steps = dict(zip(("before", "gpu", "post"), case[1:], strict=True))
    record = next(row for row in controller["results"] if row["node"] == node)
    need(record == read_json(original / "terminal.json") and record["arm"] == arm and record["post_clean"] is True,
         "Original controller or node outcome differs")
    pins = {}
    for role, ident in steps.items():
        row = actual[ident]
        data = read_file(original / (role + "-physical.json"))
        pins[role] = file_pin(data)
        physical = json.loads(data)
        need(record[role]["terminal_visible"] is True and record[role]["row"] == row
             and record[role]["client_rc"] == 0 and row[2:4] == ["COMPLETED", "0:0"]
             and row[5] == node and ("gres/dcu=8" in row[6].split(",")) == (role == "gpu")
             and physical["node"] == node and physical["step"] == ident
             and physical["kfd_holder_count"] == 0 and len(set(physical["pci"])) == 8,
             "Actual Slurm accounting or clear physical inventory differs")
        if role == "before":
            first = physical
        else:
            need(physical["boot_id"] == first["boot_id"] and physical["pci"] == first["pci"]
                 and physical["monotonic_ns"] > first["monotonic_ns"], "Physical ownership changed")
        if role == "gpu":
            kfd = physical["pci_to_kfd"]
            need(set(kfd) == set(first["pci"]) and len(set(kfd.values())) == 8,
                 "Leased eight-device KFD identity differs")
    raw = read_file(original / "phase.jsonl", 2 * 1024 * 1024)
    phases = [json.loads(line) for line in raw.splitlines()]
    need(all(row["node"] == node and row["step"] == gpu_id for row in phases), "Native phase placement differs")
    names = [row["stage"] for row in phases]
    anchors = ("assets_verified", "before_create", "created", "required_keys", "load_enter", "load_finished",
               "request_started", "request_finished", "destroyed", "normal_terminal")
    need(tuple(name for name in names if name != "load_batch") == anchors, "Incomplete native lifecycle")
    at = {row["stage"]: row for row in phases if row["stage"] != "load_batch"}
    prepared, requested = at["assets_verified"], at["request_started"]
    need(prepared["input_tokens"] == 8489 and prepared["shards"] == 46 and prepared["CABI_symbols"] == 11
         and requested["input_tokens"] == 8489 and requested["budget"] == 4096 and requested["arm"] == arm,
         "Original request or provider pin differs")
    keys = at["required_keys"]
    need((keys["selected_count"], keys["selected_sha256"], keys["unindexed_count"],
          keys["physically_missing_known_count"]) == (KEYS, KEY_DIGEST, 0, 0),
         "Original checkpoint selected keys differ")
    batches = [row for row in phases if row["stage"] == "load_batch"]
    loading = at["load_finished"]
    loaded = [0, *[row["loaded"] for row in batches]]
    need(loaded[-1] == loading["loaded"] == KEYS and all(0 < b - a <= 1024 for a, b in zip(loaded, loaded[1:]))
         and clock_value(loading["seconds"]) and loading["seconds"] > 0, "Incomplete checkpoint load")
    result, close = at["request_finished"], at["destroyed"]
    need(result["arm"] == at["normal_terminal"]["arm"] == arm and
         close["observed_owner_count"] == 8 and close["observed_owners_remaining"] == 0,
         "Native owner cleanup or arm differs")
    minimum = min(memory_after(row["memory_bytes"]) for row in phases if "memory_bytes" in row)
    minimum = min(minimum, memory_after(result["post_request_memory"]))
    return {
        "node": node, "arm": arm, "accounting": {role: actual[ident] for role, ident in steps.items()},
        "physical_receipts": pins, "phase_receipt": file_pin(raw), "checkpoint_keys": KEYS,
        "checkpoint_load_seconds_including_finish": loading["seconds"], "checkpoint_load_batches": len(batches),
        "minimum_sampled_free_bytes": minimum, "native_owner_cleanup_pass": True,
        "independent_post_clean": True, **request_metrics(result),
    }


def main():
    raw = read_file(ROOT / "terminal.json")
    controller = json.loads(raw)
    need(controller["request"] == REQUEST and controller["accounting_columns"] == COLUMNS
         and len(controller["results"]) == 3, "Original terminal identity differs")
    ids = {ident for case in CASES.values() for ident in case[1:]}
    data = subprocess.check_output(
        ["sacct", "-nP", "-j", ",".join(sorted(ids)), "--format", ",".join(COLUMNS)], text=True, timeout=12,
    )
    actual = {}
    for line in data.splitlines():
        row = line.split("|")
        if len(row) == len(COLUMNS) and row[0] in ids:
            need(row[0] not in actual, "Duplicate original accounting")
            actual[row[0]] = row
    need(set(actual) == ids, "Actual GPU or independent cleanup accounting missing")
    cases = [audit_node(node, case, controller, actual) for node, case in CASES.items()]
    result = {
        "request": REQUEST, "observed_utc": datetime.now(timezone.utc).isoformat(),
        "original_terminal_receipt": file_pin(raw), "accounting_columns": COLUMNS, "cases": cases,
        "output_ID_digests_all_equal": len({row["output_IDs_SHA256"] for row in cases}) == 1,
        "owner_rate_strictly_above_15_count": sum((row["owner_decode_tokens_s"] or 0) > 15 for row in cases),
        "allocated_DCU": 24, "measured_GPU_busy": None, "actual_fused_width_dispatch_count": None,
        "hardware_placement_confounded_with_arm": True, "formal_service_benchmark": False,
        "formal_quality_or_paired_performance": False, "raw_output_or_token_times_transferred": False,
    }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
