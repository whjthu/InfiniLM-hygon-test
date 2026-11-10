"""Verify the I03 checkpoint-load evidence, sampled memory, and independent CPU post-check."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from load_probe import EXPECTED_KEYS, FLOOR, KEYS_SHA256, NATIVE
from split_admission import NODE, STEP, require, validate_before


def verify(events, before, leased, accounting, post):
    validate_before(before, leased)
    require(leased.get("holder_scan_location") == "slurm_host_before_container",
            "missing host-side lease check")
    step = leased["step"]
    require(accounting["step"] == step and accounting["state"] == "COMPLETED"
            and accounting["exit_code"] == "0:0", "GPU Slurm step not cleanly terminal")
    require(post["kind"] == "i03_device_snapshot" and post["mode"] == "cpu"
            and post["node"] == NODE and post["boot_id"] == leased["boot_id"]
            and post["pci"] == leased["pci"] and post["kfd_holder_count"] == 0,
            "post snapshot identity or idle status differs")
    require(STEP.fullmatch(post["step"]) and int(post["step"].split(".")[1]) > int(step.split(".")[1])
            and post["monotonic_ns"] > leased["monotonic_ns"]
            and post["accounting_state"] == "COMPLETED" and post["exit_code"] == "0:0",
            "post snapshot does not follow the leased step")
    require(len(events) >= 9 and all(row["kind"] == "i03_native_weight_load"
                and row["node"] == NODE and row["step"] == step for row in events),
            "incomplete or mixed GPU events")
    admitted, baseline, created, keys, entered = events[:5]
    finished, destroyed, complete = events[-3:]
    batches = events[5:-3]
    require([row["stage"] for row in events[:5] + events[-3:]] ==
            ["admitted", "before_create", "created", "required_keys", "load_enter",
             "load_finished", "destroyed", "complete"]
            and all(row["stage"] == "load_batch" for row in batches), "unexpected load stages")
    require([admitted[key] for key in ("rows", "capacity", "tp", "pp", "native_sha256")]
            == [1, 32768, 8, 1, NATIVE], "wrong Native geometry")
    require(keys["selected_count"] == EXPECTED_KEYS and keys["unindexed_count"] == 0
            and keys["physically_missing_known_count"] == 0 and keys["selected_sha256"] == KEYS_SHA256
            and entered["expected_keys"] == EXPECTED_KEYS, "required checkpoint keys differ")
    loaded, elapsed = 0, 0.0
    for row in batches:
        count, seconds = row["loaded"], row["seconds"]
        require(type(count) is int and 0 < count - loaded <= 1024 and count <= EXPECTED_KEYS
                and type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= elapsed,
                "non-monotonic or incomplete load evidence")
        loaded, elapsed = count, seconds
    require(finished["loaded"] == loaded == EXPECTED_KEYS
            and type(finished["seconds"]) in (int, float) and math.isfinite(finished["seconds"])
            and finished["seconds"] >= elapsed, "load or finish evidence is incomplete")
    samples = [baseline, created, *batches, finished]
    for row in [*samples, destroyed]:
        values = row["memory_bytes"]
        require(len(values) == 8 and all(type(free) is int and type(total) is int
                and 0 <= free <= total for free, total in values), "invalid memory sample")
        require([total for _, total in values] == [total for _, total in baseline["memory_bytes"]],
                "visible memory geometry changed")
    require(all(free >= FLOOR for row in samples for free, _ in row["memory_bytes"]),
            "sampled headroom below floor")
    require(destroyed["observed_owner_count"] == 8 and destroyed["observed_owners_remaining"] == 0,
            "normal eight-owner destroy not verified")
    require(complete["weights_loaded"] is True and complete["native_finish_completed"] is True
            and complete["forward_calls"] == 0 and complete["service_started"] is False,
            "load-only scope did not complete")
    return {
        "passed": True, "step": step, "before_step": before["step"], "post_step": post["step"],
        "pci_to_kfd": leased["pci_to_kfd"], "selected_keys_sha256": keys["selected_sha256"],
        "loaded_key_count": loaded, "load_seconds_including_finish": finished["seconds"],
        "load_batches": len(batches),
        "sampled_min_free_bytes_by_visible_device": [
            min(row["memory_bytes"][device][0] for row in [*batches, finished]) for device in range(8)
        ],
        "post_finish_memory_bytes": finished["memory_bytes"],
        "forward_calls": 0, "service_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_names = ("events", "before", "leased", "accounting", "post")
    for name in (*input_names, "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = {}
    for name in input_names:
        path = getattr(args, name)
        require(0 < path.stat().st_size <= 2**20, "invalid evidence size")
        raw[name] = path.read_bytes()
    shas = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    leased = json.loads(raw["leased"])
    require(leased["before_sha256"] == shas["before"], "leased step used different CPU evidence")
    events = [json.loads(line) for line in raw["events"].splitlines()]
    result = verify(events, json.loads(raw["before"]), leased,
                    json.loads(raw["accounting"]), json.loads(raw["post"]))
    result["input_sha256"] = shas
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
