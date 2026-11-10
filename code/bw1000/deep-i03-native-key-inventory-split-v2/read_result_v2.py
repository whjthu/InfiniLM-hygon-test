"""Independently verify leased KFD identity, constructor evidence, and CPU post-check."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from key_inventory import FLOOR, NATIVE
from read_result import STAGES
from split_admission import NODE, STEP, require, validate_before


def verify(events, before, leased, accounting, post):
    validate_before(before, leased)
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
    require([row["stage"] for row in events] == STAGES
            and all(row["kind"] == "i03_native_key_inventory" and row["node"] == NODE
                    and row["step"] == step for row in events), "incomplete or mixed GPU events")
    admitted, baseline, created, keys, destroyed, complete = events
    require([admitted[key] for key in ("rows", "capacity", "tp", "pp", "native_sha256")]
            == [1, 32768, 8, 1, NATIVE], "wrong Native geometry")
    for phase in (baseline, created, destroyed):
        require(len(phase["memory_bytes"]) == 8 and all(type(free) is int and type(total) is int
                and 0 <= free <= total for free, total in phase["memory_bytes"]), "invalid memory sample")
    require(all(free >= FLOOR for phase in (baseline, created) for free, _ in phase["memory_bytes"]),
            "headroom below floor")
    require(keys["selected_count"] == 67569 and keys["unindexed_count"] == 0
            and keys["physically_missing_known_count"] == 0
            and re.fullmatch(r"[a-f0-9]{64}", keys["selected_sha256"]), "required checkpoint keys differ")
    require(destroyed["observed_owner_count"] == 8 and destroyed["observed_owners_remaining"] == 0,
            "normal eight-owner destroy not verified")
    require(complete["weights_loaded"] is False and complete["forward_calls"] == 0
            and complete["model_ready"] is False, "unexpected load or forward")
    return {"passed": True, "step": step, "before_step": before["step"], "post_step": post["step"],
            "pci_to_kfd": leased["pci_to_kfd"], "selected_keys_sha256": keys["selected_sha256"],
            "min_free_after_create_bytes": min(free for free, _ in created["memory_bytes"])}


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
