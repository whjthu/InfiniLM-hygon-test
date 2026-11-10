"""Verify constructor-only evidence after Slurm accounting and an independent idle check."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from key_inventory import FLOOR, NATIVE, require


STAGES = ["admitted", "before_create", "created", "required_keys", "destroyed", "complete"]


def evidence(events, accounting, before, post):
    require([item["stage"] for item in events] == STAGES, "incomplete or failed GPU sequence")
    admitted, baseline, created, keys, destroyed, complete = events
    step, node = admitted["step"], admitted["node"]
    require(re.fullmatch(r"1087656\.\d+", step) and node == "f12r3n12", "wrong resource owner")
    require(all(item["step"] == step and item["node"] == node for item in events), "mixed job evidence")
    require(accounting["step"] == step and accounting["state"] == "COMPLETED"
            and accounting["exit_code"] == "0:0", "original Slurm step is not cleanly terminal")
    before_step, post_step = before["step"], post["step"]
    require(all(re.fullmatch(r"1087656\.\d+", handle) for handle in (before_step, post_step))
            and int(before_step.split(".")[1]) < int(step.split(".")[1]) < int(post_step.split(".")[1]),
            "independent snapshots do not bracket the GPU step")
    for row in (before, post):
        require(row["node"] == node and row["kfd_holder_count"] == 0
                and row["accounting_state"] == "COMPLETED" and row["exit_code"] == "0:0",
                "node idle check is not cleanly terminal")
        require(len(row["pci_to_kfd"]) == 8 and len(set(row["pci_to_kfd"].values())) == 8,
                "node idle snapshot lacks eight distinct devices")
    require(before["pci_to_kfd"] == post["pci_to_kfd"], "node PCI/KFD identity changed")
    require([admitted[key] for key in ("rows", "capacity", "tp", "pp", "native_sha256")]
            == [1, 32768, 8, 1, NATIVE], "wrong Native geometry")
    require(keys["selected_count"] == 67569 and keys["unindexed_count"] == 0
            and keys["physically_missing_known_count"] == 0
            and re.fullmatch(r"[a-f0-9]{64}", keys["selected_sha256"]), "required checkpoint keys differ")
    for phase in (baseline, created, destroyed):
        samples = phase["memory_bytes"]
        require(len(samples) == 8 and all(type(free) is int and type(total) is int
                and 0 <= free <= total for free, total in samples), "invalid eight-device memory sample")
    require(all(free >= FLOOR for phase in (baseline, created) for free, _ in phase["memory_bytes"]),
            "headroom below floor")
    require(destroyed["observed_owner_count"] == 8 and destroyed["observed_owners_remaining"] == 0,
            "normal eight-owner destroy not verified")
    require(complete["weights_loaded"] is False and complete["forward_calls"] == 0
            and complete["model_ready"] is False, "unexpected load or forward")
    return {"passed": True, "step": step, "node": node, "before_step": before_step, "post_step": post_step,
            "required_keys": keys["selected_count"], "required_keys_sha256": keys["selected_sha256"],
            "min_free_after_create_bytes": min(free for free, _ in created["memory_bytes"]),
            "scope": "single-node TP8/PP1 32768 construction, key census and normal destroy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "accounting", "before", "post", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    for key in ("events", "accounting", "before", "post"):
        path = getattr(args, key)
        require(path.stat().st_size <= 1024 * 1024, "oversize evidence")
        inputs[key] = path.read_bytes()
    events = [json.loads(line) for line in inputs["events"].splitlines()]
    require(all(row["kind"] == "i03_native_key_inventory" for row in events), "unexpected events")
    result = evidence(events, *(json.loads(inputs[key]) for key in ("accounting", "before", "post")))
    result["input_sha256"] = {key: hashlib.sha256(raw).hexdigest() for key, raw in inputs.items()}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
