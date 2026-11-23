"""Compare registered training with its resume20 run on the same physical Hygon card."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


LOSS_ATOL = 1e-6
ADAPTER_ATOL = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    require(0 < len(raw) <= 1 << 20, "Invalid receipt size")
    return json.loads(raw)


def sealed_checkpoint(path, pin):
    import torch

    require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"]
            and 0 < pin["bytes"] <= 512 << 20, "Missing or oversized original checkpoint")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
        require(digest.hexdigest() == pin["SHA256"], "Original checkpoint hash differs")
        stream.seek(0)
        # The bounded local producer archive includes its native NumPy RNG tuple.
        return torch.load(stream, map_location="cpu", weights_only=False)


def confirm_checkpoint(arm, pin, tree_hash):
    path = arm / ("checkpoint-%d" % pin["global_step"]) / "state.private.pt"
    payload = sealed_checkpoint(path,pin)
    require(payload["metadata"]["global_step"] == pin["global_step"]
            and payload["metadata"]["next_data_cursor"] == pin["next_data_cursor"]
            and payload["lr_scheduler"] is None and set(payload["rng"]["cuda"]) == {0}
            and {"trainable","optimizer","optimizer_progress","buffers","modes"} <= set(payload["state"])
            and payload["integrity_sha256"] == pin["integrity_SHA256"]
            and payload["adapter_sha256"] == pin["adapter_SHA256"], "Native checkpoint boundary differs")
    unsigned = {k:v for k,v in payload.items() if k != "integrity_sha256"}
    require(tree_hash(unsigned) == payload["integrity_sha256"]
            and tree_hash(payload["state"]["trainable"]) == payload["adapter_sha256"],
            "Native checkpoint inner integrity differs")
    return payload


def compare(root):
    import torch
    from gencuda.hy_resume_checkpoint import tree_sha256

    require(not torch.cuda.is_available(), "Recovery comparison requires zero GRES")
    torch.set_num_threads(4)
    runs = [root / "continuous",root / "resume20"]
    results = [receipt(p/"result.json") for p in runs]
    reports = [receipt(root/(phase+"-worker-result.json")) for phase in ("continuous","resume20")]
    expected = [list(range(1,26)),list(range(21,26))]
    require(all(r["passed"] is True and r["completed_global_steps"] == n and r["actual_optimizer_updates"] == len(n)
                for r,n in zip(results,expected,strict=True)), "Wrong actual update denominator")
    require(all(r["passed"] is True for r in reports) and reports[0]["slurm"] != reports[1]["slurm"]
            and reports[0]["physical_pci"] == reports[1]["physical_pci"], "Recovery was not a fresh same-card process")
    sequences = [r["step_metrics"] for r in results]
    for rows,steps in zip(sequences,expected,strict=True):
        require([row["global_step"] for row in rows] == steps, "Step metrics do not cover the actual update window")
        require(all(row["sample_index"] == (row["global_step"] - 1) % 20 and isfinite(row["loss"])
                    and row["applied_update_seed"] == 20260911 + row["global_step"]
                    and len(row["input_SHA256"]) == 64 for row in rows), "Original data cursor, loss or seed differs")
    pairs = list(zip(sequences[0][20:],sequences[1],strict=True))
    losses = [abs(left["loss"]-right["loss"]) for left,right in pairs]
    inputs_equal = all(left["input_SHA256"] == right["input_SHA256"] for left,right in pairs)
    cp, resumed = [r["checkpoints"] for r in results]
    require([x["global_step"] for x in cp] == [10,20,25] and [x["next_data_cursor"] for x in cp] == [10,0,5]
            and [x["global_step"] for x in resumed] == [25] and resumed[0]["next_data_cursor"] == 5,
            "Unexpected durable checkpoint schedule")
    restored = results[1]["restore_info"]
    require(restored["restored_boundary_exact"] is True and restored["metadata"]["global_step"] == 20
            and restored["next"]["global_step"] == 21 and restored["next"]["sample_cursor"] == 0
            and restored["rng_devices"] == [0] and restored["checkpoint_SHA256"] == cp[1]["SHA256"],
            "Recovery did not consume the actual continuous checkpoint20")
    for pin in cp[:2]:
        confirm_checkpoint(runs[0],pin,tree_sha256)
    left = confirm_checkpoint(runs[0],cp[-1],tree_sha256)
    right = confirm_checkpoint(runs[1],resumed[0],tree_sha256)
    state_exact = {key:tree_sha256(left["state"][key]) == tree_sha256(right["state"][key])
                   for key in left["state"]}
    state_exact.update({key:tree_sha256(left[key]) == tree_sha256(right[key])
                        for key in ("metadata","scheduler","lr_scheduler","rng","integrity_sha256")})
    names = list(left["state"]["trainable"])
    require(names == list(right["state"]["trainable"]) and len(names) == 476, "Model trainable inventory differs")
    max_difference = 0.
    differing_tensors = 0
    for name in names:
        a,b = (payload["state"]["trainable"][name] for payload in (left,right))
        require(a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == "cpu"
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "Invalid paired adapter tensor")
        difference = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
        max_difference = max(max_difference,difference)
        differing_tensors += difference > 0
    return {"continuous_updates":25,"resumed_updates":5,"same_pci":True,"actual_slurm_steps":[x["slurm"] for x in reports],
            "restore_step":20,"restored_rng_devices":restored["rng_devices"],"paired_steps":expected[1],
            "paired_input_fingerprints_equal":inputs_equal,"paired_loss_abs":losses,"max_loss_abs":max(losses),
            "max_adapter_abs":max_difference,"adapter_tensors_with_differences":differing_tensors,
            "exact_checkpoint25_contents":state_exact,"checkpoint_schedule":[10,20,25],
            "original_HIP_engineering_checkpoint_gate_passed":inputs_equal and max(losses) <= LOSS_ATOL
            and max_difference <= ADAPTER_ATOL and all(state_exact.values()),
            "A100_comparison_executed":False,"formal_performance":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.root),sort_keys=True,allow_nan=False))
