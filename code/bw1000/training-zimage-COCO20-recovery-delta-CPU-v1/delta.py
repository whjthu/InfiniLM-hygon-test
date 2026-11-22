"""Measure original COCO20 recovery checkpoint differences on CPU."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def load(torch, root, pin):
    path = root / pin["relative"]
    need(path.is_file() and not path.is_symlink() and path.resolve() == path
         and path.stat().st_size == pin["bytes"] < 128 << 20, "Invalid original checkpoint extent")
    raw = path.read_bytes()
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["SHA256"],
         "Original checkpoint identity differs")
    # This private pickle is an exact SHA-pinned artifact of our own preceding work.
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    need(value["metadata"]["global_step"] == 25 and value["metadata"]["next_data_cursor"] == 5
         and value["metadata"]["next_global_step"] == 26, "Wrong COCO20 step or next cursor")
    return value


def optimizer(state, field):
    opt = state["optimizer"]
    groups = opt["param_groups"]
    names = state["optimizer_parameter_names"]
    need(len(names) == len(groups) == 1 and len(names[0]) == len(groups[0]["params"]) == 476,
         "Unexpected AdamW group or name count")
    ordered = dict(zip(groups[0]["params"], names[0], strict=True))
    need(len(ordered) == len(set(ordered.values())) == 476 and set(ordered) == set(opt["state"]),
         "AdamW parameter IDs differ")
    result = {}
    for index, name in ordered.items():
        moment = opt["state"][index]
        need(set(moment) == {"step", "exp_avg", "exp_avg_sq"} and float(moment["step"]) == 25,
             "Unexpected AdamW state or step count")
        result[name] = moment[field]
    return result


def compare(torch, a, b):
    need(len(a) == len(b) == 476 and set(a) == set(b), "Unexpected trainable names")
    count = 0
    total = 0
    squared = []
    reference = []
    changed = []
    worst = 0.0
    for name, x in a.items():
        y = b[name]
        need(x.device.type == y.device.type == "cpu" and x.shape == y.shape and x.dtype == y.dtype,
             "Checkpoint state shape, dtype or device differs")
        need(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), "Nonfinite checkpoint value")
        elements = int(torch.count_nonzero(x != y))
        count += elements
        total += x.numel()
        delta = x.double().reshape(-1) - y.double().reshape(-1)
        vector = x.double().reshape(-1)
        squared.append(float(torch.dot(delta, delta)))
        reference.append(float(torch.dot(vector, vector)))
        maximum = float(delta.abs().max())
        if elements:
            changed.append((name, maximum, elements))
        worst = max(worst, maximum)
    need(total == 9_922_560, "Wrong original model extent")
    l2 = math.sqrt(math.fsum(squared))
    scale = math.sqrt(math.fsum(reference))
    need(math.isfinite(l2) and math.isfinite(scale), "Nonfinite CPU reduction")
    largest = sorted(changed, key=lambda row: (-row[1], row[0]))[:4]
    return {"tensors": len(a), "elements": total, "changed_tensors": len(changed), "changed_elements": count,
            "max_absolute": worst, "l2": l2, "relative_to_continuous_l2": l2 / scale if scale else None,
            "largest": [dict(name=name, max_absolute=diff, changed_elements=n) for name,diff,n in largest]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--claim", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.claim.read_bytes())
    need(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")),
         "CPU audit visibility masks differ")
    import torch
    need(not torch.cuda.is_initialized(), "Accelerator initialized")
    torch.set_num_threads(4)
    a,b = [load(torch,args.root,c["checkpoints"][k]) for k in ("continuous","resume20")]
    need(a["metadata"] == b["metadata"] and a["state"]["optimizer_parameter_names"] == b["state"]["optimizer_parameter_names"],
         "The two saved states do not share the original optimizer placement")
    left, right = a["state"], b["state"]
    need(left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"], "Final AdamW groups differ")
    metrics = {"LoRA": compare(torch,left["trainable"],right["trainable"]),
               "AdamW_exp_avg": compare(torch,optimizer(left,"exp_avg"),optimizer(right,"exp_avg")),
               "AdamW_exp_avg_sq": compare(torch,optimizer(left,"exp_avg_sq"),optimizer(right,"exp_avg_sq"))}
    need(not torch.cuda.is_initialized(), "Accelerator used")
    out = {"case":"T07.sft","request":c["request"],"metrics":metrics,"same_checkpoint_metadata":True,
           "same_AdamW_groups":True,"checkpoint_SHA256":{n:v["SHA256"] for n,v in c["checkpoints"].items()},
           "actual_GPU":0,"recovery_exact":all(x["changed_elements"] == 0 for x in metrics.values()),
           "tensors_transferred":0,"formal_A100":False}
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(out,allow_nan=False))


if __name__ == "__main__":
    main()
