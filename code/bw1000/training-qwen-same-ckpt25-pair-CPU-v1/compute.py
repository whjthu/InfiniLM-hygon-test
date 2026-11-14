"""Compare actual continuous and resumed Qwen checkpoint-25 tensors on CPU."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import time

import torch
from safetensors.torch import load as loadsafe

STEP = "1087656"
ELEMENTS = 43646976
FILES = ("adapter_model.safetensors", "optimizer.pt")
ROOTS = (Path("/original/train/checkpoint-25"), Path("/resumed/train/checkpoint-25"))


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read(path, digest, size, decode):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size == size
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
        assert h.hexdigest() == digest and identity(before) == identity(opened)
        stream.seek(0)
        value = decode(stream)
        assert identity(os.fstat(stream.fileno())) == identity(opened)
    assert identity(path.lstat()) == identity(before)
    return value


def metrics(reference, candidate):
    assert set(reference) == set(candidate)
    elements = changed = changed_tensors = 0
    sum_sq = reference_sq = largest = worst_tensor = 0.0
    for key, ref in reference.items():
        got = candidate[key]
        assert ref.shape == got.shape and ref.dtype == got.dtype == torch.float32
        assert ref.device.type == got.device.type == "cpu"
        diff_sq = norm_sq = 0.0
        diff_count = 0
        for left, right in zip(ref.reshape(-1).split(262144), got.reshape(-1).split(262144)):
            assert bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
            left = left.double()
            delta = right.double() - left
            diff_sq += delta.square().sum().item()
            norm_sq += left.square().sum().item()
            largest = max(largest, delta.abs().max().item())
            diff_count += torch.count_nonzero(delta).item()
        if norm_sq:
            worst_tensor = max(worst_tensor, math.sqrt(diff_sq / norm_sq))
        sum_sq += diff_sq
        reference_sq += norm_sq
        changed += diff_count
        changed_tensors += bool(diff_count)
        elements += ref.numel()
    return dict(tensors=len(reference), elements=elements, changed_elements=changed,
                changed_tensors=changed_tensors, max_abs=largest, delta_l2=math.sqrt(sum_sq),
                relative_l2=math.sqrt(sum_sq/reference_sq) if reference_sq else None,
                max_tensor_relative_l2_nonzero_reference=worst_tensor)


def comparable_groups(groups):
    return [{k:v for k,v in row.items() if k != "params"} for row in groups]


def main():
    epoch = float(os.environ["QWEN_CPU_EPOCH"])
    deadline = float(os.environ["QWEN_CPU_DEADLINE"])
    assert deadline == epoch + 230 and epoch <= time.time() < deadline
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    assert os.environ["SLURM_JOB_ID"] == STEP
    assert all(os.environ[k] == "-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
    assert not torch.cuda.is_initialized()
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("GPU initialization is forbidden for checkpoint comparison")
    torch.cuda._lazy_init = forbidden
    torch.set_num_threads(2)
    pins = json.loads(Path("/fix/PINS.json").read_bytes())

    def pair(name, decode):
        pin = pins[name]
        return [read(folder/name, pin[f"{side}_SHA256"], pin[f"{side}_bytes"], decode)
                for folder, side in zip(ROOTS, ("continuous", "resumed"))]

    ref, got = pair(FILES[0], lambda stream: loadsafe(stream.read()))
    adapter = metrics(ref, got)
    assert adapter["tensors"] == 504 and adapter["elements"] == ELEMENTS
    del ref, got
    ref, got = pair(FILES[1], lambda stream: torch.load(stream, map_location="cpu", weights_only=True))
    assert set(ref) == set(got) == {"state", "param_groups"}
    ref_ids = [group["params"] for group in ref["param_groups"]]
    got_ids = [group["params"] for group in got["param_groups"]]
    assert ref_ids == got_ids
    ids = [i for group in ref_ids for i in group]
    assert len(ids) == len(set(ids)) == 504 and set(ids) == set(ref["state"]) == set(got["state"])
    steps = [all(values["state"][i]["step"].item() == 25 for i in ids) for values in (ref, got)]
    assert all(steps)
    for values in (ref, got):
        assert all(set(values["state"][i]) == {"step","exp_avg","exp_avg_sq"} for i in ids)
    moments = {field:metrics({i:ref["state"][i][field] for i in ids},
                             {i:got["state"][i][field] for i in ids})
               for field in ("exp_avg", "exp_avg_sq")}
    assert all(m["elements"] == ELEMENTS and m["tensors"] == 504 for m in moments.values())
    result = dict(passed=True, step=STEP+"."+os.environ["SLURM_STEP_ID"], GPU_cards=0,
                  adapter=adapter, optimizer_moments_by_serialized_parameter_id=moments,
                  optimizer_step_all_25=all(steps), optimizer_group_ids_and_order_equal=True,
                  optimizer_group_metadata_equal=comparable_groups(ref["param_groups"])==comparable_groups(got["param_groups"]),
                  optimizer_group_metadata_differing_keys=[
                      sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
                      for a,b in zip(comparable_groups(ref["param_groups"]),comparable_groups(got["param_groups"]))],
                  raw_tensor_transfer_bytes=0, exact_resume_qualified=False)
    with Path("/evidence/compute.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
