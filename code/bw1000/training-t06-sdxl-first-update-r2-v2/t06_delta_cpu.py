"""Attribute the SDXL first-step delta error using the original saved CPU tensors."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

BASE = Path("/t06-input")
REF = BASE / "reference.pt"
OBSERVED = BASE / "observed.pt"
PINS = {
    REF: "c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036",
    OBSERVED: "f44f0d40d198699fcd3a635bb8fc744b907dcba05d8fe2284a2967f34f0192a3",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(2**20), b""):
            digest.update(raw)
    return digest.hexdigest()


def cpu_update(initial, groups, gradients):
    params = {name: torch.nn.Parameter(tensor.clone()) for name, tensor in initial.items()}
    group, = groups["param_groups"]
    options = {key: value for key, value in group.items() if key not in ("params", "initial_lr")}
    assert options.pop("decoupled_weight_decay") is True
    optimizer = torch.optim.AdamW(list(params.values()), **options)
    optimizer.load_state_dict(groups)
    for name, param in params.items():
        param.grad = gradients[name].clone()
    optimizer.step()
    return {name: param.detach() for name, param in params.items()}


def metrics(a, b, initial):
    total, residual, entries = 0.0, 0.0, 0
    for name, start in initial.items():
        original = b[name].double() - start.double()
        observed = a[name].double() - start.double()
        total += float(original.square().sum())
        residual += float((original - observed).square().sum())
        entries += observed.numel()
    return dict(relative_l2=math.sqrt(residual / total), error_l2_squared=residual,
                reference_delta_l2_squared=total, trainable_elements=entries)


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    for path, pin in PINS.items():
        assert sha(path) == pin
    ref, observed = [torch.load(path, map_location="cpu", weights_only=True) for path in (REF, OBSERVED)]
    initial = ref["initial_adapter"]
    names = list(initial)
    reference_grads, actual_grads = [x["gradients_after_clip"] for x in (ref, observed)]
    error_sums = dict(same_sign=0.0, opposite_sign=0.0, zero_vs_nonzero=0.0)
    element_counts = dict(same_sign=0, opposite_sign=0, zero_vs_nonzero=0)
    for name, start in initial.items():
        assert torch.equal(start, observed["initial_adapter"][name])
        a, b = reference_grads[name].sign(), actual_grads[name].sign()
        delta_error = (ref["final_adapter"][name].double() - observed["final_adapter"][name].double()).square()
        masks = {"same_sign": a == b, "opposite_sign": a * b == -1, "zero_vs_nonzero": (a == 0) ^ (b == 0)}
        for key, mask in masks.items():
            error_sums[key] += float(delta_error[mask].sum())
            element_counts[key] += int(mask.sum())

    ref_cpu, obs_cpu = [cpu_update(initial, ref["initial_optimizer"], gradients)
                        for gradients in (reference_grads, actual_grads)]
    result = dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  GPU_initialized=torch.cuda.is_initialized(), tensors=len(names),
                  original_HIP_vs_A100=metrics(observed["final_adapter"], ref["final_adapter"], initial),
                  CPU_reference_gradients_vs_A100=metrics(ref_cpu, ref["final_adapter"], initial),
                  CPU_HIP_gradients_vs_actual_HIP=metrics(obs_cpu, observed["final_adapter"], initial),
                  CPU_HIP_gradients_vs_CPU_reference_gradients=metrics(obs_cpu, ref_cpu, initial),
                  gradient_sign_element_counts=element_counts, delta_error_l2_squared_by_sign=error_sums)
    assert math.isclose(sum(error_sums.values()),result["original_HIP_vs_A100"]["error_l2_squared"],rel_tol=1e-10)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
