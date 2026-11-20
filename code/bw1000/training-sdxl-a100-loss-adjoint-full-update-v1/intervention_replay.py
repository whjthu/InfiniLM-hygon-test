"""Isolate the effect of the original SDXL loss adjoint on the full target update."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import hygon_sdxl_first_step as replay


def digest(value):
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ExactLossAdjoint:
    def __init__(self, expected):
        self.expected = expected
        self.result = {}
        self.calls = 0

    def attach(self, actual_prediction, reference_prediction, reference_target):
        require(not self.result and actual_prediction.requires_grad, "RAW_OUTPUT_AUTOGRAD_BOUNDARY")
        require(actual_prediction.device.type == "cuda" and actual_prediction.dtype == torch.bfloat16
                and reference_prediction.dtype == torch.float32
                and actual_prediction.shape == reference_prediction.shape == reference_target.shape == (1,4,32,32),
                "ORIGINAL_PRE_LOSS_TENSOR_GEOMETRY")
        raw = reference_prediction.to(torch.bfloat16)
        require(torch.equal(raw.float(), reference_prediction), "REFERENCE_PREDICTION_BF16_EXACT")
        self.result.update(hip_raw_prediction_SHA256=digest(actual_prediction),
                           a100_raw_prediction_SHA256=digest(raw))
        require(self.result["hip_raw_prediction_SHA256"] == self.expected["hip_raw_prediction_SHA256"]
                and self.result["a100_raw_prediction_SHA256"] == self.expected["a100_raw_prediction_SHA256"],
                "ORIGINAL_FORWARD_HASHES_REQUIRED")
        grad = ((reference_prediction - reference_target.float()) * (2/reference_prediction.numel())).to(torch.bfloat16)
        self.result["a100_replacement_adjoint_SHA256"] = digest(grad)
        require(self.result["a100_replacement_adjoint_SHA256"] == self.expected["a100_original_adjoint_SHA256"],
                "ORIGINAL_A100_ADJOINT_RECONSTRUCTION")
        replacement = grad.to(actual_prediction.device)

        def replace(native):
            self.calls += 1
            require(self.calls == 1 and native.shape == replacement.shape and native.dtype == replacement.dtype
                    and native.device == replacement.device, "SINGLE_ORIGINAL_RAW_TENSOR_ADJOINT")
            self.result["hip_native_adjoint_SHA256"] = digest(native)
            require(self.result["hip_native_adjoint_SHA256"] == self.expected["hip_original_adjoint_SHA256"],
                    "ORIGINAL_HIP_ADJOINT_REQUIRED")
            self.result["applied_adjoint_SHA256"] = digest(replacement)
            return replacement

        return replace


def main():
    parser = argparse.ArgumentParser()
    for field in ("reference", "policy", "weights", "output"):
        parser.add_argument("--"+field, required=True, type=Path)
    parser.add_argument("--accelerator", choices=("hip",), required=True)
    args = parser.parse_args()
    claim = json.loads(Path(__file__).with_name("CLAIM.json").read_bytes())
    binding = replay.verify_reference(args.reference, args.policy)
    args.output.mkdir(mode=0o700, exist_ok=False)
    patch = ExactLossAdjoint(claim["intervention"])
    try:
        replay.replay(args.reference, args.weights, args.output/"replay", binding,
                      accelerator="hip", raw_adjoint=patch.attach)
        require(patch.calls == 1 and patch.result.get("applied_adjoint_SHA256")
                == claim["intervention"]["a100_original_adjoint_SHA256"], "ADJOINT_NOT_APPLIED")
    finally:
        with (args.output/"intervention.json").open("x",encoding="utf-8") as stream:
            json.dump(dict(patch.result, raw_output_hook_calls=patch.calls),stream,indent=2,allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
