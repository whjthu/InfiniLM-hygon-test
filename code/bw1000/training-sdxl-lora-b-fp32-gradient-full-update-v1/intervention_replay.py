"""Recompute only SDXL LoRA B weight gradients with an FP32 GEMM."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import torch

import hygon_sdxl_first_step as replay

B_WEIGHT = ".lora_B.default.weight"
A_WEIGHT = ".lora_A.default.weight"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class WeightGradient:
    def __init__(self, expected):
        self.expected = expected
        self.handles = []
        self.inputs = {}
        self.upstream = {}
        self.metrics = {}
        self.elements = {}
        self.result = {}

    def attach(self, model, named):
        a = {name.removesuffix(A_WEIGHT) for name in named if name.endswith(A_WEIGHT)}
        b = {name.removesuffix(B_WEIGHT) for name in named if name.endswith(B_WEIGHT)}
        require(a == b and len(a) == self.expected["lora_b_sites"]
                and len(named) == 2 * len(a), "EXACT_LORA_PARAMETER_PAIRS")
        modules = dict(model.named_modules())
        for prefix in sorted(b):
            name = prefix + B_WEIGHT
            weight = named[name]
            module = modules[name.removesuffix(".weight")]
            require(type(module) is torch.nn.Linear and module.weight is weight and module.bias is None
                    and weight.ndim == 2 and weight.shape[1] == 4
                    and weight.is_cuda and weight.dtype == torch.bfloat16,
                    "ORIGINAL_LORA_B_LINEAR")
            self.elements[name] = weight.numel()
            self.handles.append(module.register_forward_hook(partial(self.forward, name)))
            self.handles.append(weight.register_hook(partial(self.weight_grad, name)))

    def forward(self, name, module, args, output):
        require(name not in self.inputs and name not in self.metrics and len(args) == 1
                and isinstance(output, torch.Tensor) and output.requires_grad, "SINGLE_LORA_B_FORWARD")
        value, = args
        require(value.ndim >= 2 and value.dtype == output.dtype == torch.bfloat16
                and value.device == output.device == module.weight.device
                and value.shape[:-1] == output.shape[:-1]
                and value.shape[-1] == 4 and output.shape[-1] == module.out_features,
                "ORIGINAL_LORA_B_OPERANDS")
        self.inputs[name] = (value.detach(), value._version, output.shape)
        self.handles.append(output.register_hook(partial(self.capture, name)))

    def capture(self, name, gradient):
        require(name in self.inputs and name not in self.upstream, "SINGLE_LORA_B_OUTPUT_GRADIENT")
        self.upstream[name] = gradient.detach()
        # Returning None leaves the original linear input-gradient path intact.

    def weight_grad(self, name, native):
        require(name in self.inputs and name in self.upstream and name not in self.metrics,
                "LORA_B_GRADIENT_HOOK_ORDER")
        value, version, output_shape = self.inputs.pop(name)
        gradient = self.upstream.pop(name)
        require(value._version == version and gradient.shape == output_shape
                and native.dtype == gradient.dtype == value.dtype == torch.bfloat16
                and native.device == gradient.device == value.device
                and native.shape == (output_shape[-1], 4), "LORA_B_WEIGHT_GRADIENT_GEOMETRY")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            # Flatten every batch/sequence dimension just as the native linear VJP does.
            upstream = gradient.reshape(-1, native.shape[0]).float()
            inputs = value.reshape(-1, 4).float()
            replacement = (upstream.transpose(0, 1) @ inputs).to(native.dtype)
            changed = (native.contiguous().view(torch.int16) != replacement.contiguous().view(torch.int16)).sum()
            finite = torch.isfinite(native).all() & torch.isfinite(replacement).all()
            maximum = (native.float() - replacement.float()).abs().amax()
            self.metrics[name] = torch.stack((changed.float(), finite.float(), maximum))
        return replacement

    def verify_prediction(self, raw):
        encoded = raw.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        self.result["hip_raw_prediction_SHA256"] = hashlib.sha256(encoded).hexdigest()
        require(tuple(raw.shape) == (1, 4, 32, 32) and raw.dtype == torch.bfloat16 and
                self.result["hip_raw_prediction_SHA256"] == self.expected["hip_raw_prediction_SHA256"],
                "ORIGINAL_HIP_RAW_OUTPUT")

    def finish(self):
        require(self.metrics.keys() == self.elements.keys() and not self.inputs and not self.upstream,
                "COMPLETE_LORA_B_WEIGHT_GRADIENT_COVERAGE")
        rows = torch.stack(tuple(self.metrics.values())).tolist()
        require(all(row[1] == 1 for row in rows), "NONFINITE_LORA_B_GRADIENT")
        self.result.update(
            applied_weight_tensors=len(rows), replaced_elements=sum(self.elements.values()),
            changed_elements=int(sum(x[0] for x in rows)), changed_tensors=sum(x[0] > 0 for x in rows),
            max_abs_native_vs_fp32_rounded=max(x[2] for x in rows))

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.inputs.clear()
        self.upstream.clear()


def main():
    parser = argparse.ArgumentParser()
    for name in ("reference", "policy", "weights", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--accelerator", choices=("hip",), required=True)
    args = parser.parse_args()
    claim = json.loads(Path(__file__).with_name("CLAIM.json").read_bytes())
    binding = replay.verify_reference(args.reference, args.policy)
    args.output.mkdir(mode=0o700, exist_ok=False)
    patch = WeightGradient(claim["intervention"])
    try:
        replay.replay(args.reference, args.weights, args.output/"replay", binding,
                      accelerator="hip", weight_gradient=patch)
    finally:
        patch.close()
        with (args.output/"intervention.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(patch.result, observed_weight_hooks=len(patch.metrics)), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
