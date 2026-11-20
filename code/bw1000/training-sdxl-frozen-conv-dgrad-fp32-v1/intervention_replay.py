"""Keep native SDXL forward and test FP32 dInput for frozen UNet convolutions."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re

import torch

import hygon_sdxl_first_step as replay


BLOCK = re.compile(r"(?:^|\.)(down_blocks\.\d+|up_blocks\.\d+|mid_block)(?:\.|$)")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class FrozenConvInputGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, module, original, policy, name):
        output = original(value)
        require(module.weight.dtype == output.dtype == torch.bfloat16
                and value.dtype in (torch.bfloat16, torch.float32)
                and value.device == output.device == module.weight.device
                and module.padding_mode == "zeros", "FROZEN_CONV_FORWARD_BOUNDARY")
        ctx.save_for_backward(module.weight)
        ctx.shape = value.shape
        ctx.dtype = value.dtype
        ctx.geometry = module.stride, module.padding, module.dilation, module.groups
        ctx.policy = policy
        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        weight, = ctx.saved_tensors
        require(output_gradient.dtype == torch.bfloat16 and output_gradient.device == weight.device,
                "FROZEN_CONV_OUTPUT_ADJOINT")
        # The one changed operation accumulates in FP32 and rounds at the original input boundary.
        # Convolution backend TF32 is scoped only to this FP32 operation.
        flags = torch.backends.cudnn
        previous = flags.allow_tf32
        try:
            flags.allow_tf32 = False
            with torch.autocast(output_gradient.device.type, enabled=False):
                dinput = torch.nn.grad.conv2d_input(ctx.shape, weight.float(), output_gradient.float(),
                                                  *ctx.geometry).to(ctx.dtype)
        finally:
            flags.allow_tf32 = previous
        ctx.policy.record(ctx.name)
        return dinput, None, None, None, None


class InputGradientPolicy:
    def __init__(self, expected):
        self.expected = expected
        self.originals = []
        self.entered = Counter()
        self.completed = Counter()
        self.result = {}

    def attach(self, model, named):
        require(len(named) == 1120 and all(".lora_" in name for name in named), "ORIGINAL_TRAINABLES")
        for name, module in model.named_modules():
            if type(module) is not torch.nn.Conv2d:
                continue
            require(not module.weight.requires_grad and (module.bias is None or not module.bias.requires_grad)
                    and module.padding_mode == "zeros" and not isinstance(module.padding, str),
                    "EXACT_FROZEN_CONVOLUTION")
            original = module.forward
            self.originals.append((module, original, "forward" in vars(module)))
            module.forward = partial(self.forward, name, module, original)
        require(bool(self.originals), "MISSING_UNET_CONVOLUTION")

    def forward(self, name, module, original, value):
        if not torch.is_grad_enabled() or not value.requires_grad:
            return original(value)
        self.entered[name] += 1
        return FrozenConvInputGradient.apply(value, module, original, self, name)

    def record(self, name):
        self.completed[name] += 1

    def verify_prediction(self, raw):
        encoded = raw.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(encoded).hexdigest()
        require(tuple(raw.shape) == (1, 4, 32, 32) and raw.dtype == torch.bfloat16
                and digest == self.expected["hip_raw_prediction_SHA256"], "UNCHANGED_HIP_FORWARD")
        self.result["hip_raw_prediction_SHA256"] = digest

    def finish(self):
        require(self.entered == self.completed and bool(self.completed), "COMPLETE_FROZEN_CONV_DGRAD_COVERAGE")
        blocks = Counter()
        for name, count in self.completed.items():
            match = BLOCK.search(name)
            blocks[match.group(1) if match else "other"] += count
        targeted = set(self.expected["prior_high_error_blocks"])
        # Require all four actual prior error-bearing regions before using this counterfactual.
        require(targeted <= set(blocks), "FOUR_ORIGINAL_ERROR_BLOCKS_NOT_COVERED")
        self.result.update(frozen_convolutions_patched=len(self.originals), distinct_dinput_sites=len(self.completed),
                           actual_dinput_calls=sum(self.completed.values()), dinput_by_block=dict(blocks),
                           original_forward_output_exact=True, known_partial_dgrad_coverage=False)

    def close(self):
        for module, original, overridden in reversed(self.originals):
            if overridden:
                module.forward = original
            else:
                del module.forward
        self.originals.clear()


def cpu_check():
    require(torch.cuda.device_count() == 0 and not torch.cuda.is_initialized(), "CPU_CHECK_LEASE")
    torch.manual_seed(173)
    torch.set_num_threads(1)
    geometry = [(1, 1, 1, 1), (2, 2, 1, 2), (1, 2, 2, 1)]
    for stride, pad, dilation, groups in geometry:
        module = torch.nn.Conv2d(4, 8, 3, stride, pad, dilation, groups=groups, dtype=torch.bfloat16)
        module.requires_grad_(False)
        original = module.forward
        data = torch.randn(2, 4, 9, 12, dtype=torch.bfloat16, requires_grad=True)
        native = original(data)
        policy = InputGradientPolicy({})
        replaced = FrozenConvInputGradient.apply(data, module, original, policy, "fixture")
        require(torch.equal(replaced, native), "CPU_FORWARD_BYTES")
        dy = torch.randn_like(replaced)
        replaced.backward(dy)
        # Float64 is restricted to this CPU fixture.
        leaf = data.detach().double().requires_grad_()
        oracle = torch.nn.functional.conv2d(leaf, module.weight.double(), None, stride, pad, dilation, groups)
        truth, = torch.autograd.grad(oracle, leaf, dy.double())
        require(torch.equal(data.grad, truth.to(data.dtype)) and policy.completed == {"fixture": 1}
                and module.weight.grad is None and module.bias.grad is None, "CPU_DGRAD_NUMERICS")
    print(json.dumps(dict(cpu_fixture_shapes=len(geometry), gpu_initialized=torch.cuda.is_initialized())), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("reference", "policy", "weights", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--accelerator", choices=("hip",), required=True)
    args = parser.parse_args()
    claim = json.loads(Path(__file__).with_name("CLAIM.json").read_bytes())
    binding = replay.verify_reference(args.reference, args.policy)
    args.output.mkdir(mode=0o700, exist_ok=False)
    patch = InputGradientPolicy(claim["intervention"])
    try:
        replay.replay(args.reference, args.weights, args.output / "replay", binding,
                      accelerator="hip", weight_gradient=patch)
    finally:
        patch.close()
        with (args.output / "intervention.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(patch.result, observed_dinput_sites=len(patch.completed)), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        main()
