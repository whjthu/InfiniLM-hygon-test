"""Change only the frozen first ResNet time projection, with exact original hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn.functional import linear

import hygon_sdxl_first_step as frozen_replay


TARGET = "down_blocks.0.resnets.0.time_emb_proj"


def pin(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                SHA256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SplitProjection:
    def __init__(self, model_type, binding):
        self.model_type = model_type
        self.binding = binding
        self.handles = []
        self.model_calls = 0
        self.evidence = {}

    def __enter__(self):
        self.handles.append(torch.nn.modules.module.register_module_forward_pre_hook(self.attach))
        return self

    def __exit__(self, *_):
        while self.handles:
            self.handles.pop().remove()

    def attach(self, model, _args):
        if not isinstance(model, self.model_type):
            return None
        self.model_calls += 1
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        require(self.model_calls == 1 and len(trainable) == 1120 and not any(name.startswith(TARGET) for name in trainable),
                "original frozen projection and 1120 training tensors")
        projection = model.get_submodule(TARGET)
        require(type(projection) is torch.nn.Linear and projection.bias is not None, "original time projection type")
        self.handles.append(projection.register_forward_hook(self.replace))
        return None

    def replace(self, layer, args, native):
        require(not self.evidence and len(args) == 1, "original single projection execution")
        operand, = args
        require(torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16,
                "original BF16 autocast")
        require(all(t.device.type == "cuda" and t.dtype == torch.bfloat16 and not t.requires_grad
                    for t in (operand, native, layer.weight, layer.bias))
                and list(layer.weight.shape) == [320,1280] and list(layer.bias.shape) == [320],
                "frozen device projection inputs")
        self.evidence.update(operand=pin(operand),original_native=pin(native),weight=pin(layer.weight),bias=pin(layer.bias))
        require(self.evidence["operand"] == self.binding["operand"] and
                self.evidence["original_native"] == self.binding["hip_original"], "frozen input and native projection hashes")
        dot = linear(operand, layer.weight, None)
        replacement = dot + layer.bias
        require(dot.dtype == replacement.dtype == torch.bfloat16 and tuple(replacement.shape) == tuple(native.shape),
                "BF16 rounded dot and bias output schema")
        self.evidence.update(rounded_dot=pin(dot),replacement=pin(replacement))
        self.evidence["exact_A100_projection"] = self.evidence["replacement"] == self.binding["a100_original"]
        require(self.evidence["exact_A100_projection"], "device split projection does not match original A100")
        return replacement


def main():
    parser = argparse.ArgumentParser()
    for field in ("reference","policy","weights","output"):
        parser.add_argument("--"+field, required=True, type=Path)
    parser.add_argument("--accelerator", choices=("hip",), required=True)
    args = parser.parse_args()
    from diffusers import UNet2DConditionModel
    claim = json.loads(Path(__file__).with_name("CLAIM.json").read_bytes())
    verified = frozen_replay.verify_reference(args.reference, args.policy)
    args.output.mkdir(mode=0o700, exist_ok=False)
    observer = SplitProjection(UNet2DConditionModel, claim["intervention"])
    try:
        with observer:
            frozen_replay.replay(args.reference, args.weights, args.output/"replay", verified, accelerator="hip")
        require(observer.model_calls==1 and observer.evidence.get("exact_A100_projection") is True,
                "full replay did not apply the verified projection")
    finally:
        with (args.output/"intervention.json").open("x",encoding="utf8") as target:
            json.dump(observer.evidence,target,indent=2,allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())


if __name__ == "__main__":
    main()
