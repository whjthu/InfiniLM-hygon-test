"""Observe the exact native operands and the actual AdamW update in-process."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path


TOKENS = {
    ("9e44b6497ce8e086e580c84dd81be90430f38413353a406f1c2c1ab7de0e7983",
     "7203443496cbf7f99f0a8db2ed68c70814f1af26f447158f48cd95c75124b944"): "painted_stool",
    ("c589ebab6341e39916d867afe024b27b55c93e382f3253ddf4a042383d3fd3b1",
     "757a83130f1a31557f0b50ba82a83e15a8f50a3e47a5db7ff3463ab4eaf82d4e"): "tool_chest",
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def exclusive(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def tensor_bytes(tensor):
    import torch

    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(tensor, shape, device, dtype=None):
    require(tuple(tensor.shape) == shape and str(tensor.device) == device
            and (dtype is None or str(tensor.dtype) == dtype), "LIVE_OPERAND_SHAPE_DTYPE_DEVICE")
    require(bool(tensor.detach().isfinite().all()), "NONFINITE_LIVE_OPERAND")
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=device,
                SHA256=sha256(tensor_bytes(tensor)).hexdigest())


class Witness:
    def __init__(self, args, accelerator):
        import torch

        require((args.max_train_steps, args.resolution, args.rank, args.train_batch_size,
                 args.gradient_accumulation_steps, args.mixed_precision, args.lr_scheduler,
                 args.lr_warmup_steps, args.learning_rate, args.checkpointing_steps,
                 args.noise_offset) == (25, 256, 4, 1, 1, "bf16", "constant", 0, 1e-4, 1, 0)
                and not any((args.train_text_encoder, args.resume_from_checkpoint, args.gradient_checkpointing,
                             args.random_flip, args.validation_prompt, args.push_to_hub, args.dataset_name,
                             args.pretrained_vae_model_name_or_path, args.genmlu_full_checkpoint))
                and args.center_crop and args.prediction_type is None, "EXACT_FIRST_UPDATE_RECIPE")
        require(accelerator.num_processes == 1 and accelerator.is_main_process
                and accelerator.device.type == "cuda" and accelerator.device.index in (0, None)
                and torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0 and torch.version.hip,
                "EXACT_SINGLE_HIP_DEVICE")
        self.device = "cuda:0"
        self.out = Path(args.output_dir)
        self.data = Path(args.train_data_dir)

    def before_forward(self, batch, transform, **operands):
        import torch
        from PIL import Image

        keys = (batch["input_ids_one"], batch["input_ids_two"])
        require(all(tuple(value.shape) == (1, 77) and value.dtype == torch.int64
                    and str(value.device) == self.device for value in keys), "NATIVE_TOKEN_SHAPE")
        hashes = tuple(sha256(tensor_bytes(value)).hexdigest() for value in keys)
        require(hashes in TOKENS, "NO_EXACT_NATIVE_CAPTION_TOKEN_PAIR")
        role = TOKENS[hashes]
        with Image.open(self.data / (role + ".png")) as image:
            reference = transform(image.convert("RGB"))
        require(batch["original_sizes"] == [(256, 256)] and batch["crop_top_lefts"] == [(0, 0)]
                and torch.equal(batch["pixel_values"].detach().cpu(), reference[None]),
                "LIVE_RGB_DOES_NOT_MATCH_TOKEN_ROLE")
        shapes = {"pixels": (1, 3, 256, 256), "clean_latent": (1, 4, 32, 32),
                  "noise": (1, 4, 32, 32), "noisy_latent": (1, 4, 32, 32), "timestep": (1,),
                  "sequence": (1, 77, 2048), "pooled": (1, 1280), "time_ids": (1, 6)}
        require(set(operands) == set(shapes), "MISSING_NATIVE_OPERAND")
        devices = {"pixels": "torch.float32", "timestep": "torch.int64", **{
            key: "torch.bfloat16" for key in shapes if key not in ("pixels", "timestep")}}
        records = {name: tensor_record(operands[name], shape, self.device, devices[name])
                   for name, shape in shapes.items()}
        self.forward = dict(role=role, tokens_SHA256=list(hashes), operands=records,
                            timestep=int(operands["timestep"].item()))
        exclusive(self.out / "before-forward.json", self.forward)

    def before_step(self, unet, optimizer, accelerator, loss, grad_norm, prediction, target, prediction_type):
        import torch

        require(accelerator.sync_gradients and prediction_type == "epsilon", "NATIVE_UPDATE_NOT_SYNCHRONIZED")
        peft = accelerator.unwrap_model(unet).peft_config
        require(set(peft) == {"default"} and (peft["default"].r, peft["default"].lora_alpha) == (4, 4)
                and set(peft["default"].target_modules) == {"to_k", "to_q", "to_v", "to_out.0"},
                "NATIVE_LORA_CONFIGURATION_CHANGED")
        native = optimizer.optimizer
        self.named = sorted((name, value) for name, value in unet.named_parameters() if value.requires_grad)
        group_values = [value for group in native.param_groups for value in group["params"]]
        require(type(native) is torch.optim.AdamW and not native.state and len(native.param_groups) == 1
                and bool(self.named) and all("lora_" in name for name, _ in self.named)
                and {id(value) for _, value in self.named} == {id(value) for value in group_values}
                and len(self.named) == len(group_values), "EXACT_FRESH_LORA_ADAMW_OWNERSHIP")
        self.lr = float(native.param_groups[0]["lr"])
        self.loss, self.grad_norm = float(loss.detach().item()), float(grad_norm.detach().item())
        require(self.lr == 1e-4 and math.isfinite(self.loss) and self.loss >= 0
                and math.isfinite(self.grad_norm) and self.grad_norm > 0, "NATIVE_LOSS_GRAD_OR_LR")
        require(all(str(value.device) == self.device and bool(value.detach().isfinite().all())
                    for _, value in self.named), "NONFINITE_OR_WRONG_DEVICE_LORA")
        self.grad_names = {name for name, value in self.named if value.grad is not None}
        require(self.grad_names and all(bool(value.grad.detach().isfinite().all())
                                        for _, value in self.named if value.grad is not None), "NONFINITE_LORA_GRAD")
        self.before = {name: sha256(tensor_bytes(value)).hexdigest() for name, value in self.named}
        self.loss_operands = {name: tensor_record(value, (1, 4, 32, 32), self.device)
                              for name, value in (("prediction", prediction), ("target", target))}
        exclusive(self.out / "before-adamw.json", dict(applied_lr=self.lr, loss=self.loss,
                  grad_norm_preclip=self.grad_norm, gradient_tensor_count=len(self.grad_names),
                  loss_operands=self.loss_operands, parameters_before_SHA256=digest(self.before)))

    def after_step(self, optimizer, accelerator):
        native = optimizer.optimizer
        require(not optimizer.step_was_skipped and not accelerator.optimizer_step_was_skipped
                and all(int(native.state[value]["step"].item()) == 1
                        for name, value in self.named if name in self.grad_names)
                and len(native.state) == len(self.grad_names), "ADAMW_SKIPPED_OR_WRONG_COUNTER")
        require(all(bool(value.detach().isfinite().all()) for _, value in self.named), "NONFINITE_UPDATED_LORA")
        after = {name: sha256(tensor_bytes(value)).hexdigest() for name, value in self.named}
        changed = sum(self.before[name] != value for name, value in after.items())
        require(changed > 0, "NO_ACTUAL_LORA_PARAMETER_CHANGE")
        exclusive(self.out / "first-adamw.json", dict(actual_optimizer_updates=1, skipped=False,
                  applied_lr=self.lr, loss=self.loss, grad_norm_preclip=self.grad_norm, changed_tensors=changed,
                  trainable_tensors=len(after), before_SHA256=digest(self.before), after_SHA256=digest(after),
                  planned_scheduler_steps=25, consumed_role=self.forward["role"]))


def digest(mapping):
    return sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
