"""Observe the original registered native training function without changing its arithmetic."""

from math import isfinite

from domain_step import data_hash


class Witness:
    def __init__(self, torch, named, deadline, append, first_result=None, expected_norm=None):
        self.torch = torch
        self.named = named
        self.deadline = deadline
        self.append = append
        self.expected = first_result
        self.expected_norm = expected_norm
        self.operands = {}
        self.loss = None
        self.stages = []

    def prepared(self, inputs, noise, timestep_id, timestep, target):
        self.stages.append("prepared")
        self.deadline()
        self.operands = {"timestep_id": int(timestep_id.item()),
                         "cast_timestep_SHA256": data_hash(timestep),
                         "input_latent_SHA256": data_hash(inputs["input_latents"]), "noise_SHA256": data_hash(noise),
                         "noised_latent_SHA256": data_hash(inputs["latents"]), "training_target_SHA256": data_hash(target)}
        if self.expected is not None:
            expected = self.expected["noise_replay"]
            if any(self.operands[name] != expected[name] for name in (
                    "timestep_id", "cast_timestep_SHA256", "input_latent_SHA256", "noise_SHA256")):
                raise ValueError("Step 1 does not match the original HIP COCO20 operands")

    def forward_value(self, prediction, loss):
        self.stages.append("forward")
        self.loss = float(loss.detach())
        if not isfinite(self.loss) or self.loss < 0 or not bool(self.torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite native prediction or loss")
        if self.expected is not None and abs(self.loss - self.expected["replayed_loss"]) > 1e-6:
            raise ValueError("Step 1 does not match the original HIP COCO20 loss")
        self.operands["prediction_SHA256"] = data_hash(prediction)

    def gradients(self, stage, norm=None):
        self.stages.append(stage)
        if not all(p.grad is not None and bool(self.torch.isfinite(p.grad).all()) for p in self.named):
            raise ValueError("Native gradient missing or nonfinite")
        if stage != "postclip":
            return
        value = float(norm)
        if not isfinite(value) or value <= 0 or (self.expected_norm is not None and value != self.expected_norm):
            raise ValueError("Native clipping norm is invalid or differs at the original HIP first step")
        self.deadline()
        self.append("optimizer_intent")

    def updated(self):
        self.stages.append("updated")

    def bind_return(self, native):
        if self.stages != ["prepared", "forward", "preclip", "postclip", "updated"] or native["loss"] != self.loss:
            raise ValueError("Original native training boundary was incomplete")
