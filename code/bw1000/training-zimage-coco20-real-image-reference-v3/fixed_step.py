"""Run the common fixed-record path on the original full native HIP Z-Image model."""

import io
import json
from math import isfinite
import os
from pathlib import Path

import coco20_step as original


REFERENCE = Path("/original-first")
SOURCE = Path("/native-source")


def marker(root, name, value):
    with (root / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def run(model, native, torch, root, claim, deadline, absolute_deadline):
    from diffsynth.diffusion import loss
    from gencuda import hy_resume_checkpoint as common
    from gencuda.zimage_fixed_reference import load_fixed_reference
    from gencuda.zimage_native_record import _zero_fixed_forward

    need = original.require
    source = SOURCE / "diffsynth/diffusion/loss.py"
    original.exact(source, claim["loss_provider"])
    need(Path(loss.__file__).resolve() == source and native.FlowMatchSFTLoss is loss.FlowMatchSFTLoss
         and model.task == "sft", "Original HIP loss and registered recorder disagree")
    captures, backend = original.pinned_framework(claim)
    inputs, data_identity = original.prepare(root, claim, deadline)
    model.train()
    model.pipe.scheduler.set_timesteps(1000, training=True)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p, p._version) for name, p in model.named_parameters() if not p.requires_grad]
    need(len(named) == 476 and sum(p.numel() for _, p in named) == 9922560,
         "Original full LoRA boundary differs")

    progress_pin = claim["fixed_hip_reference"]["progress"]
    progress = json.loads(original.exact(REFERENCE / "progress.json", progress_pin))
    members = {pin["name"]: pin for pin in progress["files"]}
    need(len(members) == len(progress["files"]) == 12, "Original first-update archive inventory differs")
    initial = torch.load(io.BytesIO(original.exact(REFERENCE / "initial.pt", members["initial.pt"])),
                         map_location="cpu", weights_only=True)
    need(list(initial) == [name for name, _ in named]
         and all(initial[name].dtype == p.dtype and initial[name].shape == p.shape
                 and torch.equal(initial[name], p.detach().cpu()) for name, p in named),
         "Full native HIP initial adapter differs from original first step")
    del initial
    first_reference = load_fixed_reference(torch, REFERENCE, progress_pin["SHA256"])
    reference_sha = common.tree_sha256(first_reference.reference)
    old_forward = torch.load(io.BytesIO(original.exact(REFERENCE / "forward.pt", members["forward.pt"])),
                             map_location="cpu", weights_only=True)
    need(old_forward.keys() == {"noise_prediction", "loss"}, "Original native forward archive differs")
    original_loss = float(old_forward["loss"])
    prediction_shape = old_forward["noise_prediction"].shape
    prediction_sha = common.tree_sha256(old_forward["noise_prediction"])
    del old_forward

    optimizer = torch.optim.AdamW(list(model.trainable_modules()), lr=1e-4, weight_decay=.01)
    need({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for _, p in named},
         "Optimizer parameter coverage differs")
    deadline()
    preview = _zero_fixed_forward(torch, common, model, native, inputs, first_reference, optimizer)
    marker(root, "fixed-zero-forward.json", preview)
    need(preview["prepared_SHA256"] == reference_sha and isfinite(preview["loss"]),
         "Registered fixed preview used different realized inputs")

    class Recorder(captures.ZImageFixedFirstUpdateRecorder):
        def prepared(self, inputs, noise, timestep_id, timestep, target):
            fixed = dict(inputs=captures.safe_input(inputs), noise=noise, timestep_id=timestep_id, timestep=timestep,
                         training_target=target)
            need(common.tree_sha256(fixed) == reference_sha, "Recorded realized inputs differ from the fixed archive")
            super().prepared(inputs, noise, timestep_id, timestep, target)

        def forward_value(self, prediction, value):
            need(prediction.shape == prediction_shape and torch.isfinite(prediction).all().item()
                 and isfinite(float(value.detach())), "Actual full forward is invalid")
            self.prediction_sha = common.tree_sha256(prediction)
            super().forward_value(prediction, value)

        def gradients(self, stage, norm=None):
            need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for _, p in named),
                 "Actual first gradient is nonfinite or missing")
            if stage == "postclip":
                need(isfinite(float(norm)) and float(norm) > 0, "Actual first clipped norm is invalid")
            super().gradients(stage, norm)
            if stage == "postclip":
                marker(root, "optimizer-step-intent.private.json", {"planned_optimizer_updates": 1})

        def updated(self):
            marker(root, "optimizer-step-returned.private.json", {"actual_optimizer_updates": 1})
            need(all(torch.isfinite(p).all().item() for _, p in named), "Updated LoRA parameter is nonfinite")
            need(all(torch.isfinite(optimizer.state[p][key]).all().item()
                     for _, p in named for key in ("exp_avg", "exp_avg_sq")),
                 "Updated AdamW state is nonfinite")
            super().updated()

    intent = json.loads((root / "intent.json").read_bytes())
    recorder = Recorder(root / "first-update-private", named, optimizer, epoch=intent["epoch"],
                        deadline=absolute_deadline, device_ids=[0], total_limit=256 << 20,
                        binding={"case": "T07.sft", "original_hip_fixed_operands": True,
                                 "reference_progress_SHA256": progress_pin["SHA256"]})
    fixed_update = load_fixed_reference(torch, REFERENCE, progress_pin["SHA256"])
    result = captures.recorded_native_update(captures.instrument_update(backend._train_step),
                                             native, recorder, 1, model, optimizer, inputs,
                                             claim["base_seed"] + 1, 1., fixed_reference=fixed_update)
    deadline()
    need(fixed_update.applied and recorder.first_complete and recorder.completed_steps == 1 and len(recorder.files) == 12,
         "Registered fixed first-step archive is incomplete")
    need(all(p.grad is None and p._version == version for _, p, version in frozen),
         "Original frozen parameter boundary differs after fixed first update")
    recorder.finish(1, True)
    return {
        "optimizer_updates": 1, "first_update_private_files": 12, "first_update_private_bytes": recorder.total,
        "first_update_private_progress_SHA256": original.sha256((recorder.root / "progress.json").read_bytes()).hexdigest(),
        "fixed_operands_SHA256": reference_sha, "input_identity": data_identity, "zero_forward_loss": preview["loss"],
        "actual_update_loss": result["loss"], "original_archived_HIP_loss": original_loss,
        "gradient_norm": result["grad_norm"],
        "zero_prediction_matches_original_HIP": preview["prediction_SHA256"] == prediction_sha,
        "update_prediction_matches_original_HIP": recorder.prediction_sha == prediction_sha,
        "zero_and_update_prediction_match": preview["prediction_SHA256"] == recorder.prediction_sha,
        "native_full_trainable_tensors": len(named), "original_initial_adapter_exact": True,
        "A100_reference_loaded": False, "formal_numerical_qualified": False,
    }
