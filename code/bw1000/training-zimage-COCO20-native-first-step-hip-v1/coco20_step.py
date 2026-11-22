"""One native DiffSynth update on the original COCO20 media, with private first-step operands."""

import csv
from hashlib import sha256
import importlib
import io
import json
from math import isfinite
import os
from pathlib import Path

from domain_step import forward, data_hash


MEDIA = Path("/coco20-inputs")
SOURCE = Path("/native-source")
PROOF = Path("/coco20-preprocess/stdout.private")
FRAMEWORK = Path("/common-framework/generation/src/gencuda")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, pin):
    require(path.is_file() and not path.is_symlink() and path.resolve() == path
            and path.stat().st_size == pin["bytes"], "Original file missing or has wrong extent")
    raw = path.read_bytes()
    require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"],
            "Original file digest differs")
    return raw


def prepare(root, claim, deadline):
    contract = json.loads(exact(root / "COCO20_ORIGINAL.json", claim["input_contract"]))
    sources = claim["native_input_providers"]
    providers = {}
    for name, pin in sources.items():
        path = SOURCE / "diffsynth/core/data" / (name + ".py")
        exact(path, pin)
        module = importlib.import_module("diffsynth.core.data." + name)
        require(Path(module.__file__).resolve() == path, "Native image provider differs")
        providers[name] = module
    native = providers["unified_dataset"]
    require(native.ImageCropAndResize is providers["operators"].ImageCropAndResize,
            "Original image operator differs")
    index = contract["row_order"]
    prompts = contract["row_prompt_SHA256"]
    images = contract["images"]
    require(len(index) == len(prompts) == len(set(index)) == 20 and set(index) == set(images),
            "COCO20 original roster differs")
    metadata = MEDIA / "metadata-zimage-20.csv"
    values = list(csv.DictReader(io.StringIO(exact(metadata, contract["metadata"]).decode("utf-8"), newline="")))
    require(len(values) == 20, "Original CSV row count differs")
    for row, relative, prompt in zip(values, index, prompts, strict=True):
        deadline()
        p = Path(relative)
        require(len(p.parts) == 2 and p.parts[0] == "images" and p.name[:12].isdigit()
                and p.name[12:] == ".jpg" and row["image"] == relative
                and sha256(row["prompt"].encode()).hexdigest() == prompt, "Original media row differs")
        exact(MEDIA / p, images[relative])
    log = exact(PROOF, claim["preprocess_stdout"])
    [pre] = [json.loads(line) for line in log.splitlines() if line.startswith(b"{")]
    require(pre["request"] == claim["preprocess_request"] and pre["step"] == claim["preprocess_step"]
            and pre["passed"] is True and pre["decoded_occurrences"] == 25
            and pre["actual_DCU"] == 0 and pre["actual_optimizer_updates"] == 0,
            "Original preprocessing receipt differs")
    first = pre["input_values"][0]
    require(first["step"] == 1 and first["index"] == 0 and first["prompt_SHA256"] == prompts[0],
            "Original step-one routing differs")
    dataset = native.UnifiedDataset(base_path=str(MEDIA), metadata_path=str(metadata), repeat=1,
        data_file_keys=("image",), main_data_operator=native.UnifiedDataset.default_image_operator(
            base_path=str(MEDIA), height=256, width=256, height_division_factor=16, width_division_factor=16))
    require(len(dataset) == 20, "Native COCO20 dataset extent differs")
    data = dataset[0]
    pixels = sha256(data["image"].tobytes()).hexdigest()
    prompt = sha256(data["prompt"].encode()).hexdigest()
    require(data["image"].mode == "RGB" and data["image"].size == (256, 256)
            and pixels == first["RGB256_SHA256"] and prompt == first["prompt_SHA256"],
            "Actual native RGB256 or prompt differs from target preprocessing")
    original_manifest = MEDIA / "manifest-zimage-20.json"
    present = original_manifest.exists()
    if present:
        exact(original_manifest, contract["manifest"])
    return data, dict(index=0, rows=20, RGB256_SHA256=pixels, prompt_SHA256=prompt,
                      original_manifest_present=present, preprocessing_step=claim["preprocess_step"])


def pinned_framework(claim):
    for relative, pin in claim["framework_update_members"].items():
        exact(FRAMEWORK / relative, pin)
    from gencuda import zimage_first_update as captures
    from gencuda import vggt_first_update as base
    from gencuda import hy_resume_checkpoint as resume
    from gencuda.backends import zimage_training as backend

    for mod in (captures, base, resume, backend):
        path = Path(mod.__file__).resolve()
        require(path.relative_to(FRAMEWORK).as_posix() in claim["framework_update_members"],
                "Native update provider escaped the frozen framework")
    return captures, backend


def run(model, native, torch, root, claim, deadline, absolute_deadline):
    from diffsynth.diffusion import loss

    path = SOURCE / "diffsynth/diffusion/loss.py"
    exact(path, claim["loss_provider"])
    require(Path(loss.__file__).resolve() == path and native.FlowMatchSFTLoss is loss.FlowMatchSFTLoss
            and model.task == "sft", "Original loss entry differs")
    data, data_identity = prepare(root, claim, deadline)
    captures, backend = pinned_framework(claim)
    model.train()
    model.pipe.scheduler.set_timesteps(1000, training=True)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p, p._version) for name, p in model.named_parameters() if not p.requires_grad]
    require(len(named) == 476 and sum(p.numel() for _, p in named) == 9922560, "Original LoRA extent differs")
    optimizer = torch.optim.AdamW(list(model.trainable_modules()), lr=1e-4, weight_decay=.01)
    require({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for _, p in named},
            "Optimizer parameter set differs")
    versions = {name: p._version for name, p in named}
    seed = claim["base_seed"] + 1
    deadline()
    with torch.no_grad():
        loss_zero, zero, _ = forward(model, data, loss.FlowMatchSFTLoss, torch, seed, None)
    zero["cast_timestep_SHA256"] = data_hash(model.pipe.scheduler.timesteps[[zero["timestep_id"]]].to(
        dtype=model.pipe.torch_dtype, device=model.pipe.device))
    expected_loss = float(loss_zero)
    del loss_zero
    require(isfinite(expected_loss) and expected_loss >= 0 and not optimizer.state
            and all(p.grad is None and p._version == versions[name] for name, p in named),
            "Zero-update forward changed training state")

    class Recorder(captures.ZImageFirstUpdateRecorder):
        # The shared recorder's finish method certifies 25 updates. This scope has one.
        def progress(self):
            value = dict(case="T07.sft", planned_global_steps=1, planned_recorded_updates=1,
                         scope="one original COCO20 row, same HIP replay",
                         current_global_step=self.current_step, completed_global_steps=self.completed_steps,
                         first_update_complete=self.first_complete, epoch=self.epoch, deadline=self.deadline,
                         phase=self.phase, failed=self.failed, files=self.files, artifact_bytes=self.total,
                         diagnostic_seconds=self.diagnostic_seconds)
            tmp = self.root / "progress.tmp"
            tmp.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(self.root / "progress.json")

        def prepared(self, inputs, noise, timestep_id, timestep, target):
            require(int(timestep_id.item()) == zero["timestep_id"]
                    and data_hash(timestep) == zero["cast_timestep_SHA256"]
                    and data_hash(noise) == zero["noise_SHA256"]
                    and data_hash(inputs["input_latents"]) == zero["input_latent_SHA256"],
                    "Native training did not reproduce the HIP zero-forward operands")
            super().prepared(inputs, noise, timestep_id, timestep, target)

        def forward_value(self, prediction, observed_loss):
            value = float(observed_loss.detach())
            require(isfinite(value) and abs(value - expected_loss) <= 1e-6,
                    "Native training did not reproduce the HIP zero-forward loss")
            super().forward_value(prediction, observed_loss)

        def gradients(self, stage, norm=None):
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named),
                    "Native gradient missing or nonfinite")
            if stage == "postclip":
                require(isfinite(float(norm)) and float(norm) > 0, "Native clip norm invalid")
            super().gradients(stage, norm)
            if stage == "postclip":
                with (root / "optimizer-step-intent.private.json").open("x") as stream:
                    json.dump({"planned_optimizer_updates": 1}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())

        def updated(self):
            with (root / "optimizer-step-returned.private.json").open("x") as stream:
                json.dump({"actual_optimizer_updates": 1}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            require(all(bool(torch.isfinite(p).all()) for _, p in named), "Nonfinite updated LoRA")
            for _, p in named:
                for k in ("exp_avg", "exp_avg_sq"):
                    require(bool(torch.isfinite(optimizer.state[p][k]).all()), "Nonfinite native AdamW moment")
            super().updated()

    intent = json.loads((root / "intent.json").read_bytes())
    recorder = Recorder(root/"first-update-private", named, optimizer, epoch=intent["epoch"],
                        deadline=absolute_deadline, device_ids=[0], total_limit=256<<20,
                        binding=dict(case="T07.sft", input_contract_SHA256=claim["input_contract"]["SHA256"],
                                     original_media_verified=True, original_A100_operands_reused=False))
    update = captures.instrument_update(backend._train_step)
    value = captures.recorded_native_update(update, native, recorder, 1, model, optimizer, data, seed, 1.)
    deadline()
    require(recorder.first_complete and recorder.completed_steps == 1 and len(recorder.files) == 12,
            "Original-format first-step capture is incomplete")
    require(all(p.grad is None and p._version == before for _, p, before in frozen),
            "Frozen parameter received a gradient or mutated")
    recorder.phase = "HIP_COCO20_FIRST_UPDATE_RETURNED"
    recorder.progress()
    return dict(training_input_ordinal=0, input_identity=data_identity, zero_update_loss=expected_loss,
                replayed_loss=float(value["loss"]), noise_replay=zero, optimizer_updates=1,
                first_update_private_files=len(recorder.files), first_update_private_bytes=recorder.total,
                first_update_private_progress_SHA256=sha256((recorder.root/"progress.json").read_bytes()).hexdigest(),
                trainable_tensors=len(named), frozen_parameter_version_check=True,
                first_A100_timestep_equals_HIP=zero["timestep_id"]==claim["A100_first_timestep_id"],
                same_platform_only=True, A100_comparison=False, heldout_used=False)
