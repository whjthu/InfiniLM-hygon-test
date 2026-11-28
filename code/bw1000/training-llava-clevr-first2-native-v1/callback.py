"""Private, per-rank observations at native Forward and AdamW update boundaries."""

import json
import math
import os
from pathlib import Path
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.native_recipe_contract import validate_native_recipe

import probe
import train
from train import ASSETS, CALLBACK, HELPERS, IMAGE_BIND, NATIVE, need, roster


def cpu_tensor(value, integer=False):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return (value.to(torch.int64) if integer else value).contiguous()
    if not integer and isinstance(value, (list, tuple)):
        return type(value)(cpu_tensor(item) for item in value)
    raise ValueError("EXACT_COLLATED_TENSOR_CONTAINER")


def input_sha(inputs):
    names = ("input_ids", "labels", "attention_mask")
    need(all(tuple(inputs[name].shape) == tuple(inputs["input_ids"].shape)
             and not inputs[name].is_floating_point() for name in names)
         and inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[0] == 1
         and 1 < inputs["input_ids"].shape[1] <= 4096, "EXACT_FULL_UNSHIFTED_ONEVISION_INPUT")
    tensors = {name: cpu_tensor(inputs[name], integer=True) for name in names}
    tensors["pixel_values"] = cpu_tensor(inputs["pixel_values"])
    return fingerprint(tensors)["SHA256"]


def geometry(inputs):
    def describe(value):
        if torch.is_tensor(value):
            return dict(shape=list(value.shape), dtype=str(value.dtype), device=value.device.type)
        if type(value) in (tuple, list) and len(value) <= 16:
            return dict(kind=type(value).__name__, members=[describe(item) for item in value])
        return dict(kind=type(value).__name__)

    return {name: describe(inputs[name]) if name in inputs else None
            for name in ("input_ids", "labels", "attention_mask", "pixel_values")}


def snapshot(named):
    return {key: value.detach().to(device="cpu", copy=True) for key, value in named}


def verify_window(prior, local, gathered, update):
    need(len(gathered) == 8 and len(local) == 4 and all(type(row) is list and len(row) == 4 for row in gathered),
         "EXACT_8_RANKS_4_FORWARDS")
    slots = [slot for row in gathered for slot in row]
    need(len(prior) == 32 * (update - 1) and all(type(slot) is int and 1 <= slot <= 64 for slot in slots)
         and len(set(slots)) == 32 and not set(slots).intersection(prior), "EXACT_NEW_32_TRAIN_OCCURRENCES")
    return set(slots)


def learning_rate(update, rates):
    need(update in (1, 2) and rates and all(math.isfinite(rate) and
         (rate == 0 if update == 1 else rate > 0) for rate in rates), "NATIVE_ZERO_THEN_POSITIVE_LR")


class LlavaClevrFirstTwo(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.trainer = trainer
        self.request = os.environ["LLAVA_REQUEST"]
        self.deadline = float(os.environ["LLAVA_DEADLINE"])
        self.root = Path(os.environ["LLAVA_RUN"])
        self.rank = int(os.environ["RANK"])
        self.output = self.root / f"rank-{self.rank:02d}"
        self.handles = []
        self.forwards = []
        self.previous = set()
        self.steps = 0
        self.pending = None
        self.before = None
        self.after = None
        self.saved = False
        self.losses = 0
        self.prepared = None
        self.prepare_shape = None

    def report(self, name, **fields):
        probe.persist(self.output / name, dict(request=self.request, rank=self.rank, **fields))

    def agree(self, value):
        records = [None] * 8
        torch.distributed.all_gather_object(records, value)
        return records

    def on_train_begin(self, args, state, control, **kwargs):
        d = torch.distributed
        need(d.is_initialized() and (d.get_rank(), d.get_world_size()) == (self.rank, 8)
             and torch.version.hip == "6.3.26113" and not torch.version.cuda
             and torch.cuda.device_count() == 8 and torch.cuda.current_device() == self.rank
             and state.global_step == 0 and time.time() < self.deadline, "EXACT_EIGHT_FRESH_HIP_RANKS")
        self.output.mkdir(mode=0o700)
        model = kwargs["model"]
        binding = json.loads(probe.read(HELPERS / "NATIVE_BINDING.json", 2 * 1024**2, NATIVE))
        self.named, frozen = validate_native_recipe(model, args, binding)
        core = model.get_base_model()
        need(type(core) is train.EXPECTED_CLASS and getattr(core, "_vlm_onevision_rope_initialized", False)
             and self.trainer.template.template_meta.template_type == "llava_onevision1_5"
             and not args.gradient_checkpointing and args.max_steps == 25
             and args.dataloader_num_workers == 4
             and len(self.named) == 504 and sum(p.numel() for _, p in self.named) == 43_646_976
             and all(p.device.type == "cuda" and p.device.index == self.rank for _, p in self.named),
             "EXACT_T04_EAGER_FROZEN_RANK_RECIPE")
        assignment = train.CHECKPOINT_ASSIGNMENT
        need(len(assignment) == 1 and all(assignment[0][field]["count"] == 0
             for field in ("missing_keys", "unexpected_keys", "mismatched_keys")),
             "EXACT_TRAINING_CHECKPOINT_ASSIGNMENT")
        from vlm.plugins import llava_onevision
        llava_onevision._verify_loaded_native_eager_classes(core, type(core).__module__)
        initial = fingerprint(snapshot(self.named))["SHA256"]
        need(self.agree(initial) == [initial] * 8, "FRESH_504_INITIAL_VALUES_RANK_AGREEMENT")
        loader = kwargs["train_dataloader"]
        need(len(loader) == 8 and loader.num_workers == 4 and loader.in_order is True
             and not loader.drop_last and len(loader.dataset) == 64, "EXACT_NATIVE_64_ROW_DATALOADER")
        image = json.loads(probe.read(HELPERS / "INPUTS.json", 128 * 1024, IMAGE_BIND))
        rows = probe.source_rows(ASSETS)
        doc = json.loads(probe.read(HELPERS / "TENSORS.json", 32 * 1024, os.environ["LLAVA_TENSORS_SHA256"]))
        self.bounds = roster(doc, rows, image, probe.TRAIN)
        optimizer = getattr(self.trainer.optimizer, "optimizer", self.trainer.optimizer)
        need(type(optimizer) is torch.optim.AdamW and optimizer.state == {}
             and sorted(id(p) for group in optimizer.param_groups for p in group["params"])
             == sorted(id(p) for _, p in self.named), "NATIVE_ADAMW_OWNS_EXACT_504")
        self.optimizer = optimizer
        self.original_prepare = self.trainer._prepare_inputs
        self.original_loss = self.trainer.compute_loss
        self.trainer._prepare_inputs = self.prepare
        self.trainer.compute_loss = self.loss
        # Swift's multimodal hook consumes input_ids when it prepares inputs_embeds.
        self.handles.extend((model.register_forward_pre_hook(self.forward, with_kwargs=True, prepend=True),
                             optimizer.register_step_pre_hook(self.optimizer_pre),
                             optimizer.register_step_post_hook(self.optimizer_post)))
        self.report("start.json", initial_SHA256=initial, planned_forwards=8,
                    frozen=frozen, native_schedule_steps=25, trainable_count=504, optimizer="torch.optim.AdamW")
        return control

    def prepare(self, inputs, *args, **kwargs):
        need(self.prepared is None, "SINGLE_NATIVE_PREPARE_PER_FORWARD")
        key = input_sha(inputs)
        if key not in self.bounds:
            self.report("input-mismatch.json", boundary="prepare", observed_SHA256=key, observed=geometry(inputs))
            raise ValueError("LIVE_ONEVISION_PREPARE_DIFFERS_FROM_CPU_TRAIN_ONLY_PROOF")
        output = self.original_prepare(inputs, *args, **kwargs)
        self.prepared = key
        self.prepare_shape = geometry(inputs)
        return output

    def loss(self, *args, **kwargs):
        result = self.original_loss(*args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result
        need(torch.is_tensor(loss) and loss.numel() == 1 and bool(torch.isfinite(loss.detach()).all())
             and self.losses + 1 == len(self.forwards) and self.prepared is None,
             "ACTUAL_FINITE_NATIVE_FORWARD_LOSS")
        self.losses += 1
        self.report(f"loss-{self.losses:02d}.json", finite=True)
        return result

    def forward(self, module, args, inputs):
        need(not args and module.training and time.time() < self.deadline and self.steps < 2
             and self.steps * 4 <= len(self.forwards) < self.steps * 4 + 4,
             "EXACT_NATIVE_FIRST_TWO_FORWARD_GEOMETRY")
        try:
            key = input_sha(inputs)
            need(key == self.prepared and key in self.bounds, "LIVE_FORWARD_SIGNATURE_MISMATCH")
        except (KeyError, TypeError, ValueError) as error:
            self.report("input-mismatch.json", boundary="forward", prepared_SHA256=self.prepared,
                        prepared=self.prepare_shape, observed=geometry(inputs), error_class=type(error).__name__)
            raise
        row = self.bounds[key]
        self.prepared = None
        self.prepare_shape = None
        need(row["ordinal"] not in self.forwards, "DUPLICATED_LOCAL_FORWARD_OCCURRENCE")
        self.forwards.append(row["ordinal"])
        self.report(f"forward-{len(self.forwards):02d}.json", ordinal=row["ordinal"], input_sha256=key,
                    image_sha256=row["image_sha256"], row_sha256=row["row_sha256"], update=self.steps + 1,
                    accumulation=(len(self.forwards) - 1) % 4 + 1)

    def on_step_begin(self, args, state, control, **kwargs):
        need(state.global_step == self.steps < 2, "TWO_NATIVE_UPDATE_STOP_BOUNDARY")
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        local = self.forwards[4 * self.steps:]
        need(self.pending is None and state.global_step == self.steps and self.prepared is None
             and self.losses == len(self.forwards), "FRESH_OPTIMIZER_BOUNDARY")
        self.pending = verify_window(self.previous, local, self.agree(local), self.steps + 1)
        return control

    def optimizer_pre(self, optimizer, args, kwargs):
        need(optimizer is self.optimizer and self.pending is not None and self.before is None,
             "EXPECTED_NATIVE_ADAMW_ENTRY")
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        learning_rate(self.steps + 1, rates)
        self.before = (rates, snapshot(self.named))
        self.report(f"update-{self.steps + 1}-intent.json", learning_rates=rates,
                    initial_parameter_SHA256=fingerprint(self.before[1])["SHA256"])

    def optimizer_post(self, optimizer, args, kwargs):
        need(optimizer is self.optimizer and self.before is not None and self.after is None
             and all(float(optimizer.state[p]["step"].item()) == self.steps + 1 for _, p in self.named),
             "EXACT_NATIVE_ADAMW_STEP_COUNTERS")
        rates, previous = self.before
        current = snapshot(self.named)
        first, final = fingerprint(previous), fingerprint(current)
        changed = sum(not torch.equal(previous[key], current[key]) for key in current)
        squared = sum(float((previous[key].double() - value.double()).square().sum()) for key, value in current.items())
        update = self.steps + 1
        need((changed == 0 and squared == 0 and first["SHA256"] == final["SHA256"] if update == 1
              else changed > 0 and squared > 0 and first["SHA256"] != final["SHA256"])
             and math.isfinite(squared), "NATIVE_ZERO_THEN_POSITIVE_PARAMETER_DELTA")
        self.after = dict(learning_rates=rates, pre_SHA256=first["SHA256"], post_SHA256=final["SHA256"],
                          delta_l2=math.sqrt(squared), changed_tensors=changed, adamw_counters=504)
        self.before = None
        self.report(f"update-{update}-returned.json", **self.after)

    def on_optimizer_step(self, args, state, control, **kwargs):
        need(self.after is not None and self.trainer.accelerator.optimizer_step_was_skipped is False,
             "UNSKIPPED_NATIVE_ADAMW_RETURN")
        need(self.agree(self.after) == [self.after] * 8, "EIGHT_RANK_NATIVE_UPDATE_VALUE_AGREEMENT")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        need(self.after is not None and state.global_step == self.steps + 1, "EXACT_NATIVE_TRAINER_STEP_COUNT")
        self.previous.update(self.pending)
        self.pending = self.after = None
        self.steps += 1
        if self.steps == 2:
            need(self.previous == set(range(1, 65)), "EXACT_64_DISTINCT_TRAIN_ONLY_FORWARDS")
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        need(state.global_step == self.steps == 2 and not self.saved, "EXACT_NATIVE_CP2_SAVE_CALLBACK")
        self.saved = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        need(state.global_step == self.steps == 2 and len(self.forwards) == 8 and self.saved,
             "EXACT_COMPLETE_NATIVE_TWO_UPDATES")
        for handle in self.handles:
            handle.remove()
        self.trainer._prepare_inputs = self.original_prepare
        self.trainer.compute_loss = self.original_loss
        self.report("complete.json", actual_forwards=8, actual_updates=2, finite_losses=8, native_save_callback=True)
        return control


need(CALLBACK not in callbacks_map, "DUPLICATE_NATIVE_LLaVA_CALLBACK")
callbacks_map[CALLBACK] = LlavaClevrFirstTwo
