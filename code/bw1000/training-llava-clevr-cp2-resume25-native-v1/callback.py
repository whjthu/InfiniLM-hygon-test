"""Verify native CP2 state restore and every production LLaVA continuation batch."""

from functools import wraps
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.native_recipe_contract import validate_native_recipe

import probe
import restore
import resume_config as scope
import train
from first2_observe import geometry, input_sha, snapshot


class LlavaResume25(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.trainer = trainer
        self.rank = int(os.environ["RANK"])
        self.request = os.environ["LLAVA_REQUEST"]
        self.deadline = float(os.environ["LLAVA_DEADLINE"])
        self.output = Path(os.environ["LLAVA_RUN"]) / f"rank-{self.rank:02d}"
        self.step = 2
        self.forwards, self.losses, self.saves = [], 0, []
        self.events = []
        self.epoch = None
        self.prepared = self.before = self.after = None
        self.summary = None
        self.restore_calls, self.originals = restore.bind_hooks(
            trainer, scope.CP2, lambda: restore.load_rng(torch, np, scope.CP2 / f"rng_state_{self.rank}.pth"),
            self.check_local_rng, scope.source_provider)

    def check_local_rng(self, saved):
        restore.compare_rng(torch, np, saved, self.rank)
        self.report("rng-restored.json", device_rng=True, restored_CP2=True)

    def report(self, name, **fields):
        probe.persist(self.output / name, dict(request=self.request, rank=self.rank, **fields))

    def agreement(self, value):
        records = [None] * 8
        torch.distributed.all_gather_object(records, value)
        return records

    def clock(self):
        scope.need(time.time() < self.deadline, "BOUNDED_NATIVE_RESUME_DEADLINE")

    def on_train_begin(self, args, state, control, **kwargs):
        self.clock()
        d = torch.distributed
        scope.need((state.global_step, state.epoch, state.max_steps) == (2, 1.0, 25)
                   and not args.ignore_data_skip and not args.resume_only_model
                   and not args.gradient_checkpointing and args.dataloader_num_workers == 4
                   and args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 4
                   and torch.version.hip == "6.3.26113" and not torch.version.cuda
                   and d.is_initialized() and (d.get_rank(), d.get_world_size()) == (self.rank, 8)
                   and torch.cuda.device_count() == 8 and torch.cuda.current_device() == self.rank,
                   "ACTUAL_CP2_HIP_25_STEP_RECIPE_AND_CURSOR")
        self.output.mkdir(mode=0o700)
        model = kwargs["model"]
        binding = json.loads(probe.read(train.HELPERS / "NATIVE_BINDING.json", 2 * 1024**2, train.NATIVE))
        self.named, frozen = validate_native_recipe(model, args, binding)
        core = model.get_base_model()
        scope.need(type(core) is train.EXPECTED_CLASS and getattr(core, "_vlm_onevision_rope_initialized", False)
                   and self.trainer.template.template_meta.template_type == "llava_onevision1_5"
                   and len(self.named) == 504 and all(p.device.index == self.rank for _, p in self.named)
                   and len(train.CHECKPOINT_ASSIGNMENT) == 1
                   and all(train.CHECKPOINT_ASSIGNMENT[0][field]["count"] == 0 for field in
                           ("missing_keys", "unexpected_keys", "mismatched_keys")),
                   "EXACT_EAGER_T04_BASE_AND_LORA_STRUCTURE")
        scope.need(self.restore_calls["_load_from_checkpoint"] == self.restore_calls["_load_optimizer_and_scheduler"] == 1
                   and self.restore_calls["_load_rng_state"] == 0, "ACTUAL_NATIVE_ADAPTER_AND_OPTIMIZER_RESTORE_HOOKS")
        self.optimizer, evidence = restore.live_resume(torch, self.trainer, self.named, scope.CP2, fingerprint)
        scope.need(self.agreement(evidence) == [evidence] * 8, "EIGHT_RANK_CP2_RESTORE_AGREEMENT")
        self.report("restore.json", **evidence, frozen=frozen, original_cp2_step=2, no_intra_epoch_skip=True)
        images = json.loads(probe.read(train.HELPERS / "INPUTS.json", 128 << 10, train.IMAGE_BIND))
        rows = probe.source_rows(train.ASSETS)
        proof = json.loads(probe.read(train.HELPERS / "TENSORS.json", 32 << 10, os.environ["LLAVA_TENSORS_SHA256"]))
        self.bounds = train.roster(proof, rows, images, probe.TRAIN)
        loader = kwargs["train_dataloader"]
        from swift.dataloader.shard import BatchSamplerShard, DataLoaderShard
        scope.source_provider(Path(inspect.getsourcefile(DataLoaderShard)), "set_epoch")
        sampler = loader.batch_sampler
        scope.need(type(loader) is DataLoaderShard and type(sampler) is BatchSamplerShard
                   and loader.num_workers == 4 and loader.in_order is True and not loader.drop_last
                   and len(loader) == len(sampler) == 8 and len(loader.dataset) == 64
                   and (sampler.rank, sampler.world_size, sampler.base_seed, sampler.total_samples)
                   == (self.rank, 8, 42, 8), "ACTUAL_NATIVE_FOUR_WORKER_NO_SKIP_SAMPLER")
        prior = loader.set_epoch

        @wraps(prior)
        def set_epoch(epoch):
            self.clock()
            scope.need(type(epoch) is int and epoch == self.step // 2 and loader.batch_sampler is sampler,
                       "EXACT_NATIVE_CPU_BOUND_START_AND_NEXT_EPOCH")
            result = prior(epoch)
            scope.need(sampler.curr_seed == 42 + epoch and epoch != self.epoch, "ACTUAL_NATIVE_EPOCH_SEED")
            self.epoch = epoch
            self.report(f"epoch-{epoch:02d}.json", epoch=epoch, seed=sampler.curr_seed)
            return result

        self.loader, self.prior_epoch = loader, prior
        loader.set_epoch = set_epoch
        self.original_prepare, self.original_loss = self.trainer._prepare_inputs, self.trainer.compute_loss
        self.trainer._prepare_inputs, self.trainer.compute_loss = self.prepare, self.loss
        # Observe the exact collated tensor before Swift converts it to multimodal embeddings.
        self.events = [model.register_forward_pre_hook(self.forward, with_kwargs=True, prepend=True),
                       self.optimizer.register_step_pre_hook(self.adamw_pre),
                       self.optimizer.register_step_post_hook(self.adamw_post)]
        return control

    def prepare(self, inputs, *args, **kwargs):
        self.clock()
        scope.need(self.restore_calls["_load_rng_state"] == 1 and self.prepared is None and self.step < 25,
                   "PREPARE_ONLY_AFTER_ACTUAL_DEVICE_RNG_RESTORE")
        key = input_sha(inputs)
        scope.need(key in self.bounds, "ACTUAL_PREPARE_4878_COLLATOR_FINGERPRINT")
        self.prepared = (key, geometry(inputs))
        return self.original_prepare(inputs, *args, **kwargs)

    def forward(self, _, args, inputs):
        micro = len(self.forwards) - 4 * (self.step - 2)
        scope.need(not args and self.prepared is not None and 0 <= micro < 4
                   and self.epoch == self.step // 2, "EXACT_NATIVE_RESUMED_FORWARD_POSITION")
        slot = scope.UPDATE_PLAN[self.step + 1][micro * 8 + self.rank]
        try:
            key = input_sha(inputs)
            scope.need(key == self.prepared[0] == slot["input_sha256"] and all(
                self.bounds[key][field] == slot[field] for field in
                ("ordinal", "row_sha256", "image_sha256", "input_sha256")), "ACTUAL_FORWARD_MATCHES_PRIVATE_CPU_PLAN")
        except (KeyError, ValueError, TypeError) as error:
            self.report("forward-reject.json", step=self.step + 1, accumulation=micro + 1,
                        expected_SHA256=slot["input_sha256"], prepared_SHA256=self.prepared[0],
                        observed_geometry=geometry(inputs), error_type=type(error).__name__)
            raise
        self.prepared = None
        self.forwards.append(slot)
        position = dict(slot)
        position.pop("rank")
        self.report(f"forward-{len(self.forwards):02d}.json", global_step=self.step + 1, **position)

    def loss(self, *args, **kwargs):
        value = self.original_loss(*args, **kwargs)
        loss = value[0] if isinstance(value, tuple) else value
        scope.need(torch.is_tensor(loss) and loss.numel() == 1 and bool(torch.isfinite(loss.detach()).all())
                   and self.losses + 1 == len(self.forwards), "ACTUAL_NATIVE_FINITE_ONEVISION_LOSS")
        self.losses += 1
        return value

    def adamw_pre(self, optimizer, args, kwargs):
        self.clock()
        next_step = self.step + 1
        scope.need(optimizer is self.optimizer and len(self.forwards) == self.losses == 4 * (self.step - 1)
                   and self.before is None and self.after is None
                   and all(optimizer.state[param]["step"].item() == self.step for _, param in self.named),
                   "ACTUAL_504_PRE_ADAMW_STEP_COUNTERS")
        rates = [group["lr"] for group in optimizer.param_groups]
        scope.need(rates and all(math.isfinite(lr) and lr > 0 for lr in rates), "POSITIVE_NATIVE_RESUMED_LEARNING_RATE")
        snapshot_needed = next_step in (3, 10, 20, 25)
        self.before = (rates, snapshot(self.named) if snapshot_needed else None)
        self.report(f"update-{next_step:02d}-intent.json", prior_adamw_step=self.step, rates=rates)

    def adamw_post(self, optimizer, args, kwargs):
        scope.need(optimizer is self.optimizer and self.before is not None and all(
            optimizer.state[param]["step"].item() == self.step + 1 for _, param in self.named),
            "ACTUAL_504_POST_ADAMW_STEP_COUNTERS")
        rates, prior = self.before
        self.before = None
        self.after = dict(learning_rates=rates, finite_losses=4, adamw_tensors=504,
                          pre_adamw_step=self.step, post_adamw_step=self.step + 1)
        if prior is not None:
            live = snapshot(self.named)
            before, after = (fingerprint(values)["SHA256"] for values in (prior, live))
            squared = sum(float((prior[name].double() - value.double()).square().sum())
                          for name, value in live.items())
            scope.need(math.isfinite(squared) and (self.step != 2 or squared > 0 and before != after),
                       "POSITIVE_STEP3_MUST_CHANGE_ACTUAL_PARAMETERS")
            self.after.update(pre_SHA256=before, post_SHA256=after, parameter_delta_l2=math.sqrt(squared))
        self.report(f"update-{self.step + 1:02d}-returned.json", **self.after)

    def on_optimizer_step(self, args, state, control, **kwargs):
        scope.need(self.after is not None and self.trainer.accelerator.optimizer_step_was_skipped is False
                   and self.agreement(self.after) == [self.after] * 8, "EIGHT_RANK_ACTUAL_UNSKIPPED_ADAMW")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        scope.need(self.after is not None and state.global_step == self.step + 1, "EXACT_CONTINUED_GLOBAL_CURSOR")
        self.step += 1
        self.after = None
        if self.step == 25:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        scope.need(state.global_step == self.step and self.step in (10, 20, 25)
                   and self.step not in self.saves, "EXACT_NEW_NATIVE_CHECKPOINT_BOUNDARIES")
        self.saves.append(self.step)
        self.report(f"checkpoint-{self.step:02d}.json", global_step=self.step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        scope.need(self.step == state.global_step == 25 and len(self.forwards) == self.losses == 92
                   and self.saves == [10, 20, 25] and list(self.restore_calls.values()) == [1, 1, 1],
                   "ACTUAL_23_UPDATES_92_RANK_FORWARDS_AND_THREE_NATIVE_CHECKPOINTS")
        for handle in self.events:
            handle.remove()
        self.loader.set_epoch = self.prior_epoch
        self.trainer._prepare_inputs, self.trainer.compute_loss = self.original_prepare, self.original_loss
        for key, original in self.originals.items():
            setattr(self.trainer, key, original)
        self.report("complete.json", actual_forwards=92, actual_updates=23, checkpoints=self.saves,
                    actual_device_rng_restored=True, uninterrupted_baseline_evaluated=False)
        return control


scope.need(scope.CALLBACK not in callbacks_map, "EXACT_SINGLE_CONTINUATION_CALLBACK")
callbacks_map[scope.CALLBACK] = LlavaResume25
