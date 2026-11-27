"""Observe the native recovery of adapter, AdamW, schedule, RNG and five real updates."""

import functools
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import load_file
import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu
from vlm.runtime.initial_trainables import named_parameter_sha256
from vlm.runtime.native_recipe_contract import validate_native_recipe

from domain_checks import PREPARE_SHA, after_sample, before_sample, verified_domain
from scope import CALLBACK, NATIVE, REQUEST, need, pin, write_once


CHECKPOINT = Path("/gqa-recovery-cp20")


def unpack_optimizer(optimizer, named):
    current = [param for group in optimizer.param_groups for param in group["params"]]
    need(len(current) == 504 and {id(param) for param in current} == {id(param) for _, param in named},
         "Native optimizer parameter membership differs")
    return current


def validate_start(trainer, named):
    adapter = load_file(str(CHECKPOINT / "adapter_model.safetensors"), device="cpu")
    keys = {name.replace(".default.", ".") for name, _ in named}
    need(set(adapter) == keys and len(keys) == 504, "Original checkpoint tensor names differ")
    for name, value in named:
        original = adapter[name.replace(".default.", ".")]
        need(value.dtype == original.dtype and torch.equal(value.detach().cpu(), original),
             "Native Trainer did not load the original CP20 trainable values")
    opt = trainer.optimizer
    expected = torch.load(CHECKPOINT / "optimizer.pt", weights_only=True, map_location="cpu")
    current = unpack_optimizer(opt, named)
    plan = [i for group in expected["param_groups"] for i in group["params"]]
    need(len(plan) == len(current) == len(opt.state) == 504 and set(plan) == set(expected["state"]),
         "Incomplete resumed AdamW state")
    for param, key in zip(current, plan):
        saved, actual = expected["state"][key], opt.state[param]
        need(set(saved) == set(actual) == {"step", "exp_avg", "exp_avg_sq"}, "Original AdamW member layout differs")
        for field, value in saved.items():
            present = actual[field].detach().cpu()
            need(present.dtype == value.dtype and torch.equal(present, value),
                 "Native AdamW cursor or moment differs from CP20")
    planned = torch.load(CHECKPOINT / "scheduler.pt", weights_only=True, map_location="cpu")
    actual = trainer.lr_scheduler.state_dict()
    need(all(actual[key] == planned[key] for key in ("last_epoch", "_step_count", "_last_lr"))
         and actual["last_epoch"] == 20, "Native scheduler did not load CP20")
    need([group["lr"] for group in opt.param_groups] == planned["_last_lr"], "Resumed AdamW learning rate differs")
    return named_parameter_sha256(named)


def frozen_rng(rank):
    module = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    reconstruct = module._reconstruct
    allowed = [(reconstruct, "numpy.core.multiarray._reconstruct"),
               (reconstruct, "numpy._core.multiarray._reconstruct"),
               np.ndarray, np.dtype, type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(CHECKPOINT / f"rng_state_{rank}.pth", weights_only=True, map_location="cpu")


def rng_matches(saved, rank):
    np_state = np.random.get_state()
    need(torch.equal(torch.random.get_rng_state(), saved["cpu"])
         and random.getstate() == saved["python"]
         and np_state[0] == saved["numpy"][0]
         and np.array_equal(np_state[1], saved["numpy"][1])
         and np_state[2:] == tuple(saved["numpy"][2:])
         and torch.equal(torch.cuda.get_rng_state(rank), saved["cuda"][rank]),
         "Trainer did not restore the original per-rank RNG state")


class GQARecovery(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.trainer = trainer
        self.deadline = float(os.environ["T03_FIRST2_DEADLINE"])
        self.epoch = float(os.environ["T03_FIRST2_EPOCH"])
        self.rank = int(os.environ["RANK"])
        self.root = Path(os.environ["T03_FIRST2_CAPTURE"]) / f"rank-{self.rank:02d}"
        self.step = 20
        self.rows = []
        self.rng_checks = 0
        self.saves = []
        self.named = None
        self.pending = []
        self.rates = None
        self.original_prepare = trainer._prepare_inputs
        self.original_rng = trainer._load_rng_state
        saved_rng = frozen_rng(self.rank)

        @functools.wraps(self.original_rng)
        def restore(checkpoint):
            self.clock()
            need(Path(checkpoint).resolve(strict=True) == CHECKPOINT and self.rng_checks == 0,
                 "Repeated or different RNG restore")
            result = self.original_rng(checkpoint)
            rng_matches(saved_rng, self.rank)
            self.rng_checks += 1
            return result

        trainer._load_rng_state = restore

    def clock(self):
        need(self.deadline == self.epoch + NATIVE and self.epoch <= time.time() < self.deadline,
             "Original resume epoch expired")

    def on_train_begin(self, args, state, control, **kwargs):
        self.clock()
        rank = self.rank
        need(self.step == state.global_step == 20 and args.max_steps == 25 and
             args.gradient_accumulation_steps == 4 and args.ignore_data_skip is False
             and args.resume_only_model is False, "Native Trainer did not enter the original five-update continuation")
        need(torch.version.hip == "6.3.26113" and torch.cuda.device_count() == 8 and
             0 <= rank == int(os.environ["LOCAL_RANK"]) < 8 and torch.cuda.current_device() == rank and
             torch.distributed.is_initialized() and torch.distributed.get_world_size() == 8,
             "Original eight-rank HIP topology differs")
        binding = json.loads(pin(Path(os.environ["T03_FIRST2_BINDING"]), expected=os.environ["T03_FIRST2_BINDING_SHA256"]))
        model = kwargs.get("model", self.trainer.model)
        self.named, _ = validate_native_recipe(
            model, args, binding, checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        self.initial = validate_start(self.trainer, self.named)
        self.root.mkdir(mode=0o700)
        write_once(self.root / "restore.json", {
            "rank": rank, "request": REQUEST, "global_step": 20, "starting_trainables_SHA256": self.initial,
            "optimizer_moments_match": 504, "scheduler_epoch": 20})
        roster = verified_domain()
        qualified = Path(inspect.getsourcefile(self.original_prepare))
        pin(qualified, expected=PREPARE_SHA, extent=12688)

        @functools.wraps(self.original_prepare)
        def prepare(inputs):
            self.clock()
            need(self.rng_checks == 1 and self.step < 25 and len(self.pending) < 4,
                 "Native Trainer sampled outside the restored update cursor")
            value = before_sample(inputs, roster, lambda item: fingerprint(cpu(item)))
            prepared = self.original_prepare(inputs)
            row = after_sample(prepared, value, lambda item: fingerprint(cpu(item)))
            self.pending.append(row["ordinal"])
            return prepared

        self.trainer._prepare_inputs = prepare
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.clock()
        need(state.global_step == self.step and len(self.pending) == 4 and self.rates is None,
             "Resumed microbatch or optimizer cursor differs")
        rates = [float(group["lr"]) for group in self.trainer.optimizer.param_groups]
        need(rates and all(math.isfinite(lr) and lr > 0 for lr in rates), "Invalid resumed learning rate")
        self.rates = rates
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        need(state.global_step == self.step and self.rates and
             self.trainer.accelerator.optimizer_step_was_skipped is False, "Recovery update skipped")
        self.step += 1
        self.rows.append({"step": self.step, "learning_rates": self.rates, "roster_ordinals": self.pending})
        self.rates = None
        self.pending = []
        return control

    def on_save(self, args, state, control, **kwargs):
        need(state.global_step == 25 and self.saves == [], "Unexpected resumed checkpoint")
        self.saves.append(state.global_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        need(state.global_step == self.step, "Native recovered Trainer and callback steps differ")
        if state.global_step == 25:
            control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.clock()
        need(state.global_step == self.step == 25 and self.rng_checks == 1
             and self.saves == [25] and [r["step"] for r in self.rows] == list(range(21, 26)),
             "Incomplete original CP20 to CP25 continuation")
        final = named_parameter_sha256(self.named)
        need(final != self.initial, "No resumed parameter change")
        self.trainer._prepare_inputs = self.original_prepare
        self.trainer._load_rng_state = self.original_rng
        write_once(self.root / "resume25.json", {
            "request": REQUEST, "rank": self.rank, "completed_steps": self.rows,
            "cp20_trainables_SHA256": self.initial, "cp25_trainables_SHA256": final,
            "rng_restore_calls": self.rng_checks, "checkpoint_steps": self.saves})
        return control


need(CALLBACK not in callbacks_map, "Duplicate recovery callback")
callbacks_map[CALLBACK] = GQARecovery
