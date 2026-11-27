"""Verify the original CP2 restore and permit one native Swift update."""
import functools
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu
from vlm.runtime.native_recipe_contract import validate_native_recipe

import cp3_contract as contract
from domain_checks import PREPARE_SHA, after_sample, before_sample, verified_split
from live_guard import LiveCursor
from scope import CALLBACK, NATIVE, REQUEST, need, pin, write_once


def load_rng(rank):
    module = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    reconstruct = module._reconstruct
    with torch.serialization.safe_globals([
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray, np.dtype, type(np.dtype("uint32")),
    ]):
        return torch.load(contract.CHECKPOINT / f"rng_state_{rank}.pth", weights_only=True, map_location="cpu")


def confirm_rng(saved, rank):
    numpy_state = np.random.get_state()
    need(torch.equal(torch.random.get_rng_state(), saved["cpu"]) and
         random.getstate() == saved["python"] and numpy_state[0] == saved["numpy"][0] and
         np.array_equal(numpy_state[1], saved["numpy"][1]) and
         numpy_state[2:] == saved["numpy"][2:] and len(saved["cuda"]) == 8 and
         torch.equal(torch.cuda.get_rng_state(rank), saved["cuda"][rank]),
         "ACTUAL_NATIVE_CPU_AND_LOCAL_DEVICE_RNG_RESTORE")


def parameters(named):
    return fingerprint({name: cpu(parameter) for name, parameter in named})["SHA256"]


def verify_resume(trainer, named):
    need(parameters(named) == contract.CP2_FP, "LIVE_RESUMED_ADAPTER_MATCHES_EXACT_CP2")
    optimizer = trainer.optimizer
    base = getattr(optimizer, "optimizer", optimizer)
    need(type(base) is torch.optim.AdamW, "EXACT_NATIVE_ADAMW_OWNER")
    saved = torch.load(contract.CHECKPOINT/"optimizer.pt", weights_only=True, map_location="cpu")
    mapped = contract.optimizer_mapping(optimizer, named, saved)
    for key, _, parameter in mapped:
        original, actual = saved["state"][key], optimizer.state[parameter]
        need(set(original) == set(actual) == {"step", "exp_avg", "exp_avg_sq"} and
             original["step"].item() == 2 and all(
                 value.dtype == actual[field].dtype and torch.equal(value, actual[field].detach().cpu())
                 for field, value in original.items()), "EXACT_504_NAMED_RESTORED_ADAMW_STATES")
    plan = torch.load(contract.CHECKPOINT/"scheduler.pt", weights_only=True, map_location="cpu")
    current = trainer.lr_scheduler.state_dict()
    need(plan["last_epoch"] == 2 and plan["_step_count"] == 3 and
         plan["_last_lr"] == [contract.LR for _ in optimizer.param_groups] and
         all(current[key] == plan[key] for key in ("last_epoch", "_step_count", "_last_lr", "base_lrs")),
         "EXACT_CP2_SCHEDULE_CURSOR_AND_LR")
    return contract.digest(contract.encode([[key, name] for key, name, _ in mapped]))


class ContinueOnce(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.trainer = trainer
        self.rank = int(os.environ["RANK"])
        self.output = Path(os.environ["T03_FIRST2_CAPTURE"])/f"rank-{self.rank:02d}"
        self.epoch = float(os.environ["T03_FIRST2_EPOCH"])
        self.deadline = float(os.environ["T03_FIRST2_DEADLINE"])
        self.original_rng = trainer._load_rng_state
        self.original_prepare = trainer._prepare_inputs
        self.rng_calls = self.updates = self.saves = 0
        self.pending = None
        self.losses = []

        @functools.wraps(self.original_rng)
        def restore(checkpoint):
            self.clock()
            need(self.rng_calls == 0 and Path(checkpoint).resolve(strict=True) == contract.CHECKPOINT,
                 "EXACT_ONCE_NATIVE_CP2_RNG_LOAD")
            saved = load_rng(self.rank)
            result = self.original_rng(checkpoint)
            self.consensus(lambda: confirm_rng(saved, self.rank))
            self.rng_calls += 1
            return result

        trainer._load_rng_state = restore

    def clock(self):
        need(self.deadline == self.epoch + NATIVE and self.epoch <= time.time() < self.deadline,
             "NATIVE_CP3_OWNER_TIME_BUDGET")

    def consensus(self, function):
        value = failure = None
        try:
            value = function()
        except Exception as error:
            failure = error
        all_pass = [None] * 8
        torch.distributed.all_gather_object(all_pass, failure is None)
        if failure is not None:
            raise failure
        need(all(item is True for item in all_pass), "ANOTHER_NATIVE_RANK_VETOED_CP3")
        return value

    def on_train_begin(self, args, state, control, **kwargs):
        self.clock()
        need((state.global_step, state.max_steps, state.epoch) == (2, 25, 1.0) and
             args.max_steps == 25 and args.gradient_accumulation_steps == 4 and
             args.per_device_train_batch_size == 1 and args.data_seed == 42 and
             args.dataloader_num_workers == 4 and args.ignore_data_skip is False and args.resume_only_model is False,
             "EXACT_NATIVE_CP2_EPOCH_AND_RECIPE")
        need(torch.version.hip == "6.3.26113" and torch.cuda.device_count() == 8 and
             (self.rank, 8) == (torch.distributed.get_rank(), torch.distributed.get_world_size()) and
             self.rank == int(os.environ["LOCAL_RANK"]) == torch.cuda.current_device(), "ORIGINAL_HIP_EIGHT_RANKS")
        binding = json.loads(pin(Path(os.environ["T03_FIRST2_BINDING"]),
                                 expected=os.environ["T03_FIRST2_BINDING_SHA256"]))
        model = kwargs["model"]
        self.named, _ = validate_native_recipe(
            model, args, binding, checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        self.mapping = self.consensus(lambda: verify_resume(self.trainer, self.named))
        agreed = [None] * 8
        torch.distributed.all_gather_object(agreed, self.mapping)
        need(agreed == [self.mapping] * 8, "EIGHT_RANK_ADAMW_NAME_MAPPING_CONSENSUS")
        self.output.mkdir(mode=0o700)
        self.roster, matrix, order = verified_split()
        self.cursor = LiveCursor(self.rank, matrix[self.rank], order)
        self.consensus(lambda: self.cursor.attach(kwargs["train_dataloader"], pin, torch.distributed))
        pin(Path(inspect.getsourcefile(self.original_prepare)), expected=PREPARE_SHA, extent=12688)

        @functools.wraps(self.original_prepare)
        def prepare(inputs):
            self.clock()

            def observe():
                need(self.rng_calls == 1 and self.pending is None and len(self.losses) < 4 and self.updates == 0,
                     "EXACT_FOUR_CP3_PREPARES_AFTER_RNG_LOAD")
                source = before_sample(inputs, self.roster, lambda value: fingerprint(cpu(value)))
                return self.cursor.prepare(len(self.losses), source)

            self.pending = self.consensus(observe)
            return self.original_prepare(inputs)

        self.trainer._prepare_inputs = prepare
        self.forward = model.register_forward_pre_hook(self.before_forward, with_kwargs=True)
        self.forward_result = model.register_forward_hook(self.after_forward)
        write_once(self.output/"restore.json", dict(request=REQUEST, rank=self.rank, step=2,
                   adapter_sha256=contract.CP2_FP, restored_named_optimizer_states=504,
                   optimizer_index_to_name_SHA256=self.mapping))
        return control

    def before_forward(self, model, args, kwargs):
        self.clock()

        def observe():
            need(self.pending is not None and len(self.losses) < 4 and self.updates == 0,
                 "EXACT_FOUR_ACTUAL_CP3_FORWARDS")
            observed = after_sample(kwargs, self.pending, lambda value: fingerprint(cpu(value)))
            observed.update(logical=self.pending["logical"], physical=self.pending["physical"])
            self.cursor.forward(len(self.losses), observed)
            self.pending = None

        self.consensus(observe)

    def after_forward(self, model, inputs, output):
        def observe():
            loss = output["loss"]
            need(torch.is_tensor(loss) and loss.numel() == 1 and bool(torch.isfinite(loss).all()),
                 "ACTUAL_CP3_FINITE_LOSS")
            return float(loss.detach().item())

        self.losses.append(self.consensus(observe))
        write_once(self.output/f"forward-{len(self.losses)}.json",
                   dict(request=REQUEST,rank=self.rank,microbatch=len(self.losses),global_step_before=2))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.clock()

        def observe():
            need(state.global_step == 2 and self.updates == 0 and self.rng_calls == 1 and
                 self.pending is None and len(self.losses) == 4, "EXACT_CP3_ONE_UPDATE_BOUNDARY")
            self.cursor.confirmed(4)
            rates = [group["lr"] for group in self.trainer.optimizer.param_groups]
            need(rates and all(math.isfinite(lr) and lr == contract.LR for lr in rates),
                 "EXACT_POSITIVE_CP2_RESTORED_LR")
            self.rates = rates

        self.consensus(observe)
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        def observe():
            need(state.global_step == 2 and self.updates == 0 and
                 self.trainer.accelerator.optimizer_step_was_skipped is False, "EXACT_CP3_UPDATE_NOT_SKIPPED")
            optimizer = self.trainer.optimizer
            need(len(optimizer.state) == 504 and all(
                optimizer.state[parameter]["step"].item() == 3 for _, parameter in self.named),
                "EXACT_504_NATIVE_CP3_ADAMW_COUNTERS")
            final = parameters(self.named)
            need(final != contract.CP2_FP, "POSITIVE_NATIVE_CP3_PARAMETER_CHANGE")
            return final

        self.final = self.consensus(observe)
        self.updates += 1
        write_once(self.output/"update-3-observed.json",dict(request=REQUEST,rank=self.rank,
                   native_update_returned=True,global_step_before=2,AdamW_step=3,
                   final_parameter_SHA256=self.final))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.clock()
        schedule = self.trainer.lr_scheduler.state_dict()
        self.consensus(lambda: need(
            state.global_step == 3 and self.updates == 1 and schedule["last_epoch"] == 3 and
            schedule["_step_count"] == 4 and all(lr == contract.SCHEDULE_LR_AFTER for lr in schedule["_last_lr"]),
            "EXACT_CP3_SCHEDULE_AND_STOP"))
        control.should_save = True
        control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        need(state.global_step == 3 and self.saves == 0, "ONE_NATIVE_CP3_SAVE_EVENT")
        self.saves += 1
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.clock()
        need(state.global_step == 3 and self.updates == self.saves == self.rng_calls == 1,
             "EXACT_CP3_TERMINAL_CURSOR")
        self.forward.remove()
        self.forward_result.remove()
        self.cursor.detach()
        self.trainer._prepare_inputs = self.original_prepare
        self.trainer._load_rng_state = self.original_rng
        write_once(self.output/"resume3.json", dict(request=REQUEST, rank=self.rank, final_global_step=3,
                   updates=1, cp3_native_save_events=1, rng_restore_calls=1, actual_device_rng_restored=True,
                   optimizer_index_to_name_SHA256=self.mapping, initial_parameter_SHA256=contract.CP2_FP,
                   final_parameter_SHA256=self.final, learning_rates=self.rates,
                   losses=self.losses, rows=self.cursor.consumed))
        return control


need(CALLBACK not in callbacks_map, "UNIQUE_NATIVE_CP3_CALLBACK")
callbacks_map[CALLBACK] = ContinueOnce
