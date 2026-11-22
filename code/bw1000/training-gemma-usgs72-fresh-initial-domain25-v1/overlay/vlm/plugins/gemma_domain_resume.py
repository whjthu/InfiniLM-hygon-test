"""Observe the native Gemma checkpoint20 restore without replacing Trainer state."""

import hashlib
import math
import os
from pathlib import Path
import signal
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map
import transformers.trainer as trainer_module

from vlm.runtime.first_update_capture import remaining
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.resume_cursor import read_pinned_json
from vlm.runtime.resume_witness import ResumeWitness, rng_fingerprint, rng_values


KEY = "gemma_domain_resume_witness"
BINDING_SHA256 = "7bd0b8e46b164ef179ac59a2525322dfb50b6802e3a277c45113a4e9bc6cc8a1"
VIEW_SHA256 = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"


class GemmaResumeWitness(ResumeWitness):
    def after(self, module, argv, kwargs, output):
        super().after(module, argv, kwargs, output)
        loss = output.get("loss") if isinstance(output, dict) else output.loss
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise ValueError("nonfinite resumed Gemma loss")
        self.outputs[-1]["loss_scalar"] = value


def observe_native_skip(module, witness, rank):
    original = module.skip_first_batches
    seen = []

    def observed(loader, count):
        witness.check()
        if seen or len(loader) != 100 or type(count) is not int or count != 80:
            raise ValueError("native Trainer used a different Gemma loader cursor")
        tail = original(loader, count)
        if len(tail) != 20:
            raise ValueError("native Trainer returned a different Gemma tail")
        witness.write("native-data-skip.json", dict(rank=rank, before=100, skipped=80, after=20))
        seen.append(True)
        return tail

    module.skip_first_batches = observed

    def finish():
        if module.skip_first_batches is not observed:
            raise ValueError("native Trainer skip changed while being observed")
        module.skip_first_batches = original
        if not seen:
            raise ValueError("native Trainer skip was not observed")

    return seen, finish


class GemmaDomainResume(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        config = read_pinned_json(Path(os.environ["NATIVE_RESUME_CONFIG"]),
                                  os.environ["NATIVE_RESUME_CONFIG_SHA256"], maximum_bytes=65536)
        rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        loader = kwargs["train_dataloader"]
        if (config["case"] != "T02.sft" or config["dataset_SHA256"] != VIEW_SHA256
                or config["checkpoint_step"] != 20 or args.max_steps != 25
                or state.global_step != 20 or len(loader) != 100):
            raise ValueError("Gemma restore requires step 20 and the actual 100-batch rank loader")
        if (not torch.version.hip or torch.version.cuda or torch.cuda.device_count() != 8
                or int(os.environ["WORLD_SIZE"]) != 8 or rank != local or not 0 <= rank < 8
                or torch.cuda.current_device() != local):
            raise ValueError("Gemma restore requires original single-node eight-rank HIP topology")
        fresh = {"BaselineCallback", "FirstUpdates", "NativeDropoutAlignment"}
        if any(type(callback).__name__ in fresh for callback in self.trainer.callback_handler.callbacks):
            raise ValueError("fresh-run callback cannot run during restoration")
        trainer_source = Path(trainer_module.__file__).read_bytes()
        if hashlib.sha256(trainer_source).hexdigest() != config["trainer_SHA256"]:
            raise ValueError("Gemma restore received a different target Trainer source")
        binding = read_pinned_json(Path(config["binding"]), BINDING_SHA256, maximum_bytes=106000)
        model = kwargs.get("model", self.trainer.model)
        trainables, frozen = validate_native_recipe(model, args, binding)
        if any(value.device.type != "cuda" or value.device.index != local for _, value in trainables):
            raise ValueError("restored Gemma adapter was not on the assigned device")
        epoch, deadline = float(os.environ["NATIVE_RESUME_EPOCH"]), float(os.environ["NATIVE_RESUME_DEADLINE"])
        maximum = float(os.environ["NATIVE_RESUME_MAX_SECONDS"])
        self.witness = GemmaResumeWitness(
            self.trainer, model, loader, Path(args.output_dir) / "resume-witness" / f"rank-{rank:02d}",
            config["checkpoint"], config["checkpoint_files"], rank=rank, world_size=8,
            checkpoint_step=20, max_steps=25, epoch=epoch, deadline=deadline, max_seconds=maximum,
            device_rng=torch.cuda.get_rng_state_all, allow_mid_epoch=True,
        )
        if self.witness.plan["expected_batches_to_skip_in_resumed_epoch"] != 80:
            raise ValueError("Gemma checkpoint state produced a different planned cursor")
        self.skip_seen, self.finish_skip = observe_native_skip(trainer_module, self.witness, rank)
        self.witness.write("frozen-boundaries.json", frozen)
        startticks = int(Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19])
        self.witness.write("owner.json", dict(PID=os.getpid(), PGID=os.getpgrp(), rank=rank, local_rank=local,
                                               startticks=startticks, epoch=epoch, deadline=deadline))

        def expired(*_):
            error = TimeoutError("Gemma native resume deadline exhausted")
            self.witness.fail(error)
            raise error

        budget = remaining(epoch, deadline, time.time(), maximum)
        previous = signal.getitimer(signal.ITIMER_REAL)[0]
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(previous, budget) if previous else budget)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.skip_seen:
            raise ValueError("Gemma update occurred before the native data skip was observed")
        self.witness.guarded(self.witness.step_end)()
        if state.global_step == 25:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.witness.guarded(self.witness.checkpoint_saved)()
        self.witness.write(f"checkpoint-{state.global_step}-rng.json",
                           dict(rank=self.witness.rank, RNG=rng_fingerprint(rng_values(torch.cuda.get_rng_state_all))))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.witness.guarded(self.finish_skip)()
        self.witness.guarded(self.witness.finish)()
        return control


if KEY in callbacks_map:
    raise ValueError("duplicate Gemma native resume callback")
callbacks_map[KEY] = GemmaDomainResume
