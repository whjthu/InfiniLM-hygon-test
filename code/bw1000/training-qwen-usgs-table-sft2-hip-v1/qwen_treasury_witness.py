"""Observe two native Qwen updates and their step-2 checkpoint."""

import hashlib
import json
import math
import os
from pathlib import Path
import time

from peft import get_peft_model_state_dict
import torch
from swift.callbacks import TrainerCallback, callbacks_map

from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu, remaining
from vlm.runtime.initial_trainables import restore_initial_trainables
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.resume_witness import optimizer_fingerprint, rng_fingerprint, rng_values


ORIGINAL_BINDING_SHA256 = "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
TRAINER_SHA256 = "c1a56423fcfcf9cfec6847467ffb2e2c8a9a9e8cc1836b82c87ed0c81e504be0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_record(path, limit, digest):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit and hashlib.sha256(raw).hexdigest() == digest, "binding identity differs")
    return json.loads(raw)


def scalar_loss(output):
    loss = output.get("loss") if isinstance(output, dict) else getattr(output, "loss", None)
    require(isinstance(loss, torch.Tensor) and loss.numel() == 1, "missing scalar native loss")
    value = float(loss.detach().item())
    require(math.isfinite(value), "nonfinite native loss")
    return loss, value


class TreasuryBase(TrainerCallback):
    def setup(self, args, state, kw, start):
        self.config = read_record(os.environ["QWEN_TREASURY_CONFIG"], 65536,
                                  os.environ["QWEN_TREASURY_CONFIG_SHA256"])
        self.epoch, self.deadline, self.maximum = (
            float(os.environ["QWEN_TREASURY_" + key]) for key in ("EPOCH", "DEADLINE", "MAX_SECONDS"))
        self.check_time()
        self.rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        require(torch.version.hip and torch.cuda.device_count() == 8 and int(os.environ["WORLD_SIZE"]) == 8
                and self.rank == local == torch.cuda.current_device() and 0 <= local < 8,
                "native eight-rank device topology differs")
        require(state.global_step == start and args.max_steps == 2 and args.save_steps == 2
                and len(kw["train_dataloader"]) == 11, "native step, checkpoint or loader differs")
        binding = read_record(self.config["original_binding"], 2 * 1024 * 1024, ORIGINAL_BINDING_SHA256)
        require(binding["spec"]["case"] == "T01.sft" and binding["spec"]["training"]["max_steps"] == 25
                and binding["spec"]["training"]["save_steps"] == 10,
                "unexpected T01 native binding")
        binding["spec"]["training"].update(max_steps=args.max_steps, save_steps=args.save_steps)
        binding["spec"]["effective_warmup_steps"] = math.ceil(args.max_steps * binding["spec"]["training"]["warmup_ratio"])
        self.model = kw.get("model", self.trainer.model)
        named, self.frozen = validate_native_recipe(self.model, args, binding,
            checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        require(all(p.device.type == "cuda" and p.device.index == local for _, p in named),
                "trainables do not belong to the native rank device")
        self.root = Path(args.output_dir) / "treasury-witness" / f"rank-{self.rank:02d}"
        return binding, named

    def check_time(self):
        remaining(self.epoch, self.deadline, time.time(), self.maximum)


class TreasuryContinuous(TreasuryBase):
    def write(self, name, payload):
        with (self.root / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)

    def on_train_begin(self, args, state, control, **kw):
        binding, named = self.setup(args, state, kw, 0)
        callbacks = self.trainer.callback_handler.callbacks
        require(sum(cb is self for cb in callbacks) == 1
                and all(type(cb).__name__ != "BaselineCallback" for cb in callbacks),
                "unexpected original benchmark callback in the two-step domain arm")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        with (self.root / "initialization.json").open("x", encoding="utf-8") as stream:
            initial = binding["initial_state"]
            proof = restore_initial_trainables(named, self.trainer.optimizer, global_step=state.global_step,
                path=initial["path"], file_bytes=initial["bytes"], file_SHA256=initial["SHA256"],
                named_SHA256=binding["initial_trainables_SHA256"])
            json.dump(proof, stream)
        self.inputs, self.outputs, self.steps, self.saved = [], [], [], []
        self.hooks = [self.model.register_forward_pre_hook(self.before, with_kwargs=True),
                      self.model.register_forward_hook(self.after, with_kwargs=True)]
        return control

    def before(self, module, argv, kwargs):
        self.check_time()
        ordinal = len(self.inputs)
        step = self.trainer.state.global_step
        require(ordinal == len(self.outputs) and ordinal < 8 and step == ordinal // 4,
                "continuous forward is outside the declared accumulation window")
        self.inputs.append(dict(ordinal=ordinal + 1, global_step=step,
                                full_input=fingerprint(cpu(dict(args=argv, kwargs=kwargs)))))

    def after(self, module, argv, kwargs, output):
        require(len(self.inputs) == len(self.outputs) + 1, "missing continuous forward")
        loss, value = scalar_loss(output)
        self.outputs.append(dict(ordinal=len(self.outputs) + 1, loss=fingerprint(cpu(loss)), loss_scalar=value))

    def on_step_end(self, args, state, control, **kw):
        self.check_time()
        step = state.global_step
        require(step == len(self.steps) + 1 and len(self.inputs) == len(self.outputs) == step * 4,
                "continuous optimizer step is outside the declared window")
        values = dict(adapter=fingerprint(cpu(get_peft_model_state_dict(self.model))),
                      optimizer=optimizer_fingerprint(self.trainer.optimizer.state_dict()),
                      scheduler=fingerprint(cpu(self.trainer.lr_scheduler.state_dict())))
        self.steps.append(dict(global_step=step, state=values))
        if step == 2:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kw):
        require(not self.saved and state.global_step == 2,
                "unexpected continuous checkpoint callback")
        self.saved.append(state.global_step)
        if state.global_step == 2:
            self.write("checkpoint-2-rng.json", dict(rank=self.rank,
                RNG=rng_fingerprint(rng_values(torch.cuda.get_rng_state_all))))
        return control

    def on_train_end(self, args, state, control, **kw):
        require(state.global_step == 2 and len(self.steps) == 2 and self.saved == [2],
                "incomplete continuous training")
        for hook in self.hooks:
            hook.remove()
        self.write("result.json", dict(rank=self.rank, world_size=8, inputs=self.inputs, outputs=self.outputs,
                   steps=self.steps, saved_checkpoints=self.saved, frozen=self.frozen, completed_global_step=2))
        return control


require("qwen_treasury_continuous_witness" not in callbacks_map, "duplicate native training callback")
callbacks_map["qwen_treasury_continuous_witness"] = TreasuryContinuous
