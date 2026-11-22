"""Capture Gemma's native continuous arm for a checkpoint20 restore comparison."""

import json
import hashlib
import math
import os
from pathlib import Path
import time

from peft import get_peft_model_state_dict
import torch
from swift.callbacks import TrainerCallback, callbacks_map

from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu, remaining
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.resume_cursor import read_pinned_json
from vlm.runtime.resume_witness import optimizer_fingerprint, rng_fingerprint, rng_values
from vlm.runtime.vlm_update_contract import check_complete, check_update


KEY = "gemma_domain_continuous_witness"
BINDING_SHA = "7bd0b8e46b164ef179ac59a2525322dfb50b6802e3a277c45113a4e9bc6cc8a1"
DATASET_SHA = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"
CHECKPOINTS = (10, 20, 25)


class GemmaFreshDomainContinuous(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        config = read_pinned_json(Path(os.environ["NATIVE_CONTINUOUS_CONFIG"]),
                                  os.environ["NATIVE_CONTINUOUS_CONFIG_SHA256"], maximum_bytes=65536)
        self.epoch, self.deadline, self.maximum = (
            float(os.environ[f"NATIVE_CONTINUOUS_{name}"]) for name in ("EPOCH", "DEADLINE", "MAX_SECONDS")
        )
        self.check_time()
        self.rank, local = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK"))
        if (not torch.version.hip or torch.version.cuda or torch.cuda.device_count() != 8
                or int(os.environ["WORLD_SIZE"]) != 8 or self.rank != local or not 0 <= local < 8
                or torch.cuda.current_device() != local):
            raise ValueError("Gemma continuous arm requires eight local HIP ranks")
        if (config["case"] != "T02.sft" or config["dataset_SHA256"] != DATASET_SHA
                or len(kwargs["train_dataloader"]) != 100 or state.global_step != 0
                or args.max_steps != 25 or args.save_steps != 10 or args.resume_from_checkpoint):
            raise ValueError("Gemma continuous arm requires a fresh 25-step 100-batch run")
        forbidden = {"BaselineCallback", "FirstUpdates", "NativeDropoutAlignment", "GemmaDomainResume"}
        callbacks = self.trainer.callback_handler.callbacks
        if sum(c is self for c in callbacks) != 1 or any(type(c).__name__ in forbidden for c in callbacks):
            raise ValueError("conflicting callback during Gemma domain training")
        binding = read_pinned_json(Path(config["binding"]), BINDING_SHA, maximum_bytes=106000)
        self.model = kwargs.get("model", self.trainer.model)
        trainables, frozen = validate_native_recipe(self.model, args, binding)
        if len(trainables) != 476 or any(p.device.type != "cuda" or p.device.index != local for _, p in trainables):
            raise ValueError("native Gemma trainables differ from the assigned device or original scope")
        self.root = Path(args.output_dir) / "continuous-witness" / f"rank-{self.rank:02d}"
        self.root.mkdir(mode=0o700, parents=True)
        self.inputs, self.outputs, self.steps, self.saved = [], [], [], []
        proof = self.capture_fresh_initial(config, trainables, local)
        self.write("initialization.json", proof)
        startticks = int(Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19])
        self.write("owner.json", dict(PID=os.getpid(), PGID=os.getpgrp(), rank=self.rank, local_rank=local,
                                     startticks=startticks, epoch=self.epoch, deadline=self.deadline))
        self.write("frozen-boundaries.json", frozen)
        self.previous_adapter = self.adapter()
        self.progress()
        self.hooks = [self.model.register_forward_pre_hook(self.before, with_kwargs=True),
                      self.model.register_forward_hook(self.after, with_kwargs=True)]
        return control

    def capture_fresh_initial(self, config, trainables, local):
        fresh_config(config)
        if self.trainer.optimizer.state_dict()["state"]:
            raise ValueError("fresh domain optimizer already has history")
        if self.trainer.args.seed != 42 or self.trainer.args.data_seed != 42:
            raise ValueError("fresh target seed differs")
        before_rng = rng_values(lambda: [torch.cuda.get_rng_state(local)])
        before = rng_fingerprint(before_rng)
        parameters = cpu({name: value for name, value in trainables})
        named = fingerprint(parameters)
        records = [None] * 8
        torch.distributed.all_gather_object(records, dict(rank=self.rank, named=named))
        actual_named_sha = cohort(records)
        if self.rank == 0:
            with (self.root / "target-fresh-initial.pt").open("xb") as stream:
                torch.save(dict(schema="GEMMA_TARGET_FRESH_INITIAL_V1", seed=42,
                                named_SHA256=actual_named_sha, parameters=parameters), stream)
        artifacts = {}
        numpy_state = before_rng["numpy"]
        safe_rng = dict(python=before_rng["python"], cpu=before_rng["cpu"],
                        cuda_local=before_rng["cuda"][0], local_rank=local,
                        numpy=dict(bit_generator=numpy_state[0], keys=numpy_state[1].tolist(),
                                   position=numpy_state[2], has_gauss=numpy_state[3],
                                   cached_gaussian=numpy_state[4]))
        with (self.root / "target-fresh-rng.pt").open("xb") as stream:
            torch.save(safe_rng, stream)
        names = ["target-fresh-rng.pt"] + (["target-fresh-initial.pt"] if self.rank == 0 else [])
        for name in names:
            digest = hashlib.sha256()
            path = self.root / name
            with path.open("rb") as stream:
                for raw in iter(lambda: stream.read(1024**2), b""):
                    self.check_time()
                    digest.update(raw)
            artifacts[name] = dict(bytes=path.stat().st_size, SHA256=digest.hexdigest())
        after = rng_fingerprint(rng_values(lambda: [torch.cuda.get_rng_state(local)]))
        if before != after:
            raise ValueError("initial observation changed rank RNG")
        self.write("fresh-initial-cohort.json", dict(records=records, named_SHA256=actual_named_sha))
        return dict(mode="TARGET_FRESH_DOMAIN_ONLY", seed=42, data_seed=42, rank=self.rank,
                    named=named, RNG_local=before, RNG_unchanged=True,
                    artifacts=artifacts, host=os.uname().nodename,
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    target_created=True, reference_initial_read=False,
                    original_A100_value_equivalence=False)

    def check_time(self):
        remaining(self.epoch, self.deadline, time.time(), self.maximum)

    def write(self, name, data):
        with (self.root / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, allow_nan=False)

    def progress(self):
        temporary = self.root / "progress.tmp"
        temporary.write_text(json.dumps(dict(rank=self.rank, planned_updates=25, planned_microbatches=100,
            inputs=len(self.inputs), outputs=len(self.outputs), recorded_updates=len(self.steps),
            saved_checkpoints=self.saved)), encoding="utf-8")
        temporary.replace(self.root / "progress.json")

    def adapter(self):
        return fingerprint(cpu(get_peft_model_state_dict(self.model)))

    def before(self, module, argv, kwargs):
        self.check_time()
        index = len(self.inputs)
        step = self.trainer.state.global_step
        if index != len(self.outputs) or index >= 100 or step != index // 4:
            raise ValueError("continuous Gemma input is outside its accumulation group")
        self.inputs.append(dict(ordinal=index + 1, global_step=step,
            full_input=fingerprint(cpu(dict(args=argv, kwargs=kwargs))),
            optimizer_learning_rates=[float(group["lr"]) for group in self.trainer.optimizer.param_groups]))
        self.progress()

    def after(self, module, argv, kwargs, output):
        self.check_time()
        loss = output.get("loss") if isinstance(output, dict) else getattr(output, "loss", None)
        if len(self.inputs) != len(self.outputs) + 1 or not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("missing scalar loss from the expected Gemma forward")
        number = float(loss.detach().item())
        if not math.isfinite(number):
            raise ValueError("nonfinite Gemma loss")
        self.outputs.append(dict(ordinal=len(self.outputs) + 1, loss=fingerprint(cpu(loss)), loss_scalar=number))
        self.progress()

    def on_step_end(self, args, state, control, **kwargs):
        self.check_time()
        step = state.global_step
        if step != len(self.steps) + 1 or len(self.inputs) != len(self.outputs) or len(self.inputs) != step * 4:
            raise ValueError("Gemma native update was outside the declared accumulation schedule")
        adaptation = self.adapter()
        optimizer = self.trainer.optimizer.state_dict()
        counters = [float(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
                    for value in optimizer["state"].values()]
        group = self.inputs[-4:]
        rates = group[0]["optimizer_learning_rates"]
        if any(sample["optimizer_learning_rates"] != rates for sample in group):
            raise ValueError("learning rate changed during one Gemma accumulation group")
        update = check_update(step, counters, 476, adaptation != self.previous_adapter, rates)
        if step >= 20:
            update["state"] = dict(adapter=adaptation, optimizer=optimizer_fingerprint(optimizer),
                                   scheduler=fingerprint(cpu(self.trainer.lr_scheduler.state_dict())))
        self.steps.append(update)
        self.previous_adapter = adaptation
        self.progress()
        if step == 25:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.check_time()
        if len(self.saved) >= len(CHECKPOINTS) or state.global_step != CHECKPOINTS[len(self.saved)]:
            raise ValueError("unexpected Gemma continuous checkpoint")
        self.write(f"checkpoint-{state.global_step}-rng.json",
                   dict(rank=self.rank, RNG=rng_fingerprint(rng_values(torch.cuda.get_rng_state_all))))
        self.saved.append(state.global_step)
        self.progress()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.check_time()
        if state.global_step != 25 or tuple(self.saved) != CHECKPOINTS:
            raise ValueError("incomplete Gemma continuous run")
        changes = check_complete([row["adapter_changed"] for row in self.steps])
        for hook in self.hooks:
            hook.remove()
        self.write("result.json", dict(rank=self.rank, world_size=8, completed_global_step=25, inputs=self.inputs,
            outputs=self.outputs, steps=self.steps, saved_checkpoints=self.saved, parameter_change_window=changes))
        return control


if KEY in callbacks_map:
    raise ValueError("duplicate Gemma continuous callback")
callbacks_map[KEY] = GemmaFreshDomainContinuous

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def fresh_config(config):
    need(config.get("initialization_mode") == "TARGET_FRESH_DOMAIN_ONLY"
         and config.get("seed") == config.get("data_seed") == 42
         and "initial_trainables" not in config,
         "fresh domain initialization must not consume a reference initial snapshot")

def cohort(records):
    need(len(records) == 8 and [r["rank"] for r in records] == list(range(8)), "eight ordered rank records required")
    for r in records:
        f = r["named"]
        need(f.get("finite") is True and f.get("tensors") == 476
             and f.get("elements") == 29802496 and isinstance(f.get("SHA256"), str)
             and len(f["SHA256"]) == 64, "incomplete actual fresh trainable fingerprint")
    need(len({r["named"]["SHA256"] for r in records}) == 1,
         "actual fresh initial trainables differ across ranks")
    return records[0]["named"]["SHA256"]
