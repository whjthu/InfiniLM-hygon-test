"""Freeze the first two native Swift optimizer boundaries of the original MiniCPM recipe."""
from minicpm_usgs_witness import (
    TreasuryContinuous, callbacks_map, check_update, cpu, fingerprint,
    get_peft_model_state_dict, optimizer_fingerprint, require, torch,
)
from vlm.runtime.first_update_audit import parameter_delta
from scope_contract import audit_update, parameter_identity


class OffTwoUpdates(TreasuryContinuous):
    def on_train_begin(self, args, state, control, **kw):
        control = super().on_train_begin(args, state, control, **kw)
        self.named = {name: value for name, value in self.model.named_parameters() if value.requires_grad}
        self.exact_steps = []
        self.pending = None
        self.actual_optimizer = self.trainer.optimizer
        if type(self.actual_optimizer) is not torch.optim.AdamW:
            from accelerate.optimizer import AcceleratedOptimizer
            require(type(self.actual_optimizer) is AcceleratedOptimizer, "OFF_OPTIMIZER_WRAPPER")
            self.actual_optimizer = self.actual_optimizer.optimizer
        require(type(self.actual_optimizer) is torch.optim.AdamW, "OFF_NATIVE_ADAMW")
        group_ids = [id(p) for group in self.actual_optimizer.param_groups for p in group["params"]]
        require(len(group_ids) == len(self.named) and set(group_ids) == {id(p) for p in self.named.values()},
                "OFF_OPTIMIZER_PARAMETERS")
        self.step_hooks = [self.actual_optimizer.register_step_pre_hook(self.optimizer_before),
                           self.actual_optimizer.register_step_post_hook(self.optimizer_after)]
        return control

    def before(self, module, argv, kwargs):
        require(len(self.inputs) < 8, "OFF_FORWARD_BUDGET")
        super().before(module, argv, kwargs)
        require(self.inputs[-1]["tokens"]["shifted_supervised_tokens"] > 0, "OFF_SUPERVISED_TOKENS")

    def optimizer_before(self, optimizer, args, kwargs):
        self.check_time()
        step = len(self.exact_steps) + 1
        require(optimizer is self.actual_optimizer and self.pending is None and step in (1, 2)
                and self.trainer.state.global_step == step - 1
                and len(self.inputs) == len(self.outputs) == step * 4, "OFF_ADAMW_PRE_BOUNDARY")
        rates = [float(x["lr"]) for x in optimizer.param_groups]
        require(all(x["optimizer_learning_rates"] == rates for x in self.inputs[(step-1)*4:step*4]),
                "OFF_ADAMW_RATES_AT_STEP")
        before = cpu(self.named)
        fp = fingerprint(before)
        parameter_identity(fp)
        self.write(f"step-{step}-optimizer-intent.json", dict(step=step, lr=rates, before=fp))
        self.pending = (step, rates, before, fp)

    def optimizer_after(self, optimizer, args, kwargs):
        self.check_time()
        require(optimizer is self.actual_optimizer and self.pending is not None, "OFF_ADAMW_POST_BOUNDARY")
        step, rates, before, fp = self.pending
        after = cpu(self.named)
        evidence = dict(step=step, lr=rates, before=fp, after=fingerprint(after), delta=parameter_delta(before, after))
        self.write(f"step-{step}-optimizer-return.json", evidence)
        audit_update(evidence, step)
        self.exact_steps.append(evidence)
        self.pending = None

    def on_step_end(self, args, state, control, **kw):
        self.check_time()
        step = state.global_step
        require(step in (1, 2) and len(self.exact_steps) == step == len(self.steps) + 1 and self.pending is None
                and len(self.inputs) == len(self.outputs) == 4 * step, "OFF_NATIVE_STEP_END")
        evidence = self.exact_steps[-1]
        slots = self.actual_optimizer.state_dict()["state"]
        counters = [float(x["step"].item() if isinstance(x["step"], torch.Tensor) else x["step"]) for x in slots.values()]
        changed = evidence["before"]["SHA256"] != evidence["after"]["SHA256"]
        check_update(step, counters, self.expected_optimizer_parameters, changed, evidence["lr"])
        self.steps.append(dict(global_step=step, actual_optimizer=evidence, optimizer_slot_count=len(slots),
                               optimizer_step_min=min(counters), optimizer_step_max=max(counters),
                               native_optimizer=optimizer_fingerprint(self.actual_optimizer.state_dict())))
        self.progress()
        if step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kw):
        require(state.global_step == 2 and not self.saved, "OFF_CHECKPOINT_CADENCE")
        self.saved.append(2)
        self.progress()
        return control

    def on_train_end(self, args, state, control, **kw):
        require(state.global_step == 2 and self.saved == [2] and len(self.steps) == 2
                and len(self.inputs) == len(self.outputs) == 8, "OFF_INCOMPLETE_TWO_UPDATES")
        for hook in self.hooks + self.step_hooks:
            hook.remove()
        self.write("result.json", dict(request=self.config["request"], rank=self.rank, world_size=8,
                   inputs=self.inputs, outputs=self.outputs, steps=self.steps, saved_checkpoints=self.saved,
                   checkpoint_adapter=fingerprint(cpu(get_peft_model_state_dict(self.model))),
                   frozen=self.frozen, completed_global_step=2))
        return control

    def progress(self):
        temporary = self.root / "progress.tmp"
        import json
        temporary.write_text(json.dumps(dict(rank=self.rank, planned_updates=2, planned_microbatches=8,
            inputs=len(self.inputs), outputs=len(self.outputs), completed_steps=[r["global_step"] for r in self.steps],
            saved_checkpoints=self.saved), allow_nan=False), encoding="utf8")
        temporary.replace(self.root / "progress.json")


require("minicpm_off_two_updates" not in callbacks_map, "OFF_CALLBACK_DUPLICATE")
callbacks_map["minicpm_off_two_updates"] = OffTwoUpdates
