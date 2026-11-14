"""Observe native Trainer full-state recovery; never load state into a model."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
from peft import get_peft_model_state_dict
from safetensors.torch import load as load_safetensors
import torch
from transformers.trainer import safe_globals

from .first_update_audit import fingerprint
from .first_update_capture import cpu, remaining
from .resume_cursor import plan_epoch_boundary_resume


def rng_values(device_rng=None):
    value = dict(python=random.getstate(), numpy=np.random.get_state(),
                 cpu=torch.random.get_rng_state())
    if device_rng is not None:
        value['cuda'] = device_rng()
    return value


def rng_fingerprint(value):
    value = dict(value)
    state = value['numpy']
    value['numpy'] = dict(algorithm=state[0], keys=state[1].tolist(),
                         position=int(state[2]), has_gauss=int(state[3]),
                         cached_gaussian=float(state[4]))
    return fingerprint(cpu(value))


def optimizer_fingerprint(value):
    state = value['state']
    if any(type(key) is not int or key < 0 for key in state):
        raise ValueError('optimizer state must use nonnegative serialized parameter IDs')
    groups = value['param_groups']
    ids = [key for group in groups for key in group['params']]
    if len(set(ids)) != len(ids) or set(ids) != set(state):
        raise ValueError('optimizer state does not cover each parameter exactly once')
    return fingerprint(cpu(dict(state={str(k): state[k] for k in sorted(state)},
                                param_groups=groups)))


def pinned_checkpoint(root, pins, world_size):
    """Read immutable, bounded original artifacts on the same host only."""
    names = {'trainer_state.json', 'args.json', 'adapter_model.safetensors',
             'optimizer.pt', 'scheduler.pt', 'adapter_config.json'}
    names.update({'rng_state.pth'} if world_size == 1 else
                 {'rng_state_%d.pth' % rank for rank in range(world_size)})
    if set(pins) != names:
        raise ValueError('checkpoint pins must cover exactly the complete resume state')
    raw = {}
    total = 0
    for name in sorted(names):
        pin = pins[name]
        if (type(pin['bytes']) is not int or not 0 < pin['bytes'] <= 256 * 1024**2
                or len(pin['SHA256']) != 64):
            raise ValueError('invalid checkpoint file pin')
        path = Path(root) / name
        if path.is_symlink() or path.stat().st_size != pin['bytes']:
            raise ValueError('checkpoint path or byte length changed')
        with path.open('rb') as stream:
            value = stream.read(pin['bytes'] + 1)
        total += len(value)
        if (len(value) != pin['bytes'] or hashlib.sha256(value).hexdigest() != pin['SHA256']
                or total > 512 * 1024**2):
            raise ValueError('checkpoint identity or total byte bound changed')
        raw[name] = value
    def load(name):
        with safe_globals():
            return torch.load(io.BytesIO(raw[name]), map_location='cpu', weights_only=True)
    result = dict(state=json.loads(raw['trainer_state.json']),
                  args=json.loads(raw['args.json']),
                  adapter=fingerprint(load_safetensors(raw['adapter_model.safetensors'])),
                  optimizer=optimizer_fingerprint(load('optimizer.pt')),
                  scheduler=fingerprint(cpu(load('scheduler.pt'))), RNG={})
    for rank in range(world_size):
        name = 'rng_state.pth' if world_size == 1 else 'rng_state_%d.pth' % rank
        result['RNG'][rank] = rng_fingerprint(load(name))
    return result


class ResumeWitness:
    """Attach after native adapter/optimizer/scheduler restoration.

    The native RNG loader is observed at its own call boundary. It is invoked
    exactly as before; hashes/copies do not set RNG, parameters or optimizer state.
    These diagnostics are deliberately excluded from performance measurement.
    """
    def __init__(self, trainer, model, loader, root, checkpoint, pins, *,
                 rank, world_size, checkpoint_step, max_steps, epoch, deadline,
                 max_seconds, device_rng=None, entry_state_step=None, diagnostic_stop_step=None,
                 boundary_modules=None):
        if (entry_state_step is not None and (type(entry_state_step) is not int
                or not checkpoint_step <= entry_state_step < max_steps)):
            raise ValueError('entry-state step must be inside the declared resume window')
        if (diagnostic_stop_step is not None and (type(diagnostic_stop_step) is not int
                or entry_state_step is None or diagnostic_stop_step != entry_state_step + 1
                or not checkpoint_step < diagnostic_stop_step < max_steps)):
            raise ValueError('diagnostic stop must follow the observed entry by one full update')
        self.stop_step = max_steps if diagnostic_stop_step is None else diagnostic_stop_step
        self.diagnostic_stop_step = diagnostic_stop_step
        self.diagnostic_stop_requested = False
        self.entry_state_SHA256 = None
        self.entry_state_step, self.entry_state_captured = entry_state_step, False
        if boundary_modules is not None and (entry_state_step != 20 or diagnostic_stop_step != 21):
            raise ValueError('boundary trace requires checkpoint20 first-tail observation')
        self.boundary_modules, self.boundary_trace = boundary_modules, None
        self.boundary_trace_SHA256 = None
        self.trainer, self.model = trainer, model
        self.loader = loader
        self.root, self.checkpoint = Path(root), Path(checkpoint)
        self.rank, self.world_size = rank, world_size
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError('invalid resume rank')
        self.epoch, self.deadline, self.max_seconds = epoch, deadline, max_seconds
        self.anchor = time.monotonic()
        self.initial_remaining = remaining(epoch, deadline, time.time(), max_seconds)
        self.device_rng = device_rng
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.inputs, self.outputs, self.steps, self.rng_loads, self.saved = [], [], [], [], []
        self.failed = False
        self.handles = []
        self.original_rng_loader = None
        self.checkpoint_step, self.max_steps = checkpoint_step, max_steps
        self.progress()
        try:
            self.expected = pinned_checkpoint(checkpoint, pins, world_size)
            args = trainer.args
            self.plan = plan_epoch_boundary_resume(
                vars(trainer.state), self.expected['args'], checkpoint_step=checkpoint_step,
                max_steps=max_steps, world_size=world_size,
                per_device_batch=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                dataloader_batches_per_rank=len(loader))
            if (self.expected['state']['global_step'] != checkpoint_step
                    or Path(args.resume_from_checkpoint).resolve() != self.checkpoint.resolve()
                    or args.resume_only_model is not False or args.ignore_data_skip is not False
                    or args.max_steps != max_steps):
                raise ValueError('native restored checkpoint, flags or endpoint differ')
            self.accumulation = args.gradient_accumulation_steps
            self.planned_inputs = (self.stop_step - checkpoint_step) * self.accumulation
            self.plan.update(diagnostic_stop_global_step=diagnostic_stop_step,
                             declared_observation_updates=self.stop_step - checkpoint_step,
                             declared_observation_micro_forwards=self.planned_inputs)
            interval = getattr(args, 'save_steps', max_steps)
            if type(interval) is not int or interval <= 0:
                raise ValueError('resume witness requires a fixed positive save interval')
            required_final = {max_steps} if diagnostic_stop_step is None else set()
            self.expected_saved = sorted(required_final | {
                step for step in range(checkpoint_step + 1, self.stop_step + 1) if step % interval == 0})
            self.plan['expected_saved_checkpoints'] = self.expected_saved
            actual = self.training_state()
            if any(actual[key] != self.expected[key] for key in ('adapter', 'optimizer', 'scheduler')):
                raise ValueError('native restored adapter, optimizer or scheduler values differ')
            if any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                   for p in model.parameters() if p.requires_grad):
                raise ValueError('resume must begin at an empty accumulation boundary')
            self.write('restored-before-RNG.json', dict(actual=actual, plan=self.plan,
                       restored_parameter_optimizer_scheduler_equal=True))
            self.original_rng_loader = trainer._load_rng_state
            def observed(checkpoint_path):
                try:
                    self.check()
                    result = self.original_rng_loader(checkpoint_path)
                    self.after_rng_load(checkpoint_path)
                    return result
                except BaseException as error:
                    self.fail(error)
                    raise
            self.observed_rng_loader = observed
            trainer._load_rng_state = observed
            self.handles = [model.register_forward_pre_hook(self.guarded(self.before), with_kwargs=True),
                            model.register_forward_hook(self.guarded(self.after), with_kwargs=True)]
        except BaseException as error:
            self.fail(error)
            raise

    def check(self):
        remaining(self.epoch, self.deadline, time.time(), self.max_seconds)
        if time.monotonic() - self.anchor >= self.initial_remaining:
            raise TimeoutError('resume witness monotonic budget exhausted')

    def write(self, name, value):
        with (self.root / name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2)

    def progress(self):
        row = dict(rank=self.rank, restored_from_step=self.checkpoint_step,
                   planned_endpoint=self.max_steps, inputs=len(self.inputs), outputs=len(self.outputs),
                   completed_steps=[r['global_step'] for r in self.steps],
                   RNG_loads=len(self.rng_loads), saved_checkpoints=self.saved, failed=self.failed)
        row['entry_state_captured'] = self.entry_state_captured
        row.update(planned_observation_endpoint=self.stop_step,
                   diagnostic_stop_requested=self.diagnostic_stop_requested)
        temp = self.root / 'progress.tmp'
        temp.write_text(json.dumps(row), encoding='utf8')
        temp.replace(self.root / 'progress.json')

    def training_state(self):
        return dict(adapter=fingerprint(cpu(get_peft_model_state_dict(self.model))),
                    optimizer=optimizer_fingerprint(self.trainer.optimizer.state_dict()),
                    scheduler=fingerprint(cpu(self.trainer.lr_scheduler.state_dict())))

    def after_rng_load(self, checkpoint_path):
        if (Path(checkpoint_path).resolve() != self.checkpoint.resolve()
                or self.rng_loads or self.inputs or self.trainer.state.global_step != self.checkpoint_step):
            raise ValueError('unexpected native RNG restoration boundary')
        actual = rng_fingerprint(rng_values(self.device_rng))
        if actual != self.expected['RNG'][self.rank]:
            raise ValueError('native restored per-rank RNG differs from saved state')
        self.rng_loads.append(dict(global_step=self.checkpoint_step, actual=actual,
                                   saved_equal=True))
        self.progress()

    def before(self, module, argv, kwargs):
        ordinal = len(self.inputs)
        if (len(self.rng_loads) != 1 or ordinal != len(self.outputs)
                or ordinal >= self.planned_inputs
                or self.trainer.state.global_step != self.checkpoint_step + ordinal // self.accumulation):
            raise ValueError('unexpected resumed input or accumulation boundary')
        sampler = getattr(self.loader, 'sampler', None)
        batch_sampler = getattr(self.loader, 'batch_sampler', None)
        def scalar(value):
            return value if type(value) in (int, float, bool) else None
        cursor = dict(trainer_epoch=scalar(getattr(self.trainer.state, 'epoch', None)),
                      loader_iteration=scalar(getattr(self.loader, 'iteration', None)),
                      sampler_epoch=scalar(getattr(sampler, 'epoch', None)),
                      batch_sampler_epoch=scalar(getattr(getattr(batch_sampler, 'sampler', None), 'epoch', None)),
                      micro_in_update=ordinal % self.accumulation + 1,
                      actual_accumulation_group_size=scalar(getattr(
                          self.trainer, 'current_gradient_accumulation_steps', None)))
        if self.entry_state_step == self.trainer.state.global_step and not self.entry_state_captured:
            if ordinal % self.accumulation:
                raise ValueError('entry-state capture must precede the first accumulation microstep')
            from .forward_entry_state import capture_entry_state
            entry = capture_entry_state(self.model, argv, kwargs, check=self.check,
                rng_snapshot=lambda: rng_fingerprint(rng_values(self.device_rng)),
                device_type='cuda' if self.device_rng is not None else 'cpu')
            entry.update(global_step=self.trainer.state.global_step, forward_ordinal=ordinal + 1,
                         runtime_cursor=cursor,
                         loader_rng_types=[str(x) for x in (getattr(self.loader, 'rng_types', None) or [])],
                         accelerator_rng_types=[str(x) for x in (getattr(
                             getattr(self.trainer, 'accelerator', None), 'rng_types', None) or [])])
            self.write('forward-entry-state.json', entry)
            self.entry_state_SHA256 = hashlib.sha256((self.root / 'forward-entry-state.json').read_bytes()).hexdigest()
            self.entry_state_captured = True
            if self.boundary_modules is not None:
                from .forward_boundary_trace import ForwardBoundaryTrace
                def boundary_progress(value):
                    temp = self.root / 'forward-boundaries-progress.tmp'
                    temp.write_text(json.dumps(value), encoding='utf8')
                    temp.replace(self.root / 'forward-boundaries-progress.json')
                self.boundary_trace = ForwardBoundaryTrace(self.boundary_modules, check=self.check,
                    rng_snapshot=lambda: rng_fingerprint(rng_values(self.device_rng)),
                    progress=boundary_progress)
        self.inputs.append(dict(ordinal=ordinal + 1, global_step=self.trainer.state.global_step,
                                runtime_cursor=cursor,
                                full_input=fingerprint(cpu(dict(args=argv, kwargs=kwargs)))))
        self.progress()

    def after(self, module, argv, kwargs, output):
        loss = getattr(output, 'loss', None)
        if loss is None and isinstance(output, dict):
            loss = output.get('loss')
        if len(self.inputs) != len(self.outputs) + 1 or not isinstance(loss, torch.Tensor):
            raise ValueError('missing resumed forward or loss')
        if self.boundary_trace is not None and self.boundary_trace_SHA256 is None:
            boundaries = self.boundary_trace.finish()
            boundaries.update(global_step=self.trainer.state.global_step, forward_ordinal=len(self.inputs))
            self.write('forward-boundaries.json', boundaries)
            self.boundary_trace_SHA256 = hashlib.sha256((self.root / 'forward-boundaries.json').read_bytes()).hexdigest()
        self.outputs.append(dict(ordinal=len(self.outputs) + 1, loss=fingerprint(cpu(loss))))
        self.progress()

    def step_end(self):
        step = self.trainer.state.global_step
        if (step != self.checkpoint_step + len(self.steps) + 1
                or len(self.inputs) != (step - self.checkpoint_step) * self.accumulation
                or len(self.outputs) != len(self.inputs)):
            raise ValueError('unexpected resumed optimizer-step boundary')
        self.steps.append(dict(global_step=step, state=self.training_state()))
        self.progress()

    def checkpoint_saved(self):
        step = self.trainer.state.global_step
        if (len(self.saved) >= len(self.expected_saved)
                or step != self.expected_saved[len(self.saved)]
                or not self.steps or self.steps[-1]['global_step'] != step):
            raise ValueError('unexpected resumed checkpoint callback')
        self.saved.append(step)
        self.progress()

    def request_diagnostic_stop(self, control):
        """Use normal Trainer control after a complete update; never change max_steps."""
        if self.diagnostic_stop_step is not None and self.trainer.state.global_step == self.stop_step:
            if (not self.entry_state_captured or not self.steps
                    or self.steps[-1]['global_step'] != self.stop_step
                    or len(self.outputs) != self.planned_inputs):
                raise ValueError('cannot stop before the declared complete observation update')
            control.should_training_stop = True
            control.should_save = self.stop_step in self.expected_saved
            self.diagnostic_stop_requested = True
            self.progress()
        return control

    def detach(self):
        if self.boundary_trace is not None:
            self.boundary_trace.detach()
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.original_rng_loader is not None:
            if self.trainer._load_rng_state is not self.observed_rng_loader:
                raise ValueError('native RNG loader changed while under observation')
            self.trainer._load_rng_state = self.original_rng_loader
            self.original_rng_loader = None

    def fail(self, error):
        self.failed = True
        if self.boundary_trace is not None and self.boundary_trace_SHA256 is None:
            self.boundary_trace.abort(error)
        self.detach()
        self.progress()
        path = self.root / 'failure.json'
        if not path.exists():
            self.write(path.name, dict(error_type=type(error).__name__,
                       completed_steps=[row['global_step'] for row in self.steps],
                       message=str(error)[:300]))

    def guarded(self, function):
        def call(*args, **kwargs):
            try:
                self.check()
                return function(*args, **kwargs)
            except BaseException as error:
                self.fail(error)
                raise
        return call

    def finish(self):
        if (self.failed or self.trainer.state.global_step != self.stop_step
                or len(self.steps) != self.stop_step - self.checkpoint_step
                or len(self.outputs) != self.planned_inputs or self.saved != self.expected_saved
                or len(self.rng_loads) != 1
                or (self.entry_state_step is not None and not self.entry_state_captured)
                or (self.boundary_modules is not None and self.boundary_trace_SHA256 is None)
                or (self.diagnostic_stop_step is not None and not self.diagnostic_stop_requested)):
            raise ValueError('incomplete native resumed training')
        self.detach()
        self.write('result.json', dict(rank=self.rank, world_size=self.world_size, plan=self.plan,
                   RNG_loads=self.rng_loads, inputs=self.inputs, outputs=self.outputs,
                   steps=self.steps, saved_checkpoints=self.saved,
                   native_resume_window_complete=self.diagnostic_stop_step is None,
                   declared_observation_window_complete=True,
                   diagnostic_stop_step=self.diagnostic_stop_step,
                   diagnostic_stop_requested=self.diagnostic_stop_requested,
                   completed_global_step=self.trainer.state.global_step,
                   subsequent_original_inputs_equal=None,
                   entry_state_step=self.entry_state_step, entry_state_captured=self.entry_state_captured,
                   entry_state_SHA256=self.entry_state_SHA256,
                   boundary_trace_SHA256=self.boundary_trace_SHA256,
                   original_continuous_final_state_compared=False, recovery_qualified=False,
                   formal_performance=False, qualification='Observed full-state restore and native updates only; independent continuous-tail comparison remains required.'))
