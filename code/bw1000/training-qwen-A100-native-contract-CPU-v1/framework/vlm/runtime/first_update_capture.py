"""Full private first-two-update artifacts, preserving the native training path."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def cpu(value):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        if result.is_floating_point() and not bool(torch.isfinite(result).all()):
            raise RuntimeError('nonfinite captured tensor')
        return result
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(x) for x in value]
    if isinstance(value, tuple):
        return tuple(cpu(x) for x in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported captured object ' + type(value).__name__)


def remaining(epoch, deadline, now, max_seconds):
    if (not all(math.isfinite(x) for x in (epoch, deadline, now, max_seconds))
            or not 0 < max_seconds <= 21600 or epoch <= 0
            or not 0 < deadline - epoch <= max_seconds or not epoch <= now < deadline):
        raise ValueError('capture epoch is invalid or expired')
    return deadline - now


class LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0

    def write(self, data):
        if self.count + len(data) > self.limit:
            raise ValueError('full capture payload exceeds its byte budget')
        count = self.stream.write(data)
        self.count += count
        return count

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


class FirstUpdateCapture:
    """The plugin enforces real hardware/recipe; this core also admits CPU tests."""
    def __init__(self, trainer, root, named, *, rank, world_size, epoch, deadline,
                 max_seconds, rng, artifact_limit=2*1024**3-1, total_limit=8*1024**3):
        remaining(epoch, deadline, time.time(), max_seconds)
        if not 0 < artifact_limit <= 2*1024**3-1 or not 0 < total_limit <= 8*1024**3:
            raise ValueError('capture byte budget differs')
        if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
            raise ValueError('invalid capture rank')
        self.trainer, self.root, self.named = trainer, Path(root), list(named)
        if not self.named or len({n for n, _ in self.named}) != len(self.named):
            raise ValueError('empty or duplicate trainable inventory')
        self.rank, self.world_size = rank, world_size
        self.epoch, self.deadline, self.max_seconds, self.rng = epoch, deadline, max_seconds, rng
        self.anchor, self.initial_remaining = time.monotonic(), deadline-time.time()
        self.artifact_limit, self.total_limit = artifact_limit, total_limit
        self.files, self.handles, self.names = [], [], {id(p): n for n, p in self.named}
        self.inputs = self.outputs = self.updates = self.clip_calls = self.capture_bytes = 0
        self.serialization_and_hash_seconds = 0.0
        self.original_clip = None
        self.failed = self.detached = False
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.progress()

    def check_deadline(self):
        remaining(self.epoch, self.deadline, time.time(), self.max_seconds)
        if time.monotonic()-self.anchor >= self.initial_remaining:
            raise TimeoutError('capture monotonic deadline expired')

    def write_json(self, name, value):
        with (self.root/name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2)

    def progress(self):
        row = dict(rank=self.rank, world_size=self.world_size, planned_updates=2,
                   planned_micro_forwards=8, planned_native_global_step=25,
                   completed_updates=self.updates, inputs_saved=self.inputs,
                   outputs_saved=self.outputs, clip_calls=self.clip_calls,
                   failed=self.failed, files=self.files,
                   serialization_and_hash_seconds=self.serialization_and_hash_seconds)
        temp = self.root/'progress.tmp'
        temp.write_text(json.dumps(row), encoding='utf8')
        temp.replace(self.root/'progress.json')

    def save(self, name, value):
        self.check_deadline()
        started = time.monotonic()
        path = self.root/name
        with path.open('xb') as stream:
            torch.save(value, LimitedWriter(stream, min(self.artifact_limit, self.total_limit-self.capture_bytes)))
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for raw in iter(lambda: stream.read(1024**2), b''):
                h.update(raw)
        self.capture_bytes += path.stat().st_size
        self.serialization_and_hash_seconds += time.monotonic()-started
        self.files.append(dict(name=name, bytes=path.stat().st_size, SHA256=h.hexdigest()))
        self.progress()

    def parameters(self):
        return {n: cpu(p) for n, p in self.named}

    def gradients(self):
        return {n: None if p.grad is None else cpu(p.grad) for n, p in self.named}

    def optimizer(self):
        opt = self.trainer.optimizer
        groups = []
        for group in opt.param_groups:
            names = [self.names[id(p)] for p in group['params']]
            groups.append(dict(parameter_names=names, **{k: cpu(v) for k, v in group.items() if k != 'params'}))
        return dict(state_by_name={n: cpu(opt.state.get(p, {})) for n, p in self.named}, param_groups=groups)

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.original_clip is not None:
            self.trainer._clip_grad_norm = self.original_clip
        self.detached = True

    def fail(self, error):
        self.detach()
        if not self.failed:
            self.failed = True
            self.progress()
            self.write_json('failure.json', dict(error_type=type(error).__name__,
                            completed_updates=self.updates, inputs_saved=self.inputs,
                            outputs_saved=self.outputs, planned_updates=2,
                            planned_micro_forwards=8, planned_native_global_step=25))

    def guarded(self, function):
        def call(*args, **kwargs):
            try:
                self.check_deadline()
                return function(*args, **kwargs)
            except BaseException as error:
                self.fail(error)
                raise
        return call

    def attach(self, model):
        if self.handles or self.original_clip is not None or self.trainer.state.global_step != 0:
            raise ValueError('capture can attach only once at update zero')
        self.save('initial.pt', dict(parameters=self.parameters(), RNG=self.rng(),
                  manifest=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype), numel=p.numel()) for n, p in self.named],
                  global_step=0))

        def before(module, argv, kwargs):
            if self.trainer.state.global_step >= 2:
                return
            if self.inputs != self.outputs or self.inputs >= 8:
                raise ValueError('unexpected forward sequence')
            ordinal = self.inputs+1
            self.save('forward-%02d-input.pt' % ordinal,
                      dict(global_step=self.trainer.state.global_step, micro_ordinal=ordinal,
                           args=cpu(argv), kwargs=cpu(kwargs), RNG_before_forward=self.rng()))
            self.inputs = ordinal
            self.progress()

        def after(module, argv, kwargs, result):
            if self.trainer.state.global_step >= 2:
                return
            if result is None or self.inputs != self.outputs+1:
                raise ValueError('incomplete model forward')
            ordinal = self.outputs+1
            self.save('forward-%02d-output.pt' % ordinal,
                      dict(global_step=self.trainer.state.global_step, micro_ordinal=ordinal,
                           result=cpu(dict(result))))
            self.outputs = ordinal
            self.progress()

        self.handles = [model.register_forward_pre_hook(self.guarded(before), with_kwargs=True),
                        model.register_forward_hook(self.guarded(after), with_kwargs=True, always_call=True)]
        self.original_clip = self.trainer._clip_grad_norm

        def clipped(wrapped_model):
            if self.trainer.state.global_step >= 2:
                return self.original_clip(wrapped_model)
            if self.inputs != self.outputs or self.inputs != 4*(self.updates+1):
                raise ValueError('gradient accumulation sequence differs')
            ordinal = self.trainer.state.global_step+1
            self.save('update-%02d-preclip.pt' % ordinal,
                      dict(gradients=self.gradients(), global_step=self.trainer.state.global_step,
                           max_grad_norm=self.trainer.args.max_grad_norm))
            self.clip_calls += 1
            norm = self.original_clip(wrapped_model)
            self.save('update-%02d-postclip.pt' % ordinal,
                      dict(gradients=self.gradients(), reported_preclip_norm=cpu(norm),
                           global_step=self.trainer.state.global_step))
            return norm
        self.trainer._clip_grad_norm = self.guarded(clipped)

    def before_optimizer(self):
        if self.trainer.state.global_step < 2:
            if self.clip_calls != self.updates+1:
                raise ValueError('optimizer step lacks a recorded clipping boundary')
            self.save('update-%02d-before-optimizer.pt' % (self.trainer.state.global_step+1),
                      dict(parameters=self.parameters(), optimizer=self.optimizer(),
                           global_step=self.trainer.state.global_step, RNG=self.rng()))

    def after_optimizer(self):
        if self.trainer.state.global_step < 2:
            self.updates += 1
            self.save('update-%02d-after-optimizer.pt' % (self.trainer.state.global_step+1),
                      dict(parameters=self.parameters(), optimizer=self.optimizer(),
                           global_step_before_increment=self.trainer.state.global_step, RNG=self.rng()))

    def step_end(self):
        self.check_deadline()
        if self.trainer.state.global_step == 2:
            if self.failed or self.inputs != 8 or self.outputs != 8 or self.updates != 2 or self.clip_calls != 2 or len(self.files) != 25:
                raise ValueError('incomplete first-two-update capture')
            self.detach()
            self.write_json('receipt.json', dict(state='TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED',
                rank=self.rank, world_size=self.world_size, completed_global_step=2,
                micro_forwards=8, optimizer_updates=2, clip_calls=2, files=self.files,
                qualification='Full artifacts only; native25 and independent numerical comparison still required.'))

    def train_end(self):
        self.check_deadline()
        if self.failed or self.trainer.state.global_step != 25 or self.updates != 2 or not self.detached:
            raise ValueError('native 25-update training incomplete')
        self.write_json('native25-complete.json', dict(rank=self.rank, global_step=25))
