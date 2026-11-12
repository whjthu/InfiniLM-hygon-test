"""Observe original native CUDA/HIP updates with a frozen per-case model binding."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from swift.callbacks import TrainerCallback, callbacks_map

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vlm.runtime.first_update_capture import FirstUpdateCapture, remaining
from vlm.runtime.initial_trainables import named_parameter_sha256
from vlm.runtime.native_recipe_contract import model_contract, validate_native_recipe
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle

KEY = 'native_first_updates'


def rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=state[0], keys=state[1].tolist(),
                position=int(state[2]), has_gauss=int(state[3]), cached_gaussian=float(state[4])),
                torch_CPU=torch.get_rng_state(), torch_device=torch.cuda.get_rng_state(),
                device_index=torch.cuda.current_device())


def model_providers(model, expected, case='T02.sft'):
    files, classes = {}, []
    prefix = model_contract(case)['source_prefix']
    for name, module in model.named_modules():
        cls = type(module)
        if not cls.__module__.startswith(prefix):
            continue
        path = Path(inspect.getfile(cls)).resolve()
        if str(path) not in files:
            if not 0 < path.stat().st_size <= 2 * 1024**2:
                raise ValueError('native VLM provider outside source bound')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError('native VLM implementation source differs')
            files[str(path)] = dict(bytes=path.stat().st_size, SHA256=digest)
        classes.append(dict(name=name, type=cls.__module__ + '.' + cls.__qualname__, file=str(path)))
    if not classes:
        raise ValueError('native VLM provider is absent')
    return dict(files=files, classes=classes, torch=torch.__version__, HIP=torch.version.hip,
                CUDA=torch.version.cuda, qualification='Model class/source binding only; no executed kernel trace.')


class NativeFirstUpdates(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer, self.capture = trainer, None

    def on_train_begin(self, args, state, control, **kw):
        epoch = float(os.environ['FIRST_UPDATE_EPOCH'])
        deadline = float(os.environ['FIRST_UPDATE_DEADLINE'])
        maximum = float(os.environ['FIRST_UPDATE_MAX_SECONDS'])
        remaining(epoch, deadline, time.time(), maximum)
        raw = Path(os.environ['FIRST_UPDATE_BINDING']).read_bytes()
        if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != os.environ['FIRST_UPDATE_BINDING_SHA256']:
            raise ValueError('native first-update binding differs')
        binding = json.loads(raw)
        platform = os.environ['FIRST_UPDATE_PLATFORM']
        if platform not in ('cuda', 'hip') or bool(torch.version.hip) != (platform == 'hip'):
            raise ValueError('native accelerator platform differs')
        if (platform == 'cuda' and not torch.version.cuda) or torch.cuda.device_count() != 8:
            raise ValueError('expected actual eight-device native runtime')
        rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        if int(os.environ['WORLD_SIZE']) != 8 or not 0 <= rank < 8 or not 0 <= local < 8 or torch.cuda.current_device() != local:
            raise ValueError('native rank/device mapping differs')
        model = kw.get('model', self.trainer.model)
        method = getattr(self.trainer, '_prepare_gradient_checkpointing', None)
        if binding['spec']['case'] == 'T01.sft' and not binding.get('native_checkpointing_lifecycle'):
            raise ValueError('Qwen runtime observer requires qualified native checkpointing lifecycle')
        named, frozen = validate_native_recipe(model, args, binding, checkpointing_method=method)
        checkpointing = (validate_checkpointing_lifecycle(model.get_base_model(), args, binding, method)
                         if binding.get('native_checkpointing_lifecycle') else None)
        if state.global_step != 0 or any(p.device.type != 'cuda' or p.device.index != local for _, p in named):
            raise ValueError('expected fresh native rank-owned trainables')
        initial = named_parameter_sha256(named)
        if binding.get('initial_trainables_SHA256') and initial != binding['initial_trainables_SHA256']:
            raise ValueError('native initial trainables differ from original reference')
        self.capture = FirstUpdateCapture(self.trainer,
            Path(os.environ['FIRST_UPDATE_CAPTURE']) / ('rank-%02d' % rank), named,
            rank=rank, world_size=8, epoch=epoch, deadline=deadline, max_seconds=maximum, rng=rng)
        def expired(*_):
            error = TimeoutError('original native capture epoch exhausted')
            self.capture.fail(error)
            raise error
        previous = signal.getitimer(signal.ITIMER_REAL)[0]
        budget = remaining(epoch, deadline, time.time(), maximum)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(previous, budget) if previous else budget)
        self.binding = binding
        self.capture.write_json('owner.json', dict(PID=os.getpid(), PGID=os.getpgrp(),
            startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
            epoch=epoch, deadline=deadline, rank=rank, local_rank=local, accelerator=platform,
            binding_SHA256=hashlib.sha256(raw).hexdigest(), initial_trainables_SHA256=initial))
        self.capture.write_json('frozen-boundaries.json', frozen)
        if checkpointing is not None:
            self.capture.write_json('checkpointing-lifecycle.json', checkpointing)
        self.capture.write_json('runtime-providers.json', self.capture.guarded(model_providers)(
            model, binding['model_source_SHA256'], binding['spec']['case']))
        self.capture.guarded(self.capture.attach)(model)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.capture.guarded(self.capture.before_optimizer)()
        return control

    def on_optimizer_step(self, args, state, control, **kw):
        self.capture.guarded(self.capture.after_optimizer)()
        return control

    def on_step_end(self, args, state, control, **kw):
        self.capture.guarded(self.capture.step_end)()
        return control

    def on_train_end(self, args, state, control, **kw):
        self.capture.write_json('runtime-providers-final.json', self.capture.guarded(model_providers)(
            kw.get('model', self.trainer.model), self.binding['model_source_SHA256'], self.binding['spec']['case']))
        self.capture.guarded(self.capture.train_end)()
        return control


if KEY in callbacks_map:
    raise ValueError('duplicate native first-update callback')
callbacks_map[KEY] = NativeFirstUpdates
