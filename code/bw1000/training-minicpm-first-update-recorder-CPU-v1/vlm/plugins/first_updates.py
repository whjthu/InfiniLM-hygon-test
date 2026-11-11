"""Observe native HIP MiniCPM training; never replace its forward or optimizer."""
import hashlib
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
from vlm.runtime.minicpm_recipe_contract import validate_native_recipe
from vlm.runtime.training_providers import provider_inventory

KEY = 'original_first_updates'


def rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=state[0], keys=state[1].tolist(),
                position=int(state[2]), has_gauss=int(state[3]), cached_gaussian=float(state[4])),
                torch_CPU=torch.get_rng_state(), torch_device=torch.cuda.get_rng_state(),
                device_index=torch.cuda.current_device())


class FirstUpdates(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer, self.capture = trainer, None

    def on_train_begin(self, args, state, control, **kw):
        epoch = float(os.environ['FIRST_UPDATE_EPOCH'])
        deadline = float(os.environ['FIRST_UPDATE_DEADLINE'])
        max_seconds = float(os.environ['FIRST_UPDATE_MAX_SECONDS'])
        budget = remaining(epoch, deadline, time.time(), max_seconds)
        if not torch.version.hip or torch.version.cuda or torch.cuda.device_count() != 8:
            raise ValueError('expected the actual eight-device HIP runtime')
        rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        if int(os.environ['WORLD_SIZE']) != 8 or not 0 <= rank < 8 or not 0 <= local < 8 or torch.cuda.current_device() != local:
            raise ValueError('actual rank/device mapping differs')
        model = kw.get('model', self.trainer.model)
        named, frozen = validate_native_recipe(model, args, 10822656)
        if state.global_step != 0 or any(p.device.type != 'cuda' or p.device.index != local for _, p in named):
            raise ValueError('expected fresh rank-owned HIP trainables')
        if named_parameter_sha256(named) != '9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16':
            raise ValueError('first-update recorder must observe restored original trainables')
        self.capture = FirstUpdateCapture(self.trainer,
            Path(os.environ['FIRST_UPDATE_CAPTURE'])/('rank-%02d' % rank), named,
            rank=rank, world_size=8, epoch=epoch, deadline=deadline,
            max_seconds=max_seconds, rng=rng)

        def expired(*_):
            error = TimeoutError('original capture epoch deadline reached')
            self.capture.fail(error)
            raise error
        previous = signal.getitimer(signal.ITIMER_REAL)[0]
        budget = remaining(epoch, deadline, time.time(), max_seconds)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(previous, budget) if previous else budget)
        owner = dict(PID=os.getpid(), PGID=os.getpgrp(),
                     startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                     boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),
                     epoch=epoch, deadline=deadline, rank=rank, local_rank=local,
                     accelerator='hip', accelerator_version=torch.version.hip)
        self.capture.write_json('owner.json', owner)
        self.capture.write_json('frozen-boundaries.json', frozen)
        self.source_SHA256 = os.environ['FIRST_UPDATE_MODEL_SOURCE_SHA256']
        self.capture.write_json('runtime-providers.json', self.capture.guarded(provider_inventory)(model,
            model_module_prefix='transformers.models.qwen3_5.', expected_model_source_SHA256=self.source_SHA256))
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
        model = kw.get('model', self.trainer.model)
        self.capture.write_json('runtime-providers-final.json', self.capture.guarded(provider_inventory)(model,
            model_module_prefix='transformers.models.qwen3_5.', expected_model_source_SHA256=self.source_SHA256))
        self.capture.guarded(self.capture.train_end)()
        return control


if KEY in callbacks_map:
    raise ValueError('duplicate first-update diagnostic callback')
callbacks_map[KEY] = FirstUpdates
