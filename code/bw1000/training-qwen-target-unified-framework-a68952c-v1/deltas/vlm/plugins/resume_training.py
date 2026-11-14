"""HIP MiniCPM native checkpoint10/20-to25 recovery observer."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vlm.runtime.first_update_capture import remaining
from vlm.runtime.minicpm_recipe_contract import validate_native_recipe
from vlm.runtime.resume_witness import ResumeWitness

KEY = 'native_resume_witness'


class NativeResume(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        raw = Path(os.environ['NATIVE_RESUME_CONFIG']).read_bytes()
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != os.environ['NATIVE_RESUME_CONFIG_SHA256']:
            raise ValueError('resume witness configuration identity differs')
        config = json.loads(raw)
        checkpoint_step = config.get('checkpoint_step', 20)
        if type(checkpoint_step) is not int or checkpoint_step not in (10, 20):
            raise ValueError('expected a declared original checkpoint10 or paired checkpoint20')
        rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        if (not torch.version.hip or torch.version.cuda or torch.cuda.device_count() != 8
                or int(os.environ['WORLD_SIZE']) != 8 or rank != local
                or torch.cuda.current_device() != local or not 0 <= rank < 8
                or state.global_step != checkpoint_step or args.max_steps != 25):
            raise ValueError('expected actual eight-rank HIP checkpoint10/20-to25 resume')
        if any(type(c).__name__ in {'BaselineCallback', 'FirstUpdates', 'NativeDropoutAlignment'}
               for c in self.trainer.callback_handler.callbacks):
            raise ValueError('fresh-run callbacks cannot participate in resume')
        model = kw.get('model', self.trainer.model)
        named, frozen = validate_native_recipe(model, args, 10822656)
        if any(p.device.type != 'cuda' or p.device.index != local for _, p in named):
            raise ValueError('restored trainables are not on the rank-owned device')
        epoch, deadline = float(os.environ['NATIVE_RESUME_EPOCH']), float(os.environ['NATIVE_RESUME_DEADLINE'])
        maximum = float(os.environ['NATIVE_RESUME_MAX_SECONDS'])
        from vlm.runtime.forward_boundary_trace import validate_boundary_opt_in, minicpm_boundaries
        trace_enabled = validate_boundary_opt_in(config.get('first_divergence_trace', False),
            config.get('entry_state_step'), config.get('diagnostic_stop_step'))
        boundary_modules = minicpm_boundaries(model) if trace_enabled else None
        self.witness = ResumeWitness(self.trainer, model, kw['train_dataloader'],
            Path(args.output_dir) / 'resume-witness' / ('rank-%02d' % rank),
            config['checkpoint'], config['checkpoint_files'], rank=rank, world_size=8,
            checkpoint_step=checkpoint_step, max_steps=25, epoch=epoch, deadline=deadline,
            max_seconds=maximum, device_rng=torch.cuda.get_rng_state_all,
            entry_state_step=config.get('entry_state_step'),
            diagnostic_stop_step=config.get('diagnostic_stop_step'), boundary_modules=boundary_modules)
        self.witness.write('frozen-boundaries.json', frozen)
        self.witness.write('owner.json', dict(PID=os.getpid(), PGID=os.getpgrp(), rank=rank,
            local_rank=local, epoch=epoch, deadline=deadline,
            startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])))
        def expired(*_):
            error = TimeoutError('native resume epoch exhausted')
            self.witness.fail(error)
            raise error
        budget = remaining(epoch, deadline, time.time(), maximum)
        previous = signal.getitimer(signal.ITIMER_REAL)[0]
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(previous, budget) if previous else budget)
        return control

    def on_step_end(self, args, state, control, **kw):
        self.witness.guarded(self.witness.step_end)()
        self.witness.guarded(self.witness.request_diagnostic_stop)(control)
        if state.global_step == 25:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kw):
        self.witness.guarded(self.witness.checkpoint_saved)()
        return control

    def on_train_end(self, args, state, control, **kw):
        self.witness.guarded(self.witness.finish)()
        return control


if KEY in callbacks_map:
    raise ValueError('duplicate native resume callback')
callbacks_map[KEY] = NativeResume
