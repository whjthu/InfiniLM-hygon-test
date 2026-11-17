"""One common CUDA observer for T03/T04; no custom optimizer or training loop."""
import hashlib
import functools
import json
import os
from pathlib import Path
import signal
import sys

import torch
from swift.callbacks import TrainerCallback, callbacks_map

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm.runtime.cuda_resume_contract import bounded_bytes, load_config, remaining, require, verify_provider_files, verify_supervisor
from vlm.runtime.cuda_resume_witness import CudaResumeWitness
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle
from vlm.runtime.provider_sources import model_provider_sources

KEY = 'native_cuda_resume_witness'


class NativeCudaResume(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer, self.witness = trainer, None
        self.config = load_config()
        self.entry_observations = dict(initial_training_values=[], init_training_state=[])
        self.entry_methods = {}
        self.entry_root = Path(self.config['output'])
        self.entry_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.entry_path = self.entry_root / ('entry-rank-%02d.json' % int(os.environ['RANK']))
        with self.entry_path.open('x', encoding='utf-8') as stream:
            json.dump(dict(phase='CALLBACK_CREATED', planned_microbatches=20, planned_updates=5), stream)
            stream.write('\n')
        self.observe_entry('set_initial_training_values', 'initial_training_values')
        self.observe_entry('_init_training_state', 'init_training_state')

    def observe_entry(self, name, key):
        original = getattr(self.trainer, name)
        self.entry_methods[name] = original
        @functools.wraps(original)
        def call(*args, **kwargs):
            remaining(self.config)
            result = original(*args, **kwargs)
            require(isinstance(result, tuple) and all(x is None or type(x) is int for x in result),
                    'actual Trainer initial return schema differs')
            row = dict(returned_values=list(result))
            if key == 'initial_training_values':
                loader = args[1] if len(args) > 1 else kwargs['dataloader']
                row['actual_loader_length'] = len(loader)
                row['actual_loader_class'] = type(loader).__module__ + '.' + type(loader).__qualname__
            self.entry_observations[key].append(row)
            temp = self.entry_path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.entry_observations, allow_nan=False) + '\n', encoding='utf-8')
            temp.replace(self.entry_path)
            return result
        setattr(self.trainer, name, call)

    def on_train_begin(self, args, state, control, **kwargs):
        config = self.config; remaining(config)
        verify_supervisor(config)
        verify_provider_files(config)
        require(torch.version.cuda and not torch.version.hip and torch.cuda.device_count() == 8,
                'explicit CUDA eight-device runtime required')
        rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        require(int(os.environ['WORLD_SIZE']) == 8 and rank == local and 0 <= rank < 8
                and torch.cuda.current_device() == local and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() == 8 and torch.distributed.get_rank() == rank,
                'CUDA owned-rank mapping differs')
        forbidden = {'BaselineCallback', 'FirstUpdates', 'NativeFirstUpdates', 'NativeDropoutAlignment', 'NativeResume'}
        require(not any(type(callback).__name__ in forbidden for callback in self.trainer.callback_handler.callbacks),
                'fresh or HIP-only callbacks cannot join CUDA resume')
        binding_path = Path(config['binding_path'])
        raw = bounded_bytes(binding_path, 2 * 1024**2)
        require(0 < len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == config['binding_SHA256'],
                'case native binding differs')
        binding = json.loads(raw)
        require(binding['spec']['case'] == config['case'], 'CUDA resume binding case differs')
        model = kwargs.get('model', self.trainer.model)
        method = self.trainer._prepare_gradient_checkpointing
        named, frozen = validate_native_recipe(model, args, binding, checkpointing_method=method)
        require(state.global_step == 20 and all(p.device.type == 'cuda' and p.device.index == local for _, p in named),
                'restored trainables are not on actual rank-owned CUDA device')
        lifecycle = (validate_checkpointing_lifecycle(model.get_base_model(), args, binding, method)
                     if binding.get('native_checkpointing_lifecycle') else None)
        providers = model_provider_sources(model, binding['model_source_SHA256'], config['case'])
        self.witness = CudaResumeWitness(self.trainer, model, self.trainer.callback_handler.train_dataloader,
                                         config, binding, rank, self.entry_observations, self.entry_methods)
        self.witness.write('owner.json', dict(rank=rank, local_rank=local, PID=os.getpid(), PGID=os.getpgrp(),
                           host=__import__('socket').gethostname(), uid=os.getuid(),
                           startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                           epoch=config['epoch'], deadline=config['deadline'], binding_SHA256=config['binding_SHA256']))
        self.witness.write('native-recipe-and-providers.json', dict(frozen=frozen, checkpointing=lifecycle, providers=providers))
        def expired(*_):
            error = TimeoutError('original CUDA resume epoch exhausted')
            self.witness.progress(dict(type=type(error).__name__, reason=str(error)))
            raise error
        previous = signal.getitimer(signal.ITIMER_REAL)[0]
        signal.signal(signal.SIGALRM, expired)
        budget = remaining(config)
        signal.setitimer(signal.ITIMER_REAL, min(previous, budget) if previous else budget)
        try:
            self.witness.attach()
        except BaseException as error:
            self.witness.progress(dict(type=type(error).__name__, reason=str(error)[:180])); raise
        return control

    def invoke(self, method):
        require(self.witness is not None, 'CUDA resume observer not initialized')
        try:
            getattr(self.witness, method)()
        except BaseException as error:
            self.witness.progress(dict(type=type(error).__name__, reason=str(error)[:180])); raise

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.invoke('pre_optimizer'); return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.invoke('post_optimizer'); return control

    def on_step_end(self, args, state, control, **kwargs):
        self.invoke('step_end'); return control

    def on_save(self, args, state, control, **kwargs):
        self.invoke('saved'); return control

    def on_train_end(self, args, state, control, **kwargs):
        self.invoke('finish'); return control


if KEY in callbacks_map:
    raise ValueError('duplicate CUDA resume observer callback')
callbacks_map[KEY] = NativeCudaResume
