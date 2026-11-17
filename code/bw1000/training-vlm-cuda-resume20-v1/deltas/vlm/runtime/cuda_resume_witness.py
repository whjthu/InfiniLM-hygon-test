"""Observe native CUDA resume; never implement a training loop or restore state."""
import functools
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers.trainer import safe_globals

from .cuda_resume_contract import MEMBERS, bounded_bytes, checked_path, remaining, require, verify_file
from .resume_witness import rng_fingerprint, rng_values


def tree_fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            require(not cpu.is_floating_point() or bool(torch.isfinite(cpu).all()), 'nonfinite state tensor')
            digest.update(json.dumps(['tensor', str(cpu.dtype), list(cpu.shape)]).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            require(all(type(k) in (str, int) for k in item), 'unsupported state key')
            digest.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
            digest.update(b'end')
        elif item is None or type(item) in (str, int, float, bool):
            require(type(item) is not float or math.isfinite(item), 'nonfinite state scalar')
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise ValueError('unsupported state value type ' + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def comparison(expected, actual):
    """CPU reductions, tensor-at-a-time; no GPU arithmetic or tolerance relaxation."""
    result = dict(exact=True, structure_match=True, tensor_count=0, different_tensors=0,
                  max_abs=0.0, squared_error=0.0, squared_reference=0.0)
    def walk(left, right):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            result['tensor_count'] += 1
            if left.shape != right.shape or left.dtype != right.dtype:
                result.update(exact=False, structure_match=False); return
            lcpu, rcpu = left.detach().cpu(), right.detach().cpu()
            require(not lcpu.is_floating_point() or bool(torch.isfinite(lcpu).all()) and bool(torch.isfinite(rcpu).all()),
                    'nonfinite comparison tensor')
            if tree_fingerprint(lcpu) != tree_fingerprint(rcpu):
                result['exact'] = False; result['different_tensors'] += 1
            if lcpu.is_floating_point() or lcpu.dtype != torch.bool:
                l64, r64 = lcpu.to(torch.float64), rcpu.to(torch.float64)
                delta = r64 - l64
                if delta.numel():
                    result['max_abs'] = max(result['max_abs'], float(delta.abs().max()))
                result['squared_error'] += float(delta.square().sum())
                result['squared_reference'] += float(l64.square().sum())
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                result.update(exact=False, structure_match=False); return
            for key in left:
                walk(left[key], right[key])
        elif type(left) in (list, tuple) and type(left) is type(right):
            if len(left) != len(right):
                result.update(exact=False, structure_match=False); return
            for lchild, rchild in zip(left, right):
                walk(lchild, rchild)
        elif type(left) is not type(right):
            result.update(exact=False, structure_match=False)
        elif left != right:
            result['exact'] = False
    walk(expected, actual)
    result['relative_L2'] = (math.sqrt(result['squared_error'] / result['squared_reference'])
                             if result['squared_reference'] else (0.0 if not result['squared_error'] else None))
    return result


def trusted_load(root, name, pin, check):
    path = checked_path(str(Path(root) / name))
    verify_file(path, pin, check)
    before = path.stat()
    if name.endswith('.safetensors'):
        value = load_safetensors(str(path), device='cpu')
    elif name.endswith('.json'):
        require(pin['bytes'] <= 262144, 'bounded checkpoint JSON required')
        value = json.loads(path.read_bytes())
    else:
        with safe_globals():
            value = torch.load(path, map_location='cpu', weights_only=True)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'state file changed during safe decoding')
    check()
    return value


def provider_record(value):
    function = getattr(value, '__func__', value)
    partials = []
    while isinstance(function, functools.partial):
        partials.append(dict(bound_argument_types=[type(x).__name__ for x in function.args],
                             scalar_keywords={k: v for k, v in (function.keywords or {}).items()
                                              if k in {'num_workers', 'rank', 'seed', 'data_seed', 'num_processes', 'tp_size'}
                                              and (v is None or type(v) in (int, bool) or type(v) is float and math.isfinite(v))}))
        function = getattr(function.func, '__func__', function.func)
    path = Path(inspect.getsourcefile(function) or inspect.getsourcefile(type(value)) or '').resolve(strict=True)
    roots = [Path('/home/whj/nvidia-baseline/env/lib/python3.12/site-packages'),
             Path('/usr/local/lib/python3.12/dist-packages')]
    require(any(path.is_relative_to(root) for root in roots), 'runtime provider outside registered roots')
    raw = bounded_bytes(path, 1024**2, library=True)
    return dict(path=str(path), bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(),
                qualname=getattr(function, '__qualname__', type(value).__qualname__), partial_bindings=partials)


def optimizer_steps(value, expected_step, expected_parameters):
    require(set(value) == {'state', 'param_groups'}, 'optimizer serialized schema')
    ids = [key for group in value['param_groups'] for key in group['params']]
    require(len(ids) == len(set(ids)) == len(value['state']) == expected_parameters and set(ids) == set(value['state']),
            'optimizer parameter state denominator differs')
    for state in value['state'].values():
        require({'step', 'exp_avg', 'exp_avg_sq'} <= set(state), 'AdamW full state missing')
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and float(step.detach().cpu()) == expected_step,
                'AdamW actual step differs')
    return dict(parameter_states=len(ids), actual_step=expected_step)


class CudaResumeWitness:
    def __init__(self, trainer, model, loader, config, binding, rank, entry_observations, entry_methods):
        self.trainer, self.model, self.loader = trainer, model, loader
        self.config, self.binding, self.rank = config, binding, rank
        self.root = Path(config['output']) / ('rank-%02d' % rank)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.inputs, self.forward_inputs, self.losses, self.steps, self.clips = [], [], [], [], []
        self.rng_loads, self.saved_steps, self.handles = [], [], []
        self.original_methods = dict(entry_methods); self.finished = False
        self.entry_observations = entry_observations
        self.loader_epoch_calls = []
        self.expected_rng20 = None
        self.write('planned.json', dict(case=config['case'], rank=rank, planned_microbatches=20,
                                        planned_updates=[21, 22, 23, 24, 25], original_tail_micro_reference_available=False))

    def check(self):
        remaining(self.config)

    def write(self, name, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
        require(len(raw) <= 2 * 1024**2, 'bounded private witness JSON exceeded')
        path = self.root / name; temp = self.root / (name + '.tmp')
        with temp.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        temp.replace(path)

    def progress(self, error=None):
        self.write('progress.json', dict(case=self.config['case'], rank=self.rank,
                   completed_microbatches=len(self.inputs), completed_forward_calls=len(self.forward_inputs),
                   completed_training_step_returns=len(self.losses), completed_updates=self.steps,
                   clip_records=len(self.clips), rng_restore_calls=len(self.rng_loads), saved_steps=self.saved_steps,
                   planned_microbatches=20, planned_updates=5, error=error, finished=self.finished))

    def live_adapter(self):
        return get_peft_model_state_dict(self.model)

    def ref(self, key, name):
        root = self.config['checkpoint' if key == 20 else 'comparison_checkpoint']
        pins = self.config['checkpoint_files' if key == 20 else 'comparison_files']
        return trusted_load(root, name, pins[name], self.check)

    def current_rng(self):
        expected = self.expected_rng20['cuda']
        if isinstance(expected, torch.Tensor):
            require(expected.dtype == torch.uint8 and expected.ndim == 1, 'saved current-device RNG tensor differs')
            return rng_values(torch.cuda.get_rng_state)
        require(type(expected) is list and len(expected) == 8
                and all(isinstance(x, torch.Tensor) and x.dtype == torch.uint8 and x.ndim == 1 for x in expected),
                'saved all-device RNG container differs')
        return rng_values(torch.cuda.get_rng_state_all)

    def state_comparison(self, step):
        rows = {}
        expected = self.ref(step, 'adapter_model.safetensors')
        live = self.live_adapter()
        require(len(expected) == len(live) == len(self.binding['trainable_manifest']), 'adapter state name denominator')
        rows['adapter'] = comparison(expected, live)
        rows['adapter_live_SHA256'] = tree_fingerprint(live)
        del expected, live
        expected = self.ref(step, 'optimizer.pt'); optimizer_steps(expected, step, len(self.binding['trainable_manifest']))
        live = self.trainer.optimizer.state_dict(); optimizer_steps(live, step, len(self.binding['trainable_manifest']))
        rows['optimizer'] = comparison(expected, live)
        rows['optimizer_live_SHA256'] = tree_fingerprint(live)
        del expected, live
        expected = self.ref(step, 'scheduler.pt'); live = self.trainer.lr_scheduler.state_dict()
        rows['scheduler'] = comparison(expected, live)
        rows['scheduler_live_SHA256'] = tree_fingerprint(live)
        return rows

    def attach(self):
        self.check()
        args, state = self.trainer.args, self.trainer.state
        expected_providers = {row['canonical_path']: row for row in self.config['providers']}
        active = {}
        for name in ['get_train_dataloader', '_load_rng_state', '_load_optimizer_and_scheduler',
                     '_init_training_state', '_prepare_for_training', '_run_epoch',
                     'set_initial_training_values', 'training_step', '_clip_grad_norm', '_save_checkpoint']:
            observed = provider_record(self.original_methods.get(name, getattr(self.trainer, name)))
            expected = expected_providers.get(observed['path'])
            require(expected is not None and observed['bytes'] == expected['bytes']
                    and observed['SHA256'] == expected['SHA256'], 'actual Trainer method provider is not frozen')
            active[name] = observed
        self.write('trainer-method-providers.json', active)
        require(state.global_step == 20 and args.max_steps == 25 and args.gradient_accumulation_steps == 4
                and args.per_device_train_batch_size == 1 and args.dataloader_num_workers == 4
                and args.dataloader_drop_last is False and args.ignore_data_skip is False
                and args.resume_only_model is False and args.seed == args.data_seed == 42,
                'native restored recipe/state boundary differs')
        require(args.bf16 is True and args.fp16 is False and self.trainer.template.max_length == 4096
                and not self.trainer.is_deepspeed_enabled and not self.trainer.is_fsdp_enabled,
                'native CUDA recipe precision/context/distributed mode differs')
        length = len(self.loader)
        # This is the exact acquired Trainer.set_initial_training_values formula, not64/8 guesswork.
        updates_per_epoch = max(length // 4 + int(length % 4 > 0), 1)
        trained_epochs = 20 // updates_per_epoch; skipped_batches = (20 % updates_per_epoch) * 4
        initial = self.entry_observations['initial_training_values']
        initialized = self.entry_observations['init_training_state']
        require(len(initial) == len(initialized) == 1, 'actual native entry return denominator differs')
        values = initial[0]['returned_values']
        require(len(values) == 7 and values[1] == updates_per_epoch and values[2] == 64
                and values[4] == 32 and values[5] == length and values[6] == 25
                and initial[0]['actual_loader_length'] == length
                and initialized[0]['returned_values'] == [trained_epochs, skipped_batches],
                'actual Trainer return/cursor differs from observed loader')
        self.write('actual-native-entry-returns.json', self.entry_observations)
        saved_state = self.ref(20, 'trainer_state.json')
        require(skipped_batches == 0 and saved_state['global_step'] == 20 and saved_state['max_steps'] == 25
                and float(saved_state['epoch']) == trained_epochs, 'actual loader does not establish checkpoint20 epoch boundary')
        self.write('loader.json', dict(actual_len=length, updates_per_epoch=updates_per_epoch,
                   actual_saved_epoch=saved_state['epoch'], trained_epochs=trained_epochs, skipped_microbatches=skipped_batches,
                   loader_class=type(self.loader).__module__ + '.' + type(self.loader).__qualname__,
                   batch_sampler_class=type(self.loader.batch_sampler).__module__ + '.' + type(self.loader.batch_sampler).__qualname__,
                   loader_provider=provider_record(type(self.loader)), sampler_provider=provider_record(type(self.loader.batch_sampler)),
                   trainer_dataloader_provider=provider_record(self.trainer.get_train_dataloader),
                   loader_scalar_attributes={k: v for k, v in getattr(self.loader, '__dict__', {}).items()
                                             if k in {'iteration', 'epoch', 'skip_batches', 'num_workers', 'batch_size', 'drop_last'}
                                             and (v is None or type(v) in (int, bool, float, str))},
                   sampler_scalar_attributes={k: v for k, v in getattr(self.loader.batch_sampler, '__dict__', {}).items()
                                              if k in {'epoch', 'seed', 'data_seed', 'rank', 'num_replicas', 'num_processes',
                                                       'batch_size', 'drop_last', 'shuffle', 'tp_size'}
                                              and (v is None or type(v) in (int, bool, float, str))},
                   runtime_sampler_identity_only_not_historical_microinput_reference=True))
        restored = self.state_comparison(20)
        require(all(restored[key]['exact'] for key in ('adapter', 'optimizer', 'scheduler')), 'native checkpoint20 state restore differs')
        self.write('restored-state-before-rng.json', dict(states=restored, RNG_phase='NOT_YET_RESTORED_BY_NATIVE_TRAINER'))
        self.adapter20_SHA256 = restored['adapter_live_SHA256']
        self.expected_rng20 = self.ref(20, f'rng_state_{self.rank}.pth')
        self.write('saved-rng-schema.json', dict(cuda_container=type(self.expected_rng20['cuda']).__name__,
                   cuda_states=(len(self.expected_rng20['cuda']) if isinstance(self.expected_rng20['cuda'], list) else 1),
                   source_save_policy='native Trainer._save_rng_state; select observer getter from safely decoded reference'))
        self.wrap('_load_rng_state', self.observe_rng)
        self.wrap('training_step', self.observe_training_step)
        self.wrap('_clip_grad_norm', self.observe_clip)
        if callable(getattr(self.loader, 'set_epoch', None)):
            original_epoch = self.loader.set_epoch
            @functools.wraps(original_epoch)
            def observed_epoch(epoch, *args, **kwargs):
                self.check(); returned = original_epoch(epoch, *args, **kwargs)
                self.loader_epoch_calls.append(dict(epoch=epoch,
                    loader_iteration=(getattr(self.loader, 'iteration', None)
                                      if type(getattr(self.loader, 'iteration', None)) in (int, type(None)) else 'NON_SCALAR'),
                    sampler_epoch=(getattr(self.loader.batch_sampler, 'epoch', None)
                                   if type(getattr(self.loader.batch_sampler, 'epoch', None)) in (int, type(None)) else 'NON_SCALAR')))
                self.write('actual-loader-set-epoch.json', self.loader_epoch_calls)
                return returned
            self.original_loader_set_epoch = original_epoch
            self.loader.set_epoch = observed_epoch
        self.handles.append(self.model.register_forward_pre_hook(self.before_forward, with_kwargs=True))
        self.progress()

    def wrap(self, name, observer):
        original = getattr(self.trainer, name); self.original_methods[name] = original
        @functools.wraps(original)
        def call(*args, **kwargs):
            try:
                self.check(); return observer(original, *args, **kwargs)
            except BaseException as error:
                self.progress(dict(type=type(error).__name__, reason=str(error)[:180])); raise
        setattr(self.trainer, name, call)

    def observe_rng(self, original, checkpoint):
        require(Path(checkpoint).resolve() == Path(self.config['checkpoint']).resolve() and not self.rng_loads,
                'native RNG restore call differs')
        result = original(checkpoint)  # Preserve exact native call/order/allowlist; never set RNG here.
        current = rng_fingerprint(self.current_rng())
        expected = rng_fingerprint(self.expected_rng20)
        require(current == expected, 'native RNG call-return restoration differs')
        self.rng_loads.append(dict(phase='NATIVE_LOAD_RNG_CALL_RETURN', exact=True, current_SHA256=current,
                                   expected_SHA256=expected, global_step=self.trainer.state.global_step))
        self.write('rng-restoration-point.json', self.rng_loads[-1]); self.progress()
        return result

    def before_forward(self, module, argv, kwargs):
        require(len(self.rng_loads) == 1 and len(self.forward_inputs) < 20, 'forward before RNG restore or exceeds denominator')
        before = rng_fingerprint(self.current_rng())
        value = tree_fingerprint(dict(args=argv, kwargs=kwargs))
        require(before == rng_fingerprint(self.current_rng()), 'input observer consumed RNG')
        self.forward_inputs.append(dict(microbatch=len(self.forward_inputs), input_SHA256=value,
                                       post_prefetch_RNG_SHA256=before, compare_to_checkpoint20_RNG=False))

    def observe_training_step(self, original, model, inputs, *args, **kwargs):
        require(len(self.inputs) < 20 and len(self.rng_loads) == 1, 'training microbatch denominator/restore differs')
        self.inputs.append(dict(microbatch=len(self.inputs), training_inputs_SHA256=tree_fingerprint(inputs),
                                global_step_before=self.trainer.state.global_step))
        result = original(model, inputs, *args, **kwargs)
        require(isinstance(result, torch.Tensor) and result.numel() == 1 and bool(torch.isfinite(result)), 'native loss return differs')
        require(len(self.forward_inputs) == len(self.inputs), 'each native microbatch needs one full model forward')
        self.losses.append(float(result.detach().cpu())); self.progress()
        return result

    def gradient_summary(self):
        total = 0.0; count = 0; digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                require(parameter.grad is not None, 'trainable gradient missing')
                value = parameter.grad.detach().cpu()
                require(bool(torch.isfinite(value).all()), 'nonfinite native gradient')
                total += float(value.to(torch.float64).square().sum()); count += 1
                digest.update(name.encode()); digest.update(tree_fingerprint(value).encode())
        require(count == len(self.binding['trainable_manifest']), 'gradient parameter denominator differs')
        return dict(parameters=count, L2=math.sqrt(total), SHA256=digest.hexdigest())

    def observe_clip(self, original, model):
        before = self.gradient_summary(); returned = original(model); after = self.gradient_summary()
        self.clips.append(dict(global_step=self.trainer.state.global_step + 1, pre=before, post=after,
                               native_norm=float(returned.detach().cpu()) if isinstance(returned, torch.Tensor) else float(returned)))
        return returned

    def pre_optimizer(self):
        require(len(self.inputs) == (len(self.steps) + 1) * 4 and len(self.clips) == len(self.steps) + 1,
                'update lacks four complete microbatches/clip boundary')
        self.check()
        self.write('update-%02d-before.json' % (self.trainer.state.global_step + 1),
                   dict(adapter_SHA256=tree_fingerprint(self.live_adapter()),
                        optimizer_SHA256=tree_fingerprint(self.trainer.optimizer.state_dict()),
                        scheduler_SHA256=tree_fingerprint(self.trainer.lr_scheduler.state_dict()), gradient=self.clips[-1]))

    def post_optimizer(self):
        step = self.trainer.state.global_step + 1
        optimizer_steps(self.trainer.optimizer.state_dict(), step, len(self.binding['trainable_manifest']))
        self.write('update-%02d-after-optimizer.json' % step,
                   dict(adapter_SHA256=tree_fingerprint(self.live_adapter()),
                        optimizer_SHA256=tree_fingerprint(self.trainer.optimizer.state_dict()),
                        scheduler_phase='NATIVE_SCHEDULER_NOT_STEPPED_YET'))

    def step_end(self):
        step = self.trainer.state.global_step
        require(step == 21 + len(self.steps), 'native actual update sequence differs')
        self.steps.append(step)
        self.write('update-%02d-after-scheduler.json' % step,
                   dict(global_step=step, scheduler_SHA256=tree_fingerprint(self.trainer.lr_scheduler.state_dict()),
                        optimizer_SHA256=tree_fingerprint(self.trainer.optimizer.state_dict()), epoch=self.trainer.state.epoch))
        self.progress()

    def saved(self):
        require(self.trainer.state.global_step == 25, 'unexpected new checkpoint step')
        self.saved_steps.append(25); self.progress()

    def finish(self):
        self.check()
        require(len(self.inputs) == len(self.forward_inputs) == len(self.losses) == 20
                and self.steps == [21, 22, 23, 24, 25] and len(self.clips) == 5
                and self.saved_steps == [25] and len(self.rng_loads) == 1, 'native resume completion denominator differs')
        self.write('microbatches.json', dict(training_inputs=self.inputs, model_forward_inputs=self.forward_inputs,
                                            native_training_step_loss_returns=self.losses,
                                            historical_per_micro_reference_available=False))
        wait = max(0.001, min(60.0, remaining(self.config) - 5.0))
        work = torch.distributed.barrier(device_ids=[torch.cuda.current_device()], async_op=True)
        require(work.wait(timeout=datetime.timedelta(seconds=wait)) is not False, 'bounded final rank barrier failed')
        compared = self.state_comparison(25)
        compared['adapter_changed_from_checkpoint20'] = compared['adapter_live_SHA256'] != self.adapter20_SHA256
        output = Path(self.trainer.args.output_dir) / 'checkpoint-25'
        new_rng = output / f'rng_state_{self.rank}.pth'
        require(new_rng.is_file() and not new_rng.is_symlink() and 0 < new_rng.stat().st_size <= 65536,
                'new rank RNG checkpoint missing')
        with safe_globals():
            saved_rng = torch.load(new_rng, map_location='cpu', weights_only=True)
        original_rng = self.ref(25, f'rng_state_{self.rank}.pth')
        compared['saved_rank_RNG_exact'] = rng_fingerprint(saved_rng) == rng_fingerprint(original_rng)
        # All ranks validate their own RNG. Root's independent collector requires all8 records.
        self.write('final-original25-comparison.json', compared)
        if self.rank == 0:
            saved_matches_live = {}
            for name, key in [('adapter_model.safetensors', 'adapter'), ('optimizer.pt', 'optimizer'), ('scheduler.pt', 'scheduler')]:
                path = output / name
                require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 512 * 1024**2,
                        'new typed checkpoint member missing')
                before = path.stat()
                if name.endswith('.safetensors'):
                    saved_value = load_safetensors(str(path), device='cpu')
                else:
                    with safe_globals():
                        saved_value = torch.load(path, map_location='cpu', weights_only=True)
                saved_matches_live[key] = tree_fingerprint(saved_value) == compared[key + '_live_SHA256']
                del saved_value
                after = path.stat()
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                        'new checkpoint changed during safe decoding')
            require(all(saved_matches_live.values()), 'new saved checkpoint does not match actual final live states')
            self.write('new-checkpoint25-live-state.json', saved_matches_live)
            hashes = {}
            for name in sorted(MEMBERS):
                path = output / name; require(path.is_file() and not path.is_symlink(), 'new full checkpoint member missing')
                pin = dict(bytes=path.stat().st_size, SHA256='0' * 64)
                require(0 < pin['bytes'] <= 512 * 1024**2, 'new checkpoint member size differs')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        self.check(); digest.update(block)
                hashes[name] = dict(bytes=pin['bytes'], SHA256=digest.hexdigest())
            self.write('new-checkpoint25-files.json', dict(path=str(output), files=hashes,
                        qualification='actual14 saved members and hashes; typed original25 comparisons are separate'))
        self.finished = True; self.progress()
        exact = all(compared[k]['exact'] for k in ('adapter', 'optimizer', 'scheduler')) and compared['saved_rank_RNG_exact']
        self.write('receipt.json', dict(rank=self.rank, completed_global_step=25, microbatches=20, updates=5,
                   own_restore_engineering_verified=True, original25_typed_state_exact=exact,
                   original_tail_per_micro_match_qualified=False, formal_numerics=False, formal_performance=False,
                   numerical_comparison=compared))
        self.detach()

    def detach(self):
        for handle in self.handles:
            handle.remove()
        for name, original in self.original_methods.items():
            setattr(self.trainer, name, original)
        if hasattr(self, 'original_loader_set_epoch'):
            self.loader.set_epoch = self.original_loader_set_epoch
