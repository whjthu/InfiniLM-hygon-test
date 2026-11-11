"""Two original-input forward arms; diagnostic-only, no optimizer update."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import types
import numpy as np
import torch
from swift.callbacks import TrainerCallback, callbacks_map
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vlm.runtime.dropout_capture import replay_dropout, validate_record
from vlm.runtime.initial_trainables import named_parameter_sha256, restore_initial_trainables
from vlm.runtime.lora_dropout_session import input_identity
from vlm.runtime.minicpm_recipe_contract import validate_native_recipe
from vlm.runtime.provider_counterfactual import selected_callables

KEY = 'original_provider_counterfactual'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().clone()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('nonfinite forward output')
        return value
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('unsupported diagnostic value')


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def metrics(actual, reference, check):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('output shape or dtype changed')
    x = actual.reshape(-1); y = reference.reshape(-1)
    square = norm = 0.; maximum = 0.; unequal = 0
    for start in range(0, x.numel(), 1024**2):
        check(); a = x[start:start+1024**2].double(); b = y[start:start+1024**2].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite metric operands')
        delta = a-b; square += float(delta.square().sum()); norm += float(b.square().sum())
        maximum = max(maximum, float(delta.abs().max())); unequal += int((a != b).sum())
    return dict(shape=list(actual.shape), dtype=str(actual.dtype), max_abs=maximum,
                relative_l2=(square/norm)**.5 if norm else None, unequal_elements=unequal,
                exact=unequal == 0)


class OriginalProviderCounterfactual(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        path = Path(os.environ['FORWARD_PROVIDER_CONFIG'])
        assert file_sha(path) == os.environ['FORWARD_PROVIDER_CONFIG_SHA256']
        config = json.loads(path.read_bytes())
        epoch = float(os.environ['FORWARD_PROVIDER_EPOCH']); deadline = float(os.environ['FORWARD_PROVIDER_DEADLINE'])
        assert deadline == epoch+1080 and epoch <= time.time() < deadline
        def check():
            if time.time() >= deadline:
                raise TimeoutError('single-epoch forward diagnostic exhausted')
        def expired(*_):
            raise TimeoutError('forward diagnostic hard deadline')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
        rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
        assert config['case'] == 'T05.sft' and int(os.environ['WORLD_SIZE']) == 8 and rank == local
        assert torch.version.cuda == '12.9' and not torch.version.hip and torch.cuda.current_device() == local
        output = Path(config['output']) / ('rank-%02d' % rank)
        output.mkdir(parents=True, exist_ok=False)
        owner = dict(PID=os.getpid(), PGID=os.getpgrp(), rank=rank, local_rank=local,
                     startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                     epoch=epoch, deadline=deadline)
        stages = []; stage = 'initialize'; completed = []
        def save(name, value):
            with (output/name).open('x') as stream:
                json.dump(value, stream, indent=2)
        def progress():
            temp = output/'progress.tmp'
            temp.write_text(json.dumps(dict(owner=owner,stage=stage,planned_arms=['native','torch_chunk'],completed_arms=completed,optimizer_updates=0)))
            temp.replace(output/'progress.json')
        save('owner.json', owner); progress()
        try:
            model = kwargs.get('model', self.trainer.model)
            named, frozen = validate_native_recipe(model, args, 10822656)
            assert state.global_step == 0 and model.training and all(p.grad is None for _, p in named)
            row = config['ranks'][rank]; assert row['rank'] == rank
            def load(kind):
                check(); item = row[kind]; p = Path(item['path'])
                assert p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes']
                assert file_sha(p) == item['SHA256']
                return torch.load(p, map_location='cpu', weights_only=True, mmap=True)
            initial = row['initial']
            restored = restore_initial_trainables(named, self.trainer.optimizer, global_step=0,
                path=initial['path'], file_bytes=initial['bytes'], file_SHA256=initial['SHA256'],
                named_SHA256=config['initial_named_SHA256'])
            save('initialization.json', restored)
            captured = load('input'); original = load('output')['result']; packet = load('mask')
            assert captured['global_step'] == packet['identity']['global_step'] == 0
            assert captured['micro_ordinal'] == packet['identity']['micro_ordinal'] == 1
            assert packet['contract']['rank'] == rank and packet['contract']['world_size'] == 8
            assert input_identity(captured['args'], captured['kwargs']) == packet['identity']['input_SHA256']
            assert not captured['kwargs'].get('use_cache', False)
            records = packet['records']; dropouts = [(n, m) for n, m in model.named_modules() if '.lora_dropout.' in n]
            assert len(records) == len(dropouts) == 186
            assert {x['module'] for x in records} == {n for n, _ in dropouts}
            bindings = [(n, m) for n, m in model.named_modules()
                        if type(m).__module__.startswith('transformers.models.qwen3_5.')
                        and hasattr(m, 'chunk_gated_delta_rule')]
            assert len(bindings) == 18
            source = Path(inspect.getfile(type(bindings[0][1])))
            assert file_sha(source) == config['model_source_SHA256']
            defining = sys.modules[type(bindings[0][1]).__module__]
            native = defining.chunk_gated_delta_rule; fallback = defining.torch_chunk_gated_delta_rule
            assert callable(native) and callable(fallback) and native is not fallback
            assert all(m.chunk_gated_delta_rule is native for _, m in bindings)
            buffers = [(n, b, b.detach().cpu().clone()) for n, b in model.named_buffers()]
            assert sum(v.numel()*v.element_size() for _, _, v in buffers) <= 64*1024**2
            original_buffer_names = [n for n, _, _ in buffers]
            save('providers-before.json', dict(model_source_SHA256=file_sha(source),
                native=dict(module=native.__module__,name=native.__qualname__),
                fallback=dict(module=fallback.__module__,name=fallback.__qualname__),
                bound_instances=[n for n, _ in bindings],frozen=frozen,
                autocast_context='Existing prepared model forward and Trainer compute_loss_context_manager'))
            def restore_rng():
                rng = captured['RNG_before_forward']; random.setstate(rng['python'])
                nr = rng['numpy']; np.random.set_state((nr['algorithm'],np.asarray(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
                torch.set_rng_state(rng['torch_CPU']); torch.cuda.set_rng_state(rng['torch_device'],local)
            arm_outputs = {}
            for arm, replacement in [('native', None), ('torch_chunk', fallback)]:
                stage = arm; progress(); check()
                assert [n for n, _ in model.named_buffers()] == original_buffer_names
                with torch.no_grad():
                    for _, buffer, initial_buffer in buffers:
                        buffer.copy_(initial_buffer)
                assert named_parameter_sha256(named) == config['initial_named_SHA256'] and not self.trainer.optimizer.state
                restore_rng(); argv = to_device(captured['args'], torch.device('cuda',local)); kw = to_device(captured['kwargs'], torch.device('cuda',local))
                assert input_identity(argv,kw) == packet['identity']['input_SHA256']
                position = [0]; originals = []
                try:
                    for name, module in dropouts:
                        assert type(module) is torch.nn.Dropout and module.p == .05 and not module.inplace and 'forward' not in module.__dict__
                        originals.append((module,module.forward))
                        def replay(owner, value, *, _name=name):
                            i = position[0]; assert i < len(records)
                            r = records[i]; assert r['module'] == _name and r['call_ordinal'] == i
                            validate_record(value,r['native_record']); position[0] += 1
                            return replay_dropout(value,r['native_record'])
                        module.forward = types.MethodType(replay,module)
                    torch.cuda.synchronize(); assert torch.cuda.mem_get_info()[0] >= 8*1024**3
                    with selected_callables(bindings,replacement) as calls:
                        with torch.enable_grad(), self.trainer.compute_loss_context_manager():
                            result = model(*argv,**kw)
                        torch.cuda.synchronize()
                        assert position[0] == 186 and all(r['calls'] == 1 for r in calls)
                        cpu = to_cpu(dict(result)); del result
                        assert set(cpu) == set(original)
                        arm_outputs[arm] = cpu
                        for key, value in cpu.items():
                            if isinstance(value, torch.Tensor):
                                assert isinstance(original[key], torch.Tensor)
                                assert value.shape == original[key].shape and value.dtype == original[key].dtype
                        saved_path = output/(arm+'-output.pt')
                        with saved_path.open('xb') as stream:
                            torch.save(cpu,stream)
                        assert saved_path.stat().st_size <= 512*1024**2
                        save(arm+'-calls.json',dict(chunk_calls=calls,dropout_calls=position[0],actual_RNG_restored=True))
                    assert all(m.chunk_gated_delta_rule is native for _, m in bindings)
                finally:
                    for module, _ in originals:
                        del module.forward
                assert named_parameter_sha256(named) == config['initial_named_SHA256'] and not self.trainer.optimizer.state
                assert all(p.grad is None for _,p in named) and state.global_step == 0
                del argv,kw; gc.collect(); torch.cuda.synchronize()
                assert torch.cuda.mem_get_info()[0] >= 8*1024**3
                completed.append(arm); progress()
            stage='compare'; progress()
            comparisons = {}
            for label,x,y in [('native_vs_original',arm_outputs['native'],original),
                              ('torch_chunk_vs_original',arm_outputs['torch_chunk'],original),
                              ('torch_chunk_vs_native',arm_outputs['torch_chunk'],arm_outputs['native'])]:
                comparisons[label] = dict(loss=metrics(x['loss'],y['loss'],check),logits=metrics(x['logits'],y['logits'],check),
                                         actual_loss=float(x['loss']),reference_loss=float(y['loss']))
            save('result.json',dict(passed=True,owner=owner,planned_arms=2,completed_arms=completed,
                source_SHA256=file_sha(Path(__file__)),comparisons=comparisons,
                exact_native_reproduction=all(comparisons['native_vs_original'][k]['exact'] for k in ('loss','logits')),
                original_numerical_tolerances_available=False,numerical_acceptance=False,
                optimizer_updates=0,global_step=0,raw_tensor_transfer_bytes=0,Model_GO=False,
                qualification='Two full original-input forward arms. Similarity alone does not prove model numerical acceptance or heldout quality.'))
            stage='complete'; progress(); torch.distributed.barrier(); torch.distributed.destroy_process_group()
        except BaseException as error:
            if not isinstance(error,SystemExit):
                save('failure.json',dict(owner=owner,stage=stage,completed_arms=completed,planned_arms=['native','torch_chunk'],error_type=type(error).__name__,message=str(error)[:400],Model_GO=False))
            raise
        raise SystemExit(0)

if KEY in callbacks_map:
    raise ValueError('duplicate provider counterfactual callback')
callbacks_map[KEY] = OriginalProviderCounterfactual
