"""Optional private first-update evidence on the unchanged native VGGT path."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from . import hy_resume_checkpoint as common


def parameter_manifest(named):
    return [dict(name=n, shape=list(p.shape), dtype=str(p.dtype), numel=p.numel()) for n, p in named]


def validate_structure(model, config, binding):
    expected = dict(training_scope='camera_head', cache_frozen_aggregator_features=False,
                    max_steps=20, warmup_steps=5, checkpoint_steps=[10, 20, 25],
                    image_size=224, frames_per_step=2, cached_batches=4, cards=[0])
    if any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError('original VGGT camera-head recipe differs')
    if config.get('resume_from_checkpoint') or binding['case'] != 'T12.camera-head':
        raise ValueError('first-update recording requires fresh VGGT training')
    if common.json_sha256(config) != binding['config_canonical_SHA256']:
        raise ValueError('complete original VGGT configuration differs')
    if type(model).__name__ != 'VGGT' or model.aggregator.training or not model.camera_head.training:
        raise ValueError('native VGGT architecture or frozen/train mode differs')
    frozen = list(model.aggregator.parameters())
    if not frozen or any(p.requires_grad for p in frozen):
        raise ValueError('native aggregator is not fully frozen')
    named = [(n, p) for n, p in model.camera_head.named_parameters() if p.requires_grad]
    if parameter_manifest(named) != binding['trainable_manifest']:
        raise ValueError('complete native camera-head parameter manifest differs')
    if {id(p) for _, p in named} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('trainable parameters escape the native camera head')
    return named


class LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0

    def write(self, data):
        if self.count + len(data) > self.limit:
            raise ValueError('first-update artifact exceeds its byte budget')
        count = self.stream.write(data)
        self.count += count
        return count

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


class FirstUpdateRecorder:
    """Tensor-wise files bound host memory; this core also admits CPU fixtures."""
    def __init__(self, root, named, optimizer, *, epoch, deadline, device_ids, binding,
                 artifact_limit=256 * 1024**2, total_limit=12 * 1024**3):
        import torch
        self.root, self.named, self.optimizer = Path(root), list(named), optimizer
        self.epoch, self.deadline = epoch, deadline
        if (not all(math.isfinite(x) for x in (epoch, deadline)) or epoch <= 0
                or not 0 < deadline - epoch <= 3600):
            raise ValueError('invalid original recorder epoch')
        self.anchor, self.start_remaining = time.monotonic(), deadline - time.time()
        self.device_ids, self.binding = list(device_ids), binding
        if not 0 < artifact_limit <= 256 * 1024**2 or not 0 < total_limit <= 12 * 1024**3:
            raise ValueError('invalid recorder byte limits')
        self.artifact_limit, self.total_limit = artifact_limit, total_limit
        self.files, self.total, self.completed_steps = [], 0, 0
        self.current_step, self.phase, self.failed, self.first_complete = None, 'created', False, False
        self.diagnostic_seconds = 0.0
        if (not self.named or len({n for n, _ in self.named}) != len(self.named)
                or len({id(p) for _, p in self.named}) != len(self.named)):
            raise ValueError('empty or duplicate trainable inventory')
        if any(type(p) is not torch.nn.Parameter or not p.requires_grad or p.is_meta
               or not p.is_leaf or p.grad is not None for _, p in self.named):
            raise ValueError('expected fresh materialized trainable parameters')
        owned = [p for group in optimizer.param_groups for p in group['params']]
        if [id(p) for p in owned] != [id(p) for _, p in self.named] or optimizer.state:
            raise ValueError('expected exact fresh optimizer ownership')
        self.check()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        with (self.root / 'binding.json').open('x') as stream:
            json.dump(binding, stream, indent=2)
        self.progress()

    def check(self):
        if not self.epoch <= time.time() < self.deadline or time.monotonic() - self.anchor >= self.start_remaining:
            raise TimeoutError('original first-update epoch exhausted')
        if self.failed:
            raise RuntimeError('first-update recorder is poisoned')

    def progress(self):
        value = dict(planned_global_steps=25, planned_recorded_updates=1,
                     planned_tensor_groups=['initial', 'preclip', 'postclip', 'updated', 'optimizer'],
                     planned_fixed_files=['initial_optimizer.pt', 'input.pt', 'target.pt',
                                          'forward.pt', 'clip_norm.pt', 'optimizer_groups.pt'],
                     trainable_manifest=parameter_manifest(self.named), completed_global_steps=self.completed_steps,
                     current_global_step=self.current_step, first_update_complete=self.first_complete,
                     epoch=self.epoch, deadline=self.deadline, device_ids=self.device_ids,
                     phase=self.phase, failed=self.failed, files=self.files,
                     artifact_bytes=self.total, diagnostic_seconds=self.diagnostic_seconds)
        tmp = self.root / 'progress.tmp'
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(self.root / 'progress.json')

    def guard(self, function, *args, **kwargs):
        try:
            self.check()
            return function(*args, **kwargs)
        except BaseException as error:
            self.failed, self.phase = True, 'FAILED:' + type(error).__name__
            self.progress()
            raise

    def _cpu(self, value):
        import torch
        if isinstance(value, torch.Tensor):
            result = value.detach().cpu().contiguous()
            if result.is_floating_point() and not bool(torch.isfinite(result).all()):
                raise ValueError('nonfinite captured tensor')
            return result
        if isinstance(value, dict):
            return {k: self._cpu(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._cpu(v) for v in value)
        if value is None or type(value) in (bool, int, float, str, bytes):
            if type(value) is float and not math.isfinite(value):
                raise ValueError('nonfinite captured scalar')
            return value
        raise TypeError('unsupported private capture value')

    def _save(self, name, value):
        import torch
        self.check()
        started = time.monotonic()
        path = self.root / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        partial = path.with_suffix('.partial')
        with partial.open('xb') as stream:
            writer = LimitedWriter(stream, min(self.artifact_limit, self.total_limit - self.total))
            torch.save(self._cpu(value), writer)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for data in iter(lambda: stream.read(1024**2), b''):
                digest.update(data)
        size = partial.stat().st_size
        os.link(partial, path)
        partial.unlink()
        self.total += size
        self.files.append(dict(name=name, bytes=size, SHA256=digest.hexdigest()))
        self.diagnostic_seconds += time.monotonic() - started
        with (self.root / 'files.ndjson').open('a') as stream:
            stream.write(json.dumps(self.files[-1]) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _group(self, stage, gradients=False):
        for index, (name, parameter) in enumerate(self.named):
            value = parameter.grad if gradients else parameter
            if value is None:
                raise ValueError('missing native gradient: ' + name)
            self._save(stage + '/%04d.pt' % index, dict(name=name, value=value))

    def enter(self, step, batch):
        self.check()
        if type(step) is not int or step != self.completed_steps + 1 or not 1 <= step <= 25:
            raise ValueError('native global update order differs')
        self.current_step = step
        if step != 1:
            return
        if self.optimizer.state or any(p.grad is not None for _, p in self.named):
            raise ValueError('first recorded update is not fresh')
        self.phase = 'initial'
        self._group('initial')
        self._save('initial_optimizer.pt', self.optimizer.state_dict())
        import numpy as np
        import torch
        numpy = np.random.get_state()
        rng = dict(python=random.getstate(), numpy=dict(algorithm=numpy[0], keys=numpy[1].tolist(),
                   position=int(numpy[2]), has_gauss=int(numpy[3]), cached_gaussian=float(numpy[4])),
                   torch_CPU=torch.get_rng_state(), device={})
        for device in self.device_ids:
            torch.cuda.synchronize(device)
            rng['device'][device] = torch.cuda.get_rng_state(device).cpu()
        self.batch = batch
        self.input_SHA256 = common.tree_sha256(batch)
        self._save('input.pt', dict(batch=batch, RNG=rng, input_SHA256=self.input_SHA256))
        self.phase = 'input'
        self.progress()

    def targets(self, target, valid_frames):
        if self.current_step == 1:
            if self.phase != 'input':
                raise ValueError('target capture order differs')
            self._save('target.pt', dict(pose_encoding=target, valid_frames=valid_frames))
            self.phase = 'target'

    def forward(self, predictions, loss):
        if self.current_step == 1:
            if self.phase != 'target' or common.tree_sha256(self.batch) != self.input_SHA256:
                raise ValueError('native target order or input values changed')
            self._save('forward.pt', dict(pose_enc_list=predictions['pose_enc_list'], loss=loss))
            del self.batch
            self.phase = 'forward'
            self.progress()

    def gradients(self, stage, norm=None):
        if self.current_step != 1:
            return
        if stage not in ('preclip', 'postclip') or self.phase != ('forward' if stage == 'preclip' else 'preclip'):
            raise ValueError('native gradient capture order differs')
        self._group(stage, gradients=True)
        if stage == 'postclip':
            self._save('clip_norm.pt', norm)
        self.phase = stage
        self.progress()

    def updated(self):
        if self.current_step == 1:
            if self.phase != 'postclip':
                raise ValueError('native update capture order differs')
            self._group('updated')
            for index, (name, parameter) in enumerate(self.named):
                state = self.optimizer.state.get(parameter)
                if not state or not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state) or float(state['step']) != 1:
                    raise ValueError('missing first native AdamW state')
                self._save('optimizer/%04d.pt' % index, dict(name=name, state=state))
            self._save('optimizer_groups.pt', self.optimizer.state_dict()['param_groups'])
            self.first_complete = True
        self.completed_steps = self.current_step
        self.phase = 'updated'
        self.progress()

    def finish(self, global_step, frozen_aggregator_unchanged):
        self.check()
        if global_step != 25 or self.completed_steps != 25 or not self.first_complete or not frozen_aggregator_unchanged:
            raise ValueError('native25 or original frozen aggregator did not complete')
        self.phase = 'NATIVE25_RETURNED'
        self.progress()


def create_vggt_recorder(model, optimizer, config, identity, evidence):
    path = os.environ.get('VGGT_FIRST_UPDATE_BINDING')
    if path is None:
        return None
    source = Path(path)
    if not source.is_file() or source.is_symlink() or not 0 < source.stat().st_size <= 2 * 1024**2:
        raise ValueError('VGGT first-update binding path or length differs')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['VGGT_FIRST_UPDATE_BINDING_SHA256']:
        raise ValueError('VGGT first-update binding identity differs')
    binding = json.loads(raw)
    named = validate_structure(model, config, binding)
    if (identity != binding['native_identity']
            or evidence['frozen_model']['weights_model_pt'] != binding['model_weights_SHA256']
            or evidence['data']['selected_batch_content_sha256'] != binding['selected_batch_SHA256']):
        raise ValueError('original native weight/input/source identity differs')
    import torch
    platform = binding['platform']
    if (platform not in ('cuda', 'hip') or bool(torch.version.hip) != (platform == 'hip')
            or (platform == 'cuda' and not torch.version.cuda) or torch.cuda.device_count() != 1
            or torch.cuda.current_device() != 0 or any(p.device.type != 'cuda' or p.device.index != 0 for _, p in named)):
        raise ValueError('expected one actual native rank-owned accelerator')
    return FirstUpdateRecorder(os.environ['VGGT_FIRST_UPDATE_OUTPUT'], named, optimizer,
        epoch=float(os.environ['VGGT_FIRST_UPDATE_EPOCH']), deadline=float(os.environ['VGGT_FIRST_UPDATE_DEADLINE']),
        device_ids=[0], binding=binding)
