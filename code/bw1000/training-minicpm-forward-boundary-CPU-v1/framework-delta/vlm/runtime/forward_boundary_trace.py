"""Opt-in, bounded output hashes for one native forward; no value export."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time


def validate_boundary_opt_in(enabled, entry_step, stop_step):
    if type(enabled) is not bool:
        raise ValueError('boundary trace opt-in must be an explicit boolean')
    if enabled and (type(entry_step) is not int or type(stop_step) is not int
                    or entry_step != 20 or stop_step != 21):
        raise ValueError('boundary trace requires first checkpoint20 tail and normal stop21')
    return enabled


def minicpm_boundaries(model):
    """Resolve only the original native52 boundaries, including all24 mixers."""
    core = model.get_base_model()
    mini_sha = 'a37ef94154e0b0ef3918ae60dfb43e871a5964d6a3cf48cd501dc36bf863119b'
    qwen_sha = '395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6'

    def native(module, class_name, expected):
        if type(module).__name__ != class_name:
            raise ValueError('boundary native class differs')
        path = Path(inspect.getfile(type(module)))
        if not 0 < path.stat().st_size <= 128 * 1024:
            raise ValueError('boundary native source exceeds bound')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('boundary native provider differs')

    native(core, 'MiniCPMV4_6ForConditionalGeneration', mini_sha)
    native(core.get_submodule('model.vision_tower'), 'MiniCPMV4_6VisionModel', mini_sha)
    native(core.get_submodule('model.merger'), 'MiniCPMV4_6Merger', mini_sha)
    language = core.get_submodule('model.language_model')
    native(language, 'Qwen3_5TextModel', qwen_sha)
    kinds = list(language.config.layer_types)
    if (len(language.layers) != 24 or len(kinds) != 24 or kinds.count('linear_attention') != 18
            or kinds.count('full_attention') != 6 or core.is_gradient_checkpointing):
        raise ValueError('boundary model inventory or checkpointing differs')
    names = ['model.language_model.embed_tokens', 'model.vision_tower', 'model.merger']
    for index, (layer, kind) in enumerate(zip(language.layers, kinds)):
        native(layer, 'Qwen3_5DecoderLayer', qwen_sha)
        attribute = 'linear_attn' if kind == 'linear_attention' else 'self_attn'
        native(getattr(layer, attribute),
               'Qwen3_5GatedDeltaNet' if kind == 'linear_attention' else 'Qwen3_5Attention', qwen_sha)
        names.extend(['model.language_model.layers.%d.%s' % (index, attribute),
                      'model.language_model.layers.%d' % index])
    names.append('model.language_model.norm')
    result = [(name, core.get_submodule(name)) for name in names]
    if len(result) != 52 or len({id(module) for _, module in result}) != 52:
        raise ValueError('boundary modules alias or are absent')
    return result


class ForwardBoundaryTrace:
    """Output-only hooks; caller supplies its unchanged absolute deadline/RNG.

    Hashes synchronize/copy device tensors and can perturb execution and kernel
    choice. The trace is diagnostic, never a formal performance measurement.
    """
    def __init__(self, modules, *, check, rng_snapshot, progress,
                 max_logical_bytes=2 * 1024**3):
        if (not callable(check) or not callable(rng_snapshot) or not callable(progress)
                or type(max_logical_bytes) is not int or not 0 < max_logical_bytes <= 2 * 1024**3
                or not 1 <= len(modules) <= 64):
            raise ValueError('invalid boundary trace callbacks or bounds')
        self.planned = [name for name, _ in modules]
        if (any(not isinstance(name, str) or not name or len(name) > 160 for name in self.planned)
                or len(set(self.planned)) != len(modules)
                or len({id(module) for _, module in modules}) != len(modules)):
            raise ValueError('duplicate or invalid boundary modules')
        self.check, self.rng_snapshot, self.progress = check, rng_snapshot, progress
        self.limit, self.copied = max_logical_bytes, 0
        self.records, self.handles, self.failed = [], [], False
        self.failure_error_type = None
        self.closed = False
        self.started = time.monotonic()
        self.observer_seconds = 0.0
        self._emit()
        try:
            for name, module in modules:
                self.handles.append(module.register_forward_hook(self._hook(name)))
        except BaseException as error:
            self.abort(error)
            raise

    def _emit(self):
        self.progress(dict(schema=1, planned_boundaries=self.planned, records=self.records,
                           completed_boundaries=len(self.records), planned_count=len(self.planned),
                           copied_logical_bytes=self.copied, max_logical_bytes=self.limit,
                           observer_seconds=self.observer_seconds, failed=self.failed,
                           closed=self.closed, complete=False, raw_values_exported=False,
                           failure_error_type=self.failure_error_type))

    def _hook(self, name):
        def observe(module, args, output):
            del args
            try:
                self.check()
                if self.closed or any(row['name'] == name for row in self.records):
                    raise ValueError('unexpected repeated or closed forward boundary')
                started = time.monotonic()
                before = self.rng_snapshot()
                # Ensure the callback returns summaries rather than raw RNG tensors.
                json.dumps(before, allow_nan=False)
                tensors, structure = self._inventory(output)
                size = sum(t.numel() * t.element_size() for _, t in tensors)
                if self.copied + size > self.limit:
                    raise ValueError('boundary logical-byte limit exceeded')
                from .forward_entry_state import tensor_digest
                summaries = []
                for path, value in tensors:
                    self.check()
                    digest = tensor_digest(value, check=self.check)
                    self.copied += digest['logical_bytes']
                    summaries.append(dict(path=path, shape=list(value.shape), dtype=str(value.dtype),
                                          stride=list(value.stride()), storage_offset=value.storage_offset(),
                                          **digest))
                after = self.rng_snapshot()
                if before != after:
                    raise ValueError('boundary observer changed RNG')
                values = [dict(path=r['path'], shape=r['shape'], dtype=r['dtype'],
                               SHA256=r['SHA256'], logical_bytes=r['logical_bytes']) for r in summaries]
                value_sha = hashlib.sha256(json.dumps(dict(structure=structure, tensors=values),
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                elapsed = time.monotonic() - started
                self.observer_seconds += elapsed
                self.records.append(dict(name=name, module_class=type(module).__module__ + '.' + type(module).__qualname__,
                    ordinal=len(self.records) + 1, tensors=summaries, structure=structure,
                    value_SHA256=value_sha, observer_RNG_unchanged=True, observer_seconds=elapsed))
                self._emit()
            except BaseException as error:
                self.abort(error)
                raise
            # Returning None preserves the native module output object and autograd graph.
            return None
        return observe

    def _inventory(self, output):
        import torch
        tensors = []
        visited = 0
        def visit(value, path, depth):
            nonlocal visited
            self.check()
            visited += 1
            if visited > 512 or depth > 16:
                raise ValueError('boundary output structure exceeds bound')
            if isinstance(value, torch.Tensor):
                if len(tensors) >= 128:
                    raise ValueError('boundary tensor count exceeds bound')
                tensors.append((path, value))
                return dict(tensor=path)
            if value is None or type(value) in (bool, int, float, str):
                if isinstance(value, str) and len(value) > 256:
                    raise ValueError('boundary scalar string exceeds bound')
                json.dumps(value, allow_nan=False)
                return dict(scalar=value)
            if isinstance(value, dict):
                if any(not isinstance(k, str) or len(k) > 128 for k in value):
                    raise ValueError('boundary mapping keys differ')
                return dict(mapping={k: visit(value[k], path + '/' + k, depth + 1) for k in sorted(value)})
            if isinstance(value, (tuple, list)):
                return dict(sequence=[visit(v, path + '/' + str(i), depth + 1) for i, v in enumerate(value)])
            raise ValueError('unsupported boundary output object')
        structure = visit(output, 'output', 0)
        if not tensors:
            raise ValueError('boundary output has no tensor')
        return tensors, structure

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.closed = True

    def abort(self, error):
        self.failed = True
        self.failure_error_type = type(error).__name__
        self.detach()
        self._emit()

    def finish(self):
        self.detach()
        if self.failed or {r['name'] for r in self.records} != set(self.planned):
            self.failed = True
            self.failure_error_type = self.failure_error_type or 'IncompleteBoundaryTrace'
            self._emit()
            raise ValueError('incomplete forward boundary trace')
        self.check()
        return dict(schema=1, complete=True, failed=False, planned_boundaries=self.planned,
                    planned_count=len(self.planned), completed_boundaries=len(self.records),
                    records=self.records, copied_logical_bytes=self.copied, max_logical_bytes=self.limit,
                    observer_seconds=self.observer_seconds, total_wall_seconds=time.monotonic() - self.started,
                    raw_values_exported=False, observer_RNG_unchanged=True,
                    numerical_acceptance=False, operator_cause_proven=False, formal_performance=False)


def compare_boundary_traces(left, right):
    """Locate the earliest observed value difference, not its operator cause."""
    for trace in (left, right):
        planned, rows = trace['planned_boundaries'], trace['records']
        if (trace.get('schema') != 1 or trace.get('complete') is not True or trace.get('failed') is not False
                or trace.get('raw_values_exported') is not False
                or type(trace.get('max_logical_bytes')) is not int
                or not 0 < trace['max_logical_bytes'] <= 2 * 1024**3
                or type(trace.get('copied_logical_bytes')) is not int
                or not 0 <= trace['copied_logical_bytes'] <= trace['max_logical_bytes']
                or len(planned) != len(set(planned)) or not 1 <= len(planned) <= 64
                or len(rows) != len(planned) or {r['name'] for r in rows} != set(planned)
                or trace['planned_count'] != len(planned) or trace['completed_boundaries'] != len(rows)
                or any(r['ordinal'] != i + 1 or r.get('observer_RNG_unchanged') is not True
                       or len(r['value_SHA256']) != 64
                       or any(c not in '0123456789abcdef' for c in r['value_SHA256'])
                       for i, r in enumerate(rows))):
            raise ValueError('incomplete or invalid boundary evidence')
    if left['planned_boundaries'] != right['planned_boundaries']:
        raise ValueError('boundary scope differs')
    a, b = left['records'], right['records']
    if [r['name'] for r in a] != [r['name'] for r in b]:
        return dict(state='OBSERVED_ORDER_DIFFERS', first_value_difference=None,
                    numerical_acceptance=False, operator_cause_proven=False)
    different = [i for i, (x, y) in enumerate(zip(a, b)) if x['value_SHA256'] != y['value_SHA256']]
    index = different[0] if different else None
    return dict(state='OBSERVED_BOUNDARY_VALUES_DIFFER' if different else 'OBSERVED_BOUNDARY_VALUES_EQUAL',
                boundary_denominator=len(a), equal_boundaries=len(a) - len(different),
                first_value_difference=None if index is None else dict(ordinal=index + 1, name=a[index]['name']),
                numerical_acceptance=False, operator_cause_proven=False, formal_performance=False)
