"""Ordered native LoRA dropout masks for bounded training-alignment diagnostics.

This session changes no probabilities and delegates every native sampler call.
Capture returns native outputs. Replay uses the real recorded masks while still
advancing the destination backend RNG. Callers must include recomputation in the
scope; the first Swift integration explicitly requires checkpointing disabled.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import types
from pathlib import Path
import torch
from .dropout_capture import capture_dropout, replay_dropout, validate_record


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def _pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def input_identity(args, kwargs):
    """Bind the full structured model input, including actual image/label/mask bytes."""
    h = hashlib.sha256()
    def walk(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            h.update(json.dumps(['tensor', str(value.dtype), list(value.shape)]).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError('model input keys must be strings')
            h.update(b'dict')
            for key in sorted(value):
                h.update(json.dumps(key).encode()); walk(value[key])
            h.update(b'end-dict')
        elif isinstance(value, (tuple, list)):
            h.update(type(value).__name__.encode())
            for item in value:
                walk(item)
            h.update(b'end-sequence')
        elif value is None or isinstance(value, (str, bool, int, float)):
            h.update(json.dumps([type(value).__name__, value], allow_nan=False).encode())
        else:
            raise ValueError('unsupported model input identity type: ' + type(value).__name__)
    walk(args); walk(kwargs)
    return h.hexdigest()


class LoRADropoutSession:
    """Record/replay complete ordered forward packets without changing model state."""
    def __init__(self, model, output, *, mode, binding, rank, world_size,
                 forwards=8, accumulation=4, reference=None, reference_manifest_SHA256=None,
                 expected_input_features=None, byte_budget=1024**3):
        if mode not in ('capture', 'replay') or (mode == 'replay') != (reference is not None):
            raise ValueError('expected capture or replay with an explicit reference')
        if not (isinstance(forwards, int) and forwards > 0 and accumulation > 0
                and forwards % accumulation == 0 and 0 <= rank < world_size and byte_budget > 0):
            raise ValueError('invalid rank, forward or accumulation budget')
        self.mode, self.binding, self.rank, self.world_size = mode, binding, rank, world_size
        self.forwards, self.accumulation, self.byte_budget = forwards, accumulation, byte_budget
        self.active = None; self.files = []; self.mask_bytes = 0; self.closed = False
        self.failed = False; self.originals = []; self.calls = 0
        self.modules = [(name, module) for name, module in model.named_modules()
                        if '.lora_dropout.' in name]
        if not self.modules:
            raise ValueError('no native LoRA dropout modules')
        self.inventory = []
        for name, module in self.modules:
            if (type(module) is not torch.nn.Dropout or module.p != .05 or module.inplace
                    or 'forward' in module.__dict__
                    or getattr(module.forward, '__func__', None) is not torch.nn.Dropout.forward):
                raise ValueError('LoRA dropout differs from native non-inplace p=.05: ' + name)
            self.inventory.append({'name': name, 'p': module.p, 'inplace': module.inplace})
        self.contract = {'version': 1, 'binding': binding, 'rank': rank, 'world_size': world_size,
                         'planned_forwards': forwards, 'accumulation': accumulation,
                         'module_inventory': self.inventory}
        # JSON serialization rejects nonprimitive identities before patching any module.
        json.dumps(self.contract, allow_nan=False)
        self.contract = json.loads(json.dumps(self.contract, allow_nan=False))
        self.expected_input_features = expected_input_features
        if expected_input_features is not None and set(expected_input_features) != {n for n, _ in self.modules}:
            raise ValueError('LoRA input feature binding differs from module inventory')
        self.reference = None if reference is None else Path(reference)
        if self.reference is not None:
            manifest = self.reference / 'complete.json'
            if reference_manifest_SHA256 is None or _sha(manifest) != reference_manifest_SHA256:
                raise ValueError('reference manifest identity differs')
            source = json.loads(manifest.read_bytes())
            if source['mode'] != 'capture':
                raise ValueError('reference must contain actual native capture packets')
            if source['contract'] != self.contract or len(source['files']) != forwards:
                raise ValueError('reference model/rank/recipe/module/forward contract differs')
            self.reference_files = source['files']
            for ordinal, row in enumerate(self.reference_files, 1):
                if (row['name'] != f'forward-{ordinal:04d}-masks.pt'
                        or not 0 < row['bytes'] <= byte_budget or len(row['SHA256']) != 64):
                    raise ValueError('invalid reference packet manifest')
        self.output = Path(output); self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._progress()
        for name, module in self.modules:
            original = module.forward
            def forward(owner, value, *, _name=name, _original=original):
                if not owner.training:
                    return _original(value)
                if owner.p != .05 or owner.inplace:
                    raise ValueError('native LoRA dropout settings changed during the diagnostic')
                return self._dropout(_name, value)
            module.forward = types.MethodType(forward, module)
            self.originals.append(module)

    def _progress(self, error=None):
        value = {'contract': self.contract, 'mode': self.mode, 'files': self.files,
                 'completed_forwards': len(self.files), 'planned_forwards': self.forwards,
                 'native_dropout_calls': self.calls, 'mask_bytes': self.mask_bytes,
                 'active_ordinal': self.active['micro_ordinal'] if self.active else None,
                 'active_completed_calls': len(self.active['observed']) if self.active else 0,
                 'failure': error}
        p = self.output / 'progress.tmp'; p.write_bytes(json.dumps(value).encode())
        p.replace(self.output / 'progress.json')

    def begin_forward(self, *, global_step, micro_ordinal, input_SHA256):
        if self.closed or self.failed or self.active is not None:
            raise ValueError('closed, failed or nested forward scope')
        if (micro_ordinal != len(self.files) + 1 or micro_ordinal > self.forwards
                or global_step != (micro_ordinal - 1) // self.accumulation
                or len(input_SHA256) != 64 or any(c not in '0123456789abcdef' for c in input_SHA256)):
            raise ValueError('forward ordinal, update or complete input identity differs')
        active = {'global_step': global_step, 'micro_ordinal': micro_ordinal,
                  'input_SHA256': input_SHA256, 'observed': []}
        if self.mode == 'replay':
            info = self.reference_files[micro_ordinal - 1]; p = self.reference / info['name']
            if p.stat().st_size != info['bytes'] or _sha(p) != info['SHA256']:
                raise ValueError('reference packet bytes differ')
            packet = torch.load(p, map_location='cpu', weights_only=True)
            if (packet['contract'] != self.contract or packet['identity'] != {k: active[k] for k in
                    ('global_step', 'micro_ordinal', 'input_SHA256')} or not packet['records']):
                raise ValueError('reference packet identity differs')
            active['records'] = packet['records']
        self.active = active; self._progress()

    def _dropout(self, name, value):
        if self.closed or self.failed or self.active is None:
            raise RuntimeError('native training dropout outside an active diagnostic scope')
        if self.expected_input_features is not None and (value.ndim < 1 or value.shape[-1] != self.expected_input_features[name]):
            raise ValueError('native LoRA input width differs from bound weight')
        active = self.active; ordinal = len(active['observed'])
        if self.mask_bytes + value.numel() > self.byte_budget:
            raise RuntimeError('complete mask byte budget exhausted')
        if self.mode == 'capture':
            output, record = capture_dropout(value, .05)
        else:
            if ordinal >= len(active['records']):
                raise ValueError('extra native LoRA dropout call')
            item = active['records'][ordinal]
            if item['module'] != name or item['call_ordinal'] != ordinal:
                raise ValueError('LoRA module call order differs')
            record = item['native_record']
            validate_record(value, record)
            if record['probability'] != .05:
                raise ValueError('record changed frozen dropout probability')
            output = replay_dropout(value, record)
        active['observed'].append({'module': name, 'call_ordinal': ordinal, 'native_record': record})
        self.mask_bytes += value.numel(); self.calls += 1
        return output

    def end_forward(self):
        if self.active is None or self.closed or self.failed:
            raise ValueError('no active forward')
        active = self.active
        if not active['observed'] or (self.mode == 'replay' and len(active['observed']) != len(active['records'])):
            raise ValueError('missing native LoRA dropout calls')
        identity = {k: active[k] for k in ('global_step', 'micro_ordinal', 'input_SHA256')}
        # Capture packets contain full real masks. Replay writes only an audit receipt.
        name = f"forward-{active['micro_ordinal']:04d}-masks.pt"
        if self.mode == 'capture':
            p = self.output / name
            with p.open('xb') as stream:
                torch.save({'contract': self.contract, 'identity': identity,
                            'records': active['observed']}, stream)
            if p.stat().st_size > self.byte_budget:
                raise RuntimeError('serialized mask packet budget exhausted')
            row = {'name': name, 'bytes': p.stat().st_size, 'SHA256': _sha(p)}
        else:
            row = {'name': name, 'reference_SHA256': self.reference_files[active['micro_ordinal'] - 1]['SHA256']}
        row.update(identity); row['native_calls'] = len(active['observed'])
        row['mask_identity_SHA256'] = _pin([{'module': x['module'], 'call_ordinal': x['call_ordinal'],
                                           'mask_SHA256': x['native_record']['mask_SHA256']}
                                          for x in active['observed']])
        self.files.append(row); self.active = None; self._progress()

    def abort(self, error):
        self.failed = True; self._progress({'type': type(error).__name__, 'message': str(error)[:300]})
        self._restore(); self.closed = True

    def _restore(self):
        for module in self.originals:
            del module.forward
        self.originals.clear()

    def close(self):
        if self.closed or self.failed:
            raise ValueError('already closed dropout session')
        if self.active is not None or len(self.files) != self.forwards:
            error = ValueError('incomplete dropout session')
            self.abort(error)
            raise error
        value = {'contract': self.contract, 'mode': self.mode, 'files': self.files,
                 'native_dropout_calls': self.calls, 'mask_bytes': self.mask_bytes,
                 'formal_performance': False}
        with (self.output / 'complete.json').open('x') as stream:
            json.dump(value, stream, indent=2)
        self._restore(); self.closed = True
        return value

    @contextlib.contextmanager
    def forward(self, *, global_step, micro_ordinal, input_SHA256):
        try:
            self.begin_forward(global_step=global_step, micro_ordinal=micro_ordinal, input_SHA256=input_SHA256)
            yield
            self.end_forward()
        except BaseException as error:
            self.abort(error)
            raise
