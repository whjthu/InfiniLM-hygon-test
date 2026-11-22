"""One rank-local default HIP generator intervention for a fresh native process."""

from hashlib import sha256
import math
import os
from pathlib import Path


ARMS = ('a1', 'b', 'a2')
SEED_BASE = 0x4D43504D3230
ROTARY = ('inv_freq', 'original_inv_freq', 'attention_scaling', 'max_seq_len_cached', 'original_max_seq_len')
CACHE = ('cache', 'past_key_values', 'rope_deltas', 'rotary_cache', 'kv_cache', '_cache')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def raw_sha(tensor, torch):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return sha256(memoryview(value)).hexdigest()


def rng_snapshot(torch, rng_values, rng_fingerprint):
    state = rng_values(torch.cuda.get_rng_state_all)
    device = state['cuda']
    need(len(device) == 8, 'expected eight native default device generators')
    return device, dict(full=rng_fingerprint(state), non_device=rng_fingerprint(
        {key: value for key, value in state.items() if key != 'cuda'}),
        devices=[raw_sha(value, torch) for value in device])


def apply_rng(torch, rng_values, rng_fingerprint, restored, rank, arm):
    need(arm in ARMS and 0 <= rank < 8 and torch.cuda.current_device() == rank, 'wrong rank or RNG arm')
    states, before = rng_snapshot(torch, rng_values, rng_fingerprint)
    need(before['full'] == restored, 'default generators changed after native checkpoint restore')
    state = states[rank]
    seed = None
    if arm == 'b':
        seed = SEED_BASE + rank
        generator = torch.Generator(device=torch.device('cuda', rank))
        generator.manual_seed(seed)
        state = generator.get_state()
    # Both control arms execute the same setter using the exact original state.
    requested = raw_sha(state, torch)
    torch.cuda.set_rng_state(state, device=rank)
    _, after = rng_snapshot(torch, rng_values, rng_fingerprint)
    check_transition(before, after, arm, rank, requested)
    return dict(arm=arm, rank=rank, seed=seed, before=before, after=after, requested_device_SHA256=requested)


def check_transition(before, after, arm, rank, requested):
    need(arm in ARMS and 0 <= rank < 8 and len(before['devices']) == len(after['devices']) == 8,
         'malformed generator boundary')
    changed = [r for r in range(8) if before['devices'][r] != after['devices'][r]]
    expected = [rank] if arm == 'b' else []
    need(before['non_device'] == after['non_device'] and changed == expected
         and (before['full'] != after['full']) == bool(expected)
         and after['devices'][rank] == requested, 'RNG intervention escaped its single device or had no effect')


def atom(value):
    if type(value) in (str, int, bool, type(None)) or type(value) is float and math.isfinite(value):
        return value
    raise TypeError('unobserved input or module scalar type: ' + type(value).__name__)


def describe_inputs(value, torch):
    if isinstance(value, torch.Tensor):
        return dict(tensor=True, identity=id(value), version=value._version, dtype=str(value.dtype),
                    shape=list(value.shape), stride=list(value.stride()), device=str(value.device))
    if isinstance(value, dict):
        need(all(type(k) is str for k in value), 'ambiguous native input keys')
        return {key: describe_inputs(member, torch) for key, member in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return dict(sequence=type(value).__name__, items=[describe_inputs(item, torch) for item in value])
    return atom(value)


def observed_model(torch, model, check_time):
    """Small registered buffers and named rotary state, plus local parameter versions."""
    parameter_ids = {name: (id(p), p._version) for name, p in model.named_parameters()}
    modes = {}
    values = {}
    opaque = []
    total = 0

    def describe(value):
        nonlocal total
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
            need(total <= 64 * 2**20, 'registered buffer/rotary observation byte bound')
            return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                        SHA256=raw_sha(value, torch))
        return atom(value)

    for name, module in model.named_modules():
        check_time()
        modes[name] = bool(module.training)
        values[name] = dict(buffers={key: describe(value) for key, value in sorted(module._buffers.items())},
                            rotary={}, cache_attrs={})
        if 'RotaryEmbedding' in type(module).__name__:
            values[name]['rotary'] = {key: describe(getattr(module, key)) for key in ROTARY
                                      if key not in module._buffers and hasattr(module, key)}
        for key in CACHE:
            if key in module.__dict__ and key not in module._buffers:
                value = module.__dict__[key]
                try:
                    values[name]['cache_attrs'][key] = describe(value)
                except TypeError:
                    opaque.append([name, key, type(value).__module__ + '.' + type(value).__qualname__])
    need(0 < len(parameter_ids) <= 50000 and 0 < len(modes) <= 20000, 'native model inventory out of bounds')
    return parameter_ids, dict(training=modes, observed_values=values, opaque_cache_attrs=opaque,
                               observed_bytes=total, parameter_count=len(parameter_ids))


def owner(proc=Path('/proc')):
    pid = os.getpid()
    fields = (proc / 'self/stat').read_text().rsplit(') ', 1)[1].split()
    identity = dict(pid=pid, startticks=int(fields[19]),
                    boot_id=(proc / 'sys/kernel/random/boot_id').read_text().strip())
    need(pid > 1 and identity['startticks'] > 0 and identity['boot_id'], 'missing strong rank process identity')
    return identity
