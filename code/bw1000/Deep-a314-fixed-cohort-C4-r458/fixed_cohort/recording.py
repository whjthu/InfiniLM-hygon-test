"""CPU only libffi leaves. No Native/SDK is loaded or executed."""
import ctypes as C
import json
from types import SimpleNamespace
import numpy as np
from .checkpoint_loader import declare


class Recording:
    def __init__(self, vocab=129280, fail_decode=False):
        self.vocab = vocab; self.ctx = 0x12345678ABCDEF01
        self.calls = []; self.lengths = [0]*4; self.decoded = [0]*4
        self.decode_calls = 0; self.reset_calls = 0; self.destroy_calls = 0
        self.fail_decode = fail_decode; self.keep = []
        self.fp32 = 0; self.hc = [0]*3; self.tail = [0]*5; self.layout = [0]*4
        self.lib = SimpleNamespace()
        P, I, Z = C.c_void_p, C.c_int, C.c_size_t
        signatures = {
            'modern_native_engine_create': ([C.c_char_p, Z, Z], P),
            'modern_native_engine_error': ([], C.c_char_p),
            'modern_native_engine_keys': ([P], C.c_char_p),
            'modern_native_engine_load': ([P, C.c_char_p], I),
            'modern_native_engine_forward': ([P, Z, Z] + [P]*8, I),
        }
        for name in ['finish', 'reset', 'destroy', 'rank_logits_equal']:
            signatures['modern_native_engine_'+name] = ([P], I)
        for name in ['deep_hc_canonical_policy_r7130', 'modern_deep_c4_tile_policy_r5920', 'modern_deep_allselected_set_r813']:
            signatures[name] = ([I], I)
        for name in ['deep_hc_canonical_get_r7130', 'modern_deep_layout_get_r7052', 'deep_moe_tail_get_r7211',
                     'modern_deep_c4_tile_get_r5920', 'deep_hc_model_epilogue_get_r366', 'deep_hc_fullpre_enabled_r377']:
            signatures[name] = ([], I)
        for name, count in [('deep_output_tile_calls_r4808', 1), ('modern_deep_c4_tile_calls_r5920', 2),
                ('modern_deep_FP32_calls_r3597', 1), ('deep_hc_canonical_calls_r7130', 2),
                ('deep_moe_tail_calls_r7211', 2), ('modern_deep_layout_calls_r7052', 2)]:
            signatures[name] = ([I]*count, C.c_uint64)
        self.callback_errors = []
        for name, (args, result) in signatures.items():
            callback_result = P if result is C.c_char_p else result
            def callback(*args, name=name):
                try:
                    return self.leaf(name, *args)
                except BaseException as e:
                    self.callback_errors.append(repr(e)); return 1
            cb = C.CFUNCTYPE(callback_result, *args)(callback)
            fn = C.CFUNCTYPE(callback_result, *args)(C.cast(cb, P).value)
            fn.argtypes, fn.restype = args, result
            self.keep.extend([cb, fn]); setattr(self.lib, name, fn)
        self.binding = declare(self.lib)

    def text(self, value):
        b = C.create_string_buffer(value.encode()); self.keep.append(b)
        return C.addressof(b)

    def leaf(self, name, *a):
        if name == 'modern_native_engine_create':
            assert a[1:] == (4, 32768)
            self.calls.append(dict(kind='create', rows=4, capacity_per_row=32768))
            return self.ctx
        if name == 'modern_native_engine_error': return self.text('recorded Native leaf failure')
        if name == 'modern_native_engine_keys': return self.text('[]')
        if name.startswith('modern_native_engine_'):
            assert a[0] == self.ctx, 'high ctx truncated'
        if name == 'modern_native_engine_reset':
            self.lengths = [0]*4; self.decoded = [0]*4; self.reset_calls += 1
            self.calls.append(dict(kind='reset')); return 0
        if name == 'modern_native_engine_destroy':
            self.destroy_calls += 1; self.calls.append(dict(kind='destroy')); return 0
        if name == 'modern_native_engine_forward':
            _, tokens, count, *p = a
            rows = list((C.c_int32*count).from_address(p[0]))
            past = list((C.c_int32*count).from_address(p[1]))
            total = list((C.c_int32*count).from_address(p[2]))
            offsets = list((C.c_int32*(count+1)).from_address(p[3]))
            positions = list((C.c_int64*tokens).from_address(p[5]))
            assert len(set(rows)) == count and offsets[0] == 0 and offsets[-1] == tokens
            widths = [b-a for a,b in zip(offsets, offsets[1:])]
            decode = all(n == 1 and v > 0 for n,v in zip(widths,past))
            for i, row in enumerate(rows):
                assert row in range(4) and past[i] == self.lengths[row]
                assert total[i] == past[i] + widths[i] <= 32768
                assert positions[offsets[i]:offsets[i+1]] == list(range(past[i], total[i]))
            self.calls.append(dict(kind='forward', rows=rows, past=past, widths=widths, decode=decode))
            if decode:
                self.decode_calls += 1
                if self.fail_decode: return 1
            self.fp32 += 43 * int(count > 1)
            self.hc[0 if tokens == 512 else 2] += 86
            target = count == 1 and tokens == 512 and all(v == 0 for v in past)
            self.tail[3 if target else 4] += 43; self.layout[3] += 43
            np.ctypeslib.as_array((C.c_uint16*(count*self.vocab)).from_address(p[6]))[:] = 0
            output = (C.c_int64*count).from_address(p[7])
            for i,row in enumerate(rows):
                if decode: self.decoded[row] += 1
                output[i] = 1 if decode and self.decoded[row] >= row+2 else 100+row
                self.lengths[row] = total[i]
            return 0
        if name == 'modern_deep_FP32_calls_r3597': return self.fp32
        if name == 'deep_hc_canonical_calls_r7130': return self.hc[a[1]]
        if name == 'deep_moe_tail_calls_r7211': return self.tail[a[1]]
        if name == 'modern_deep_layout_calls_r7052': return self.layout[a[1]]
        return 0


class RecordingTorch:
    cuda = SimpleNamespace(mem_get_info=lambda device: (30*1024**3, 64*1024**3))
