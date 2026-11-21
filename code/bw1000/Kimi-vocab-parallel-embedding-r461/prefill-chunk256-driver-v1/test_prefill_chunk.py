import ctypes as C
import time
from types import SimpleNamespace
import unittest

import numpy as np
import model_driver


class Recorder:
    def __init__(self, prompt):
        self.prompt = prompt
        self.calls = []
        self.events = []
        self.resets = 0
        self.lib = SimpleNamespace()
        setattr(self.lib, model_driver.SYMBOL, SimpleNamespace())
        for name in ('load', 'finish', 'reset'):
            setattr(self.lib, 'modern_kimi_engine_' + name + '_r2475', SimpleNamespace())
        self.lib.modern_kimi_engine_error_r2475 = lambda: b'recording failure'
        self.lib.modern_kimi_shard_head_expected_r9183 = SimpleNamespace()
        self.lib.modern_kimi_shard_head_error_r9183 = SimpleNamespace()
        self.library = self.create_functions()

    def create_functions(self):
        forward = C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int,C.c_size_t,C.c_size_t,C.c_int,
                              *([C.c_void_p]*9))(self.forward)
        reset = C.CFUNCTYPE(C.c_int,C.c_void_p)(self.reset)
        setattr(self.lib, model_driver.SYMBOL, forward)
        self.lib.modern_kimi_engine_reset_r2475 = reset
        return self.lib

    def forward(self, ctx, kind, width, count, flags, *args):
        before = C.cast(args[1], C.POINTER(C.c_int32))[0]
        after = C.cast(args[2], C.POINTER(C.c_int32))[0]
        offsets = tuple((C.c_int32*2).from_address(args[3]))
        ids = tuple((C.c_int64*width).from_address(args[4]))
        positions = tuple((C.c_int64*width).from_address(args[5]))
        self.calls.append((kind, before, after, offsets, ids, positions))
        self.events.append(('native', kind, before, width))
        np.ctypeslib.as_array((C.c_uint16*163840).from_address(args[6])).fill(0x3f80)
        C.cast(args[7], C.POINTER(C.c_int64))[0] = (163585 if kind else 17 if after==len(self.prompt) else 5)
        C.cast(args[8], C.POINTER(C.c_int32))[0] = 3
        return 0

    def reset(self, ctx):
        self.resets += 1
        return 0

    def collect(self, label, packet):
        return [dict(packet, node=node) for node in range(4)]

    def run(self, width=512, first_guard=None):
        return model_driver.generate(self.library, 1, 0, self, self.prompt, 4096, time.monotonic()+40,
            lambda name, packet: self.events.append((name, packet)),
            memory_guard=lambda phase: self.events.append(('memory_guard', phase)),
            after_first_prefill=first_guard, prefill_chunk_tokens=width)


class ChunkTest(unittest.TestCase):
    def test_both_widths_replay_every_input_in_order_and_emit_only_after_full_prompt(self):
        for length in (1, 255, 256, 511, 512, 513, 1025, 28672):
            tokens = list(range(length))
            for width in (256,512):
                with self.subTest(length=length,width=width):
                    runner=Recorder(tokens)
                    out=runner.run(width)
                    prefill, decode = runner.calls[:-1], runner.calls[-1]
                    self.assertEqual(out['prefill_calls'], (length+width-1)//width)
                    self.assertEqual(out['prefill_chunk_tokens'],width)
                    actual_ids = [token for call in prefill for token in call[4]]
                    self.assertEqual(actual_ids,tokens)
                    for i, (kind,before,after,offsets,ids,positions) in enumerate(prefill):
                        self.assertEqual((kind,before,after),(0,i*width,min((i+1)*width,length)))
                        self.assertEqual(offsets,(0,len(ids)))
                        self.assertEqual(positions,tuple(range(before,after)))
                    self.assertEqual(decode,(1,length,length+1,(0,1),(17,),(length,)))
                    self.assertEqual(out['generated_ids'],[17,163585])
                    self.assertIsNone(out['backend_visible_decode_tps'])
                    self.assertTrue(out['natural_EOS'])
                    self.assertEqual(runner.resets,2)
                    emit = next(i for i,event in enumerate(runner.events) if event[0]=='emit')
                    self.assertEqual(sum(event[0]=='native' for event in runner.events[:emit]),len(prefill))

    def test_invalid_chunk_width_fails_before_any_native_call(self):
        for width in (0,128,257,513,'256',256.0,True,None):
            runner=Recorder([1,2])
            with self.assertRaisesRegex(ValueError,'Frozen prefill chunk size'):
                runner.run(width)
            self.assertEqual(runner.calls,[])
            self.assertEqual(runner.resets,0)

    def test_early_guard_failure_never_signals_a_completed_request(self):
        runner=Recorder(list(range(513)))
        def stop():
            raise RuntimeError('recorded first chunk guard failed')
        with self.assertRaisesRegex(RuntimeError,'first chunk guard'):
            runner.run(256,stop)
        self.assertEqual(len(runner.calls),1)
        self.assertEqual(len(runner.calls[0][4]),256)
        self.assertEqual(runner.resets,2)
        self.assertFalse(any(event[0]=='emit' for event in runner.events))

    def test_four_node_geometry_must_match(self):
        peers=[dict(node=i,prefill_chunk_tokens=256) for i in range(4)]
        model_driver.check_tp_prefill_width(peers,256)
        for bad in (peers[:3], peers[:3]+[peers[0]], peers[:3]+[dict(node=3,prefill_chunk_tokens=512)]):
            with self.assertRaisesRegex(ValueError,'All TP nodes'):
                model_driver.check_tp_prefill_width(bad,256)


if __name__ == '__main__':
    unittest.main()
