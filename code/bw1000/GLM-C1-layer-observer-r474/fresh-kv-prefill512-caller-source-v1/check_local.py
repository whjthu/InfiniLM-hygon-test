"""Verify the production-shaped source edits and staged CPU-only bounds."""

import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render


class StubCall:
    def __init__(self, symbol, ledger):
        self.symbol, self.ledger = symbol, ledger
        self.status = 0

    def __call__(self, *args):
        self.ledger.append((self.symbol, args))
        if self.symbol == render.SYMBOL:
            report = args[9]
            assert args[10] == 5 and tuple(args[3]) == (0, 0, 512)
            report[0], report[1], report[2] = (78, 77, 1)
        if self.symbol.endswith("error_r470"):
            return b"recorded failure"
        return self.status


def binding():
    scope = {}
    exec(compile(render.sources()["typed_binding.py"], "typed_binding.py", "exec"), scope)
    calls, lib = [], type("StubLibrary", (), {})()
    names = ["modern_glm_global_" + key + "_r470" for key in scope["SIGNATURES"]]
    names += ["modern_glm_global_fresh_kv_decode_r474", render.SYMBOL]
    for name in names:
        setattr(lib, name, StubCall(name, calls))
    obj = scope["GlobalTP16Context"].__new__(scope["GlobalTP16Context"])
    obj.ready, obj.poisoned, obj.ctx, obj.thread, obj.past = True, False, 1, threading.get_ident(), 0
    obj.f = scope["attach"](lib)
    return obj, calls, scope


class CallerTests(unittest.TestCase):
    def test_causal_and_post_materialize_code_preserved(self):
        old, new = render.original(), render.sources()
        before = b"// Reject all external metadata before either indexer or latent cache mutates."
        stop = b'glm_layer_mark_r5670(r,"attn.projectedKV.begin");'
        after = b"work.held.insert(work.held.end(),{fresh_all.first,fresh_all.second});"
        self.assertEqual(old["NoV_tp16.cpp"].split(before)[1].split(stop)[0],
                         new["NoV_tp16.cpp"].split(before)[1].split(stop)[0])
        self.assertEqual(old["NoV_tp16.cpp"].split(after)[1], new["NoV_tp16.cpp"].split(after)[1])
        self.assertEqual(old["owner.py"], new["owner.py"])
        self.assertEqual(old["glm_moe_dsa_attention_prepare.hpp"], new["glm_moe_dsa_attention_prepare.hpp"])
        self.assertIn(old["fresh_kv_materialize.inc"].split(b"rt::Tensor value;\n")[1], new["fresh_kv_materialize.inc"])
        self.assertIn(b"phase!='prefill' or tokens!=512 or past!=0", new["typed_binding.py"])

    def test_only_explicit_initial_trial_hits_added_typed_abi(self):
        ctx, runs, scope = binding()
        self.assertEqual(ctx.f["prefill512_forward"].argtypes, scope["SIGNATURES"]["forward"][0]+[C.POINTER(C.c_size_t),C.c_size_t])
        self.assertIsNone(ctx.forward(512,0,3,4,5,6,phase="prefill"))
        self.assertEqual(runs[-1][0],"modern_glm_global_forward_r470")
        self.assertIsNone(ctx.forward(1,512,3,4,5,6,phase="decode"))
        self.assertEqual(runs[-1][0],"modern_glm_global_fresh_kv_decode_r474")
        with self.assertRaisesRegex(ValueError,"initial C1 T512"):
            ctx.forward(512,513,3,4,5,6,phase="prefill",prefill512=True)
        unused, runs, _ = binding()
        report = unused.forward(512,0,3,4,5,6,phase="prefill",prefill512=True)
        self.assertEqual(runs[-1][0],render.SYMBOL)
        self.assertEqual(tuple(report.values()),(78,77,1,0,0))
        self.assertEqual(unused.past,512)

    def test_rejected_cases_do_not_invoke_native(self):
        for count, phase, stream in ((511,"prefill",6),(1,"prefill",6),(512,"prefill",0),(1,"decode",6)):
            with self.subTest(count=count, phase=phase, stream=stream):
                ctx, calls, _ = binding()
                with self.assertRaises(ValueError):
                    ctx.forward(count,0,3,4,5,stream,phase=phase,prefill512=True)
                self.assertEqual(calls,[])
                self.assertEqual(ctx.past,0)

    def test_native_failure_persists_scope_counts(self):
        ctx, _, _ = binding()
        ctx.f["prefill512_forward"].status = 1
        with self.assertRaisesRegex(RuntimeError,"recorded failure"):
            ctx.forward(512,0,3,4,5,6,phase="prefill",prefill512=True)
        self.assertEqual(tuple(ctx.last_prefill512.values()),(78,77,1,0,0))
        self.assertTrue(ctx.poisoned)
        self.assertEqual(ctx.past,0)


def compile_cpp(compiler):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, raw in render.sources().items():
            (directory/name).write_bytes(raw)
        binary = directory/"check_prefill512.exe"
        subprocess.run([compiler,"-std=c++17","-O2","-Wall","-Wextra","-Werror","-pedantic",
                        "-I"+str(directory),str(HERE/"check_materializer.cpp"),"-o",str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CallerTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    compile_cpp(args.cxx)
