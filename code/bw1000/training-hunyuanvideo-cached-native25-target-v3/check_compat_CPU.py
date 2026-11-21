"""Execute the exact patched FA2/FA3 source loops with CPU arrays and stub kernels.

This verifies ABI plumbing, not the numerical FlashAttention kernel or a model.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import flash_compat as compat

ORIGINAL = None


class Box:
    def __init__(self, data): self.data = np.asarray(data)
    @property
    def shape(self): return self.data.shape
    def unbind(self, dim): return tuple(Box(np.take(self.data, i, axis=dim)) for i in range(self.shape[dim]))


def rearrange(box, pattern, **kw):
    x = box.data
    if pattern == "b s three h d -> b s (three h d)": out = x.reshape(x.shape[0], x.shape[1], -1)
    elif pattern == "nnz (three h d) -> nnz three h d": out = x.reshape(x.shape[0], kw["three"], kw["h"], -1)
    elif pattern == "nnz h d -> nnz (h d)": out = x.reshape(x.shape[0], -1)
    elif pattern == "b s (h d) -> b s h d": out = x.reshape(x.shape[0], x.shape[1], kw["h"], -1)
    elif pattern == "b s h d -> b s (h d)": out = x.reshape(x.shape[0], x.shape[1], -1)
    elif pattern == "nnz (h d) -> nnz h d": out = x.reshape(x.shape[0], kw["h"], -1)
    else: raise AssertionError("Unexpected original source rearrange: " + pattern)
    return Box(out)


def module_namespace(patched):
    tree = ast.parse(patched)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = {"rearrange": rearrange}
    exec(compile(tree, "<actual-patched-FA2-FA3-source-loop>", "exec"), namespace)
    return namespace


class Checks(unittest.TestCase):
    def test_exact_source_patch_inverse_and_compile(self):
        patched = compat.patch_bytes(ORIGINAL)
        ast.parse(patched, feature_version=(3, 10))
        self.assertEqual(len(compat.assignments(ast.parse(patched), wrapped=True)), 4)
        with self.assertRaises(ValueError): compat.patch_bytes(ORIGINAL + b"\n")

    def test_strict_first_four_object_identity(self):
        helper = module_namespace(compat.patch_bytes(ORIGINAL))[compat.HELPER]
        objects = [object() for _ in range(5)]
        for fields in (tuple(objects[:4]), tuple(objects), objects[:4], objects):
            result = helper(fields)
            self.assertEqual(len(result), 4)
            self.assertTrue(all(result[i] is fields[i] for i in range(4)))
        for fields in (objects[:3], objects + [object()], iter(objects[:4]), None, "1234"):
            with self.assertRaises(ValueError): helper(fields)

    def run_loop(self, function, arity):
        namespace = module_namespace(compat.patch_bytes(ORIGINAL))
        counters = {"unpad": 0, "kernel": 0}
        def unpad(x, mask):
            counters["unpad"] += 1
            flat = x.data.reshape(6, -1)
            indices = np.flatnonzero(mask.reshape(-1))
            first = [Box(flat[indices]), indices, np.array([0, 2, 3], dtype=np.int32), 2]
            return tuple((first + [object(), object()])[:arity])
        def pad(x, indices, batch, seqlen):
            out = np.zeros((batch * seqlen, *x.shape[1:]), dtype=x.data.dtype)
            out[indices] = x.data
            return Box(out.reshape(batch, seqlen, *x.shape[1:]))
        def fa2(qkv, cu, maximum, dropout, **kw):
            counters["kernel"] += 1
            self.assertTrue(np.array_equal(cu, [0, 2, 3])); self.assertEqual(maximum, 2)
            return Box(qkv.data[:, 0])  # Stub only; no attention-math claim.
        def fa3(q, k, v, cuq, cuk, maxq, maxk, **kw):
            counters["kernel"] += 1
            self.assertTrue(np.array_equal(cuq, [0, 2, 3])); self.assertTrue(np.array_equal(cuk, [0, 2, 3]))
            self.assertEqual((maxq, maxk), (2, 2))
            return (Box(q.data.copy()), None)  # Exercise original tuple-output branch.
        package = ModuleType("flash_attn"); package.__path__ = []
        package.flash_attn_varlen_qkvpacked_func = fa2
        padding = ModuleType("flash_attn.bert_padding"); padding.unpad_input = unpad; padding.pad_input = pad
        interface = ModuleType("flash_attn_interface"); interface.flash_attn_varlen_func = fa3
        modules = {"flash_attn": package, "flash_attn.bert_padding": padding, "flash_attn_interface": interface}
        qkv = Box(np.arange(36, dtype=np.float32).reshape(2, 3, 3, 1, 2))
        mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        with patch.dict(sys.modules, modules):
            if arity not in (4, 5):
                with self.assertRaises(ValueError): namespace[function](qkv, mask)
                self.assertEqual(counters["kernel"], 0)
                return None
            output = namespace[function](qkv, mask)
        self.assertEqual(output.shape, (2, 3, 1, 2))
        self.assertTrue(np.isfinite(output.data).all())
        self.assertEqual(counters["kernel"], 1)
        self.assertEqual(counters["unpad"], 1 if function == "flash_attn_no_pad" else 3)
        return output.data

    def test_actual_FA2_source_loop(self):
        self.assertTrue(np.array_equal(self.run_loop("flash_attn_no_pad", 4), self.run_loop("flash_attn_no_pad", 5)))
        self.run_loop("flash_attn_no_pad", 3); self.run_loop("flash_attn_no_pad", 6)

    def test_actual_FA3_source_loop(self):
        self.assertTrue(np.array_equal(self.run_loop("flash_attn_no_pad_v3", 4), self.run_loop("flash_attn_no_pad_v3", 5)))
        self.run_loop("flash_attn_no_pad_v3", 3); self.run_loop("flash_attn_no_pad_v3", 6)

    def test_real_command_wiring_and_clean_imports(self):
        import commands
        import controller
        import native_worker
        import review
        args = commands.argv("gpu", 60, 100., "a"*64, "b"*64)
        self.assertIn(str(compat.BACKING) + ":" + str(compat.ALIAS) + ":ro", args)
        self.assertNotIn("torch", sys.modules)
        # Both actual worker and observer compile successfully without importing runtime providers.
        ast.parse(Path(native_worker.__file__).read_bytes(), feature_version=(3, 10))

    def test_actual_CPU_receipt_contract_before_native(self):
        receipt = {"passed": True, "actual_return_arity": 4, "finite": True,
                   "padded_roundtrip_shape": [2, 3, 4], "first_four_semantic_identities":
                   {name: True for name in ("unpadded_values", "indices", "cu_seqlens", "max_seqlen", "padded_roundtrip")},
                   "actual_current_provider": {"bytes": compat.PROVIDER_BYTES, "SHA256": compat.PROVIDER_SHA}}
        pin = {"bytes": compat.CPU_RECEIPT_BYTES, "SHA256": compat.CPU_RECEIPT_SHA}
        with patch.object(compat, "load", return_value=(receipt, pin)):
            self.assertIs(compat.load_cpu_receipt()[0], receipt)
            receipt["first_four_semantic_identities"]["padded_roundtrip"] = False
            with self.assertRaises(ValueError): compat.load_cpu_receipt()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    args = p.parse_args()
    ORIGINAL = args.original.read_bytes()
    compat.patch_bytes(ORIGINAL)  # Exact 3909B/e983 source gate before any test.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
