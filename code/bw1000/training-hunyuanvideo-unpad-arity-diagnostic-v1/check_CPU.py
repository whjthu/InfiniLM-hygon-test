"""Pure stdlib tests; they do not claim the real installed provider was executed."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import source_probe
import probe_provider_CPU as provider


class Checks(unittest.TestCase):
    def test_same_exception_requires_no_model(self):
        bad = source_probe.reproduce_assignment(4)
        self.assertEqual(bad["message"], "not enough values to unpack (expected 5, got 4)")
        self.assertFalse(bad["passed"])
        self.assertTrue(source_probe.reproduce_assignment(5)["passed"])

    def test_source_contract_does_not_guess_nested_returns(self):
        result = source_probe.return_arities(b"def unpad_input(a,b):\n def inner(): return 1,2,3,4,5\n return 1,2,3,4\n")
        self.assertEqual(result["literal_return_arities"], [4])
        self.assertEqual(result["unknown_return_expressions"], 0)
        result = source_probe.return_arities(b"def unpad_input(a,b):\n return helper(a,b)\n")
        self.assertEqual(result["literal_return_arities"], [])
        self.assertEqual(result["unknown_return_expressions"], 1)

    def test_active_path_resolution_without_import(self):
        package = SimpleNamespace(submodule_search_locations=["/active-dtk/flash_attn"])
        module = SimpleNamespace(origin="/active-dtk/flash_attn/bert_padding.py")
        with patch.object(provider.PathFinder, "find_spec", side_effect=[package, module]) as called:
            self.assertEqual(provider.resolve_provider(), Path(module.origin))
            self.assertEqual(called.call_args_list[0].args[0], "flash_attn")
            self.assertEqual(called.call_args_list[1].args,
                             ("flash_attn.bert_padding", ["/active-dtk/flash_attn"]))
        with patch.object(provider.PathFinder, "find_spec", return_value=None):
            with self.assertRaises(ImportError): provider.resolve_provider()

    def test_wrong_source_rejected_before_Torch(self):
        before = "torch" in sys.modules
        with tempfile.TemporaryDirectory(prefix="t09-provider-source-reject-") as folder:
            path = Path(folder) / "flash_attn/bert_padding.py"
            path.parent.mkdir()
            path.write_text("raise AssertionError('must never execute')\n")
            with self.assertRaises(ValueError): provider.probe(path, "0"*64)
        self.assertEqual("torch" in sys.modules, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
