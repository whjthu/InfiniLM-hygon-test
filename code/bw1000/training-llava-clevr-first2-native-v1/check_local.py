"""Local structural and failure regression, using only the original public launch document."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train
import verify


class CandidateTests(unittest.TestCase):
    def test_dynamic_modules_use_eight_distinct_new_cache_directories(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ):
            scratch = Path(root)
            paths = set()
            for rank in range(8):
                path = train.bind_dynamic_module_cache(scratch, rank)
                self.assertTrue(path.is_dir())
                self.assertEqual(os.environ["HF_MODULES_CACHE"], str(path))
                paths.add(path)
            self.assertEqual(len(paths), 8)
            with self.assertRaises(FileExistsError):
                train.bind_dynamic_module_cache(scratch, 0)
            with self.assertRaisesRegex(ValueError, "BIND_RANK_CACHE_BEFORE_TRANSFORMERS_IMPORT"):
                train.bind_dynamic_module_cache(scratch, 8)

    def test_native_cli_preserves_exact_t04_recipe(self):
        repo = Path(__file__).resolve().parents[3]
        path = repo / "code/bw1000/training-llava-A100-first-updates-v1/original-launch.json"
        raw = path.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(), train.ORIGINAL)
        original = json.loads(raw)
        argv = train.cli(original, "/model", "/bound", "/new/train", "/callback.py")
        original = original["command"][2:]
        del original[original.index("--external_plugins") + 1]
        changed = {i for i, (before, after) in enumerate(zip(original, argv)) if before != after}
        self.assertEqual(changed, {original.index(name) + 1 for name in
                                   ("--model", "--dataset", "--output_dir", "--callbacks", "--external_plugins")}
                         )
        for name, value in {"--attn_impl": "eager", "--gradient_checkpointing": "false", "--max_steps": "25",
                            "--max_length": "4096", "--gradient_accumulation_steps": "4", "--optim": "adamw_torch"}.items():
            self.assertEqual(argv[argv.index(name) + 1], value)

    def test_roster_rejects_crossbound_image_and_duplicate_tensor(self):
        from copy import deepcopy

        rows = [(None, f"{i:064x}") for i in range(64)]
        binding = {"rows": [{"sha256": f"{100+i:064x}"} for i in range(64)]}
        entries = [dict(ordinal=i+1, row_sha256=raw, image_sha256=media["sha256"], input_sha256=f"{200+i:064x}")
                   for i, ((_, raw), media) in enumerate(zip(rows, binding["rows"]))]
        proof = dict(binding_sha256=train.IMAGE_BIND, train_sha256="train", rows=entries)
        self.assertEqual(len(train.roster(proof, rows, binding, "train")), 64)
        for field in ("image_sha256", "input_sha256"):
            bad = deepcopy(proof)
            bad["rows"][1][field] = bad["rows"][0][field]
            with self.assertRaisesRegex(ValueError, "EXACT_NATIVE_TENSOR_AND_SOURCE_ROW"):
                train.roster(bad, rows, binding, "train")

    def test_partial_denominator_remains_64_and_16(self):
        with tempfile.TemporaryDirectory() as path:
            result = verify.reconcile(Path(path).resolve(), "0" * 32)
        self.assertFalse(result["passed"])
        self.assertEqual((result["planned_forwards"], result["recorded_forwards"], result["planned_rank_updates"]),
                         (64, 0, 16))

    def test_real_optimizer_window_guards(self):
        import math

        path = Path(__file__).with_name("callback.py")
        tree = ast.parse(path.read_bytes())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"verify_window", "learning_rate"}]
        namespace = {"need": train.need, "math": math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        verify_window, learning_rate = (namespace[key] for key in ("verify_window", "learning_rate"))
        gathered = [list(range(start, start+4)) for start in range(1, 33, 4)]
        self.assertEqual(verify_window(set(), gathered[0], gathered, 1), set(range(1, 33)))
        gathered[0][0] = 32
        with self.assertRaisesRegex(ValueError, "EXACT_NEW_32_TRAIN_OCCURRENCES"):
            verify_window(set(), gathered[0], gathered, 1)
        learning_rate(1, [0.0, 0.0])
        learning_rate(2, [1e-4, 1e-4])
        with self.assertRaisesRegex(ValueError, "NATIVE_ZERO_THEN_POSITIVE_LR"):
            learning_rate(2, [0.0])


if __name__ == "__main__":
    unittest.main()
