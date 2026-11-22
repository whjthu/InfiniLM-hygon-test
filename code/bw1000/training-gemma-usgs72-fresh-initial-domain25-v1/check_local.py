"""Pure-stdlib ingress/source/plan tests. No native encoding, Torch or devices."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

from fresh_contract import clock, cohort, fresh_config
from stable_io import read_exact
from worker import authorize_resume, admission_gate, CHECKPOINT_FILES
from stage import TRAINING_REQUEST

D = Path(__file__).resolve().parent


class Boundaries(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(clock(100, 2500, 101), 2399)
        for args in ((float("nan"), 2, 1), (100, 2501, 101), (101, 200, 100), (100, 200, 200)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                clock(*args)

    def test_no_reference_initial(self):
        c = dict(initialization_mode="TARGET_FRESH_DOMAIN_ONLY", seed=42, data_seed=42)
        fresh_config(c)
        for diff in ({"initial_trainables": "/forbidden"}, {"seed": 41}, {"initialization_mode": "REFERENCE"}):
            with self.subTest(diff=diff), self.assertRaises(ValueError):
                fresh_config(dict(c, **diff))

    def test_actual_initial_cohort(self):
        rows = [dict(rank=i, named=dict(SHA256="a"*64, finite=True, tensors=476, elements=29802496)) for i in range(8)]
        self.assertEqual(cohort(rows), "a"*64)
        for mutate in (lambda r: r.pop(), lambda r: r[4]["named"].update(SHA256="b"*64),
                       lambda r: r[4]["named"].update(finite=False)):
            x = copy.deepcopy(rows)
            mutate(x)
            with self.assertRaises(ValueError):
                cohort(x)

    def test_resume_after_actual_continuous_only(self):
        gate = dict(passed=True, independent_passed=True, accounting_C0=True, owned_cleanup_passed=True,
            request_id=TRAINING_REQUEST, arm="continuous", actual_updates=25, ranks_completed=8,
            microbatches_per_rank=100, checkpoint20_files={n: {} for n in CHECKPOINT_FILES})
        self.assertEqual(set(authorize_resume(gate)), CHECKPOINT_FILES)
        for key in ("independent_passed", "accounting_C0", "owned_cleanup_passed"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                authorize_resume(dict(gate, **{key: False}))
        with self.assertRaises(ValueError):
            authorize_resume(dict(gate, checkpoint20_files={}))

    def test_admission_freshness_and_scope(self):
        c = dict(request_id=TRAINING_REQUEST, node="f13r1n12", step="1103606.1", kfd_holder_count=0,
            full_fd_inventory_performed=True, master_port=29932, master_port_available=True,
            pci=list(range(8)), boot_id="actual", monotonic_ns=100)
        admission_gate(c, dict(boot_id="actual"), "1103606.2", 101)
        for diff in ({"kfd_holder_count": 1}, {"full_fd_inventory_performed": False},
                     {"request_id": "another"}, {"pci": [1]*8}):
            with self.subTest(diff=diff), self.assertRaises(ValueError):
                admission_gate(dict(c, **diff), dict(boot_id="actual"), "1103606.2", 101)
        with self.assertRaises(ValueError):
            admission_gate(c, dict(boot_id="actual"), "1103606.2", 45_000_000_101)

    def test_stable_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"file"
            p.write_bytes(b"actual fixture bytes")
            pin = dict(bytes=p.stat().st_size, SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
            # This is a Windows fixture adapter only. Production Linux compares
            # real fstat and lstat; its five-field contract is unchanged.
            adapter = patch("stable_io.os.fstat", lambda fd: p.stat()) if os.name == "nt" else nullcontext()
            with adapter:
                raw, result = read_exact(p, pin, lambda: None)
            self.assertEqual(raw, b"actual fixture bytes")
            self.assertEqual(result["SHA256"], pin["SHA256"])
            with self.assertRaises(ValueError):
                read_exact(p, dict(pin, SHA256="0"*64), lambda: None)
            def mutate():
                if p.stat().st_size == pin["bytes"]:
                    p.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                read_exact(p, pin, mutate)

    def test_opened_identity_mismatch_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"file"
            p.write_bytes(b"identity fixture")
            expected = dict(bytes=p.stat().st_size, SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
            before = p.stat()
            wrong = SimpleNamespace(**{k:getattr(before,k) for k in
                ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")})
            wrong.st_ino += 1
            with patch("stable_io.os.fstat", lambda fd: wrong), self.assertRaises(ValueError):
                read_exact(p, expected, lambda: None)

    def test_real_plans_keep_native_recipe(self):
        plans = {a: json.loads((D/(a+"-plan.json")).read_bytes()) for a in ("continuous", "resumed")}
        for arm, plan in plans.items():
            c = plan["command"]
            arg = lambda n: c[c.index(n)+1]
            self.assertTrue(c[0].endswith("/framework/vlm/runtime/swift_entry.py"))
            self.assertTrue(arg("--model").endswith("/training-gemma-original-model-v1/gemma-3-4b-it"))
            self.assertEqual(arg("--model"), plan["required_assets"]["model_directory"])
            for key, value in {"--max_length":"512", "--max_steps":"25", "--gradient_accumulation_steps":"4",
                "--attn_impl":"sdpa", "--gradient_checkpointing":"false", "--seed":"42", "--data_seed":"42"}.items():
                self.assertEqual(arg(key), value)
            self.assertNotIn("nvidia_baseline", c)
            self.assertNotIn("initial_trainables", json.dumps(plan))
        c = plans["resumed"]["command"]
        self.assertTrue(c[c.index("--resume_from_checkpoint")+1].endswith("/continuous/train/checkpoint-20"))
        self.assertNotEqual(plans["resumed"]["command"][plans["resumed"]["command"].index("--output_dir")+1],
                            plans["continuous"]["command"][plans["continuous"]["command"].index("--output_dir")+1])

    def test_original_math_and_training_methods_identical(self):
        repo = D.parents[2]
        old = repo/"code/bw1000/training-gemma-usgs-shared-resume-v1/gemma_domain_continuous.py"
        before = ast.parse(old.read_bytes())
        after = ast.parse((D/"gemma_domain_continuous.py").read_bytes())
        a = next(n for n in before.body if isinstance(n, ast.ClassDef))
        b = next(n for n in after.body if isinstance(n, ast.ClassDef))
        for n in a.body:
            if isinstance(n, ast.FunctionDef) and n.name != "on_train_begin":
                new = next(x for x in b.body if isinstance(x, ast.FunctionDef) and x.name == n.name)
                self.assertEqual(ast.dump(n), ast.dump(new), n.name)
        text = (D/"gemma_domain_continuous.py").read_text()
        self.assertNotIn("restore_initial_trainables", text)
        self.assertNotIn("b85fdfc7", text)
        # The target must execute exactly the copy that was tested here.
        self.assertEqual((D/"gemma_domain_continuous.py").read_bytes(),
            (D/"overlay/vlm/plugins/gemma_domain_continuous.py").read_bytes())


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries))
    receipt = dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), actual_updates=0,
        native_encode_calls=0, Torch_imported="torch" in __import__("sys").modules,
        target_stage_executed=False, GPU_invocations=0,
        Windows_fixture_fstat_adapter=os.name == "nt",
        source_hashes={p.relative_to(D).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in D.rglob("*.py")})
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
