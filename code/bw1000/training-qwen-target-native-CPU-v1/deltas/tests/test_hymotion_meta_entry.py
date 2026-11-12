"""CPU admission boundaries for the native meta structure checker."""
import hashlib
import importlib.util
import importlib
import types
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "tools/hygon_hymotion_meta_check.py"
SPEC = importlib.util.spec_from_file_location("hymotion_meta_check", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class NativeMetaAdmissionTests(unittest.TestCase):
    def test_exhausted_and_rebased_epochs_rejected(self):
        for epoch, end, now in ((1, 221, 221), (1, 222, 2), (3, 2, 3), (4, 5, 3)):
            with self.assertRaises(TimeoutError):
                module.deadline_check(epoch, end, now)
        module.deadline_check(1, 221, 220)

    def test_nonfinite_clock_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                module.deadline_check(1, 220, bad)

    def test_changed_metadata_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spec.json"
            path.write_bytes(b'{"frames":128}')
            pin = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(module.pinned_json(path, pin)["frames"], 128)
            path.write_bytes(b'{"frames":60}')
            with self.assertRaises(ValueError):
                module.pinned_json(path, pin)

    def test_source_mutation_and_incomplete_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pins = {}
            for i in range(16):
                name = f"module{i}.py"
                (root / name).write_bytes(b"value = 1\n")
                pins[name] = {"bytes": 10, "SHA256": hashlib.sha256(b"value = 1\n").hexdigest()}
            self.assertEqual(len(module.verify_sources(root, pins)), 16)
            with self.assertRaises(ValueError):
                module.verify_sources(root, dict(list(pins.items())[:15]))
            (root / "module0.py").write_bytes(b"value = 2\n")
            with self.assertRaises(ValueError):
                module.verify_sources(root, pins)


    def test_failed_deadline_is_recorded_without_runtime_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = root / "failed.json"
            command = [sys.executable, "-B", str(PATH), "--spec", str(root / "missing.json"),
                       "--spec-sha256", "0" * 64, "--source-root", str(root),
                       "--output", str(output), "--epoch", "1", "--deadline", "2"]
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            value = json.loads(output.read_bytes())
            self.assertEqual(value["error_type"], "TimeoutError")
            self.assertFalse(value["passed"])
            self.assertFalse(value["Model_GO"])
            self.assertIsNone(value["CUDA_initialized"])
            original = output.read_bytes()
            subprocess.run(command, capture_output=True, timeout=10)
            self.assertEqual(output.read_bytes(), original)


    def test_optional_gpu_extensions_blocked_before_loader_and_scope_restored(self):
        before = list(sys.meta_path)
        with module.without_optional_gpu_extensions():
            for name in ("flash_attn", "flash_attn_2_cuda", "flash_attn.child"):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(name)
        self.assertEqual(sys.meta_path, before)
        self.assertFalse(any(n.startswith("flash_attn") for n in sys.modules))

    def test_preloaded_gpu_extension_is_not_silently_qualified(self):
        sys.modules["flash_attn"] = types.ModuleType("flash_attn")
        before = list(sys.meta_path)
        try:
            with self.assertRaises(RuntimeError):
                with module.without_optional_gpu_extensions():
                    self.fail("preloaded extension accepted")
            self.assertEqual(sys.meta_path, before)
        finally:
            del sys.modules["flash_attn"]

    def test_source_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                module.verify_sources(Path(d), {"../outside.py": {}})


if __name__ == "__main__":
    unittest.main()
