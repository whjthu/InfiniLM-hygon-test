"""Small synthetic ingress/deadline/export checks; no target/model/provider qualification."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import commands
import contract as c
import controller
import review


def fixture_bytes(path, limit=65536):
    """Actual tiny fixture bytes, isolating metadata logic from POSIX fstat admission."""
    import hashlib
    path = Path(path)
    raw = path.read_bytes()
    if not 0 < len(raw) <= limit:
        raise ValueError("Fixture extent differs")
    return raw, {"path": str(path), "bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}


def preflight():
    return {"state": "T09_CACHED_NATIVE_TARGET_PREFLIGHT_PASS", "passed": True, "root_GO": True,
            "request_id": c.REQUEST, "parent": c.PARENT, "node": c.NODE, "run": str(c.RUN),
            "epoch": 100., "deadline": 1540., "framework_manifest_SHA256":
            "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
            "model": {"path": str(c.WEIGHT), "bytes": c.MODEL_STAMP[2], "SHA256": c.MODEL_SHA,
                      "file_identity": c.COMPUTE_MODEL_STAMP[:], "host_file_identity": c.HOST_MODEL_STAMP[:],
                      "verification": "EXECUTION_MOUNT_FULL_SHA256",
                      "full_SHA_receipt": {"path": str(c.FULL_SHA_RECEIPT), "bytes": 932,
                                           "SHA256": c.FULL_SHA_RECEIPT_SHA}},
            "cache": {"path": str(c.CACHE), "bytes": 3521630, "SHA256": c.CACHE_SHA},
            "config": {"path": str(c.CONFIG), "bytes": 968, "SHA256": c.CONFIG_SHA},
            "native_source_manifest": {"path": str(c.RUN / "src/NATIVE_BINDING.json"), "bytes": 10,
                "SHA256": "26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7"},
            "runtime_manifest": {"path": str(c.RUNTIME / "receipt.json"), "bytes": 2124,
                "SHA256": "56f65e63b28f8ec30c7dc1efb3f6dc1df291d4a5d981f91d681f2fad41a03b6a"}}


class Checks(unittest.TestCase):
    def test_ingress_and_full_sha_rebinding(self):
        p = preflight()
        c.validate_preflight(p, 100.)
        for key, value in [("request_id", "wrong"), ("node", "wrong"), ("root_GO", False)]:
            q = copy.deepcopy(p); q[key] = value
            with self.assertRaises(ValueError): c.validate_preflight(q, 100.)
        p["model"]["file_identity"][0] = 47
        with self.assertRaises(ValueError): c.validate_preflight(p, 100.)
        p["model"]["file_identity"][0] = 49
        del p["model"]["full_SHA_receipt"]
        with self.assertRaises(ValueError): c.validate_preflight(p, 100.)
        p["model"]["full_SHA_receipt"] = {"path": str(c.FULL_SHA_RECEIPT), "bytes": 932, "SHA256": c.FULL_SHA_RECEIPT_SHA}
        c.validate_preflight(p, 100.)  # Declaration only; actual receipt consumed by validate_assets.
        p["model"]["file_identity"][1] += 1
        with self.assertRaises(ValueError): c.validate_preflight(p, 100.)

    def test_exact_dual_mount_observations(self):
        p = preflight()
        self.assertEqual(c.validate_model_observation(p, c.HOST_MODEL_STAMP, execution_mount=False),
                         c.HOST_MODEL_STAMP)
        self.assertEqual(c.validate_model_observation(p, c.COMPUTE_MODEL_STAMP, execution_mount=True),
                         c.COMPUTE_MODEL_STAMP)
        for mode, actual in [(False, c.COMPUTE_MODEL_STAMP), (True, c.HOST_MODEL_STAMP),
                             (False, [55, *c.MODEL_STAMP[1:]]), (True, [50, *c.MODEL_STAMP[1:]])]:
            with self.assertRaises(ValueError): c.validate_model_observation(p, actual, execution_mount=mode)
        q = copy.deepcopy(p); q["model"]["host_file_identity"][1] += 1
        with self.assertRaises(ValueError): c.validate_model_observation(q, c.HOST_MODEL_STAMP, execution_mount=False)
        with self.assertRaises(ValueError): c.validate_model_observation(p, c.COMPUTE_MODEL_STAMP, execution_mount=None)

    def test_metadata_fd_identity_guard(self):
        from types import SimpleNamespace
        import stat
        with tempfile.TemporaryDirectory(prefix="t09-fd-equality-fixture-") as folder:
            path = Path(folder) / "tiny.bin"
            path.write_bytes(b"abc")
            good = SimpleNamespace(st_dev=49, st_ino=123, st_size=3, st_mtime_ns=456,
                                   st_ctime_ns=789, st_mode=stat.S_IFREG)
            changed = SimpleNamespace(**{**vars(good), "st_ino": 124})
            with patch.object(Path, "lstat", return_value=good), patch("contract.os.fstat", return_value=good):
                self.assertEqual(c.read(path, 3)[0], b"abc")
            with patch.object(Path, "lstat", return_value=good), patch("contract.os.fstat", return_value=changed):
                with self.assertRaises(ValueError): c.read(path, 3)
            with patch.object(Path, "lstat", side_effect=[good, changed]), patch("contract.os.fstat", return_value=good):
                with self.assertRaises(ValueError): c.read(path, 3)

    def test_clock_and_actual_late_argv(self):
        b = c.Budget(100, now=100, monotonic=20)
        self.assertEqual(b.left(now=200, monotonic=120), 1340)
        for origin in (float("nan"), float("inf"), 101, -1, -1400):
            with self.assertRaises(ValueError): c.Budget(origin, now=100, monotonic=20)
        class Late:
            def left(self, offset=c.WHOLE): return 70 if offset == c.WORK else 300
        seconds = commands.allowance(Late(), "gpu")
        self.assertEqual(seconds, 60)
        argv = commands.argv("gpu", seconds, 100, "a"*64, "b"*64)
        self.assertIn("--time=00:01:00", argv)
        self.assertIn(f"{c.NATIVE}:{c.NATIVE_ALIAS}:ro", argv)
        class Expired:
            def left(self, offset=c.WHOLE): return 0
        with self.assertRaises(ValueError): commands.allowance(Expired(), "gpu")

    def test_accounting_fail_closed(self):
        parent = f"{c.PARENT}|parent|RUNNING|0:0|1|{c.NODE}|cpu=320"
        row = f"{c.PARENT}.99|{c.LABEL}-gpu|COMPLETED|0:0|20|{c.NODE}|cpu=8,gres/dcu=1"
        self.assertEqual(controller.accounting_rows([parent, row], "gpu")[0], c.PARENT + ".99")
        for rows in ([row], [parent, row, row], [parent, "malformed"], [parent, row.replace(c.NODE, "other")]):
            with self.assertRaises(ValueError): controller.accounting_rows(rows, "gpu")

    @patch("contract.read", side_effect=fixture_bytes)
    @patch("review.read", side_effect=fixture_bytes)
    def test_stored_update_denominator_and_bounded_export(self, _read, _load_read):
        with tempfile.TemporaryDirectory(prefix="t09-metadata-only-fixture-") as folder:
            root = Path(folder)
            rows = [{"global_step": i, "loss": .1, "grad_norm": .2, "step_seconds": 1.,
                     "forward_seconds": .4, "backward_update_seconds": .6} for i in range(1, 26)]
            def write_rows(items):
                # Fresh independent fixture file: don't rely on Windows overwrite stat/fstat cache timing.
                (root / "steps.ndjson").unlink(missing_ok=True)
                (root / "steps.ndjson").write_text("".join(json.dumps(row)+"\n" for row in items))
            write_rows(rows[:-1])
            result = review.native(root)
            self.assertEqual((result["completed_updates"], result["missing_updates"]), (24, 1))
            self.assertFalse(result["native_evidence_complete"])
            rows[3]["loss"] = float("nan"); write_rows(rows)
            result = review.native(root)
            self.assertEqual((result["completed_updates"], result["missing_updates"]), (3, 22))
            self.assertEqual(result["failure_phase"], "steps")
            rows[3]["loss"] = .1; write_rows(rows)
            summary = {"optimizer": "Muon", "trainable_parameters": 1769472,
                "lora_targets": ["img_attn_q", "img_attn_v"], "lora_rank": 4,
                "cache_sha256": c.CACHE_SHA, "configured_warmup_steps": 5,
                "configured_measured_steps": 20, "completed_global_step_at_start": 0, "external_resume": False,
                "invocation": {"executed_global_steps": 25, "executed_warmup_steps": 5,
                               "executed_measured_steps": 20, "first_global_step": 1},
                "adapter_sha_before": "a"*64, "adapter_sha_after": "b"*64,
                "checks": {key: True for key in ("finite_tensors", "nonzero_finite_gradients", "current_adapter",
                    "main_forward_device", "attention_cuda_sdpa_fallback", "peak_reserved_below_75gb", "muon_optimizer")}}
            summary["checks"]["checkpoint_resume"] = None
            (root / "summary.json").write_text(json.dumps(summary))
            for step in (10, 20, 25):
                (root / f"checkpoint-{step}").mkdir()
                (root / f"checkpoint-{step}/state.pt").write_bytes(b"SYNTHETIC_METADATA_TEST_NOT_TORCH")
            (root / "adapter-final.safetensors").write_bytes(b"SYNTHETIC_METADATA_TEST_NOT_WEIGHTS")
            import hashlib
            artifacts = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in [root / "summary.json", root / "adapter-final.safetensors",
                          *[root / f"checkpoint-{step}/state.pt" for step in (10, 20, 25)]]]
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "artifacts": artifacts,
                "execution": {"status": "completed"}, "runtime": {"accelerator": "hip"}}))
            self.assertTrue(review.native(root)["native_evidence_complete"])
            summary["adapter_sha_after"] = summary["adapter_sha_before"]
            (root / "summary.json").unlink()
            (root / "summary.json").write_text(json.dumps(summary))
            rejected = review.native(root)
            self.assertFalse(rejected["native_evidence_complete"])
            self.assertEqual(rejected["failure_phase"], "summary_checks")
            (root / "large.json").write_bytes(b"x"*101)
            with self.assertRaises(ValueError): c.read(root / "large.json", 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
