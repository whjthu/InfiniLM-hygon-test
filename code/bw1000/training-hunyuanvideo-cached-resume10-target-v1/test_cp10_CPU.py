"""Local checks for step-10 restart evidence and immutable invocation binding."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import compare_resume
import contract
import preflight_ingress
import review


SOURCE = Path(__file__).resolve().parent


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"global_step": step, "loss": step / 32, "grad_norm": step / 8}
            for step in range(1, 26)
        ]
        self.summary = {
            "external_resume": True,
            "completed_global_step_at_start": 10,
            "optimizer": "Muon",
            "checks": {"muon_optimizer": True, "checkpoint_resume": None},
            "invocation": {
                "executed_global_steps": 15,
                "executed_warmup_steps": 0,
                "executed_measured_steps": 15,
                "first_global_step": 11,
            },
        }
        self.final = {name: {"observed": name} for name in compare_resume.CHECKPOINT_FIELDS}

    def compare(self, resumed=None, final=None):
        def ndjson(rows):
            return "\n".join(json.dumps(row) for row in rows)

        return compare_resume.compare(
            ndjson(self.rows), ndjson(self.rows[10:] if resumed is None else resumed),
            self.summary, self.final, self.final if final is None else final,
            lambda value: json.dumps(value, sort_keys=True),
        )

    def test_all_fifteen_exact_and_optional_disabled(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual((result["resumed_updates"], result["first_step"], result["last_step"]), (15, 11, 25))
        self.assertEqual([row["global_step"] for row in result["per_step_absolute_error"]], list(range(11, 26)))
        self.assertEqual(result["max_absolute_error"], {"loss": 0, "grad_norm": 0})
        self.assertIsNone(result["optional_in_process_check"])

    def test_reject_missing_updated_step(self):
        with self.assertRaisesRegex(ValueError, "step sequence"):
            self.compare(resumed=self.rows[10:-1])

    def test_report_separate_loss_and_gradient_deviations(self):
        resumed = deepcopy(self.rows[10:])
        resumed[1]["loss"] += 0.25
        resumed[10]["grad_norm"] += 0.125
        result = self.compare(resumed=resumed)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_absolute_error"], {"loss": 0.25, "grad_norm": 0.125})
        self.assertEqual(result["per_step_absolute_error"][10]["global_step"], 21)

    def test_reject_explicit_failure_or_untyped_optional_assertion(self):
        for value in (False, 0, 1, "true"):
            with self.subTest(value=value):
                self.summary["checks"]["checkpoint_resume"] = value
                with self.assertRaisesRegex(ValueError, "Optional"):
                    self.compare()

    def test_reject_incorrect_start_position(self):
        self.summary["completed_global_step_at_start"] = 20
        with self.assertRaisesRegex(ValueError, "checkpoint 10"):
            self.compare()

    def test_reject_mutated_final_rng(self):
        final = deepcopy(self.final)
        final["rng"]["observed"] = "changed"
        result = self.compare(final=final)
        self.assertFalse(result["passed"])
        self.assertFalse(result["final_checkpoint_exact"]["rng"])


class AdmissionTest(unittest.TestCase):
    def test_exact_published_precedent_creates_fresh_binding(self):
        raw = (SOURCE / "REFERENCE-830.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), preflight_ingress.PROOF_SHA)
        proof = json.loads(raw)
        old = {
            "request_id": preflight_ingress.OLD_REQUEST,
            "run": str(Path(contract.ROOT) / "results" / preflight_ingress.OLD_LABEL),
            "epoch": 100,
            "parent": contract.PARENT,
            "node": contract.NODE,
            "native_source_manifest": {"path": "old/path"},
        }
        derived = preflight_ingress.inherit(old, proof, 200, SOURCE)
        self.assertEqual((derived["request_id"], derived["epoch"], derived["deadline"]),
                         (contract.REQUEST, 200, 200 + contract.WHOLE))
        self.assertEqual(derived["new_resume_after"], 10)
        self.assertEqual(derived["native_source_manifest"]["path"], str(SOURCE / "NATIVE_BINDING.json"))
        self.assertEqual(old["epoch"], 100)
        proof["original_GPU_slurm_state"] = "COMPLETED"
        with self.assertRaisesRegex(ValueError, "precedent"):
            preflight_ingress.inherit(old, proof, 200, SOURCE)

    def test_freeze_source_and_use_one_budget(self):
        manifest = SOURCE / "SOURCE_FILES.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        # Windows os.fstat omits fields used by the deployment's Linux inode check.
        test_stamp = (lambda value: (value.st_size, value.st_mtime_ns)) if os.name == "nt" else contract.stamp
        with patch.object(contract, "stamp", test_stamp):
            self.assertEqual(contract.verify_package(SOURCE, digest)["SHA256"], digest)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                contract.verify_package(SOURCE, "f" * 64)
        budget = contract.Budget(100, now=110, monotonic=20)
        self.assertEqual(budget.left(contract.WORK, now=130, monotonic=35), 1210)
        self.assertEqual(budget.left(now=1540, monotonic=1450), 0)

    def test_separate_exact_resource_terminal_steps(self):
        items = {}
        for index, phase in enumerate(("pre", "gpu", "post"), 1):
            tres = "node=1,cpu=8,mem=125G,gres/dcu=1" if phase == "gpu" else "node=1,cpu=1,mem=1G"
            items[phase] = {
                "row": [f"{contract.PARENT}.{index}", contract.LABEL + "-" + phase, "COMPLETED", "0:0",
                        "10", contract.NODE, tres],
                "child_reaped": True,
                "passed": True,
            }
        review.terminal(items)
        invalid = deepcopy(items)
        invalid["gpu"]["row"][6] = "node=1,cpu=8,mem=125G,gres/dcu=2"
        with self.assertRaisesRegex(ValueError, "resources differ"):
            review.terminal(invalid)
        items["post"]["row"][0] = items["gpu"]["row"][0]
        with self.assertRaisesRegex(ValueError, "separate ordered"):
            review.terminal(items)


if __name__ == "__main__":
    unittest.main()
