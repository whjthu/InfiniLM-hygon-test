"""Exercise exact request/accounting binding and serialized adapter comparisons."""

from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from unittest import TestCase

import read_qwen_usgs_resumed_adapter_values as audit


def encoded(value):
    return json.dumps(value).encode("utf-8")


class RecoveryAdapterTest(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        intent = dict(request_id=audit.REQUEST, publication="a" * 40, source_manifest_SHA256="b" * 64)
        self.store("intent.json", intent)
        self.row = ["1103606.700", audit.LABEL + "-gpu", "COMPLETED", "0:0", "100", audit.NODE,
                    "cpu=64,gres/dcu=8,mem=224000M,node=1"]
        self.store("controller-outcome.json", dict(passed=True, stages=dict(
            gpu=dict(row=self.row, passed=True, terminal_visible=True, rc=0))))
        self.report = dict(state="QWEN_USGS25_AND_RESUME20_INDEPENDENT_METADATA_PASS", passed=True,
                           publication=intent["publication"], source_manifest_SHA256=intent["source_manifest_SHA256"],
                           request=audit.REQUEST, actual_global_optimizer_invocations=30, actual_GPU_cards=8,
                           paired_microbatches=160)
        self.adapter("continuous", 1.0, "continuous")
        self.adapter("resumed", 1.0, "resumed")

    def store(self, name, value):
        (self.root / name).write_bytes(encoded(value))

    def adapter(self, arm, value, provenance):
        path = self.root / arm / "train/checkpoint-25" / audit.CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        header = encoded({
            "__metadata__": {"provenance": provenance},
            "base.lora_A.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        })
        raw = struct.pack("<Q", len(header)) + header + struct.pack("<ff", 0.75, value)
        path.write_bytes(raw)
        files = dict.fromkeys(audit.INVENTORY, {"bytes": 1, "SHA256": "0" * 64})
        files[audit.CHECKPOINT] = {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}
        self.store(f"{arm}-checkpoint-25-files.json",
                   dict(arm=arm, global_step=25, member_count=14, files=files))

    def compare(self, report=None, accounting=None, pin=None):
        raw = encoded(self.report if report is None else report)
        if accounting is None:
            accounting = "|".join(self.row).encode("utf-8")
        return audit.run(self.root, accounting, raw, pin or sha256(raw).hexdigest())

    def test_two_serializations_with_equal_tensor_values_pass(self):
        result = self.compare()
        self.assertTrue(result["paired_numeric_equal"])
        self.assertEqual(result["original_GPU_step"], "1103606.700")
        self.assertEqual(result["adapter_stats"]["all"]["elements"], 2)
        self.assertNotEqual(result["checkpoint_adapters"]["stock"], result["checkpoint_adapters"]["paired"])

    def test_paired_numeric_drift_remains_a_failure(self):
        self.adapter("resumed", 1.5, "resumed")
        result = self.compare()
        self.assertFalse(result["paired_numeric_equal"])
        self.assertEqual(result["adapter_stats"]["all"]["differing_values"], 1)
        self.assertEqual(result["adapter_stats"]["all"]["max_abs"], 0.5)
        self.assertEqual(result["compared_checkpoints"], 2)

    def test_wrong_run_or_ambiguous_accounting_does_not_read_adapters(self):
        (self.root / "continuous/train/checkpoint-25" / audit.CHECKPOINT).unlink()
        with self.assertRaisesRegex(ValueError, "metadata or request"):
            self.compare(self.report | {"request": "0" * 32})
        duplicate = ("|".join(self.row) + "\n") * 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.compare(accounting=duplicate.encode())

    def test_original_report_and_adapter_pins_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "report identity"):
            self.compare(pin="0" * 64)
        path = self.root / "resumed/train/checkpoint-25" / audit.CHECKPOINT
        raw = path.read_bytes()
        path.write_bytes(raw[:-4] + struct.pack("<f", 5.0))
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.compare()
