from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import read_qwen417_adapter_drift as drift


class AdapterDriftTest(TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name).resolve()

    def payload(self, path, rows, metadata=None):
        header = {}
        if metadata is not None:
            header["__metadata__"] = metadata
        blob = bytearray()
        for name, dtype, array in rows:
            data = np.asarray(array, dtype=drift.DTYPE[dtype])
            start = len(blob)
            blob.extend(data.tobytes())
            header[name] = dict(dtype=dtype, shape=list(data.shape), data_offsets=[start, len(blob)])
        encoded = json.dumps(header).encode()
        payload = struct.pack("<Q", len(encoded)) + encoded + blob
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return dict(bytes=len(payload), sha256=sha256(payload).hexdigest())

    def pair(self, stock, paired, *, metadata=None):
        a, b = self.root / "a.safetensors", self.root / "b.safetensors"
        pins = dict(stock=self.payload(a, stock, metadata=metadata),
                    paired=self.payload(b, paired, metadata={"format": "pt"}))
        return a, b, pins

    def test_changed_headers_and_payload_order_can_preserve_all_values(self):
        rows = [("m.lora_A.weight", "F32", [1., 2.]), ("m.lora_B.weight", "BF16", [0x3f80])]
        a, b, pins = self.pair(rows, rows[::-1], metadata={"other": "provenance"})
        self.assertNotEqual(pins["stock"]["sha256"], pins["paired"]["sha256"])
        summary, tensors = drift.compare(a, b, pins)
        self.assertEqual(summary["all"]["differing_values"], 0)
        self.assertEqual(summary["all"]["differing_tensor_bodies"], 0)
        self.assertEqual(summary["all"]["elements"], 3)
        self.assertEqual(set(tensors), {row[0] for row in rows})

    def test_f32_and_bf16_maxima_and_full_denominators(self):
        stock = [("m.lora_A.weight", "F32", [1, 2, 3, 4, 5, 6]),
                 ("m.lora_B.weight", "BF16", [0x3f80, 0x4000])]
        paired = [("m.lora_A.weight", "F32", [1, 2, 3, 4, 5, 8]),
                  ("m.lora_B.weight", "BF16", [0x3f80, 0xc000])]
        a, b, pins = self.pair(stock, paired)
        with patch.object(drift, "CHUNK_BYTES", 16):
            summary, tensors = drift.compare(a, b, pins)
        overall = summary["all"]
        self.assertEqual((overall["elements"], overall["differing_values"], overall["max_abs"]), (8, 2, 4))
        self.assertAlmostEqual(overall["relative_l2_to_stock_tensor_values"], (20/96)**.5)
        self.assertEqual(summary["by_family"]["lora_A"]["max_abs"], 2)
        self.assertEqual(tensors["m.lora_B.weight"]["delta_squared"], 16)

    def test_signed_zero_records_bitwise_difference_without_numeric_drift(self):
        a, b, pins = self.pair([("m.lora_A.weight", "F16", [-0.0])],
                               [("m.lora_A.weight", "F16", [0.0])])
        summary, _ = drift.compare(a, b, pins)
        self.assertEqual(summary["all"]["differing_values"], 0)
        self.assertEqual(summary["all"]["differing_tensor_bodies"], 1)
        self.assertIsNone(summary["all"]["relative_l2_to_stock_tensor_values"])

    def test_file_sha_inventory_dtype_shape_and_nonfinite_rejected(self):
        row = ("m.lora_A.weight", "F32", [1, 2])
        a, b, pins = self.pair([row], [row])
        b.write_bytes(b.read_bytes()[:-1] + b"*")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            drift.compare(a, b, pins)
        for other, reason in ((("other", "F32", [1, 2]), "inventory"),
                              ((row[0], "F16", [1, 2]), "dtype or shape"),
                              ((row[0], "F32", [[1, 2]]), "dtype or shape"),
                              ((row[0], "F32", [1, np.inf]), "nonfinite")):
            with self.subTest(reason=reason):
                a, b, pins = self.pair([row], [other])
                with self.assertRaisesRegex(ValueError, reason):
                    drift.compare(a, b, pins)

    def test_overlapping_and_duplicate_header_entries_rejected(self):
        field = b'{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
        for second, message in ((b"a", "duplicate"), (b"b", "overlap")):
            raw = b'{"a":' + field + b',"' + second + b'":' + field + b"}"
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                drift.tensor_header(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)

    def test_exact_checkpoint_paths_and_partial_results(self):
        archive = self.root / "original"
        archive.mkdir()
        outcome = archive / "outcome.json"
        outcome.write_bytes(b"frozen controller fixture")
        summary_file = self.root / "summary.json"
        pins = {arm: [] for arm in drift.ARMS}
        for arm in drift.ARMS:
            for step in drift.STEPS:
                path = archive / arm / "train" / f"checkpoint-{step}" / "adapter_model.safetensors"
                file_pin = self.payload(path, [("m.lora_A.weight", "F32", [step])])
                pins[arm].append(dict(global_step=step, adapter=file_pin))
        (archive / "paired/train/checkpoint-20/adapter_model.safetensors").write_bytes(b"invalid")
        expected = sha256(outcome.read_bytes()).hexdigest()
        summary_file.write_bytes(json.dumps(dict(original_outcome_SHA256=expected,
            independent_post_clean=True, checkpoint_adapter_payloads_rehashed=6,
            by_arm={arm: dict(checkpoints=rows) for arm, rows in pins.items()})).encode())
        saved = []
        with patch.multiple(drift, ARCHIVE=archive, EVIDENCE=summary_file, OUTCOME_SHA=expected,
                            EVIDENCE_SHA=sha256(summary_file.read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError, "adapter path or size"):
                drift.measure(lambda: None, lambda row, _: saved.append(row))
        self.assertEqual([row["step"] for row in saved], [10])
