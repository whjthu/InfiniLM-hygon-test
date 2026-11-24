"""Verify the split binder against original 16-range evidence and hash-only fixtures."""

import argparse
from base64 import b64decode, b64encode
from copy import deepcopy
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import bind
from load import load_buffers, load_checkpoint, stage_payloads


def fixture_range_document():
    path = "checkpoint-input-v2/CHECKPOINT_METADATA.json"
    metadata = json.loads(bind.pinned(bind.BASE / path, 65536, bind.DEP[path]))
    return {"schema": 1, "layer": 2, "original_config_SHA256": metadata["original_config_SHA256"],
            "original_index_SHA256": metadata["original_index_SHA256"], "total_selected_weight_bytes": 13144320,
            "header_SHA256": {key: row["header_SHA256"] for key, row in metadata["headers"].items()},
            "whole_shard_hashed": False, "Native_dlopen": False, "Runtime": False, "GPU": False,
            "weight_values_exported": False, "weights": {name: dict(bytes=row["bytes"], SHA256="a" * 64)
                                                        for name, row in metadata["weights"].items()}}


def envelope(document):
    data = bind.canonical(document)
    sha = bind.digest(data)
    original = deepcopy(wrapper)
    original["observed"]["files"]["original_seven"] = dict(
        exists=True, bytes=len(data), SHA256=sha, evidence_base64=b64encode(data).decode())
    return bind.canonical(original), sha


class BinderTest(unittest.TestCase):
    def test_real_16_and_absent_7_cannot_export_full(self):
        report, full = bind.inspect_private_wrapper(bind.canonical(wrapper))
        self.assertEqual((report["validated_non_indexer_ranges"], report["validated_indexer_ranges"]), (16, 0))
        self.assertEqual(report["missing_indexer_payload_bytes"], 13144320)
        self.assertEqual(report["original_indexer_status"], "UNKNOWN_RESULT_ABSENT")
        self.assertIsNone(full)

    def test_visible_original_still_requires_independent_digest(self):
        value, _ = envelope(fixture_range_document())
        view, full = bind.inspect_private_wrapper(value)
        self.assertEqual(view["original_indexer_status"], "VISIBLE_INDEPENDENT_PIN_PENDING")
        self.assertIsNone(full)
        with self.assertRaisesRegex(ValueError, "independent original indexer digest"):
            bind.inspect_private_wrapper(value, "b" * 64)

    def test_hash_only_fixture_materializes_exact_23_receipt(self):
        value, sha = envelope(fixture_range_document())
        view, receipt = bind.inspect_private_wrapper(value, sha)
        self.assertTrue(view["full_payload_pins_complete"])
        self.assertFalse(view["load_or_GPU_qualified"])
        self.assertEqual(view["merged_receipt_SHA256"], bind.digest(receipt))
        content = json.loads(receipt)
        self.assertEqual((len(content["weights"]), sum(p["bytes"] for p in content["weights"].values())), (23, 170605056))
        self.assertEqual(content["indexer_original_SHA256"], sha)

    def test_forged_count_config_and_narrowed_indexer_fail(self):
        for name, value in (("total_selected_weight_bytes", 2), ("original_index_SHA256", "b" * 64),
                            ("weights", {"compressor.ape": {"bytes": 4096, "SHA256": "a" * 64}})):
            with self.subTest(field=name), self.assertRaises(ValueError):
                doc = fixture_range_document()
                doc[name] = value
                packet, sha = envelope(doc)
                bind.inspect_private_wrapper(packet, sha)

    def test_modified_original_non_indexer_bytes_refused(self):
        doc = deepcopy(wrapper)
        doc["observed"]["files"]["non_indexer"]["SHA256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "private file binding differs"):
            bind.inspect_private_wrapper(bind.canonical(doc))


def payload_fixture(root):
    shard = "fixture.safetensors"
    header = b"{}"
    data = {"a": b"0123456", "z": b"789"}
    (root / shard).write_bytes(struct.pack("<Q", len(header)) + header + b"".join(data.values()))
    source = root / shard
    metadata = {"headers": {shard: dict(file_bytes=source.stat().st_size, header_bytes=len(header),
                                        header_SHA256=bind.digest(header))}, "weights": {}}
    offset = 8 + len(header)
    for name, value in data.items():
        metadata["weights"][name] = dict(shard=shard, absolute_offset=offset, bytes=len(value))
        offset += len(value)
    return data, metadata, {name: dict(bytes=len(value), SHA256=bind.digest(value)) for name, value in data.items()}


class LoadTest(unittest.TestCase):
    def test_incomplete_original_fails_before_native_or_checkpoint(self):
        files = wrapper["observed"]["files"]
        trial = Mock()
        metadata, non = (b64decode(files[name]["evidence_base64"]) for name in ("metadata", "non_indexer"))
        with self.assertRaisesRegex(ValueError, "before any checkpoint access"):
            load_checkpoint(trial, metadata_raw=metadata, non_indexer_raw=non, original_indexer=None,
                            checkpoint_root="unused", runtime_config_sha=None, deadline=lambda _: None)
        self.assertEqual(trial.mock_calls, [])

    def test_staging_checks_late_range_before_any_native_write(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, metadata, pins = payload_fixture(root)
            last_bad = deepcopy(pins)
            last_bad["z"]["SHA256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "original payload digest differs: z"):
                stage_payloads(root, metadata, last_bad, lambda _: None)
            buffers = stage_payloads(root, metadata, pins, lambda _: None)
            self.assertEqual({name: buffer.raw for name, buffer in buffers.items()}, data)
            self.assertEqual(load_buffers(Mock(), buffers, lambda _: None), (["a", "z"], 10))
            self.assertEqual(buffers, {})

    def test_native_failure_and_postload_deadline_close_partially_mutated_trial(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, metadata, pins = payload_fixture(root)
            for failing_call in ("load", "checkpoint.postload"):
                with self.subTest(failing_call=failing_call):
                    trial = Mock()
                    if failing_call == "load":
                        trial.call.side_effect = RuntimeError("Native failure")

                    def deadline(name):
                        if name == failing_call:
                            raise RuntimeError("deadline")

                    buffers = stage_payloads(root, metadata, pins, lambda _: None)
                    with self.assertRaises(RuntimeError):
                        load_buffers(trial, buffers, deadline)
                    trial.close.assert_called_once_with()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_wrapper", type=Path)
    path = parser.parse_args().private_wrapper
    wrapper = json.loads(path.read_bytes())
    unittest.main(argv=[__file__], verbosity=2)
