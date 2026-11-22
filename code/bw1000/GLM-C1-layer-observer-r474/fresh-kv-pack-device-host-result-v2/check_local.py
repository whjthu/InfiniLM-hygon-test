"""Verify the result adapter against the original STARTED then FAIL schema."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inspect_result import inspect_probe


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOST = load(HERE.parent / "fresh-kv-pack-device-host-v1/host_once.py", "host_v1")
PROBE, _, _ = HOST.verify_stage(HERE.parents[3])
BOOT = "1e3ff586-fd62-4b74-bf36-0388e04c761c"
STEP = "1103606.932"
ROW = [STEP, HOST.LABEL, "FAILED", "1:0", "1", HOST.NODE, "cpu=4,mem=16G,node=1,gres/dcu=1"]
INTENT = "a" * 64
BEFORE = {"boot_id": BOOT, "pci": ["0000:91:00.0"]}
OWNER = {"pid": 4242, "startticks": 50930, "boot_id": BOOT}
START = dict(state="STARTED", label=HOST.LABEL, request=HOST.REQUEST, step=STEP, node=HOST.NODE,
             intent_SHA256=INTENT, native_SHA256=PROBE.NATIVE_SHA, planned=PROBE.PLANNED, **OWNER)
FAIL = dict(state="FAIL", type="ValueError", error="unexpected canonical HIP provider")


class PreBoundFailure(unittest.TestCase):
    def probe_file(self, *records, before=None, suffix=b"", prior=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            raw = b"".join((json.dumps(item) + "\n").encode() for item in records) + suffix
            path.write_bytes(raw)
            args = path, ROW, INTENT, before or BEFORE, PROBE
            if prior:
                return HOST.inspect_probe(*args)
            result = inspect_probe(*args, HOST)
            self.assertEqual(result["trace_SHA256"], sha256(raw).hexdigest())
            return result

    def test_original_failure_preserves_identity_error_and_both_denominators(self):
        original = self.probe_file(START, FAIL, prior=True)
        self.assertEqual(original["error"], "KeyError: 'physical_device'")
        repaired = self.probe_file(START, FAIL)
        self.assertEqual(repaired["binding_state"], "FAILED_BEFORE_BOUND")
        self.assertEqual(repaired["worker_failure"], {"type": "ValueError", "error": FAIL["error"]})
        self.assertEqual(repaired["worker"], OWNER)
        self.assertEqual({key: len(ids) for key, ids in repaired["expected"].items()}, {"valid": 4, "rejected": 36})
        self.assertEqual(repaired["passed"], {"valid": 0, "rejected": 0})
        self.assertFalse(repaired["trace_pass"])
        self.assertFalse({"physical_device", "last_started", "error"} & repaired.keys())

    def test_malformed_or_extended_failure_is_unverified(self):
        for records, suffix in (((START, {**FAIL, "type": []}), b""),
                                ((START, {**FAIL, "error": ""}), b""),
                                ((START, FAIL), b'{"state":"BOUND"}\n'),
                                ((START, FAIL), b'{"state":')):
            with self.subTest(records=records, suffix=suffix):
                result = self.probe_file(*records, suffix=suffix)
                self.assertEqual(result["binding_state"], "UNVERIFIED")
                self.assertNotIn("worker_failure", result)
                self.assertEqual(result["worker"], OWNER)
                self.assertFalse(result["trace_pass"])

    def test_mismatched_original_worker_identity_is_refused(self):
        for start, before in (({**START, "step": "1103606.900"}, BEFORE),
                              ({**START, "intent_SHA256": "b"*64}, BEFORE), (START, {"boot_id": "wrong"})):
            with self.subTest(start=start, before=before):
                result = self.probe_file(start, FAIL, before=before)
                self.assertEqual(result["binding_state"], "UNVERIFIED")
                self.assertIsNone(result["worker"])
                self.assertNotIn("worker_failure", result)

    def test_bound_record_uses_original_parser(self):
        physical = {"pci": BEFORE["pci"][0], "visible_devices": 1,
                    "kfd": {"pci": BEFORE["pci"][0], "gpu_id": 10}}
        bound = dict(state="BOUND", physical_device=physical, providers={"count": 71}, hip_SHA256=PROBE.HIP_SHA)
        result = self.probe_file(START, bound)
        self.assertEqual(result, self.probe_file(START, bound, prior=True))
        self.assertEqual(result["physical_device"], physical)
        self.assertFalse(result["trace_pass"])


if __name__ == "__main__":
    unittest.main()
