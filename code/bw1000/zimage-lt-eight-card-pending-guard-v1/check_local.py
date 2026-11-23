"""Exercise the guarded original standby path without Slurm or target hardware."""

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pending
import render_bridge


BACKFILL = "a" * 32
REFILL = "b" * 32
PROPOSAL = {"parent": pending.PARENT, "pci": sorted(pending.PCI)}


def write(path, item):
    raw = (json.dumps(item, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest()


class GuardTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.results = Path(folder.name)
        self.directory = self.results / pending.MARKER

    def reserve(self):
        with patch.object(pending, "shared_lock"), patch.object(pending, "fsync_directory"), \
             patch.object(pending, "durable", side_effect=write):
            return pending.reserve_locked(self.results, object())

    def decision(self, request=BACKFILL, proposal=PROPOSAL):
        return pending.decision(self.results, request, proposal)

    def test_clear_then_persist_through_legacy_unknown_and_corrupt_claim(self):
        self.assertFalse(self.decision()["blocked"])
        original = self.results / pending.ORIGINAL
        original.mkdir()
        write(original / "gpu-attempt.json", {"request": pending.REQUEST, "epoch": 10, "deadline_epoch": 250})
        with patch.object(pending.time, "time", return_value=10 ** 6):
            self.assertTrue(self.decision()["blocked"])
        self.directory.mkdir()
        (self.directory / "claim.json").write_text("{")
        self.assertTrue(self.decision()["blocked"])
        with patch.object(pending, "present", side_effect=PermissionError):
            self.assertTrue(self.decision()["blocked"])

    def test_marker_exists_before_transport_and_duplicate_reserve_refused(self):
        first = self.reserve()
        self.assertEqual(self.decision()["marker_SHA256"], first)
        with self.assertRaisesRegex(ValueError, "ORIGINAL_REQUIRES_RECONCILIATION"):
            self.reserve()
        self.assertFalse((self.results / pending.ORIGINAL).exists())

    def test_single_refill_then_release_with_original_evidence_chain(self):
        marker = self.reserve()
        eligibility = {"root_refill_eligibility": True, "PCI": sorted(pending.PCI),
                       "post_SHA256": "1" * 64, "boot_id_SHA256": "2" * 64}
        host = SimpleNamespace(digest=lambda raw: sha256(raw).hexdigest(), encoded=lambda scope: json.dumps(scope).encode(),
                               read_exact=lambda path, scope_pin: None,
                               refill_eligibility_locked=lambda *args: eligibility,
                               consume_refill=lambda *args: {"actual_computing_cards": 8, "refill_step": "1103606.2204"})
        with patch.object(pending, "shared_lock"), patch.object(pending, "bound_host", return_value=host), \
             patch.object(pending, "durable", side_effect=write), \
             patch.object(pending.os, "getuid", return_value=123, create=True):
            grant_pin = pending.authorize_refill_locked(self.results, object(), Path("unused"), lambda: {}, {}, {}, REFILL)
            self.assertEqual(pending.marker_document(self.results), marker)
            self.assertTrue(self.decision()["blocked"])
            self.assertFalse(self.decision(REFILL)["blocked"])
            self.assertTrue(self.decision(REFILL, {**PROPOSAL, "pci": ["wrong"] * 8})["blocked"])
            with self.assertRaisesRegex(ValueError, "REFILL_DIFFERS"):
                pending.finish_refill_locked(self.results, object(), Path("unused"), lambda: {}, {"request": BACKFILL})
            pending.finish_refill_locked(self.results, object(), Path("unused"), lambda: {}, {"request": REFILL})
        release = json.loads((self.directory / "released.json").read_bytes())
        self.assertEqual(release["grant_SHA256"], grant_pin)
        self.assertFalse(self.decision()["blocked"])
        (self.directory / "released.json").write_text("{}")
        self.assertTrue(self.decision()["blocked"])

    def test_real_controller_skips_one_pending_node_inside_original_lock(self):
        self.reserve()
        other = "f12r2n10"
        bridge_dir = self.results / "bridge"
        bridge_dir.mkdir()
        calls = []
        locks = []
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda file, flags: locks.append((file, flags)))
        namespace = {"__name__": "local_guarded_frozen_controller"}
        base = render_bridge.CONTROLLER
        with patch.dict(sys.modules, {"fcntl": fake_fcntl}):
            raw = render_bridge.bind(base, render_bridge.CONTROLLER_SHA)
            exec(compile(raw, str(base), "exec"), namespace)

        def live():
            self.assertEqual(len(locks), 1)
            self.assertFalse(locks[0][0].closed)
            self.assertEqual(Path(locks[0][0].name), self.results / pending.LOCK)
            return {pending.NODE: pending.PARENT, other: pending.PARENT}, set()

        def start(root, name, row, seconds, request):
            calls.append(name)
            return {"node": name, "launch_record": True}

        namespace.update(RESULTS=self.results, live_inventory=live, spawn=start)
        data = {"request": BACKFILL, "publication": "0" * 40, "admission_SHA256": "1" * 64, "seconds": 120,
                "nodes": {pending.NODE: PROPOSAL, other: PROPOSAL},
                "worker": base.with_name("worker.py").read_bytes(),
                "claim": json.loads(base.with_name("CLAIM.json").read_bytes())}
        with patch.object(pending, "durable", side_effect=write), redirect_stdout(io.StringIO()):
            pending.install(namespace, data, bridge_dir)
            namespace["run"](data)
        self.assertEqual(calls, [other])
        record = json.loads((bridge_dir / "o01-pending-decision.json").read_bytes())
        self.assertTrue(record["blocked"])
        actual = self.results / ("training80-dcu-standby-" + BACKFILL) / "submission-receipt.json"
        result = json.loads(actual.read_bytes())
        self.assertEqual(result["changed_before_launch"], [pending.NODE])
        self.assertEqual(result["submit_records"], [{"node": other, "launch_record": True}])

    def test_renderer_changes_only_bridge_call_anchor(self):
        raw = render_bridge.bind(render_bridge.BASE, render_bridge.BASE_SHA)
        patched = render_bridge.render()
        added = b'    pending_namespace = {"__name__": "o01_pending_guard"}\n'
        first = patched.index(added)
        last = patched.index(b'    try:\n        namespace["run"](inputs)\n', first)
        self.assertEqual(patched[:first] + patched[last:], raw)
        self.assertIn(b'pending_namespace["install"](namespace, inputs, directory)', patched[first:last])


if __name__ == "__main__":
    unittest.main()
