"""Check the narrow compatibility overlay without sending an envelope."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refill


class Check(unittest.TestCase):
    def test_original_sources_and_current_observer_boundary(self):
        raw = refill.observer_source()
        compile(raw, "observer", "exec")
        self.assertIn(refill.CURRENT_STAGE.encode(), raw)
        self.assertIn(b"o01-memory48-pending-decision.json", raw)
        self.assertNotIn(refill.PREVIOUS_STAGE.encode(), raw)

    def test_grant_binds_new_stage_bridge_and_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            runner = refill.transport()
            # renderer import is from the same frozen published package.
            sys.modules["render"].build(stage)
            launched = refill.launch_adapter(stage, Path(temporary))
            self.assertIn(refill.CURRENT_STAGE, launched.HOST)
            self.assertNotIn(refill.PREVIOUS_STAGE, launched.HOST)
            encoded, _ = launched.encode_remote({"synthetic_test": True})
            self.assertTrue(encoded)
            pending, _, bridge = launched.prepare.load_sources()
            self.assertEqual(pending.REQUEST, refill.REQUEST)
            self.assertEqual(refill.sha256(bridge).hexdigest(), refill.BRIDGE_SHA)
            self.assertEqual(runner.REQUEST, refill.REQUEST)
            request = "b" * 32
            marker = "a" * 64
            grant = {"original_request": refill.REQUEST, "refill_request": request, "marker_SHA256": marker,
                     "scope_SHA256": "c" * 64, "replay_allowed": False, "eligibility": {"root_refill_eligibility": True,
                     "PCI": sorted(pending.PCI), "boot_id_SHA256": "d" * 64, "post_SHA256": "e" * 64}}
            raw = json.dumps(grant).encode()
            digest = refill.sha256(raw).hexdigest()
            self.assertEqual(launched.prepare.validate_grant(pending, raw, request, digest, marker), grant)
            grant["original_request"] = "e23d24ea0c9c4d00af06154d82dc22e8"
            raw = json.dumps(grant).encode()
            with self.assertRaises(ValueError):
                launched.prepare.validate_grant(pending, raw, request, refill.sha256(raw).hexdigest(), marker)


if __name__ == "__main__":
    unittest.main()
