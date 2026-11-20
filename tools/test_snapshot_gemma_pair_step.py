import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapshot_gemma_pair_step import capture, seal_accounting
from test_read_gemma_domain_terminal import LABEL, NODE, REQUEST, digest, fixture, write_json


class GemmaStepBinding(unittest.TestCase):
    def test_preflight_persists_only_its_assigned_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            intent = root / "intent.json"
            write_json(intent, dict(request_id=REQUEST, label=LABEL, master_port=29920))
            intent_sha = digest(intent.read_bytes())
            host_receipt = dict(node=NODE, step="1103606.800", full_fd_inventory_performed=True, kfd_holder_count=0)
            env = {"SLURM_JOB_ID": "1103606", "SLURM_STEP_ID": "800"}
            with (patch.dict(os.environ, env), patch("snapshot_gemma_pair_step.socket.gethostname", return_value=NODE),
                  patch("snapshot_gemma_pair_step.socket.socket"),
                  patch("snapshot_gemma_pair_step.pinned_module", return_value={"snapshot": lambda _: host_receipt})):
                self.assertTrue(capture(root, root, intent_sha, "pre"))
                with self.assertRaises(FileExistsError):
                    capture(root, root, intent_sha, "pre")
            pre = json.loads((root / "before.json").read_bytes())
            self.assertEqual(pre["step"], "1103606.800")
            self.assertEqual(pre["intent_SHA256"], intent_sha)
            self.assertNotIn("gpu_step", pre)
            self.assertEqual(digest(intent.read_bytes()), intent_sha)

    def test_accounting_seal_preserves_original_terminal_failure_and_rejects_live_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _, binding = fixture(root)
            final = root / "terminal-accounting.json"
            raw = json.loads(final.read_bytes())
            final.unlink()
            raw["rows"]["gpu"][2] = "RUNNING"
            with self.assertRaisesRegex(ValueError, "still active"):
                seal_accounting(root, binding["intent_sha"], raw)
            self.assertFalse(final.exists())
            raw["rows"]["gpu"][2:4] = ["TIMEOUT", "0:9"]
            seal_accounting(root, binding["intent_sha"], raw)
            sealed = json.loads(final.read_bytes())
            self.assertEqual(sealed["rows"]["gpu"][2:4], ["TIMEOUT", "0:9"])
            self.assertEqual(sealed["intent_SHA256"], binding["intent_sha"])


if __name__ == "__main__":
    unittest.main()
