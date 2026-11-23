"""Offline boundary checks for the HIP-only preflight."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mask_preflight as mask


class MaskTests(unittest.TestCase):
    def test_inherited_masks(self):
        inherited = {"ROCR_VISIBLE_DEVICES": "GPU-a,GPU-b", "GPU_DEVICE_ORDINAL": "7,0"}
        self.assertEqual(mask.selectors(inherited), list(map(str, range(8))))
        tokens = "7,4,6,2,5,3,1,0"
        supplied = dict(inherited, HIP_VISIBLE_DEVICES=tokens, CUDA_VISIBLE_DEVICES=tokens)
        self.assertEqual(mask.selectors(supplied), tokens.split(","))
        self.assertEqual(mask.child_environment("7", supplied), {
            **inherited, "HIP_VISIBLE_DEVICES": "7", "CUDA_VISIBLE_DEVICES": "7",
        })
        for bad in ("", "7,7,6,2,5,3,1,0", "4,7,6,2,5,3,1,0"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "SELECTOR_UNPROVED"):
                mask.selectors(dict(supplied, CUDA_VISIBLE_DEVICES=bad))

    def test_actual_child_surfaces_mismatches(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(parent_pid=100, token="7", pci=mask.PROBE_PCI, out=Path(root) / "mask.json")
            with patch.dict(mask.os.environ, mask.child_environment("7", {}), clear=True), \
                 patch.object(mask.os, "getpid", return_value=101), patch.object(mask.os, "getppid", return_value=100), \
                 patch.object(mask, "visible", return_value=["0000:36:00.0"]) as visible:
                with self.assertRaisesRegex(ValueError, "MASKED_HIP_PCI_MISMATCH"):
                    mask.check_child({"request": mask.REQUEST}, "1103606.9999", args)
                self.assertFalse(args.out.exists())
                visible.return_value = [mask.PROBE_PCI]
                mask.check_child({"request": mask.REQUEST}, "1103606.9999", args)
                visible.assert_called_with(1)
                child = json.loads(args.out.read_bytes())
                self.assertEqual((child["visible_count"], child["visible_ordinal"], child["physical_PCI"]),
                                 (1, 0, mask.PROBE_PCI))

    def test_prove_partial_and_full_denominator(self):
        devices = sorted(mask.PCI_SET, reverse=True)
        argv_masks = "7,6,5,4,3,2,1,0"
        name = "1103606.9999"
        for fail_at in (None, 3, 6):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as root:
                count = 0

                def spawn(command, *, env, stdin, stdout, stderr):
                    nonlocal count
                    index = count
                    count += 1
                    option = dict(zip(command[5::2], command[6::2]))
                    self.assertEqual(option["--token"], argv_masks.split(",")[index])
                    self.assertEqual(option["--pci"], devices[index])
                    self.assertEqual(env["HIP_VISIBLE_DEVICES"], option["--token"])
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], option["--token"])
                    self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "preserved")
                    pid = 2001 + index
                    receipt = {
                        "request": mask.REQUEST, "step": name, "PID": pid,
                        "parent_PID": int(option["--parent-pid"]), "physical_PCI": option["--pci"],
                        "visible_ordinal": 0, "visible_count": 1, "selector": option["--token"],
                    }
                    if fail_at == 6 and index == fail_at:
                        receipt["physical_PCI"] = devices[0]
                    rc = 1 if fail_at == 3 and index == fail_at else 0
                    if not rc:
                        mask.write(Path(option["--out"]), receipt)
                    return SimpleNamespace(pid=pid, wait=lambda timeout: rc, poll=lambda: rc)

                args = SimpleNamespace(scope=Path(root) / "original.json", out=Path(root), scope_sha256="a" * 64)
                result = {"launched_children": [], "child_outcomes": [], "child_receipts": []}
                env = {"HIP_VISIBLE_DEVICES": argv_masks, "CUDA_VISIBLE_DEVICES": argv_masks,
                       "ROCR_VISIBLE_DEVICES": "preserved"}
                with patch.dict(mask.os.environ, env, clear=True), patch.object(mask, "visible", return_value=devices), \
                     patch.object(subprocess, "Popen", side_effect=spawn):
                    if fail_at is None:
                        mask.prove({"deadline_epoch": time.time() + 60}, name, args, result)
                        self.assertEqual(result["verified_mapping"]["probe_PCI"], mask.PROBE_PCI)
                        self.assertEqual(len(result["verified_mapping"]["standby_PCIs"]), 7)
                    else:
                        error = "EIGHT_CHILD_DENOMINATOR" if fail_at == 3 else "CHILD_BINDING_DIFFERS"
                        with self.assertRaisesRegex(ValueError, error):
                            mask.prove({"deadline_epoch": time.time() + 60}, name, args, result)
                        self.assertNotIn("mask_bijection_passed", result)
                self.assertEqual(len(result["child_outcomes"]), 8)
                self.assertEqual(len(result["child_receipts"]), 7 if fail_at == 3 else 8)

    def test_2075_never_admitted(self):
        with patch.dict(mask.os.environ, {"SLURM_JOB_ID": mask.PARENT, "SLURM_STEP_ID": "2075"}):
            scope = {"request": mask.REQUEST, "label": mask.LABEL, "fresh_root_admitted": True, "parent": mask.PARENT}
            with self.assertRaisesRegex(ValueError, "NEW_OWNED_STEP_REQUIRED"):
                mask.admission(scope)


if __name__ == "__main__":
    unittest.main()
