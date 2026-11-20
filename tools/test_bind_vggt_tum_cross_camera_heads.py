import json
from pathlib import Path
import tempfile
import unittest

from bind_vggt_tum_cross_camera_heads import ARMS, BUNDLE, HEAD, NODE, SOURCES, UPDATES, bind_head, digest, parse_accounting


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, sort_keys=True).encode()
    path.write_bytes(raw)
    return digest(raw)


def fixture(results):
    arm = "lowlr25"
    gpu, request, clean, _ = ARMS[arm]
    label, backend = SOURCES[arm]
    root = results / label
    intent = dict(request=request, publication="1" * 40, parent="1103606", node=NODE, epoch=12000, deadline=13800)
    write(root / "intent.json", intent)
    pci = [f"0000:0{index}:00.0" for index in range(8)]
    rows, stages = {}, {}
    for phase, handle in (("pre", "1103606.488"), ("gpu", gpu), ("post", clean)):
        resources = "cpu=4,gres/dcu=1,mem=64G,node=1" if phase == "gpu" else "cpu=2,mem=2G,node=1"
        row = [handle, f"{label}-{phase}", "COMPLETED", "0:0", "25", NODE, resources]
        rows[handle] = row
        stages[phase] = dict(row=row, terminal_visible=True, passed=True, rc=0)
        write(root / f"{phase}-accounting.json", stages[phase])
        if phase != "gpu":
            snapshot = "before" if phase == "pre" else "after"
            write(root / f"{snapshot}.json", dict(pci=pci, boot_id="boot", node=NODE, step=handle,
                                                  full_fd_inventory_performed=True, kfd_holder_count=0))
    write(root / "assignment.json", dict(step=gpu, node=NODE, pci=pci[0]))
    checkpoint = root / HEAD
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixture bytes that must not be read by reconciliation")
    head_sha = "f" * 64
    module_sha = "e" * 64
    native_sha = write(root / "arms/continuous/native-result.json", {
        "metrics": dict(global_optimizer_step=25, executed_optimizer_steps=25, completed_global_step_before_resume=0,
                        checkpoint_identity="checkpoint", final_state_sha256="a" * 64, camera_head_sha_after=module_sha),
        "checks": dict.fromkeys(("camera_head_weights_changed", "frozen_aggregator_unchanged", "main_forward_device",
                                 "nonzero_finite_camera_head_gradients", "resume_boundary"), True),
        "artifacts": [dict(path="/pilot/" + HEAD, sha256=head_sha)],
    })
    receipts = {}
    for name, start in UPDATES.items():
        receipt = dict(arm=name, backend_SHA256=backend, bundle_SHA256=BUNDLE, epoch=12000, deadline=13700,
                       executed_optimizer_steps=25-start, steps=[
                           dict(global_optimizer_step=step) for step in range(start+1, 26)],
                       final_state_SHA256="a" * 64, checkpoint_identity="checkpoint", result_SHA256=native_sha)
        receipt_sha = write(root / "arms" / name / "receipt.json", receipt)
        receipts[name] = dict(value=receipt, SHA256=receipt_sha)
    combined_sha = write(root / "native-result.json", dict(receipts=receipts, actual_DCU=1,
                                                           comparison=dict(comparison_passed=True),
                                                           formal_CO3D_eligible=False))
    write(root / "outcome.json", dict(request=request, publication=intent["publication"],
                                     stages=stages, passed=True, result_SHA256=combined_sha))
    return root, rows, head_sha, module_sha, combined_sha


class OriginalHeadBindingTests(unittest.TestCase):
    def test_binds_original_declarations_without_reading_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary).resolve()
            root, rows, head_sha, module_sha, combined_sha = fixture(results)
            proof = bind_head(results, "lowlr25", rows)
            self.assertEqual((proof["file_SHA256"], proof["module_SHA256"]), (head_sha, module_sha))
            self.assertEqual(proof["original_combined_native_SHA256"], combined_sha)
            self.assertEqual(proof["actual_checkpoint_bytes_read"], 0)
            self.assertEqual(proof["head_bytes_stat"], (root / HEAD).stat().st_size)

    def test_rejects_modified_native_metadata_and_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary).resolve()
            root, rows, _, _, _ = fixture(results)
            _, _, clean, _ = ARMS["lowlr25"]
            missing = {key: val for key, val in rows.items() if key != clean}
            with self.assertRaisesRegex(ValueError, "Slurm terminal"):
                bind_head(results, "lowlr25", missing)
            path = root / "arms/continuous/native-result.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "receipt SHA"):
                bind_head(results, "lowlr25", rows)

    def test_accounting_rejects_duplicate_numbered_steps(self):
        raw = b"1103606.489|native|COMPLETED|0:0|25|f13r4n09|cpu=4,gres/dcu=1,mem=64G,node=1\n"
        self.assertEqual(len(parse_accounting(raw)), 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_accounting(raw + raw)


if __name__ == "__main__":
    unittest.main()
