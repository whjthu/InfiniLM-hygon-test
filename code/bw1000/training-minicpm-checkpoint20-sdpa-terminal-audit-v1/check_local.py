"""Exercise actual terminal-reader gates without making cluster requests."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_terminal as audit

POST = "1103606.2304"
PINS = {str(i): "a" * 64 for i in range(96)}


def accounting():
    return "\n".join("|".join([step, audit.LABEL + "-" + phase, "COMPLETED", "0:0", "60", audit.NODE, resources])
                     for step, phase, resources in ((audit.PRE, "pre", "cpu=2,mem=2G,node=1"),
                         (audit.GPU, "gpu", "cpu=64,gres/dcu=8,mem=224000M,node=1"),
                         (POST, "post", "cpu=2,mem=2048M,node=1")))


def evidence():
    pci = [f"0000:{rank:02x}:00.0" for rank in range(8)]
    common = dict(request=audit.REQUEST, node=audit.NODE, boot_id="same-boot", pci=pci)
    snapshot = dict(common, master_port=audit.PORT, pci_to_kfd=None, kfd_holder_count=0, full_fd_inventory_performed=True,
                    holder_inventory_basis="kernel_KFD_plus_sameUID_FD_complete")
    compare = dict(actual_step=audit.GPU, planned_forwards=32, verified_forwards=32, missing_forwards=0,
                   comparable_ranks=8, complete=True, rows=[dict(rank=rank) for rank in range(8)], errors=[])
    out = dict(request=audit.REQUEST, epoch=42.5, planned_arms=list(audit.ARMS), planned_forwards=32,
               planned_optimizer_updates=0, passed=True, arms=[dict(
                   arm=arm, passed=True, cleanup_known=True, cleanup=dict(owned_processes_remaining=0)) for arm in audit.ARMS])
    return {
        "diagnostic-intent.json": dict(request=audit.REQUEST, parent=audit.PARENT, node=audit.NODE, epoch=42.5,
             port=audit.PORT, source_manifest_SHA256=audit.MANIFEST, planned_forwards=32, planned_optimizer_updates=0),
        "pre-device-inventory.json": dict(snapshot, step=audit.PRE, phase="pre", monotonic_ns=10),
        "post-device-inventory.json": dict(snapshot, step=POST, phase="post", monotonic_ns=100,
                                           owned_registry_complete=True, owned_processes_remaining=[]),
        "worker-assigned.json": dict(common, epoch=42.5, actual_step=audit.GPU, visible_PCI_by_index=pci[::-1],
             CPU_affinity=list(range(64)), leased_snapshot=dict(common, step=audit.GPU, pci_to_kfd=dict.fromkeys(pci, 1),
                 monotonic_ns=20, kfd_holder_count=0, full_fd_inventory_performed=False, holder_inventory_basis="kernel_KFD_proc_only")),
        "worker-outcome.json": out,
        "diagnostic-outcome.json": dict(request=audit.REQUEST, epoch=42.5, clean_post=True, complete=True),
        "sdpa-repeatability-comparison.json": dict(compare, native_state_receipts_SHA256=PINS, missing_or_invalid_receipt_files=[]),
    }, compare


class Terminal(unittest.TestCase):
    def run_scope(self, edit=None, observed=None):
        files, comparison = evidence()
        rows = audit.select_rows(accounting()) if observed is None else observed
        if edit:
            edit(files)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for name, row in files.items():
                (root / name).write_text(json.dumps(row), encoding="utf8")
            return audit.inspect(root, root, rows, compare=lambda *_: (comparison, PINS, []))

    def test_valid_complete_and_same_boot_cleanup(self):
        def include_descendant_registry(files):
            rows = files["worker-outcome.json"]["arms"]
            for arm in rows:
                arm["owners"] = {1000 + i: {"startticks": i, "state": "T", "PGID": 1000} for i in range(210)}
            self.assertGreater(len(json.dumps(files["worker-outcome.json"])), 32768)

        result = self.run_scope(include_descendant_registry)
        self.assertTrue(result["diagnosis_complete"])
        self.assertTrue(result["clean_independent_post"])
        self.assertTrue(result["original_worker_comparison_agrees"])
        self.assertEqual((result["native_metadata_members"], result["actual_optimizer_updates"]), (96, 0))
        self.assertFalse(result["formal_recovery_performance_or_effect_qualified"])

    def test_wrong_accounting_step_duplicate_resources_and_node_rejected(self):
        for raw in (accounting().replace(audit.GPU, "1103606.17"), accounting() + "\n" + accounting(),
                    accounting().replace("gres/dcu=8", "gres/dcu=4"), accounting().replace(audit.NODE, "other")):
            with self.assertRaises(ValueError):
                audit.select_rows(raw)

    def test_running_does_not_read_or_claim_progress(self):
        rows = audit.select_rows(accounting())
        rows["gpu"][2] = "RUNNING"
        self.assertFalse(self.run_scope(observed=rows)["actual_gpu_terminal"])
        rows["gpu"][2] = "COMPLETED"
        rows["post"] = None
        self.assertFalse(self.run_scope(observed=rows)["diagnosis_complete"])

    def test_cleanup_or_freshness_cannot_be_inferred_from_c0(self):
        for edit in (lambda f: f["post-device-inventory.json"].update(owned_processes_remaining=[100]),
                     lambda f: f["post-device-inventory.json"].update(boot_id="changed"),
                     lambda f: f["worker-assigned.json"]["leased_snapshot"].update(monotonic_ns=46_000_000_000)):
            report = self.run_scope(edit)
            self.assertFalse(report["clean_independent_post"])
            self.assertFalse(report["diagnosis_complete"])

    def test_worker_summary_and_denominator_checked_independently(self):
        report = self.run_scope(lambda f: f["sdpa-repeatability-comparison.json"].update(verified_forwards=31))
        self.assertFalse(report["original_worker_comparison_agrees"])
        self.assertFalse(report["diagnosis_complete"])
        with self.assertRaises(ValueError):
            self.run_scope(lambda f: f["worker-outcome.json"].update(arms=deepcopy(f["worker-outcome.json"]["arms"][::-1])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
