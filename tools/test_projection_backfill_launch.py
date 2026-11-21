"""Exercise the actual projection backfill selectors against a synthetic fleet."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

from report_training80_backlog import REPO, TRAINING80_LEDGER, fleet_view, plan
import test_report_training80_backlog as fixtures


FLEET = REPO / "code/bw1000/fleet-qwen-packed-projection-backfill-v1"
RUNTIME = REPO / "code/bw1000/fleet-lora-isolated-backfill-v1"


def source(name):
    spec = importlib.util.spec_from_file_location("_projection_" + name, FLEET / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(RUNTIME), *sys.path]):
        spec.loader.exec_module(module)
    return module


class ProjectionBackfillTest(unittest.TestCase):
    def setUp(self):
        self.fleet = fixtures.BacklogTest()
        self.fleet.setUp()
        self.addCleanup(self.fleet.doCleanups)
        self.launch = source("launch")
        for row in self.fleet.snapshot['nodes'].values():
            row.update(kfd_process_count=0, own_device_fd_process_count=0)

    def selection(self, snapshot, jobs, holds=()):
        schedule = plan(snapshot, dict(minimum_ready_training=5, items=jobs),
                        repo=self.fleet.repo, holds=holds,
                        now=self.fleet.now, boot_anchor=self.fleet.anchor)
        return self.launch.pick(snapshot, self.launch.protected_nodes(schedule), self.fleet.now,
                                self.fleet.now + self.fleet.anchor, TRAINING80_LEDGER, fleet_view)

    def test_ready_training_and_validation_keep_their_nodes(self):
        training = self.fleet.trial(1, "training", "f13r1n10")
        validation = self.fleet.trial(2, "validation", "a01r4n16")
        selection = self.selection(self.fleet.verified_snapshot(), [training, validation])
        self.assertNotIn("f13r1n10", selection["PCI"])
        self.assertNotIn("a01r4n16", selection["PCI"])
        self.assertIn("f13r1n11", selection["PCI"])

    def test_clean_half_of_a_held_training_handoff_is_reserved(self):
        protected = self.launch.protected_nodes(dict(queue_held_nodes=["owner"], original_launch_holds={},
            rows=[dict(kind="training", dispatch_state="awaiting_original_queue_release",
                       placement={"owner": 8, "ready-peer": 8})]))
        self.assertEqual(protected, {"owner", "ready-peer"})

    def test_expired_collection_never_dispatches(self):
        with self.assertRaisesRegex(ValueError, "fresh fleet receipt"):
            self.selection(self.fleet.verified_snapshot(age=90), [])

    def test_contradictory_device_probe_isolates_only_its_node(self):
        self.fleet.snapshot['nodes']['f13r1n11']['kfd_process_count'] = 1
        selected = self.selection(self.fleet.verified_snapshot(), [])
        self.assertNotIn('f13r1n11', selected['PCI'])
        self.assertIn('f13r1n11', selected['defer'])
        self.assertIn('f13r1n10', selected['PCI'])

    def test_shard_is_stable_capped_and_single_group(self):
        shard = source("dispatch").shard_nodes
        owners = {f"train-{i}": ("1087656", "T01") for i in range(5)}
        owners.update({"infer": ("1087656", "I03"), "other-parent": ("1103606", "T01")})
        occupied = {"1087656": {"train-0"}, "1103606": set()}
        result = shard(reversed(list(owners)), owners, occupied)
        self.assertEqual(result, shard(owners, owners, occupied))
        self.assertEqual(len(result), 3)
        for parent, nodes in result.values():
            self.assertTrue(1 <= len(nodes) <= 4)
            self.assertEqual(len({owners[node] for node in nodes}), 1)
            self.assertEqual(owners[nodes[0]][0], parent)
            self.assertNotIn("train-0", nodes)


if __name__ == "__main__":
    unittest.main()
