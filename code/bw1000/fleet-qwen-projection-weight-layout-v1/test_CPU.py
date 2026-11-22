"""Exercise the original scope, atomic handoff, and useful work denominator locally."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import launch
import layout_scope as scope


def pci():
    return [f"0000:{i:02x}:00.0" for i in range(8)]


def evidence():
    a, b = "checkpoint_transpose_view", "pretransposed_contiguous"
    row = lambda seconds: dict(seconds=seconds, drains=2, projections=128, rows=184*128)
    return dict(M=184, K=4096, N=1024, sync_depth=64, distinct_inputs=2, numeric=True,
                arm_order=[a,b,b,a],layout_strides={a:[1,4096],b:[1024,1]}, packed_logical_bytes=8388608,
                pack_wall_s=0.001,extra_resident_allocated_bytes=8388608,
                measurements={a:[row(8),row(8)],b:[row(6),row(6)]})


class Scope(unittest.TestCase):
    def setUp(self):
        self.files = launch.source(launch.REPO)
        self.roster = scope.owners(json.loads(self.files["ledger.json"]))
        self.selected = {node:pci() for node in self.roster if node not in scope.KNOWN_FOREIGN_KFD}

    def test_exact_membership_full_request_and_consistent_shard_ids(self):
        self.assertRegex(scope.REQUEST, r"^[0-9a-f]{32}$")
        self.assertEqual(json.loads(self.files["CLAIM.json"])["request"],scope.REQUEST)
        groups = scope.shards(self.selected, self.roster)
        self.assertEqual(sum(len(row["nodes"]) for row in groups.values()),78)
        self.assertEqual(len({row["request"] for row in groups.values()}),len(groups))
        self.assertTrue(all(re.fullmatch("[0-9a-f]{32}",row["request"]) and 1<=len(row["nodes"])<=4
                            for row in groups.values()))
        bad = {**self.selected,next(iter(scope.KNOWN_FOREIGN_KFD)):pci()}
        with self.assertRaises(ValueError):
            scope.shards(bad,self.roster)

    def test_admission_keeps_exact_holds_and_other_clean_nodes(self):
        nodes = list(self.selected)[:6]
        current = {n:dict(admission="clear",parent=self.roster[n][0]) for n in self.roster}
        snapshot = dict(local_receipt=dict(payload_SHA256="a"*64),
                        nodes={n:dict(PCIs=pci(),active_steps=[],kfd_process_count=0,
                                      own_device_fd_process_count=0) for n in nodes})
        snapshot["nodes"][nodes[3]]["kfd_process_count"] = 1
        current[nodes[4]]["admission"]="unknown"
        schedule=dict(queue_held_nodes=[nodes[0]],original_launch_holds={"example":{"nodes":[nodes[1]]}},
                      rows=[dict(kind="training",dispatch_state="proposed_for_root",placement=[nodes[2]])])
        with patch.object(launch,"fleet_view",return_value=(None,current,True)):
            result=launch.make_plan(snapshot,schedule,50,200,self.roster,nodes)
        self.assertEqual(set(result["PCI"]),{nodes[5]})
        self.assertEqual(len(result["groups"]),1)

    def test_reserve_wave_uses_current_production_handoff_only_on_test_db(self):
        selected = {node:pci() for node in list(self.selected)[:6]}
        campaign=dict(groups=scope.shards(selected,self.roster))
        with TemporaryDirectory() as directory:
            queue=launch.Queue(Path(directory)/"test.sqlite")
            try:
                receipt=dict(payload_SHA256="a"*64,raw_response_SHA256="b"*64,request_id="c"*32)
                queue.db.execute("BEGIN IMMEDIATE")
                holds=launch.reserve_wave(queue,campaign,"d"*40,receipt,400,"e"*64,Path("local-only/test"))
                counted={node for item in holds.values() for node in item["handle"]["nodes"]}
                self.assertEqual(counted,set(selected))
                self.assertEqual(queue.db.execute("SELECT count(*) FROM runs WHERE state='DISPATCHING'").fetchone()[0],
                                 len(holds))
                queue.db.rollback()
                self.assertEqual(queue.db.execute("SELECT count(*) FROM runs").fetchone()[0],0)
            finally:
                queue.db.close()

    def test_complete_valid_paired_counts_and_lossless_bf16_layout(self):
        prior=launch.REPO/launch.PRIOR
        spec=spec_from_file_location("weight_layout_dispatch",launch.HERE/"dispatch.py")
        dispatch=module_from_spec(spec)
        with patch.object(sys,"path",[str(prior),*sys.path]):
            spec.loader.exec_module(dispatch)
        value=evidence()
        pairs, ratio=dispatch.checked_trial(value)
        self.assertEqual(len(pairs),2)
        self.assertAlmostEqual(ratio,4/3)
        for bad in (dict(value,numeric=False),dict(value,sync_depth=8),dict(value,arm_order=value["arm_order"][:3])):
            with self.assertRaises(ValueError):
                dispatch.checked_trial(bad)
        value["measurements"]["checkpoint_transpose_view"][0]["rows"] -= 1
        with self.assertRaises(ValueError):
            dispatch.checked_trial(value)


if __name__ == "__main__":
    unittest.main()
