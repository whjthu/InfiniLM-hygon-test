"""CPU fixtures for fan-out pulse identity and full fleet busy denominators."""

from copy import deepcopy
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

import collect_training80_busy_pulses as sampler


REQUEST = "5" * 32
PIN = "a" * 64
CARDS = [f"0000:{i:02x}:00.0" for i in range(8)]


def fleet():
    nodes = {parent: [f"f{index}r1n{node:02d}" for node in range(40)]
             for index, parent in enumerate(sorted(sampler.PARENTS), 11)}
    owners = sampler.node_map({"groups": {"training": {"allocations": nodes}}})
    return nodes, owners, {node: CARDS[:] for node in owners}


def responder(node, parent, event="pulse", nonce="b" * 32):
    identity = dict(event=event, node=node, parent=parent, step=parent + ".42", request=REQUEST,
                    pci_pins_SHA256=PIN)
    if event == "pulse":
        identity.update(sequence=0, nonce=nonce, busy=dict.fromkeys(CARDS, 81), failed_pci=[], error=None,
                        local_read_ns=9000)
    return identity


def slices(owners, points):
    return [dict(sent_ns=1_000_000_000 + sequence * sampler.INTERVAL_NS, nonce=f"{sequence:032x}",
                 rows={node: (responder(node, parent, nonce=f"{sequence:032x}"),
                              1_100_000_000 + sequence * sampler.INTERVAL_NS)
                       for node, (parent, _) in owners.items()})
            for sequence in range(points)]


class PulseTest(unittest.TestCase):
    def test_dual_parent_full_denominators_and_strict_threshold(self):
        _, owners, pins = fleet()
        for planned in (3, 5):
            with self.subTest(planned=planned):
                observations = slices(owners, planned)
                receipt = sampler.summarize(owners, pins, observations, 250_000_000, True)
                self.assertTrue(receipt["complete"])
                self.assertEqual(receipt["expected_samples"], 640 * planned)
                self.assertEqual(receipt["valid_samples"], 640 * planned)
                self.assertEqual(receipt["observed_unique_node_PCI"], 640)
                self.assertEqual(receipt["fleet_mean_busy_percent"], 81)
                self.assertTrue(receipt["all_fleet_points_strictly_above_80_percent"])
                self.assertFalse(receipt["sustained_interval_qualified"])
                for row, _ in observations[0]["rows"].values():
                    row["busy"] = dict.fromkeys(CARDS, 80)
                result = sampler.summarize(owners, pins, observations, 250_000_000, True)
                self.assertEqual(result["points"][0]["fleet_mean_busy_percent"], 80)
                self.assertFalse(result["all_fleet_points_strictly_above_80_percent"])

    def test_missing_node_card_and_unattempted_points_count_in_denominator(self):
        _, owners, pins = fleet()
        observations = slices(owners, 5)
        first, second = list(owners)[:2]
        observations[0]["rows"].pop(first)
        observations[1]["rows"][second][0]["busy"].pop(CARDS[2])
        observations[1]["rows"][second][0]["failed_pci"] = [CARDS[2]]
        observations[4] = {"sent_ns": None, "rows": {}}
        result = sampler.summarize(owners, pins, observations, 250_000_000, True)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["fleet_mean_busy_percent"])
        self.assertEqual(result["missing_samples"], 649)
        self.assertEqual(result["points"][0]["missing_nodes"], [first])
        self.assertEqual(result["points"][1]["missing_PCI_by_node"], {second: [CARDS[2]]})
        self.assertEqual(result["points"][4]["responded_nodes"], 0)

    def test_late_full_response_and_unverified_accounting_do_not_qualify(self):
        _, owners, pins = fleet()
        observations = slices(owners, 3)
        node = next(iter(owners))
        original = observations[0]["rows"][node]
        observations[0]["rows"][node] = original[0], 1_250_000_001
        result = sampler.summarize(owners, pins, observations, 250_000_000, True)
        self.assertEqual(result["valid_samples"], 1920)
        self.assertEqual(result["points"][0]["sample_span_upper_bound_ns"], 250_000_001)
        self.assertIsNone(result["points"][0]["fleet_mean_busy_percent"])
        self.assertIsNone(result["fleet_mean_busy_percent"])
        observations[0]["rows"][node] = original
        result = sampler.summarize(owners, pins, observations, 250_000_000, False)
        self.assertTrue(result["points"][0]["complete"])
        self.assertIsNone(result["points"][0]["fleet_mean_busy_percent"])
        self.assertFalse(result["complete"])

    def test_worker_binds_nonce_then_reads_and_reports_each_pulse(self):
        placement, _, _ = fleet()
        parent, nodes = next(iter(placement.items()))
        identity = responder(nodes[0], parent, "ready")
        packet = [dict(request=REQUEST, nonce=f"{seq:032x}", sequence=seq) for seq in range(3)]
        wire = io.BytesIO("".join(json.dumps(row) + "\n" for row in packet).encode())
        sent = []

        def read_after_ready(drm, expected):
            self.assertGreater(len(sent), 0)
            self.assertEqual(expected, CARDS)
            return dict.fromkeys(CARDS, 27), []

        with patch.object(sampler, "read_counters", side_effect=read_after_ready) as reads:
            sampler.run_worker(identity, CARDS, 3, wire, sent.append, Path("."))
            self.assertEqual(reads.call_count, 3)
        self.assertEqual([row["nonce"] for row in sent[1:]], [row["nonce"] for row in packet])
        self.assertEqual(sent[3]["busy"], dict.fromkeys(CARDS, 27))
        packet[1]["nonce"] = packet[0]["nonce"]
        duplicate = io.BytesIO("".join(json.dumps(row) + "\n" for row in packet).encode())
        with patch.object(sampler, "read_counters", return_value=(dict.fromkeys(CARDS, 27), [])) as reads:
            with self.assertRaisesRegex(ValueError, "duplicate pulse"):
                sampler.run_worker(identity, CARDS, 3, duplicate, sent.append, Path("."))
            self.assertEqual(reads.call_count, 1)

    def test_read_actual_fixture_counters_and_keep_invalid_card_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            drm = Path(directory)
            for index, card in enumerate(CARDS):
                device = drm / f"renderD{128 + index}" / "device"
                device.mkdir(parents=True)
                (device / "gpu_busy_percent").write_bytes(b"101\n" if index == 2 else b"83\n")
            original_resolve = Path.resolve

            def resolve(path, strict=False):
                if path.name == "device":
                    index = int(path.parent.name[7:]) - 128
                    return PurePosixPath("/devices") / CARDS[index]
                return original_resolve(path, strict=strict)

            with patch.object(Path, "resolve", resolve):
                busy, failed = sampler.read_counters(drm, CARDS)
                self.assertEqual(busy, {card: 83 for card in CARDS if card != CARDS[2]})
                self.assertEqual(failed, [CARDS[2]])
                with self.assertRaisesRegex(ValueError, "PCI set"):
                    sampler.read_counters(drm, CARDS[1:])

    def test_controller_rejects_foreign_duplicate_or_wrong_nonce(self):
        placement, owners, pins = fleet()
        parent, nodes = next(iter(placement.items()))
        receiver = sampler.Receiver(owners, REQUEST, PIN, pins)
        for node in nodes[:2]:
            receiver.accept(responder(node, parent, "ready"), parent, "ready", receiver.ready, 1)
        rows = {}
        packet = {"sequence": 0, "nonce": "b" * 32}
        valid = responder(nodes[0], parent)
        receiver.accept(valid, parent, "pulse", rows, 50_000, packet, 40_000)
        with self.assertRaisesRegex(ValueError, "duplicate node"):
            receiver.accept(valid, parent, "pulse", rows, 51_000, packet, 40_000)
        foreign = responder(nodes[1], parent)
        foreign["step"] = parent + ".43"
        with self.assertRaisesRegex(ValueError, "nonce and sequence"):
            receiver.accept(foreign, parent, "pulse", rows, 52_000, packet, 40_000)
        duplicate_nonce = responder(nodes[1], parent, nonce="f" * 32)
        with self.assertRaisesRegex(ValueError, "nonce and sequence"):
            receiver.accept(duplicate_nonce, parent, "pulse", rows, 53_000, packet, 40_000)
        with self.assertRaisesRegex(ValueError, "worker interval"):
            receiver.accept(responder(nodes[1], parent), parent, "pulse", rows, 53_000, packet, 45_000)
        foreign = responder("foreign", parent)
        with self.assertRaisesRegex(ValueError, "foreign worker"):
            receiver.accept(foreign, parent, "pulse", rows, 54_000, packet, 40_000)

    def test_exact_zero_gres_accounting_and_arguments(self):
        placement, _, _ = fleet()
        steps = {parent: parent + ".42" for parent in placement}
        rows = {parent: [step, "busy-label", "COMPLETED", "0:0", "8", parent,
                         "cpu=40,mem=10G,node=40"] for parent, step in steps.items()}
        hostnames = lambda parent: set(placement[parent])
        self.assertTrue(sampler.qualified_accounting(rows, steps, placement, hostnames))
        unknown = deepcopy(rows)
        unknown[next(iter(unknown))][2] = "RUNNING"
        self.assertFalse(sampler.qualified_accounting(unknown, steps, placement, hostnames))
        foreign = deepcopy(rows)
        foreign[next(iter(foreign))][6] += ",gres/dcu=1"
        self.assertFalse(sampler.qualified_accounting(foreign, steps, placement, hostnames))
        parent = next(iter(placement))
        argv = sampler.slurm_argv(parent, placement[parent], "busy-label", REQUEST, 5, Path("pins.json"), PIN)
        for expected in ("--gres=none", "--overlap", "--input=all", "--ntasks=40", "--nodes=40", "--time=00:01:00"):
            self.assertIn(expected, argv)


if __name__ == "__main__":
    unittest.main()
