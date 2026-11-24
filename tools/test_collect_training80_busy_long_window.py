"""Offline checks for the exact full-hour driver-counter denominator and gaps."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from collect_training80_busy_pulses import Receiver
import collect_training80_busy_long_window as window


REQUEST = "1" * 32
PIN = "a" * 64
CARDS = [f"0000:{idx:02x}:00.0" for idx in range(8)]
START = 10_000_000_000
DEADLINE = START + window.DURATION_NS + window.INTERVAL_NS


def fleet(ready=True):
    owners = {f"f{idx}r1n{node:02d}": (parent, "fixture")
              for idx, parent in enumerate(window.PARENTS, 12) for node in range(1, 41)}
    receiver = Receiver(owners, REQUEST, PIN, {node: CARDS for node in owners})
    if ready:
        for node, (parent, _) in owners.items():
            row = dict(node=node, parent=parent, step=parent + ".42", request=REQUEST,
                       pci_pins_SHA256=PIN, event="ready")
            receiver.accept(row, parent, "ready", receiver.ready, START - 1)
    return receiver


class Clock:
    def __init__(self):
        self.now = START

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Transport:
    def __init__(self, receiver, clock, mutate=None):
        self.receiver = receiver
        self.clock = clock
        self.mutate = mutate
        self.packets = []

    def send(self, packet, deadline):
        if deadline <= self.clock.now:
            raise ValueError("fixture send passed the deadline")
        self.packets.append((packet, self.clock.now))

    def receive_until(self, cutoff):
        packet, sent = self.packets[-1]
        self.clock.now = sent + 5_000_000
        if self.clock.now > cutoff:
            raise ValueError("fixture reply passed the deadline")
        for node, (parent, _) in self.receiver.owners.items():
            row = dict(**packet, node=node, parent=parent, step=parent + ".42", event="pulse",
                       pci_pins_SHA256=PIN, busy=dict.fromkeys(CARDS, 82), failed_pci=[], error=None, local_read_ns=100)
            if self.mutate:
                yield from self.mutate(parent, row, self.clock.now)
            else:
                yield parent, row, self.clock.now


def collect(folder, receiver, clock, traffic, stop=None, nonce=None):
    values = iter(f"{i:032x}" for i in range(window.POINTS))
    return window.collect_window(folder, receiver, traffic, START, DEADLINE,
        stopping=stop or (lambda: False), clock=clock.monotonic_ns, sleep=clock.sleep, nonce=nonce or values.__next__)


class LongWindowTest(unittest.TestCase):
    def test_full_hour_uses_both_endpoints_and_exact_absolute_cadence(self):
        receiver, clock = fleet(), Clock()
        clock.now += 20_000_000
        actual_first = clock.now
        traffic = Transport(receiver, clock)
        emitted = []

        def count(point, record):
            if point.name.startswith("point-"):
                emitted.append((record["sequence"], record["valid_samples"]))
            return sha256(point.name.encode()).digest()

        with TemporaryDirectory() as temp, patch.object(window, "write_receipt", side_effect=count):
            result = collect(Path(temp) / "once", receiver, clock, traffic)
        self.assertEqual(emitted, [(n, 640) for n in range(1801)])
        self.assertEqual([sent for _, sent in traffic.packets],
                         [actual_first + n * window.INTERVAL_NS for n in range(1801)])
        self.assertEqual(result["expected_samples"], result["valid_samples"])
        self.assertEqual(result["expected_samples"], 1_152_640)
        self.assertEqual(result["qualified_device_seconds"], 2_304_000)
        self.assertEqual(result["unknown_device_seconds"], 0)
        self.assertEqual(result["measured_send_span_ns"], window.DURATION_NS)
        self.assertTrue(result["hour_interval_coverage_complete"])
        self.assertEqual(result["full_hour_interval_busy_percent"], 82.0)
        self.assertFalse(result["zero_GRES_terminal_accounting_verified"])
        self.assertFalse(result["sustained_interval_qualified"])

    def test_missing_interior_card_invalidates_two_intervals_but_observed_zero_is_valid(self):
        receiver, clock = fleet(), Clock()
        first = next(iter(receiver.owners))

        def manipulate(parent, row, arrived):
            if row["node"] == first:
                if row["sequence"] == 1:
                    del row["busy"][CARDS[0]]
                    row["failed_pci"].append(CARDS[0])
                row["busy"][CARDS[1]] = 0
            yield parent, row, arrived

        traffic = Transport(receiver, clock, manipulate)
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "once"
            result = collect(folder, receiver, clock, traffic, lambda: clock.now >= START + 3 * window.INTERVAL_NS)
            first_point = json.loads((folder / "point-0000.json").read_bytes())
            middle = json.loads((folder / "point-0001.json").read_bytes())
            final = json.loads((folder / "point-0002.json").read_bytes())
            previous_bytes = (folder / "outcome.json").read_bytes()
            with self.assertRaises(FileExistsError):
                collect(folder, receiver, clock, traffic)
            self.assertEqual((folder / "outcome.json").read_bytes(), previous_bytes)
        self.assertEqual(first_point["valid_samples"], 640)
        self.assertEqual(first_point["nonzero_cards"], 639)
        self.assertEqual(middle["missing_PCI_by_node"], {first: [CARDS[0]]})
        self.assertEqual(middle["preceding_interval"]["unknown_device_seconds"], 2)
        self.assertEqual(final["preceding_interval"]["unknown_device_seconds"], 2)
        self.assertEqual(result["qualified_device_seconds"], 2 * (1280 - 2))
        self.assertEqual(result["unknown_device_seconds"], 2_304_000 - 2 * (1280 - 2))
        self.assertEqual(result["unattempted_range"], [3, 1800])
        self.assertIsNone(result["full_hour_interval_busy_percent"])
        self.assertEqual(result["stop_reason"], "STOP_REQUESTED")

    def test_overrun_leaves_missed_point_and_does_not_retime_the_remaining_window(self):
        receiver, clock = fleet(), Clock()
        traffic = Transport(receiver, clock)
        original_write = window.write_receipt

        def advance(path, record):
            digest = original_write(path, record)
            if path.name == "point-0000.json":
                clock.now = START + window.INTERVAL_NS + window.MAX_START_LAG_NS + 1
            return digest

        with TemporaryDirectory() as temp, patch.object(window, "write_receipt", side_effect=advance):
            folder = Path(temp) / "once"
            result = collect(folder, receiver, clock, traffic, lambda: clock.now >= START + 3 * window.INTERVAL_NS)
            missed = json.loads((folder / "point-0001.json").read_bytes())
        self.assertEqual(missed["status"], "MISSED_SCHEDULE")
        self.assertEqual(missed["valid_samples"], 0)
        self.assertEqual([sent for _, sent in traffic.packets], [START, START + 2 * window.INTERVAL_NS])
        self.assertEqual(result["qualified_device_seconds"], 0)
        self.assertEqual(result["unknown_device_seconds"], 2_304_000)

    def test_foreign_nonce_fails_this_point_and_preserves_raw_observed_count(self):
        receiver, clock = fleet(), Clock()
        first = next(iter(receiver.owners))

        def manipulate(parent, row, arrived):
            if row["node"] == first:
                yield parent, {**row, "nonce": "f" * 32}, arrived
            yield parent, row, arrived

        traffic = Transport(receiver, clock, manipulate)
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "once"
            result = collect(folder, receiver, clock, traffic, lambda: clock.now >= START + window.INTERVAL_NS)
            point = json.loads((folder / "point-0000.json").read_bytes())
        self.assertEqual(point["observed_samples"], 640)
        self.assertEqual(point["valid_samples"], 0)
        self.assertEqual(point["status"], "FRAME_REJECTED")
        self.assertEqual(result["missing_samples"], 1_152_640)

    def test_late_node_does_not_invalidate_other_nodes(self):
        receiver, clock = fleet(), Clock()
        first = next(iter(receiver.owners))

        def late(parent, row, arrived):
            if row["node"] == first and row["sequence"] == 1:
                yield parent, row, arrived + window.SPAN_NS
            else:
                yield parent, row, arrived

        traffic = Transport(receiver, clock, late)
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "once"
            result = collect(folder, receiver, clock, traffic, lambda: clock.now >= START + 3 * window.INTERVAL_NS)
            point = json.loads((folder / "point-0001.json").read_bytes())
        self.assertEqual(point["late_frames"], 1)
        self.assertEqual(point["valid_samples"], 632)
        self.assertEqual(point["missing_nodes"], [first])
        self.assertEqual(point["preceding_interval"]["unknown_device_seconds"], 16)
        self.assertEqual(result["qualified_device_seconds"], 2 * 1264)
        self.assertFalse(result["complete_counter_window"])

    def test_absolute_deadline_stops_tail_even_after_a_good_first_point(self):
        receiver, clock = fleet(), Clock()
        traffic = Transport(receiver, clock)
        writer = window.write_receipt

        def after_deadline(path, record):
            pin = writer(path, record)
            if path.name == "point-0000.json":
                clock.now = DEADLINE + 1
            return pin

        with TemporaryDirectory() as temp, patch.object(window, "write_receipt", side_effect=after_deadline):
            result = collect(Path(temp) / "once", receiver, clock, traffic)
        self.assertEqual(result["complete_points"], 1)
        self.assertEqual(result["unattempted_range"], [1, 1800])
        self.assertEqual(result["stop_reason"], "DEADLINE")
        self.assertIsNone(result["measured_send_span_ns"])
        self.assertEqual(result["unknown_device_seconds"], 2_304_000)
        self.assertFalse(result["hour_interval_coverage_complete"])

    def test_incomplete_start_and_ambiguous_send_do_not_restart(self):
        for ready in (False, True):
            receiver, clock = fleet(ready), Clock()
            traffic = Transport(receiver, clock)
            with TemporaryDirectory() as temp, patch.object(traffic, "send", side_effect=OSError("fixture")) as call:
                result = collect(Path(temp) / "once", receiver, clock, traffic)
            self.assertEqual(call.call_count, int(ready))
            self.assertEqual(result["stop_reason"], "TRANSPORT_ERROR" if ready else "STARTUP_INCOMPLETE")
            self.assertEqual(result["unattempted_range"], [int(ready), 1800])
            self.assertEqual(result["missing_samples"], 1_152_640)
            self.assertEqual(result["unknown_device_seconds"], 2_304_000)
            self.assertFalse(result["hour_interval_coverage_complete"])


if __name__ == "__main__":
    unittest.main()
