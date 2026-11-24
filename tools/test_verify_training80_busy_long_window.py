"""Independent hour-denominator verification using small and streamed offline fixtures."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_collect_training80_busy_long_window import CARDS, Clock, DEADLINE, PIN, REQUEST, START, Transport, collect, fleet
import verify_training80_busy_long_window as verify


def wire(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


class VerifyWindowTest(unittest.TestCase):
    def test_independent_reader_detects_changed_interval_and_keeps_unfinished_tail_unknown(self):
        receiver, clock = fleet(), Clock()
        transport = Transport(receiver, clock)
        with TemporaryDirectory() as temp:
            output = Path(temp) / "once"
            measured = collect(output, receiver, clock, transport, lambda: clock.now >= START + 3 * verify.INTERVAL_NS)
            report = verify.verify(output, receiver.owners, receiver.pins, REQUEST, PIN)
            self.assertEqual(report["qualified_device_seconds"], measured["qualified_device_seconds"])
            self.assertEqual(report["expected_device_seconds"], 2_304_000)
            self.assertFalse(report["hour_interval_coverage_complete"])
            # Missing outcome describes a recoverable interrupted read, never a complete hour.
            (output / "outcome.json").unlink()
            recovered = verify.verify(output, receiver.owners, receiver.pins, REQUEST, PIN)
            self.assertFalse(recovered["terminal_outcome_present"])
            self.assertEqual(recovered["unknown_device_seconds"], 2_304_000 - 2_560)
            changed = output / "point-0001.json"
            point = json.loads(changed.read_bytes())
            point["preceding_interval"]["valid_device_seconds"] -= 2
            changed.write_bytes(wire(point))
            with self.assertRaisesRegex(ValueError, "does not recompute"):
                verify.verify(output, receiver.owners, receiver.pins, REQUEST, PIN)

    def test_independent_reader_recomputes_full_hour_with_one_streamed_point_at_a_time(self):
        receiver = fleet()
        request = receiver.request
        steps = receiver.parent_steps
        intents = dict(request=request, parents=verify.PARENTS, worker_steps=steps, pci_pins_SHA256=PIN, ready_nodes=80,
                       planned_points=1801, expected_samples=1_152_640, expected_device_seconds=2_304_000,
                       earliest_first_monotonic_ns=START, earliest_final_planned_monotonic_ns=START+verify.DURATION_NS,
                       absolute_deadline_ns=DEADLINE, interval_ns=verify.INTERVAL_NS, span_limit_ns=verify.SPAN_NS,
                       GPU_requested=0, replay_allowed=False)
        intent_raw = wire(intents)
        def raw_point(seq):
            sent = START + seq * verify.INTERVAL_NS
            arrived = sent + 5_000_000
            nonce = f"{seq:032x}"
            point = dict(sequence=seq, expected_samples=640, observed_samples=640, responded_nodes=80,
                         status="COMPLETE", complete=True, valid_samples=640, missing_samples=0, rejected_frames=0,
                         nonce=nonce, planned_controller_monotonic_ns=sent, controller_send_monotonic_ns=sent,
                         controller_last_receipt_monotonic_ns=arrived, sample_span_upper_bound_ns=arrived-sent,
                         worker_reports=[dict(node=node, parent=parent, step=steps[parent], event="pulse",
                             pci_pins_SHA256=PIN, request=request, sequence=seq, nonce=nonce, local_read_ns=100,
                             controller_receipt_monotonic_ns=arrived, busy=dict.fromkeys(CARDS, 90),
                             failed_pci=[], error=None) for node,(parent,_) in receiver.owners.items()])
            if seq:
                point["preceding_interval"] = dict(start_planned_monotonic_ns=sent-verify.INTERVAL_NS,
                    end_planned_monotonic_ns=sent, expected_device_seconds=1280, valid_device_seconds=1280,
                    unknown_device_seconds=0, busy_percent_device_seconds=1280*90)
            return wire(point)

        chain = bytes(32)
        for seq in range(1801):
            chain = sha256(chain + sha256(raw_point(seq)).digest()).digest()
        outcome = wire(dict(request=request, persisted_points=1801, complete_points=1801, expected_samples=1_152_640,
                        valid_samples=1_152_640, observed_samples=1_152_640, missing_samples=0,
                        expected_device_seconds=2_304_000, qualified_device_seconds=2_304_000, unknown_device_seconds=0,
                        measured_send_span_ns=verify.DURATION_NS, hour_interval_coverage_complete=True,
                        full_hour_interval_busy_percent=90, stop_reason=None, intent_SHA256=sha256(intent_raw).hexdigest(),
                        point_digest_chain_SHA256=chain.hex()))
        def read(path):
            if path.name == "intent.json":
                return intent_raw
            if path.name == "outcome.json":
                return outcome
            return raw_point(int(path.stem.split("-")[1]))
        with patch.object(verify, "bounded", side_effect=read), patch.object(Path, "exists", return_value=True):
            report = verify.verify(Path("streamed-fixture"), receiver.owners, receiver.pins, request, PIN)
        self.assertTrue(report["hour_interval_coverage_complete"])
        self.assertEqual(report["qualified_device_seconds"], 2_304_000)
        self.assertEqual(report["full_hour_interval_busy_percent"], 90)
        self.assertFalse(report["zero_GRES_terminal_accounting_verified"])
        self.assertFalse(report["sustained_interval_qualified"])


if __name__ == "__main__":
    unittest.main()
