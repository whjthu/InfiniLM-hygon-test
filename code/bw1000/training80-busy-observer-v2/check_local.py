import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import prepare
import pulse_schedule

sys.path.insert(0, str(prepare.REPO / "tools"))
import collect_training80_busy_pulses as stable


REQUEST = "c" * 32
CARDS = [f"0000:{i:02x}:00.0" for i in range(8)]


def pulse(sequence, nonce=None):
    return dict(request=REQUEST, sequence=sequence, nonce=nonce or f"{sequence:032x}")


def frames(rows):
    return io.BytesIO(b"".join(json.dumps(row).encode() + b"\n" for row in rows))


class ObserverProtocolTest(unittest.TestCase):
    def test_missed_345_keeps_346_and_last_sample_without_reading_the_gap(self):
        ready = []
        packets = [pulse(344), pulse(346), pulse(1800), pulse_schedule.finish_packet(REQUEST, 1801)]
        with patch.object(stable, "read_counters", return_value=({p: 97 for p in CARDS}, [])) as read:
            pulse_schedule.run_worker(stable, {"request": REQUEST}, CARDS, 1801,
                                      frames(packets), ready.append, Path("/dev/not-used"))
        self.assertEqual([(r["event"], r.get("sequence")) for r in ready],
                         [("ready", None), ("pulse", 344), ("pulse", 346), ("pulse", 1800)])
        self.assertEqual(read.call_count, 3)

    def test_duplicate_foreign_out_of_order_and_unacknowledged_eof_are_rejected(self):
        wrong = dict(pulse(1), request="d" * 32)
        cases = ([pulse(1), pulse(1)], [pulse(10), pulse(2)], [wrong], [pulse(1801)],
                 [pulse(0), pulse(2, pulse(0)["nonce"])], [pulse(100)],
                 [pulse_schedule.finish_packet(REQUEST, 1801), pulse(7)])
        with patch.object(stable, "read_counters", return_value=({}, list(CARDS))):
            for packets in cases:
                with self.subTest(packets=packets), self.assertRaises(ValueError):
                    pulse_schedule.run_worker(stable, {"request": REQUEST}, CARDS, 1801,
                                              frames(packets), lambda _: None, Path("/dev/not-used"))

    def test_trailing_missed_slots_still_close_with_zero_counter_reads(self):
        report = []
        with patch.object(stable, "read_counters") as read:
            pulse_schedule.run_worker(stable, {"request": REQUEST}, CARDS, 1801,
                                      frames([pulse_schedule.finish_packet(REQUEST, 1801)]),
                                      report.append, Path("/dev/not-used"))
        self.assertEqual([r["event"] for r in report], ["ready"])
        read.assert_not_called()

    def test_rendered_controller_keeps_original_full_hour_coverage_gate(self):
        raw = prepare.packet()[prepare.BASE]
        self.assertIn(b'transport.send(pulse_schedule.finish_packet(request, window.POINTS)', raw)
        self.assertIn(b'full_hour_observer_C0=verified and not interrupted and schedule_ended', raw)
        self.assertIn(b'"persisted_points") == window.POINTS', raw)
        self.assertIn(b'and schedule_error is None and set(released) == set(expected)', raw)


if __name__ == "__main__":
    unittest.main()
