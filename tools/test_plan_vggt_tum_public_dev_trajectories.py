import math
import unittest

from plan_vggt_tum_public_dev_trajectories import parse_trajectory, plan, pose_angles


def row(time, x):
    return f"{time} {x} 0 0 0 0 0 1\n".encode()


class TumPublicDevelopmentTrajectories(unittest.TestCase):
    def test_keeps_no_arbitrary_pose_when_timestamp_values_conflict(self):
        raw = row(1, 0) + row(1, 0) + row(2, 1) + row(2, 3) + row(3, 3)
        accepted, report = parse_trajectory(raw)
        self.assertEqual([r[0] for r in accepted], [1, 3])
        self.assertEqual(report["identical_duplicate_rows_collapsed"], 1)
        self.assertEqual(report["conflicting_timestamp_rows_excluded"], 2)

    def test_deterministic_separate_windows_and_three_temporal_gaps(self):
        data, _ = parse_trajectory(b"".join(row(i / 100, i / 200) for i in range(3601)))
        rows = plan(data)
        self.assertEqual(rows, plan(data))
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["requested_gap_seconds"] for r in rows}, {0.15, 0.30, 0.45})
        times = [time for pair in rows for time in pair["original_pose_times"]]
        self.assertEqual(len(set(times)), 24)
        self.assertTrue(all(r["state"] == "POSE_PAIR_CANDIDATE" for r in rows))

    def test_rejects_bad_quaternion_and_retains_static_windows(self):
        data, _ = parse_trajectory(b"".join(row(i / 100, 0) for i in range(3601)))
        self.assertTrue(all(r["state"] == "LOW_TRANSLATION" for r in plan(data)))
        with self.assertRaisesRegex(ValueError, "quaternion"):
            parse_trajectory(b"0 0 0 0 0 0 0 0\n" + row(1, 1))
        ninety = (0, 0, 0, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        self.assertAlmostEqual(pose_angles(data[0], ninety), 90.0)


if __name__ == "__main__":
    unittest.main()
