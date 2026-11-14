from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gencuda.backends.hunyuanvideo_data_parallel import TIMING_BOUNDARY, rank_window_metrics


class RankWindowTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.logs = []
        for rank in range(2):
            (self.root / f"rank-{rank}").mkdir()

    def write_summary(self, rank, *, start=0, steps=20, seconds=100, boundary=TIMING_BOUNDARY):
        path = self.root / f"rank-{rank}" / "summary.json"
        raw = json.dumps({
            "rank": rank, "completed_global_step_at_start": start,
            "measurement_steps": steps, "measurement_wall_seconds": seconds,
            "timing_boundary": boundary,
        }).encode()
        path.write_bytes(raw)
        self.logs.append({"path": str(path), "sha256": sha256(raw).hexdigest()})
        return path

    def test_rank_one_can_determine_throughput(self):
        self.write_summary(0, seconds=100)
        self.write_summary(1, seconds=105)
        metrics = rank_window_metrics(self.root, self.logs[::-1], warmup=5, measured=20, resumed=False)
        self.assertEqual(metrics["rank_measurement_wall_seconds"], [100, 105])
        self.assertEqual(metrics["measurement_updates_per_second"], 20 / 105)
        self.assertTrue(metrics["complete_5_warmup_20_measured_window"])

    def test_resume_has_its_own_window(self):
        self.write_summary(0, start=20, steps=5, seconds=25)
        self.write_summary(1, start=20, steps=5, seconds=24)
        metrics = rank_window_metrics(self.root, self.logs, warmup=5, measured=20, resumed=True)
        self.assertEqual(metrics["measurement_updates_per_second"], 5 / 25)
        self.assertFalse(metrics["complete_5_warmup_20_measured_window"])

    def test_disagreement_even_with_equal_measured_counts(self):
        self.write_summary(0, start=3)
        self.write_summary(1, start=4)
        with self.assertRaisesRegex(RuntimeError, "starts disagree"):
            rank_window_metrics(self.root, self.logs, warmup=5, measured=20, resumed=True)

    def test_modified_summary_refused(self):
        path = self.write_summary(0)
        self.write_summary(1)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            rank_window_metrics(self.root, self.logs, warmup=5, measured=20, resumed=False)

    def test_wrong_boundary_refused(self):
        self.write_summary(0)
        self.write_summary(1, boundary="wall time including warmup")
        with self.assertRaisesRegex(RuntimeError, "timing boundary mismatch"):
            rank_window_metrics(self.root, self.logs, warmup=5, measured=20, resumed=False)


if __name__ == "__main__":
    unittest.main()
