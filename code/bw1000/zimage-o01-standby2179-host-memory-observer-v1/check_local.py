"""Offline cgroup parser and accounting boundary checks."""

from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observer as o


def fixture(directory, files):
    for key, val in files.items():
        (directory / key).write_text(val + "\n", encoding="ascii")


class ObserverTest(unittest.TestCase):
    def test_controller_selection_identifies_exact_step(self):
        task = "/slurm/uid_42/job_1103606/step_2179/user/task_0"
        expected = PurePosixPath("/slurm/uid_42/job_1103606/step_2179")
        self.assertEqual(o.memory_controller(f"0::{task}\n"), (2, PurePosixPath(task), expected))
        self.assertEqual(o.memory_controller(f"0::/\n7:cpu:/else\n8:memory:{task}\n")[0], 1)
        for wrong in (task.replace("step_2179", "step_21790"), "/../" + task[1:]):
            with self.assertRaises(ValueError):
                o.memory_controller(f"0::{wrong}\n")
        with self.assertRaisesRegex(ValueError, "UNRESOLVED"):
            o.memory_controller(f"0::{task}\n0::{task}\n")

    def test_prioritizes_most_specific_mount_and_decodes_escapes(self):
        step = PurePosixPath("/slurm/job_1103606/step_2179")
        rows = "\n".join([
            "19 1 0:4 / /sys/other rw - cgroup cgroup rw,memory",
            "20 1 0:5 / /sys/fs/cgroup rw - cgroup2 cgroup rw",
            r"21 1 0:5 /slurm /sys/fs/cgroup/slurm\040bind ro - cgroup2 cgroup rw",
        ])
        self.assertEqual(o.mount_candidates(rows, 2, step),
                         {(PurePosixPath("/slurm"), PurePosixPath("/sys/fs/cgroup/slurm bind"))})
        self.assertEqual(o.mount_candidates(rows, 1, step),
                         {(PurePosixPath("/"), PurePosixPath("/sys/other"))})

    def test_v2_distinguishes_hard_limit_counters_and_optional_peak(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = root / "job_1103606"
            step = job / "step_2179"
            step.mkdir(parents=True)
            fixture(job, {"memory.max": "1000", "memory.current": "200"})
            fixture(step, {"memory.max": "max", "memory.current": "90", "memory.events": "max 2\noom 1\noom_kill 0",
                           "memory.stat": "anon 70\nfile 15\nshmem 2\npgfault 100"})
            result = o.memory_counters(root, step, 2)
            row = result["step_memory"]
            self.assertEqual(result["minimum_visible_hard_limit_bytes"], 1000)
            self.assertFalse(row["peak_bytes"]["available"])
            self.assertTrue(row["limit_bytes"]["unlimited"])
            self.assertEqual(row["events"]["value"]["oom_kill"], 0)
            self.assertEqual(row["classified_bytes"], {"anon": 70, "file": 15, "shmem": 2})

    def test_v1_preserves_failcnt_without_inventing_oom_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            step = root / "step_2179"
            step.mkdir()
            fixture(root, {"memory.limit_in_bytes": "9223372036854771712", "memory.usage_in_bytes": "200"})
            fixture(step, {"memory.limit_in_bytes": "256", "memory.usage_in_bytes": "90",
                           "memory.max_usage_in_bytes": "120", "memory.failcnt": "7",
                           "memory.oom_control": "under_oom 0\noom_kill_disable 0",
                           "memory.stat": "cache 30\nrss 60\ntotal_rss 60\npgfault 100"})
            row = o.memory_counters(root, step, 1)["step_memory"]
            self.assertEqual(row["peak_bytes"]["value"], 120)
            self.assertEqual(row["limit_exceed_count"]["value"], 7)
            self.assertNotIn("oom_kill", row["oom_control"]["value"])
            self.assertIn("total_rss", row["classified_bytes"])
            with self.assertRaisesRegex(ValueError, "DUPLICATE_COUNTER"):
                o.table("rss 4\nrss 5")

    def test_exact_row_and_published_identity(self):
        source = Path(__file__).resolve().parent.parent / "zimage-lt-eight-card-attested-entry-v1" / "node_entry.py"
        identity = o.load_identity(source)
        self.assertTrue(identity.task_pid("PID JOBID STEPID\n982368 1103606 2179 0", o.TARGET, o.WORKER_PID))
        self.assertFalse(identity.task_pid("982368 1103606 2178 0", o.TARGET, o.WORKER_PID))
        row = f"StepId={o.TARGET} Name={o.TARGET_LABEL} State=RUNNING NodeList={o.NODE} TRES=cpu=8,gres/dcu=8,mem=16G,node=1"
        self.assertEqual(o.known_step("StepId=other\n" + row, o.TARGET, o.TARGET_LABEL, 8, 8, "16G")["StepId"], o.TARGET)
        for bad in (row + "\n" + row, row.replace("RUNNING", "COMPLETED"), row.replace("gres/dcu=8", "gres/dcu=1")):
            with self.assertRaises(ValueError):
                o.known_step(bad, o.TARGET, o.TARGET_LABEL, 8, 8, "16G")


if __name__ == "__main__":
    unittest.main()
