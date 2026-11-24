"""Exercise PID reuse and the two independent procfs access boundaries."""

from contextlib import contextmanager
import errno
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from probe import compare, inspect_fd
from launch_once import LABEL, NODE, PARENT, srun


class OwnedProcTest(unittest.TestCase):
    def run_view(self, *, owner_start=77, link=lambda _: "/dev/null", scandir=None):
        with TemporaryDirectory() as directory:
            proc = Path(directory)
            process = proc / "17"
            process.mkdir()
            path = process / "fd"
            path.mkdir()
            (path / "4").touch()
            fields = ["S", "3", *(["0"] * 17), "77"]
            (process / "stat").write_text("17 (n ) a me) " + " ".join(fields), encoding="utf-8")
            uid = process.stat().st_uid
            by = dict(readlink=link)
            if scandir is not None:
                by["scandir"] = scandir
            return inspect_fd(dict(pid=17, ppid=3, startticks=owner_start, uid=uid, fd=4), uid=uid, proc=proc, **by)

    def test_sentinel_read_and_foreign_link(self):
        self.assertEqual(self.run_view(), dict(enumerate="ok", readlink="ok"))
        self.assertEqual(self.run_view(link=lambda _: "/other")["readlink"], "foreign_sentinel")

    def test_pid_reuse_refuses_before_descriptor_scan(self):
        def forbidden(_):
            self.fail("reused PID was observed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_view(owner_start=76, scandir=forbidden)

    def test_enumeration_and_link_permission_errors_are_distinct(self):
        def denied(_):
            raise PermissionError(errno.EACCES, "private")

        @contextmanager
        def listed(_):
            yield [SimpleNamespace(name="4")]

        self.assertEqual(self.run_view(scandir=denied), dict(enumerate="errno_13", readlink="ok"))
        self.assertEqual(self.run_view(link=denied, scandir=listed), dict(enumerate="ok", readlink="errno_13"))

    def test_equality_never_qualifies_holder_scan(self):
        view = dict(namespace=[1, 2], root_observer={}, child_observer={})
        result = compare(view, view)
        self.assertTrue(result["same_PID_namespace"])
        self.assertFalse(result["full_holder_scan_qualified"])
        self.assertFalse(result["original_2292_denied_process_identified"])

    def test_exact_cohost_launch_is_one_cpu_zero_gres(self):
        launch = srun()
        self.assertEqual(launch[0], "srun")
        self.assertIn(f"--jobid={PARENT}", launch)
        self.assertIn(f"--job-name={LABEL}", launch)
        self.assertIn(f"--nodelist={NODE}", launch)
        self.assertEqual([x for x in launch if x.startswith("--gres=")], ["--gres=none"])
        self.assertIn("--cpus-per-task=1", launch)
        self.assertIn("--time=00:01:00", launch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
