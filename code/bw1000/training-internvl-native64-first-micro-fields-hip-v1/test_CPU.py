import tempfile
import unittest
from pathlib import Path

from metadata_result import inspect_stop
from plugin.fields import REQUEST, document, write_once


class MetadataResultTest(unittest.TestCase):
    def test_closes_only_the_complete_deliberate_stop(self):
        expected = "a" * 64
        actual = "b" * 64
        roster = {str(rank): [expected] for rank in range(8)}
        rows = [
            dict(path=[], kind="dict", children=2),
            dict(path=["args"], kind="tuple", children=0),
            dict(path=["kwargs"], kind="dict", children=0),
        ]
        cleanup = dict(owned_processes_remaining=0, termination_needed=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root / "native.log"
            log.write_text(f"[rank0]: hip_hook.FirstMicroCaptured: {REQUEST}:DIFFERENT_INPUT_DIAGNOSTIC_STOP\n",
                           encoding="utf-8")
            for rank in range(8):
                write_once(root / f"rank-{rank:02d}.json", document("HIP", rank, actual, expected, rows))

            def report(returncode=1, state=cleanup):
                return inspect_stop(root, log, roster, returncode=returncode,
                                    controller_error=None, cleanup=state)

            result = report()
            self.assertTrue(result["diagnostic_complete"])
            self.assertEqual((result["matched_ranks"], result["verified_ranks"]), (0, 8))
            self.assertFalse(report(returncode=0)["diagnostic_complete"])
            self.assertFalse(report(returncode=-15)["diagnostic_complete"])
            self.assertFalse(report(state={**cleanup, "termination_needed": True})["diagnostic_complete"])
            log.write_text(f"SomeError: {REQUEST}:DIFFERENT_INPUT_DIAGNOSTIC_STOP\n", encoding="utf-8")
            self.assertFalse(report()["diagnostic_complete"])
            (root / "rank-07.json").unlink()
            self.assertEqual(report()["verified_ranks"], 7)


if __name__ == "__main__":
    unittest.main()
