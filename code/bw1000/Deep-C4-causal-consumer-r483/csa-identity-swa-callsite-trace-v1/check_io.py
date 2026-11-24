"""Exercise exact trace transfer against the frozen two-row CSA schedule."""

from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

from trace_io import validate, verify_trace, write_trace


base = Path(__file__).resolve().parents[1]
plan = runpy.run_path(str(base / "csa-identity-swa-commit-differential-v1/compare.py"))["plan"]()


class NativeTraceTest(unittest.TestCase):
    def test_all_five_captures_in_both_arms_preserve_exact_raw_trace(self):
        selected = [(i, unit) for i, unit in enumerate(plan["steps"]) if unit["capture"]]
        self.assertEqual(len(selected), 5)
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            for flag in (0, 1):
                path = root / str(flag)
                path.mkdir()
                for index, unit in selected:
                    key = unit["requests"]
                    direct = int(bool(flag and unit["direct"]))
                    row = [index+1, sum(n for _, _, n in key), len(key), sum(min(n, 128) for _, _, n in key),
                           direct, 1-direct, 1, 1]
                    proof = write_trace(path, index, key, flag, row)
                    self.assertEqual(verify_trace(path, index, key, flag, proof)["index_select_submissions"], 1-direct)

    def test_foreign_ordinal_missing_store_and_cropped_129_cannot_pass_as_direct(self):
        requests = [[0, 8625, 129]]
        invalid = ([1, 129, 1, 128, 0, 1, 1, 1], [2, 129, 1, 128, 0, 1, 0, 1],
                   [2, 129, 1, 128, 1, 0, 1, 1], [2, 129, 1, 128, 0, 1, 1, 0])
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate(row, 1, requests, 1)

    def test_raw_trace_bytes_must_match_capture_sha_and_counter(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            requests = [[0, 32767, 1]]
            proof = write_trace(root, 0, requests, 1, [1, 1, 1, 1, 1, 0, 1, 1])
            with self.assertRaisesRegex(ValueError, "capture counter"):
                verify_trace(root, 0, requests, 1, dict(proof, commit_IndexSelect_calls=1))
            with (root / "000-commit-route.json").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "trace bytes"):
                verify_trace(root, 0, requests, 1, proof)


if __name__ == "__main__":
    unittest.main(verbosity=2)
