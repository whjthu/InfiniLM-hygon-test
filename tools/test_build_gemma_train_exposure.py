from collections import Counter
import json
import tempfile
import unittest
from pathlib import Path

from build_gemma_train_exposure import exposure_plan, digest, read_pinned_file


def training_rows():
    return [(json.dumps({"images": [f"fixture-{i % 12}.png"], "messages": [
        {"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i)},
    ]}) + "\n").encode("utf-8") for i in range(72)]


class GemmaExposureTests(unittest.TestCase):
    def test_full_updates_and_resume_from_step_twenty(self):
        source = training_rows()
        view, receipt = exposure_plan(source)
        lines = view.splitlines(keepends=True)
        self.assertEqual(len(lines), 800)
        self.assertEqual(Counter(lines), {line: 12 if i < 8 else 11 for i, line in enumerate(source)})
        self.assertEqual(receipt["expected_microbatches_per_rank"], 100)
        self.assertEqual(receipt["expected_resume_skip_microbatches_per_rank"], 80)
        self.assertEqual(receipt["expected_resume_tail_microbatches_per_rank"], 20)
        self.assertEqual(receipt["expected_continuous_updates"], 25)
        self.assertFalse(receipt["training_ready"])

    def test_partial_and_duplicate_rows_are_rejected(self):
        source = training_rows()
        for rows in (source[:71], source[:71] + source[:1], source[:71] + [source[-1].rstrip()]):
            with self.assertRaises(ValueError):
                exposure_plan(rows)

    def test_user_only_input_is_rejected(self):
        source = training_rows()
        row = json.loads(source[0])
        row["messages"].pop()
        source[0] = (json.dumps(row) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "supervised"):
            exposure_plan(source)

    def test_pinned_reader_checks_the_exact_bytes_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "input.jsonl"
            file.write_bytes(b"a\n")
            self.assertEqual(read_pinned_file(file, 2, digest(b"a\n")), b"a\n")
            with self.assertRaises(ValueError):
                read_pinned_file(file, 2, digest(b"b\n"))
            with self.assertRaises(ValueError):
                read_pinned_file(file, 1, digest(b"a\n"))


if __name__ == "__main__":
    unittest.main()
