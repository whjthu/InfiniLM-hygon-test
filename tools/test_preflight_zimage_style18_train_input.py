"""Protect the preflight's fixed denominator and training-only input boundary."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import preflight_zimage_style18_train_input as gate


class Image:
    size = (256, 256)
    mode = "RGB"

    def tobytes(self):
        return b"image fixture"


class ZImageInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.train = Path(self.temp.name).resolve() / "train"
        (self.train / "images").mkdir(parents=True)
        self.source = b"source fixture"
        self.prompt = "training fixture"
        self.binding = {"rows": []}
        manifest = self.train / "metadata.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "prompt"])
            for idx in range(gate.TRAIN_COUNT):
                path = f"images/{idx:03d}.png"
                (self.train / path).write_bytes(self.source)
                writer.writerow([path, self.prompt])
                self.binding["rows"].append(dict(
                    ordinal=idx, path=path, image=self.pin(self.source),
                    prompt_SHA256=gate.digest(self.prompt.encode()), RGB256_SHA256=gate.digest(Image().tobytes()),
                ))
        self.binding["CSV"] = self.pin(manifest.read_bytes())

    @staticmethod
    def pin(raw):
        return {"bytes": len(raw), "SHA256": gate.digest(raw)}

    def test_accepts_exact_train_files_and_does_not_require_other_splits(self):
        self.assertEqual(gate.inspect_inputs(self.train, self.binding), self.train / "metadata.csv")

    def test_changed_image_rejected_before_native_import(self):
        (self.train / "images/011.png").write_bytes(self.source[::-1])
        with self.assertRaisesRegex(ValueError, "frozen input bytes differ"):
            gate.inspect_inputs(self.train, self.binding)

    def test_other_split_rejected(self):
        wrong_dir = self.train.with_name("dev")
        self.train.rename(wrong_dir)
        with self.assertRaisesRegex(ValueError, "canonical train split"):
            gate.inspect_inputs(wrong_dir, self.binding)

    def test_native_bad_prompt_and_pixel_stay_in_twelve_item_denominator(self):
        samples = [dict(prompt=self.prompt, image=Image()) for _ in range(gate.TRAIN_COUNT)]
        samples[4]["prompt"] += "changed"
        samples[-1]["image"].size = (255, 256)
        rows = gate.compare_samples(samples, self.binding, Image)
        self.assertEqual([row["ordinal"] for row in rows], list(range(12)))
        self.assertEqual([row["ordinal"] for row in rows if not row["prompt_matches"]], [4])
        self.assertEqual([row["ordinal"] for row in rows if not row["image_matches"]], [11])
        with self.assertRaisesRegex(ValueError, "dataset count differs"):
            gate.compare_samples(samples[:-1], self.binding, Image)


if __name__ == "__main__":
    unittest.main()
