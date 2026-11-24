"""Exercise the scoring inventory boundary without loading the native pipeline."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import audit_zimage_coco_development_generations as collector
import render_zimage_coco_development_pairs as renderer
from render_zimage_coco_development_pairs import DEVELOPMENT_COUNT, shard_plan


NOISE = sha256(b"fixture noise").hexdigest()


class Replay:
    def sample(self, image, prompt, seed):
        return b"fixture noise", NOISE


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = [{"image_id": 1000 + index, "seed": 2000 + index,
                      "prompt_SHA256": sha256(str(index).encode()).hexdigest()}
                     for index in range(DEVELOPMENT_COUNT)]

    def shard(self, name, first, stop):
        directory = self.root / name
        directory.mkdir()
        (directory / "plan.jsonl").write_text(json.dumps(shard_plan(first, stop)) + "\n", encoding="utf-8")
        return directory

    def event(self, directory, ordinal, arm, *, completed=True):
        row = self.rows[ordinal]
        record = dict(row, ordinal=ordinal, arm=arm, completed=completed)
        if not completed:
            return dict(record, error_type="RuntimeError")
        image = Image.new("RGB", (256, 256), (2, 5, 9))
        path = directory / f"{ordinal:03d}-{arm}.private.png"
        image.save(path)
        raw = path.read_bytes()
        return dict(record, image_relative=path.name, image_bytes=len(raw), image_SHA256=sha256(raw).hexdigest(),
                    initial_noise_SHA256=NOISE, RGB_SHA256=sha256(image.tobytes()).hexdigest(),
                    inference_steps=8, guidance=1.0, width=256, height=256)

    def write_events(self, directory, events, trailing=""):
        text = "".join(json.dumps(event) + "\n" for event in events) + trailing
        (directory / "generations.jsonl").write_text(text, encoding="utf-8")

    def test_all_missing_and_failed_arms_stay_in_full_denominator(self):
        directory = self.shard("one", 0, 3)
        events = [self.event(directory, 0, "before"), self.event(directory, 0, "after"),
                  self.event(directory, 1, "after", completed=False)]
        self.write_events(directory, events, trailing='{"uncommitted":')
        result = collector.audit([directory], self.rows, Replay())
        self.assertEqual((result["planned_pairs"], result["planned_outputs"], result["complete_pairs"]), (256, 512, 1))
        self.assertEqual(result["per_arm"]["before"], {"COMPLETED": 1, "FAILED": 0, "MISSING": 255})
        self.assertEqual(result["per_arm"]["after"], {"COMPLETED": 1, "FAILED": 1, "MISSING": 254})
        self.assertEqual(result["sources"][0]["uncertain_trailing_bytes"], 15)
        self.assertFalse(result["complete_development_inventory"])

    def test_corrupt_image_is_rejected(self):
        directory = self.shard("one", 0, 1)
        before = self.event(directory, 0, "before")
        self.write_events(directory, [before])
        (directory / before["image_relative"]).write_bytes(b"wrong image")
        with self.assertRaisesRegex(ValueError, "generated image bytes differ"):
            collector.audit([directory], self.rows, Replay())

    def test_wrong_noise_is_rejected(self):
        directory = self.shard("one", 0, 1)
        before = self.event(directory, 0, "before")
        self.write_events(directory, [dict(before, initial_noise_SHA256="0" * 64)])
        with self.assertRaisesRegex(ValueError, "actual noise differs"):
            collector.audit([directory], self.rows, Replay())

    def test_swapped_arms_and_overlapping_shards_are_rejected(self):
        directory = self.shard("one", 0, 2)
        self.write_events(directory, [self.event(directory, 0, "after")])
        with self.assertRaisesRegex(ValueError, "order differs"):
            collector.audit([directory], self.rows, Replay())
        self.write_events(directory, [])
        with self.assertRaisesRegex(ValueError, "overlapping generation shards"):
            collector.audit([directory, self.shard("two", 1, 3)], self.rows, Replay())

    def test_finished_log_rejects_any_extra_event_bytes(self):
        directory = self.shard("one", 0, 1)
        events = [self.event(directory, 0, arm, completed=False) for arm in ("before", "after")]
        self.write_events(directory, events, trailing="{")
        with self.assertRaisesRegex(ValueError, "extra generation events"):
            collector.audit([directory], self.rows, Replay())

    def test_verified_checkpoint_bytes_are_the_only_bytes_deserialized(self):
        path = self.root / 'state.private.pt'
        fixture = b'small checkpoint fixture'
        checkpoint = {
            'metadata': {'global_step': 25, 'completed_optimizer_steps': 25, 'next_data_cursor': 5},
            'integrity_sha256': renderer.CHECKPOINT_INTEGRITY_SHA256,
            'adapter_sha256': renderer.ADAPTER_SHA256,
        }

        def load(stream, **kwargs):
            self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': False})
            path.write_bytes(b'changed after verification')
            self.assertEqual(stream.read(), fixture)
            return checkpoint

        torch = Mock()
        torch.load.side_effect = load
        deadline = Mock()
        with patch.multiple(renderer, CHECKPOINT_BYTES=len(fixture), CHECKPOINT_SHA256=sha256(fixture).hexdigest()):
            for bad in (b'', fixture[:-1], fixture + b'x', b'x' * len(fixture)):
                path.write_bytes(bad)
                with self.assertRaisesRegex(ValueError, 'step-25 checkpoint bytes differ'):
                    renderer.read_original_checkpoint(path, torch, deadline)
            torch.load.assert_not_called()
            path.write_bytes(fixture)
            self.assertIs(renderer.read_original_checkpoint(path, torch, deadline), checkpoint)
            checkpoint['metadata']['completed_optimizer_steps'] = 24
            path.write_bytes(fixture)
            with self.assertRaisesRegex(ValueError, 'completed checkpoint state differs'):
                renderer.read_original_checkpoint(path, torch, deadline)
        self.assertEqual(torch.load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
