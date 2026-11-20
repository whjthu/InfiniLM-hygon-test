import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from run_gemma_domain_pair import REQUIRED_CHECKPOINT_FILES, command_owned, inventory, run_arm


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class GemmaNativePairRunner(unittest.TestCase):
    def test_uses_native_swift_args_and_hashes_original_checkpoint_members(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work).resolve()
            checkpoint = root / "continuous/train/checkpoint-20"
            checkpoint.mkdir(parents=True)
            for name in REQUIRED_CHECKPOINT_FILES:
                (checkpoint / name).write_bytes(b"fixture payload")
            write(checkpoint / "trainer_state.json", dict(global_step=20, max_steps=25, train_batch_size=1))
            write(checkpoint / "args.json", dict(max_steps=25, per_device_train_batch_size=1,
                                                  gradient_accumulation_steps=4))
            (checkpoint / "training_args.bin").write_bytes(b"unused format")
            files = inventory(root, "continuous", 20, lambda: None)
            self.assertIn("args.json", files)
            self.assertNotIn("training_args.bin", files)
            self.assertEqual(len(files), 14)
            result = json.loads((root / "continuous-checkpoint-20-files.json").read_bytes())
            self.assertEqual(result["files"], files)
            with self.assertRaises(FileExistsError):
                inventory(root, "continuous", 20, lambda: None)

    def test_exact_owned_command_match(self):
        arm = Path("/test/gemma/continuous")
        cmd = [b"--output_dir", os.fsencode(arm / "train"), b"--callbacks", b"gemma_domain_continuous_witness"]
        self.assertTrue(command_owned(cmd, arm))
        self.assertFalse(command_owned(cmd, arm.parent / "resumed"))
        self.assertFalse(command_owned(cmd[:-1] + [b"qwen_treasury_continuous_witness"], arm))

    def test_preserves_real_failed_subprocess_exit_in_original_arm_receipt(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work).resolve()
            worker = root / "worker_fixture.py"
            worker.write_text("raise SystemExit(7)\n", encoding="utf-8")
            fake_release = {"snapshot": lambda: {}, "refresh": lambda *_: {},
                            "cleanup": lambda *_, **__: {"termination_needed": False, "owned_processes_remaining": 0}}
            now = time.time()
            run = dict(request_id="a" * 32, steps={"gpu": "1103606.999"}, epoch=now, maximum_seconds=60)
            passed = run_arm(root, "continuous", dict(command=[str(worker)], environment_overrides={}),
                             {}, run, now + 30, lambda: None, fake_release)
            self.assertFalse(passed)
            decision = json.loads((root / "continuous/native-outcome.json").read_bytes())
            self.assertEqual(decision["rc"], 7)
            self.assertFalse(decision["passed"])
            self.assertEqual(decision["ranks"], [])
            self.assertFalse((root / "resumed").exists())


if __name__ == "__main__":
    unittest.main()
