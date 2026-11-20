import json
from pathlib import Path, PurePosixPath
import unittest

from prepare_gemma_domain_pair import ORIGINAL, candidate


def option(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f"expected one {name}")
    return argv[argv.index(name) + 1]


class GemmaPairedCommands(unittest.TestCase):
    def test_both_arms_use_the_same_native_training_configuration(self):
        repo = Path(__file__).resolve().parents[1]
        root = PurePosixPath("/local-pair-fixture")
        sources, plans = candidate(repo, root, root / "models", root / "plugins", 29920)
        left, right = plans["continuous"], plans["resumed"]
        original = json.loads((repo / ORIGINAL[0]).read_bytes())["command"]
        for flag in ("--model_type", "--template", "--torch_dtype", "--max_length", "--max_steps",
                     "--gradient_accumulation_steps", "--per_device_train_batch_size", "--save_steps",
                     "--optim", "--lr_scheduler_type", "--data_seed", "--gradient_checkpointing", "--packing"):
            self.assertEqual(option(left["command"], flag), option(original, flag))
            self.assertEqual(option(right["command"], flag), option(original, flag))
        self.assertEqual(left["command"][0], right["command"][0])
        self.assertEqual(option(left["command"], "--dataset"), option(right["command"], "--dataset"))
        self.assertEqual(option(right["command"], "--resume_from_checkpoint"),
                         option(left["command"], "--output_dir") + "/checkpoint-20")
        self.assertNotIn("--resume_from_checkpoint", left["command"])
        self.assertEqual((left["planned_optimizer_updates"], right["planned_optimizer_updates"]), (25, 5))
        self.assertEqual((left["expected_checkpoint_steps"], right["expected_checkpoint_steps"]),
                         ([10, 20, 25], [25]))
        for plan, key, filename in (
            (left, "gemma_domain_continuous_witness", "gemma_domain_continuous.py"),
            (right, "gemma_domain_resume_witness", "gemma_domain_resume.py"),
        ):
            self.assertEqual(plan["environment_overrides"]["NPROC_PER_NODE"], "8")
            self.assertEqual(option(plan["command"], "--callbacks"), key)
            self.assertEqual(plan["required_assets"]["plugins"], [f"{root}/framework/vlm/plugins/{filename}"])
            self.assertIn(f"vlm/plugins/{filename}", sources)
            self.assertFalse(plan["training_ready"])
            self.assertNotIn("nvidia_baseline", plan["command"])

    def test_rejects_nonabsolute_or_traversal_namespace(self):
        repo = Path(__file__).resolve().parents[1]
        root = PurePosixPath("/fixture")
        for path in ("relative", "/fixture/../other"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "absolute"):
                candidate(repo, PurePosixPath(path), root / "models", root / "plugins", 29920)


if __name__ == "__main__":
    unittest.main()
