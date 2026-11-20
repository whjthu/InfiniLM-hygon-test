import json
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from build_vlm_shared_resume_overlay import SOURCES, build_sources, pinned


BASE = Path(__file__).resolve().parents[1]
RECIPES = {
    "T01.sft": "training-qwen-A100-native-contract-CPU-v1",
    "T02.sft": "training-gemma-A100-native-contract-CPU-v2",
    "T03.sft": "training-internvl-A100-native-contract-CPU-v1",
    "T04.sft": "training-llava-A100-native-contract-CPU-v1",
    "T05.sft": "training-minicpm-resume-witness-CPU-v1",
}


def load(raw, path):
    scope = {"__name__": "test_shared_resume_overlay", "__file__": path}
    exec(compile(raw, path, "exec"), scope)
    return scope


def launch(build, case, resume=None, callback=None):
    root = PurePosixPath("/gemma-resume-local-fixture")
    original = BASE / "code/bw1000" / RECIPES[case] / "original-launch.json"
    return build(case, original.read_bytes(), root / "swift_entry.py", root / "models",
                 root / "train800-swift.jsonl", root / "plugins", root / "resumed", 29920,
                 diagnostic_plugin=root / "witness.py" if callback is not None else None,
                 diagnostic_callback=callback, resume_from_checkpoint=root / resume if resume else None)


class SharedResumeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = build_sources(BASE)
        entry = "tools/hygon_vlm_recipe.py"
        cls.build = staticmethod(load(cls.files[entry], entry)["build_launch"])
        cls.old_build = staticmethod(load(pinned(BASE, *SOURCES[entry]), entry)["build_launch"])
        cursor = "vlm/runtime/resume_cursor.py"
        cls.plan = staticmethod(load(cls.files[cursor], cursor)["plan_epoch_boundary_resume"])

    def test_original_fresh_command_for_all_five_cases_is_byte_equivalent(self):
        for case in RECIPES:
            with self.subTest(case=case):
                result = launch(self.build, case)
                self.assertEqual(json.dumps(result), json.dumps(launch(self.old_build, case)))

    def test_gemma_checkpoint20_rebind_retains_the_original_optimization_recipe(self):
        fresh = launch(self.build, "T02.sft")["command"]
        result = launch(self.build, "T02.sft", "continuous/checkpoint-20", "gemma_domain_resume_witness")
        argv = result["command"]
        for key, expected in (("--template", "gemma3_vision"), ("--max_length", "512"),
                              ("--gradient_checkpointing", "false"), ("--attn_impl", "sdpa"),
                              ("--per_device_train_batch_size", "1"), ("--gradient_accumulation_steps", "4"),
                              ("--max_steps", "25"), ("--resume_only_model", "false"),
                              ("--ignore_data_skip", "false"), ("--load_args", "false")):
            self.assertEqual(argv[argv.index(key) + 1], expected)
        for key in ("--learning_rate", "--lr_scheduler_type", "--optim", "--data_seed", "--save_steps"):
            self.assertEqual(argv[argv.index(key) + 1], fresh[fresh.index(key) + 1])
        self.assertEqual(result["environment_overrides"]["NPROC_PER_NODE"], "8")
        self.assertNotIn("nvidia_baseline", argv)

    def test_preserves_minicpm_and_qwen_opt_in_and_fails_other_cases(self):
        self.assertEqual(
            launch(self.build, "T05.sft", "continuous/checkpoint-20", "native_resume_witness"),
            launch(self.old_build, "T05.sft", "continuous/checkpoint-20", "native_resume_witness"),
        )
        launch(self.build, "T01.sft", "continuous/checkpoint-20", "qwen_treasury_resume_witness")
        for case in ("T02.sft", "T03.sft", "T04.sft"):
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "case-specific"):
                launch(self.build, case, "continuous/checkpoint-20", "native_resume_witness")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            launch(self.build, "T02.sft", "resumed/checkpoint-20", "gemma_domain_resume_witness")

    def test_checkpoint20_cursor_requires_explicit_opt_in_and_full_loader(self):
        state = dict(global_step=20, max_steps=25, train_batch_size=1, epoch=0.8)
        args = dict(max_steps=25, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                    ignore_data_skip=False, resume_only_model=False, callbacks=["gemma_domain_continuous_witness"])
        params = dict(checkpoint_step=20, max_steps=25, world_size=8, per_device_batch=1,
                      gradient_accumulation_steps=4, dataloader_batches_per_rank=100)
        actual = self.plan(state, args, allow_mid_epoch=True, **params)
        self.assertEqual(actual["expected_batches_to_skip_in_resumed_epoch"], 80)
        self.assertEqual(actual["planned_micro_forwards_per_rank"], 20)
        self.assertEqual(actual["remaining_optimizer_updates"], 5)
        self.assertFalse(actual["recovery_qualified"])
        with self.assertRaisesRegex(ValueError, "mid-epoch"):
            self.plan(state, args, **params)
        with self.assertRaisesRegex(ValueError, "partial accumulation"):
            self.plan(state, args, allow_mid_epoch=True, **(params | {"dataloader_batches_per_rank": 9}))

    def test_native_callback_registers_and_rejects_a_short_or_filtered_loader(self):
        registry = {}
        config = {"case": "T02.sft", "checkpoint_step": 20,
                  "dataset_SHA256": "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"}
        constructor = Mock()
        witness = type("ResumeWitness", (), {"__init__": constructor})
        trainer = SimpleNamespace(skip_first_batches=lambda loader, count: loader[count:])
        modules = {
            "torch": SimpleNamespace(),
            "transformers": SimpleNamespace(trainer=trainer),
            "transformers.trainer": trainer,
            "swift.callbacks": SimpleNamespace(TrainerCallback=object, callbacks_map=registry),
            "vlm.runtime.first_update_capture": SimpleNamespace(remaining=Mock()),
            "vlm.runtime.native_recipe_contract": SimpleNamespace(validate_native_recipe=Mock()),
            "vlm.runtime.resume_cursor": SimpleNamespace(read_pinned_json=Mock(return_value=config)),
            "vlm.runtime.resume_witness": SimpleNamespace(ResumeWitness=witness, rng_fingerprint=Mock(),
                                                          rng_values=Mock()),
        }
        env = {"NATIVE_RESUME_CONFIG": "fixture", "NATIVE_RESUME_CONFIG_SHA256": "fixture",
               "RANK": "0", "LOCAL_RANK": "0"}
        path = "vlm/plugins/gemma_domain_resume.py"
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env):
            module = load(self.files[path], path)
            callback = registry["gemma_domain_resume_witness"]()
            for length in (9, 99):
                with self.subTest(length=length), self.assertRaisesRegex(ValueError, "actual 100-batch"):
                    callback.on_train_begin(SimpleNamespace(max_steps=25), SimpleNamespace(global_step=20),
                                            None, train_dataloader=range(length))
            constructor.assert_not_called()
            original_skip = trainer.skip_first_batches
            trace = Mock()
            seen, finish = module["observe_native_skip"](trainer, trace, 0)
            with self.assertRaisesRegex(ValueError, "different Gemma loader cursor"):
                trainer.skip_first_batches(range(100), 79)
            self.assertEqual(trainer.skip_first_batches(range(100), 80), range(80, 100))
            trace.write.assert_called_once_with("native-data-skip.json", dict(rank=0, before=100, skipped=80, after=20))
            self.assertTrue(seen)
            with self.assertRaisesRegex(ValueError, "different Gemma loader cursor"):
                trainer.skip_first_batches(range(100), 80)
            finish()
            self.assertIs(trainer.skip_first_batches, original_skip)


if __name__ == "__main__":
    unittest.main()
