"""Run the regenerated owner and assembly boundaries with local fixtures."""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import render


TRAINER = """
class Trainer:
    def train(self, checkpoint):
        self._load_from_checkpoint(checkpoint)

    def _prepare_for_training(self, checkpoint):
        self._load_optimizer_and_scheduler(checkpoint)

    def _inner_training_loop(self, checkpoint):
        self._prepare_for_training(checkpoint)
        self.on_train_begin()
        self._run_epoch()

    def _init_training_state(self, num_update_steps_per_epoch):
        epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
        steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
        return epochs_trained, steps_trained_in_current_epoch

    def _run_epoch(self):
        if steps_trained_in_current_epoch == 0:
            self._load_rng_state(original_checkpoint)
        self.get_batch_samples()
"""


def load(name):
    source = render.render()[name + ".py"]
    namespace = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    namespace.__file__ = str(Path(__file__).with_name(name + ".py"))
    exec(compile(source, namespace.__file__, "exec"), namespace.__dict__)
    return namespace


class OwnerTests(unittest.TestCase):
    def test_generated_actual_mounts_and_separate_output(self):
        sys.modules["resume_layout"] = load("resume_layout")
        owner = load("owner")
        request = "f" * 32
        cmd = owner.runtime_command(dict(vision_root="/vision", port=29717, request=request),
                                    Path("/persist"), Path("/tmp-private"))
        self.assertIn("/persist:/llava-run:rw", cmd)
        self.assertIn("/tmp-private:/tmp:rw", cmd)
        self.assertIn("/persist/RESUME-AUTH.json:/llava-resume-authority/AUTH.json:ro", cmd)
        cp2 = owner.layout.CP2.as_posix()
        self.assertIn(f"{cp2}:{cp2}:ro", cmd)
        self.assertIn("--nproc-per-node", cmd)
        self.assertIn("8", cmd)
        self.assertIn(owner.layout.PLAN.as_posix() + ":/llava-resume-authority/CP2_NATIVE_RESUME_PLAN.private.json:ro", cmd)
        self.assertIn(b"APPTAINERENV_LLAVA_RESUME_AUTH_SHA256=resume_sha", render.render()["owner.py"])

    def test_chronology_rejects_rng_after_first_fetch(self):
        sys.modules["resume_layout"] = load("resume_layout")
        sys.modules["owner"] = load("owner")
        assembly = load("assembly")
        facts = assembly.chronology(ast.parse(TRAINER))
        self.assertLess(facts["rng_at_epoch_boundary"], facts["first_fetch"])
        wrong = TRAINER.replace("        if steps_trained_in_current_epoch == 0:",
                                "        self.get_batch_samples()\n        if steps_trained_in_current_epoch == 0:")
        with self.assertRaises(ValueError):
            assembly.chronology(ast.parse(wrong))

    def test_exact_zero_gres_cpu_row_and_resume_derived_auth(self):
        sys.modules["resume_layout"] = load("resume_layout")
        owner = load("owner")
        row = owner.layout.CPU_ROW
        self.assertTrue(owner.tres(row[6], 0, 16, 32768))
        self.assertFalse(owner.tres(row[6], 8, 16, 32768))
        proof = owner.layout.future_auth("0" * 32)
        self.assertEqual(proof["plan_SHA256"], owner.layout.PLAN_SHA)
        self.assertEqual(proof["original_CP2_members_verified"], 17)


if __name__ == "__main__":
    unittest.main()
