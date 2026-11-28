"""Check the regenerated Swift entry and exact restore hook wrapper semantics."""

import ast
import tempfile
from pathlib import Path
import unittest

import render
import restore


class FixtureTrainer:
    def __init__(self):
        self.events = []

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.events.append(("adapter", resume_from_checkpoint, model))
        return "native-adapter-result"

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.events.append(("optimizer_and_scheduler", checkpoint))
        return "native-optimizer-and-scheduler-result"

    def _load_rng_state(self, checkpoint):
        self.events.append(("rng", checkpoint))
        return "native-rng-result"


class ResumeTests(unittest.TestCase):
    def test_exact_original_bodies_and_native_main(self):
        files = render.render()
        self.assertEqual(len(files), 8)
        self.assertIn(b'runpy.run_path(str(entry), run_name="__main__")', files["train.py"])
        self.assertIn(b"resume_config.resume_argv(arguments, CALLBACK)", files["train.py"])
        self.assertIn(b"EXPECT" + b"ED_CLASS = target_eager.preimport", files["train.py"])
        self.assertIn(b"prepend=True", files["callback.py"])
        self.assertIn(b"learning_rates=rates", files["callback.py"])
        self.assertIn(b"resume_only_model", files["resume_config.py"])
        baseline = (render.FIRST2 / "callback.py").read_text(encoding="utf-8")
        after = files["first2_observe.py"].decode()
        methods = lambda text: {item.name: ast.dump(item) for item in ast.parse(text).body if isinstance(item, ast.FunctionDef)}
        self.assertEqual(methods(after), {name: methods(baseline)[name] for name in methods(after)})

    def test_wrappers_delegate_and_preserve_order_keyword_and_result(self):
        trainer = FixtureTrainer()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()
            cp2 = path / "checkpoint-2"
            cp2.mkdir()
            verified = []

            def verify_rng(value):
                self.assertEqual(value, "old-rng")
                self.assertEqual(trainer.events[-1], ("rng", str(cp2)))
                verified.append(value)

            def provider(path, method):
                self.assertEqual(path.resolve(), Path(__file__).resolve())
                self.assertTrue(method.startswith("_load"))

            counts, _ = restore.bind_hooks(trainer, cp2, lambda: "old-rng", verify_rng, provider)
            self.assertEqual(trainer._load_from_checkpoint(resume_from_checkpoint=str(cp2), model="live"),
                             "native-adapter-result")
            self.assertEqual(trainer._load_optimizer_and_scheduler(str(cp2)), "native-optimizer-and-scheduler-result")
            self.assertEqual(trainer._load_rng_state(str(cp2)), "native-rng-result")
            self.assertEqual(list(counts.values()), [1, 1, 1])
            self.assertEqual(verified, ["old-rng"])
            with self.assertRaisesRegex(ValueError, "EXACT_ONCE_ORIGINAL_CP2"):
                trainer._load_rng_state(str(cp2))
            self.assertEqual(len(trainer.events), 3)

    def test_wrong_path_fails_before_any_native_body(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()
            trainer = FixtureTrainer()
            cp = path / "checkpoint-2"
            cp.mkdir()
            restore.bind_hooks(trainer, cp, lambda: None, lambda _: None, lambda *_: None)
            with self.assertRaisesRegex(ValueError, "EXACT_ONCE_ORIGINAL_CP2"):
                trainer._load_from_checkpoint(str(path))
            self.assertEqual(trainer.events, [])


if __name__ == "__main__":
    unittest.main()
