"""Verify the checkpoint pin exception and the five-step controller's call path."""

import ast
import unittest

import recovery_binding as binding
import render


class RecoveryTests(unittest.TestCase):
    def test_only_exact_optimizer_pin_can_exceed_per_file_limit(self):
        baseline = (render.BASE.parent / "training-qwen-target-unified-framework-a68952c-v1"
                    / "deltas" / binding.WITNESS).read_bytes()
        trusted = render.runtime(render.BASE.parent / binding.LABEL)
        old_patch = {}
        exec(compile(trusted["mid_epoch_recovery.py"], "mid_epoch_recovery.py", "exec"), old_patch)
        old = old_patch["patch_witness"](baseline)
        new = binding.patched_witness(old)
        self.assertEqual(binding.digest(new), "92f786ddc670883c94eda55ba25f60ebf881257810024cfccddcd5d89cefd9b3")

        def reject_pin(source):
            function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                            and node.name == "pinned_checkpoint")
            condition = next(node.test for node in ast.walk(function) if isinstance(node, ast.If)
                             and "invalid checkpoint file pin" in ast.unparse(node))
            return compile(ast.Expression(condition), "optimizer_checkpoint_gate", "eval")

        self.assertTrue(eval(reject_pin(old), {}, dict(name="optimizer.pt", pin=binding.OPTIMIZER)))
        gate = reject_pin(new)
        self.assertFalse(eval(gate, {}, dict(name="optimizer.pt", pin=binding.OPTIMIZER)))
        for name, pin in (
            ("optimizer.pt", dict(binding.OPTIMIZER, SHA256="0" * 64)),
            ("optimizer.pt", dict(binding.OPTIMIZER, bytes=binding.OPTIMIZER["bytes"] + 1)),
            ("adapter_model.safetensors", binding.OPTIMIZER),
        ):
            self.assertTrue(eval(gate, {}, dict(name=name, pin=pin)))

    def test_controller_never_calls_the_continuous_arm(self):
        files = render.runtime(render.BASE.parent / binding.LABEL)
        controller = ast.parse(files["controller.py"])
        calls = [node for node in ast.walk(controller) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "run_arm"]
        self.assertEqual([node.args[0].value for node in calls], ["resumed"])
        self.assertIn(b"compare(ORIGINAL,RUN)", files["controller.py"])
        self.assertIn(b'planned_updates_by_arm={"resumed":5}', files["receive.py"])


if __name__ == "__main__":
    unittest.main()
