import ast
import json
import math
import pickle
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src/gencuda/backends/hunyuanvideo_sage_grpo.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
backend = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run_hunyuanvideo_sage_grpo")
training = next(node.body for node in backend.body if isinstance(node, ast.Try))
start = next(i for i, node in enumerate(training) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "first_global_step")
end = next(i for i, node in enumerate(training) if isinstance(node, ast.FunctionDef)
           and node.name == "restore_recovery_checkpoint")
window = compile(ast.Module(body=training[start:end], type_ignores=[]), str(SOURCE), "exec")


checkpoint_functions = compile(ast.Module(body=[
    node for node in tree.body if isinstance(node, ast.FunctionDef)
    and node.name in {"_save_rl_checkpoint", "_load_rl_checkpoint"}
], type_ignores=[]), str(SOURCE), "exec")


class PolicyCheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_both_device_rngs_anchor_and_optimizer_cursor(self):
        restored = {}
        rng = lambda label: SimpleNamespace(cpu=lambda: label)
        cuda = SimpleNamespace(get_rng_state=lambda rank: rng(rank + 7),
                               set_rng_state=lambda value, rank: restored.__setitem__(rank, value))
        torch = SimpleNamespace(cuda=cuda, get_rng_state=lambda: 6,
                                set_rng_state=lambda value: restored.__setitem__("torch", value),
                                save=lambda obj, path: path.write_bytes(pickle.dumps(obj)),
                                load=lambda path, **kwargs: pickle.loads(path.read_bytes()))
        scope = dict(Path=Path, Any=Any, _adapter_state=lambda model: model,
                     _cpu_tree=lambda x: x, _optimizer_to_device=lambda x: None,
                     _load_state=lambda model, state: model.update(state),
                     random=SimpleNamespace(getstate=lambda: 4, setstate=lambda v: restored.update(python=v)),
                     np=SimpleNamespace(random=SimpleNamespace(get_state=lambda: 5,
                                        set_state=lambda v: restored.update(numpy=v))))
        exec(checkpoint_functions, scope)
        optimizer = SimpleNamespace(state_dict=lambda: {"updates": 20},
                                    load_state_dict=lambda state: restored.update(optimizer=state))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-15" / "state.pt"
            contract = {"schema": "sage-policy-resume-v1", "warmup_steps": 5}
            scope["_save_rl_checkpoint"](checkpoint, torch, {"adapter": 1}, optimizer,
                                         {"anchor": 2}, 15, contract)
            self.assertEqual(pickle.loads(checkpoint.read_bytes())["global_step"], 20)
            model = {}
            for wrong_contract, last in [(dict(contract, warmup_steps=4), 20), (contract, 14)]:
                with self.assertRaisesRegex(ValueError, "checkpoint recipe or optimizer-step cursor"):
                    scope["_load_rl_checkpoint"](checkpoint, torch, model, optimizer,
                                                 wrong_contract, max_measured_step=last)
                self.assertFalse(restored or model)
            cursor, anchor = scope["_load_rl_checkpoint"](checkpoint, torch, model, optimizer,
                                                          contract, max_measured_step=19)
            self.assertEqual((cursor, anchor, model), (15, {"anchor": 2}, {"adapter": 1}))
            self.assertEqual(restored, {"optimizer": {"updates": 20}, "python": 4, "numpy": 5,
                                        "torch": 6, 0: 7, 1: 8})


class PolicyWindowTests(unittest.TestCase):
    def test_drains_and_all_measured_checkpoint_io_are_inside_the_boundary(self):
        events = []
        times = iter([100.0, 180.0])

        def clock():
            events.append(("clock",))
            return next(times)

        def step(index, *, save_outputs):
            events.append(("step", index, save_outputs))
            return {"global_step": index}

        def checkpoint(path, torch, model, optimizer, anchor_state, measured_step, resume_contract):
            events.append(("checkpoint", measured_step))
            path.parent.mkdir()
            path.write_bytes(b"checkpoint")

        def digest(path):
            events.append(("hash", path.parent.name))
            return "checkpoint-sha"

        with TemporaryDirectory() as tmp:
            ctx = dict(
                warmup_steps=1, max_steps=3, checkpoint_steps={2}, resumed_steps=0, resume_contract={},
                output_dir=Path(tmp), json=json, math=math, Any=object, rl_step=step,
                perf_counter=clock, _save_rl_checkpoint=checkpoint, sha256_path=digest,
                torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: events.append(("sync", device)))),
                actor_device="actor", reward_device="reward", model=None, optimizer=None, anchor_state=None,
            )
            exec(window, ctx)
            self.assertEqual(ctx["measurement_seconds"], 80.0)
            self.assertEqual(len(ctx["measured"]), 3)
            self.assertEqual([json.loads(line)["global_step"] for line in
                              (Path(tmp) / "steps.ndjson").read_text().splitlines()], [1, 2, 3, 4])
            fresh_events = list(events)
            with TemporaryDirectory() as resumed_tmp:
                ctx.update(output_dir=Path(resumed_tmp), resumed_steps=2, perf_counter=iter([200.0, 230.0]).__next__)
                events.clear()
                exec(window, ctx)
                self.assertEqual(ctx["measurement_seconds"], 30.0)
                self.assertEqual(ctx["measured"], [{"global_step": 4}])
                self.assertFalse((Path(resumed_tmp) / "checkpoint-2").exists())
                self.assertEqual(events, [("sync", "actor"), ("sync", "reward"), ("step", 4, True),
                                          ("checkpoint", 3), ("hash", "checkpoint-3"),
                                          ("sync", "actor"), ("sync", "reward")])
        self.assertEqual(fresh_events, [
            ("step", 1, False), ("sync", "actor"), ("sync", "reward"), ("clock",),
            ("step", 2, True), ("step", 3, True), ("checkpoint", 2), ("hash", "checkpoint-2"),
            ("step", 4, True), ("checkpoint", 3), ("hash", "checkpoint-3"),
            ("sync", "actor"), ("sync", "reward"), ("clock",),
        ])


if __name__ == "__main__":
    unittest.main()
