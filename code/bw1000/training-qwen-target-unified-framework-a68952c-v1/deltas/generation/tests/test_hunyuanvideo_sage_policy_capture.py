import ast
import contextlib
from typing import Any
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import hygon_training_recipe as recipe

SOURCE = ROOT / "generation/src/gencuda/backends/hunyuanvideo_sage_grpo.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_policy_trace")
namespace = {"math": math}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
trace = namespace["_policy_trace"]
loader = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "_load_first_trajectory")
namespace["hashlib"] = hashlib
exec(compile(ast.Module(body=[loader], type_ignores=[]), str(SOURCE), "exec"), namespace)
load_trajectory = namespace["_load_first_trajectory"]


class PolicyCaptureTests(unittest.TestCase):
    def test_actor_trace_removes_forward_hooks_before_backward(self):
        tracer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_trace_actor_block_boundaries")
        ns = {"contextmanager": contextlib.contextmanager, "Any": Any,
              "tree_sha256": lambda tensor: tensor.label}
        exec(compile(ast.Module(body=[tracer], type_ignores=[]), str(SOURCE), "exec"), ns)
        trace_blocks = ns["_trace_actor_block_boundaries"]

        def module():
            obj = SimpleNamespace(callback=None)

            def register(callback):
                obj.callback = callback
                return SimpleNamespace(remove=lambda: setattr(obj, "callback", None))

            obj.register_forward_hook = register
            obj.register_forward_pre_hook = lambda callback: (
                setattr(obj, "pre_callback", callback)
                or SimpleNamespace(remove=lambda: setattr(obj, "pre_callback", None))
            )
            return obj

        def tensor(label, requires_grad):
            obj = SimpleNamespace(label=label, requires_grad=requires_grad)
            obj.register_hook = lambda callback: setattr(obj, "gradient", callback)
            return obj

        double, single = module(), module()
        model = SimpleNamespace(double_blocks=[double], single_blocks=[single])
        image, text, joined = tensor("img", True), tensor("txt", False), tensor("joined", True)
        with trace_blocks(model, True) as evidence:
            double.callback(double, (), (image, text))
            single.callback(single, (), joined)
        self.assertIsNone(double.callback)
        self.assertIsNone(single.callback)
        joined.gradient(tensor("d_joined", False))
        image.gradient(tensor("d_img", False))
        self.assertEqual(evidence["double_blocks.0"], [
            {"output_sha256": "img", "requires_grad": True, "output_gradient_sha256": "d_img"},
            {"output_sha256": "txt", "requires_grad": False},
        ])
        self.assertEqual(evidence["single_blocks.0"][0]["output_gradient_sha256"], "d_joined")
        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            with trace_blocks(model, True):
                pass
        self.assertIsNone(double.callback)
        self.assertIsNone(single.callback)
        with trace_blocks(model, False) as disabled:
            self.assertIsNone(disabled)

        children = {
            f"{stream}_{name}": module()
            for stream in ("img", "txt")
            for name in ("attn_q", "attn_k", "attn_v", "attn_proj", "mlp")
        }
        double._checkpoint_wrapped_module = SimpleNamespace(**children)
        with trace_blocks(model, True, detail_block=0) as detail:
            double.callback(double, (), (tensor("img_out", False), tensor("txt_out", False)))
            single.callback(single, (), tensor("joined_out", False))
            for name, child in children.items():
                child.callback(child, (), tensor(name, False))
                if name.endswith(("attn_proj", "mlp")):
                    child.pre_callback(child, (tensor(name + "_input", False),))
        self.assertEqual(len(detail), 16)
        self.assertEqual(detail["double_blocks.0.img_attn_proj.input"][0]["output_sha256"],
                         "img_attn_proj_input")
        self.assertIsNone(children["img_attn_proj"].pre_callback)
        self.assertIsNone(children["img_attn_proj"].callback)
    def test_transition_records_backward_boundaries_in_order(self):
        step = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "rl_step")
        body = next(node.body for node in ast.walk(step) if isinstance(node, ast.For)
                    and any(ast.unparse(item) == "(loss / transition_count).backward()"
                            for item in node.body))
        index = next(i for i, item in enumerate(body)
                     if ast.unparse(item) == "(loss / transition_count).backward()")
        boundary = compile(ast.Module(body=body[index - 1:index + 2], type_ignores=[]),
                           str(SOURCE), "exec")
        events = []
        prediction = SimpleNamespace(grad=None, retain_grad=lambda: events.append("retain"))

        class Loss:
            def __truediv__(self, count):
                events.append(count)
                return self

            def backward(self):
                self.assert_retained()
                prediction.grad = "output gradient"

            def assert_retained(self):
                if events != ["retain", 16]:
                    raise AssertionError(events)

            def detach(self):
                return self

            def cpu(self):
                return 0.25

        state = {"transition_losses": [], "transition_boundaries": []}
        exec(boundary, dict(update_state=state, prediction=prediction, loss=Loss(), transition_count=16,
                            sample_index=2, index=1, anchor_mean="anchor", tree_sha256=lambda value: value,
                            named_trainable={"lora": SimpleNamespace(grad="parameter gradient")}))
        self.assertEqual(state["transition_losses"], [0.25])
        self.assertEqual(state["transition_boundaries"], [{
            "rollout": 2, "transition": 1, "actor_output_sha256": prediction,
            "actor_output_gradient_sha256": "output gradient", "anchor_mean_sha256": "anchor",
            "accumulated_lora_gradients_sha256": {"lora": "parameter gradient"},
        }])

    def test_rollout_rows_match_optimizer_iteration_order(self):
        old = list(range(16))
        new = [value + 0.25 for value in old]
        ratios = [1.25] * 16
        rows = trace(old, new, ratios, [-2, -1, 1, 2], old, new)
        self.assertEqual([row["rollout"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(rows[2], {
            "rollout": 2, "advantage": 1, "old_log_probs": old[8:12],
            "new_log_probs": new[8:12], "ratios": ratios[8:12],
            "policy_losses": old[8:12], "moving_kls": new[8:12],
        })

    def test_bad_transition_evidence_is_rejected(self):
        values = [1.0] * 16
        for ratios, advantages, message in (
            (values[:-1], [1] * 4, "four transitions"),
            (values, [0, 0, float("nan"), 0], "non-finite"),
            ([0, *values[1:]], [1] * 4, "non-positive"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                trace(values, values, ratios, advantages, values, values)

    def test_unified_entry_only_changes_opt_in_config(self):
        frozen = {"config": {"warmup_steps": 5, "max_steps": 20,
                             "num_generations": 4, "sampling_steps": 4},
                  "weights": "/original/weights"}
        raw = json.dumps(frozen).encode()
        binding = (hashlib.sha256(raw).hexdigest(), "hunyuanvideo-1.5", "rl", 2)
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch.json"
            launch.write_bytes(raw)
            assets = Path(directory) / "assets.json"
            assets.write_text(json.dumps({"/original/weights": "/target/weights"}), encoding="utf-8")
            args = ["T09.sage-grpo", "--a100-launch", str(launch), "--asset-map", str(assets), "--run"]
            with patch.dict(recipe.GENERATION, {"T09.sage-grpo": binding}):
                with patch.object(recipe, "run_generation", return_value=0) as run:
                    self.assertEqual(recipe.main(args), 0)
                    self.assertEqual(run.call_args.args[0]["config"], frozen["config"])
                    self.assertEqual(recipe.main(args + ["--capture-first-policy"]), 0)
                    plan = run.call_args.args[0]
                    self.assertEqual(plan["cards"], 2)
                    self.assertEqual(plan["config"], dict(frozen["config"], capture_first_policy=True))
                    self.assertEqual(recipe.main(args + ["--capture-first-policy-only"]), 0)
                    self.assertEqual(run.call_args.args[0]["config"], dict(
                        frozen["config"], capture_first_policy=True, capture_first_policy_only=True))
                    self.assertEqual(recipe.main(args + ["--replay-first-trajectory", "/capture/step-0001.pt",
                                                        "--replay-first-trajectory-sha256", "a" * 64]), 0)
                    self.assertEqual(run.call_args.args[0]["config"], dict(
                        frozen["config"], replay_first_trajectory="/capture/step-0001.pt",
                        replay_first_trajectory_sha256="a" * 64))
                    trace_option = ["--trace-first-actor-blocks"]
                    self.assertEqual(recipe.main(args + ["--capture-first-policy-only"] + trace_option,
                                                 generation_accelerator="cuda"), 0)
                    self.assertTrue(run.call_args.args[0]["config"]["trace_first_actor_blocks"])
                    replay = ["--replay-first-trajectory", "/capture/step-0001.pt",
                              "--replay-first-trajectory-sha256", "a" * 64]
                    separate = ["--sage-vae-dir", "/staged/vae",
                                "--sage-scheduler-root", "/staged/sage"]
                    self.assertEqual(recipe.main(args + replay + separate), 0)
                    routed = run.call_args.args[0]["config"]
                    self.assertEqual(routed["sage_vae_dir"], "/staged/vae")
                    self.assertEqual(routed["sage_scheduler_root"], "/staged/sage")
                    self.assertEqual(routed["replay_first_trajectory_sha256"], "a" * 64)
                    resumed = ["--resume-from-checkpoint", "/previous/checkpoint-20"]
                    self.assertEqual(recipe.main(args + resumed + separate), 0)
                    self.assertEqual(run.call_args.args[0]["config"], dict(
                        frozen["config"], resume_from_checkpoint="/previous/checkpoint-20",
                        sage_vae_dir="/staged/vae", sage_scheduler_root="/staged/sage"))
                    for flags, accelerator in ((resumed + replay, "hip"), (replay + separate, "cuda")):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            recipe.main(args + flags, generation_accelerator=accelerator)
                    detail = ["--trace-first-actor-detail-block", "50"]
                    self.assertEqual(recipe.main(args + replay + trace_option + detail,
                                                 generation_accelerator="cuda"), 0)
                    self.assertEqual(run.call_args.args[0]["config"]["trace_first_actor_detail_block"], 50)
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        recipe.main(args + replay + detail, generation_accelerator="cuda")
                    for accelerator, flags in (("cuda", []), ("hip", ["--capture-first-policy-only"])):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            recipe.main(args + flags + trace_option, generation_accelerator=accelerator)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            recipe.main(["T09.dp2", "--run", "--capture-first-policy"])

    def test_backend_routes_separately_staged_scheduler(self):
        cli_source = ROOT / "generation/src/gencuda/cli.py"
        cli_tree = ast.parse(cli_source.read_bytes())
        names = {"run_backend", "_source_dir"}
        functions = [node for node in cli_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        call = lambda *args: args
        ns = {"Path": Path, "ModelSpec": Any, "Any": Any, "_call": call,
              "validate_resume_request": lambda *args: None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(cli_source), "exec"), ns)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "hunyuanvideo-1.5").mkdir()
            staged = root / "separate-sage"
            path = staged / "hyvideo/schedulers/scheduling_flow_match_discrete.py"
            spec = SimpleNamespace(name="hunyuanvideo-1.5")
            config = {"sage_scheduler_root": str(staged), "replay_first_trajectory": "/trajectory.pt"}
            with self.assertRaisesRegex(ValueError, "file is missing"):
                ns["run_backend"]("rl", spec, root, config, root, root)
            path.parent.mkdir(parents=True)
            path.touch()
            args = ns["run_backend"]("rl", spec, root, config, root, root)
            self.assertEqual(args[-2:], (root / "hunyuanvideo-1.5", staged))
            (root / "sage-grpo").mkdir()
            args = ns["run_backend"]("rl", spec, root, {}, root, root)
            self.assertEqual(args[-1], root / "sage-grpo")

    def test_replay_loader_refuses_altered_bytes_before_deserializing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-0001.pt"
            path.write_bytes(b"original")
            def deserialize(stream, *, map_location, weights_only):
                self.assertEqual(stream.read(), b"original")
                self.assertEqual((map_location, weights_only), ("cpu", True))
                return {"global_step": 1, "adapter_sha": "other", "input_identity": {}}
            torch = SimpleNamespace(load=deserialize)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_trajectory(None, path, "0" * 64, "adapter", {}, (), None, None)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_trajectory(torch, path, hashlib.sha256(b"original").hexdigest(),
                                "adapter", {}, (), None, None)

    def test_capture_only_result_is_accepted_by_the_real_manifest_merger(self):
        sys.path.insert(0, str(ROOT / "generation/src"))
        from gencuda.manifest import merge_backend_result, new_manifest, sha256_path

        backend = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "run_hunyuanvideo_sage_grpo")
        training = next(node.body for node in backend.body if isinstance(node, ast.Try))
        branch = next(node for node in training if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "replay_path is not None or capture_only")
        wrapper = ast.parse("def diagnostic():\n    pass")
        wrapper.body[0].body = [branch]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "first-policy-update.json"
            trajectory = root / "step-0001.pt"
            update_state = root / "first-policy-update-state.pt"
            policy.write_text("{}", encoding="utf-8")
            trajectory.write_bytes(b"test")
            update_state.write_bytes(b"state")
            steps = []
            scope = dict(replay_path=None, capture_only=True, policy_path=policy,
                         update_state_path=update_state,
                         trajectory_paths=[trajectory], sha256_path=sha256_path, load_seconds=1.0,
                         hook_handle=SimpleNamespace(remove=lambda: None),
                         rl_step=lambda index, save_outputs: steps.append((index, save_outputs)))
            exec(compile(wrapper, str(SOURCE), "exec"), scope)
            result = scope["diagnostic"]()
            manifest = new_manifest("rl", "hunyuanvideo-1.5", {}, {})
            merge_backend_result(manifest, result)
            self.assertEqual(steps, [(1, False)])
            self.assertEqual(manifest["artifacts"][0]["path"], str(policy))
            self.assertIn({"path": str(update_state), "sha256": sha256_path(update_state)}, manifest["logs"])
            self.assertFalse(manifest["metrics"]["performance_accepted"])


if __name__ == "__main__":
    unittest.main()
