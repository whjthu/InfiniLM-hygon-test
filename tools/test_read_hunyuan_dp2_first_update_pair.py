import importlib.util
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import read_hunyuan_dp2_first_update_pair as pair


class ScalarOnly:
    @staticmethod
    def is_tensor(_value):
        return False


class PairTests(unittest.TestCase):
    def test_exact_scalar_differences_and_zero_reference(self):
        result = pair.compare_tree(ScalarOnly, np, {"step": 1, "value": [0.0, 0.0]},
                                   {"step": 1, "value": [3.0, 4.0]})
        self.assertEqual(result["scalar_leaves"], 3)
        self.assertEqual(result["different_scalars"], 2)
        self.assertEqual(result["absolute_l2"], 5)
        self.assertEqual(result["relative_l2"], 5)
        self.assertIsNone(pair.compare_tree(ScalarOnly, np, 0.0, 1.0)["relative_l2"])

    def test_incomplete_or_nonfinite_tree_rejected(self):
        for ref, target in (({"a": 1}, {"b": 1}), ([1], [1, 2]), (1.0, math.inf), ("a", "b")):
            with self.subTest(ref=ref, target=target), self.assertRaises(ValueError):
                pair.compare_tree(ScalarOnly, np, ref, target)

    def test_target_device_assignment_is_independent_by_rank(self):
        def rank(step, pci):
            return ({"slurm_device": {"step": step, "node": "f13r3n13", "pci": pci}},)
        left = rank("1103606.1612", "0000:09:00.0")
        right = rank("1103606.1612", "0000:36:00.0")
        self.assertEqual(pair.target_scope([left, right])["PCI_count"], 2)
        with self.assertRaises(ValueError):
            pair.target_scope([left, left])
        with self.assertRaises(ValueError):
            pair.target_scope([left, rank("1103606.1613", "0000:36:00.0")])

    def test_bound_run_metadata_rejects_a_substituted_flow(self):
        plan_file = Path(__file__).resolve().parents[1] / (
            "code/bw1000/training-hunyuanvideo-dp2-first-flow-replay-v1/CAPTURE_PLAN.json")
        plan = pair.read_binding_file(plan_file, pair.PLAN_SHA256)
        config = {"parallel_mode": "data_parallel", "world_size": 2, "sp": 1, "seed": plan["seed"],
                  "training_cache_sha": plan["cache_SHA256"], "transformer_version": "fixture",
                  "lora_target_modules": ["q", "v"], "lora_rank": 4, "lora_alpha": 8,
                  "learning_rate": 1e-5, "weight_decay": 0.01, "max_grad_norm": 1.0,
                  "warmup_steps": 5, "max_steps": 20, "checkpoint_steps": [10, 20, 25],
                  "deterministic_algorithms": True, "gradient_checkpointing": True,
                  "first_flow_replay": {"plan_SHA256": pair.PLAN_SHA256, "manifest_SHA256": "a" * 64}}

        def write(root, name, value):
            data = json.dumps(value).encode()
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return sha256(data).hexdigest()

        with TemporaryDirectory() as temp:
            for wrong in (False, True):
                root = Path(temp).resolve() / str(wrong)
                logs = []
                for rank in (0, 1):
                    name = f"rank-{rank}"
                    summary = {"rank": rank, "completed_global_step_at_start": 0, "measurement_steps": 0,
                               "formal_performance_eligible": False, "adapter_before": plan["initial_adapter_SHA256"],
                               "adapter_after": "e" * 64, "process_identity_SHA256": str(rank) * 64,
                               "first_update_artifacts": [
                                   {"path": f"/private/{name}/first-update-{phase}.pt"} for phase in pair.PHASES]}
                    step = {"rank": rank, "global_step": 1, "flow_input_sha256": plan["expected_flow_SHA256"][str(rank)],
                            "replica_sha256": "a" * 64, "rank_state_sha256": "b" * 64, "rng_after_sha256": "c" * 64}
                    if wrong and rank == 1:
                        step["flow_input_sha256"] = plan["expected_flow_SHA256"]["0"]
                    for leaf, data in (("summary.json", summary), ("steps.ndjson", step)):
                        rel = f"{name}/{leaf}"
                        logs.append({"path": f"/private/{rel}", "sha256": write(root, rel, data)})
                request_sha = write(root, "dp2-request.json", {"config": config})
                result_sha = write(root, "dp2-result.json", {
                    "checks": {"replicated_state_agreement": True}, "logs": logs,
                    "metrics": {"total_global_steps": 1, "world_size": 2, "parallel_mode": "data_parallel",
                                "steps": 1, "warmup_steps": 0, "performance_accepted": False}})
                evidence_sha = write(root, "dp2-evidence.json", {
                    "schema": "hunyuanvideo-dp2-post-torchrun-evidence-v1", "torchrun_exit_code": 0,
                    "backend_SHA256": pair.BACKEND_SHA256, "request_SHA256": request_sha, "result_SHA256": result_sha,
                    "rank_window_metrics": {"measurement_updates_per_second": None,
                                            "complete_5_warmup_20_measured_window": False}})
                if wrong:
                    with self.assertRaisesRegex(ValueError, "provenance differs for rank 1"):
                        pair.read_run(root, evidence_sha, plan, pair.recipe(config))
                else:
                    self.assertEqual(len(pair.read_run(root, evidence_sha, plan, pair.recipe(config))[1]), 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU tensor runtime")
    def test_real_bf16_cpu_parameter_delta(self):
        import torch

        def state(values):
            return {"trainable": {"adapter": torch.tensor(values, dtype=torch.bfloat16, device="cpu")}}

        ref = pair.updates(torch, state([3, 4, 5]), state([6, 4, 5]))
        desired = pair.updates(torch, state([3, 4, 5]), state([5, 4, 9]))
        result = pair.compare_tree(torch, np, ref, desired)
        self.assertEqual(result["tensor_elements"], 3)
        self.assertEqual(result["different_elements"], 2)
        self.assertEqual(result["max_absolute"], 4.0)
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(17) / 3)


if __name__ == "__main__":
    unittest.main()
