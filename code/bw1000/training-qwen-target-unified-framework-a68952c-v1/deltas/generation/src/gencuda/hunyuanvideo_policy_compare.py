"""Compare two pinned SAGE-GRPO first-policy updates on CPU."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from .checkpoint_compare import compare_trees


def _read_pinned(path, expected_sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"SHA256 differs for {path.name}")
    return raw


def _load(root, policy_sha):
    import torch

    policy = json.loads(_read_pinned(root / "first-policy-update.json", policy_sha))
    raw = _read_pinned(root / "first-policy-update-state.pt", policy["update_state"]["sha256"])
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (policy["global_step"] != 1 or state["global_step"] != 1
            or state["schema"] != "hunyuanvideo-sage-policy-update-v1"):
        raise ValueError("expected a first-policy update")
    expected = [(sample, step) for sample in range(4) for step in range(4)]
    if [(row["rollout"], row["transition"]) for row in state["transition_boundaries"]] != expected:
        raise ValueError("incomplete or reordered policy transitions")
    losses = state["transition_losses"]
    if len(losses) != 16 or not all(map(math.isfinite, losses)):
        raise ValueError("incomplete or non-finite transition losses")
    return policy, state


def _tensor_delta(ref, actual):
    import torch

    if not isinstance(ref, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ValueError("expected tensors in the trainable state")
    if ref.shape != actual.shape or ref.dtype != actual.dtype:
        raise ValueError("trainable tensor shape or dtype differs")
    a, b = ref.double(), actual.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite trainable tensor")
    delta = a - b
    return a.square().sum().item(), delta.square().sum().item(), delta.abs().max().item(), not torch.equal(ref, actual)


def _group_metrics(groups):
    metrics = {}
    for name, rows in sorted(groups.items()):
        reference, squared_error, maximum, different = zip(*rows)
        reference_sq = sum(reference)
        metrics[name] = {
            "tensors": len(rows),
            "different_tensors": sum(different),
            "max_abs": max(maximum),
            "relative_l2": math.sqrt(sum(squared_error) / reference_sq) if reference_sq else None,
        }
    return metrics


def compare_updates(reference_root, candidate_root, reference_policy_sha, candidate_policy_sha):
    reference, a = _load(reference_root, reference_policy_sha)
    candidate, b = _load(candidate_root, candidate_policy_sha)
    if (reference["trajectory"]["sha256"] != candidate["trajectory"]["sha256"]
            or reference["adapter_before"] != candidate["adapter_before"]
            or a["optimizer_parameter_names"] != b["optimizer_parameter_names"]):
        raise ValueError("trajectory, initial adapter or optimizer parameter identity differs")
    names = set(a["parameters_before"])
    fields = ("parameters_before", "parameters_after", "gradients_before_clip", "gradients_after_clip")
    if not names or any(set(state[field]) != names for state in (a, b) for field in fields):
        raise ValueError("trainable tensor names differ")
    initial_adapter = compare_trees(a["parameters_before"], b["parameters_before"])
    initial_optimizer = compare_trees(a["optimizer_before"], b["optimizer_before"])
    if not initial_adapter["tree_equal"] or not initial_optimizer["tree_equal"]:
        raise ValueError("initial trainable tensors or optimizer state differ")

    grouped = {field: {} for field in (*fields[2:], "parameter_updates")}
    for name in sorted(names):
        matches = [f"{projection}.{adapter}" for projection in ("img_attn_q", "img_attn_v")
                   for adapter in ("lora_A", "lora_B")
                   if projection in name.split(".") and adapter in name.split(".")]
        if len(matches) != 1:
            raise ValueError(f"unexpected trainable parameter {name}")
        group = matches[0]
        for field in fields[2:]:
            grouped[field].setdefault(group, []).append(_tensor_delta(a[field][name], b[field][name]))
        _tensor_delta(a["parameters_before"][name], a["parameters_after"][name])
        _tensor_delta(b["parameters_before"][name], b["parameters_after"][name])
        update_a = a["parameters_after"][name].double() - a["parameters_before"][name].double()
        update_b = b["parameters_after"][name].double() - b["parameters_before"][name].double()
        grouped["parameter_updates"].setdefault(group, []).append(_tensor_delta(update_a, update_b))

    return {
        "schema": "hunyuanvideo-sage-policy-comparison-v1",
        "trajectory_sha256": reference["trajectory"]["sha256"],
        "reference_policy_sha256": reference_policy_sha,
        "candidate_policy_sha256": candidate_policy_sha,
        "initial_adapter_equal": True,
        "initial_optimizer_equal": True,
        "trainable_tensors": len(names),
        "policy_equal": {field: reference[field] == candidate[field] for field in
                         ("reward", "rollouts", "reference_reward", "recomputed_old_log_probs", "adapter_after")},
        "transition_losses_equal": a["transition_losses"] == b["transition_losses"],
        "transition_boundary_mismatches": {
            key: [[x["rollout"], x["transition"]] for x, y in zip(a["transition_boundaries"], b["transition_boundaries"])
                  if x[key] != y[key]]
            for key in ("actor_output_sha256", "actor_output_gradient_sha256", "anchor_mean_sha256",
                        "accumulated_lora_gradients_sha256")
        },
        "grouped_tensor_metrics": {name: _group_metrics(groups) for name, groups in grouped.items()},
        "optimizer_after": compare_trees(a["optimizer_after"], b["optimizer_after"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-policy-sha256", required=True)
    parser.add_argument("--candidate-policy-sha256", required=True)
    args = parser.parse_args(argv)
    result = compare_updates(args.reference, args.candidate, args.reference_policy_sha256,
                             args.candidate_policy_sha256)
    print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
