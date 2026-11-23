"""Localize SDXL first-update error using the original .596/.598 scalar comparison."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = ROOT / "training-sdxl-sdpa-adjoint-fp32-v1-a93d41a0d31140f1ad99bb105a024e99"
COMPARISON = "cbcf56ab5a83e89c2a8e772eed8f89743365a07ede763be3a5b5c7d6585b176c"
POLICY = "075e86e54a3eeefb8d5aad219a23f16959d14bd42b22103cd5fe5302c823f6b5"
REFERENCE = "c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036"
OBSERVED = "bda1a100837fa939137c925f14b1f60bffddde0bc38aa5af94514f0e51f5d37c"
TENSOR_COUNT = 1120
NAME = re.compile(
    r"(?P<block>down_blocks\.\d+|mid_block|up_blocks\.\d+)"
    r"\.attentions\.\d+\.transformer_blocks\.\d+\."
    r"(?P<attention>attn[12])\.(?P<projection>to_q|to_k|to_v|to_out\.0)"
    r"\.lora_(?P<factor>[AB])\.default\.weight"
)


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate object key in archived scalar evidence")
    return result


def pinned_json(path, expected_bytes, expected_sha256):
    with path.open("rb") as stream:
        raw = stream.read(expected_bytes + 1)
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"mismatched archived scalar evidence: {path.name}")
    return json.loads(raw, object_pairs_hook=unique_object)


def validate_metric(metric, names):
    rows = metric["per_tensor"]
    if len(rows) != TENSOR_COUNT or rows.keys() != names:
        raise ValueError("incomplete or nonmatching trainable parameter coverage")
    sums = {}
    for key in ("reference_l2_squared", "observed_l2_squared", "difference_l2_squared"):
        values = [row[key] for row in rows.values()]
        if any(type(v) not in (float, int) or v < 0 or not math.isfinite(v) for v in values):
            raise ValueError(f"invalid squared norm: {key}")
        sums[key] = math.fsum(values)
    reference = math.sqrt(sums["reference_l2_squared"])
    observed = math.sqrt(sums["observed_l2_squared"])
    error = math.sqrt(sums["difference_l2_squared"])
    if not reference or not all(math.isclose(metric[key], actual, rel_tol=1e-10)
                                for key, actual in (("reference_l2", reference), ("observed_l2", observed),
                                                    ("relative_l2", error / reference))):
        raise ValueError("per-tensor squared norms do not reproduce the original full comparison")
    return sums


def reduce(comparison, policy):
    if (comparison["stage_id"] != "sdxl-sft" or comparison["accelerator"] != "hip"
            or comparison["boundary"] != "fixed_realized_inputs" or comparison["passed"] is not False
            or comparison["checks"]["parameter_update"] is not False
            or comparison["reference_sha256"] != REFERENCE or comparison["observed_sha256"] != OBSERVED
            or comparison["policy_sha256"] != POLICY or policy["stage_id"] != "sdxl-sft"
            or policy["reference_payload_sha256"] != REFERENCE):
        raise ValueError("wrong SDXL first-update comparison or policy identity")
    gate = policy["metrics"]["parameter_update_relative_l2_max"]
    if gate != 0.05:
        raise ValueError("wrong threshold for the original first-update gate")
    update = comparison["metrics"]["parameter_update"]
    before = comparison["metrics"]["gradients_before_clip"]
    after = comparison["metrics"]["gradients_after_clip"]
    names = update["per_tensor"].keys()
    update_norms, _, gradient_norms = (validate_metric(metric, names) for metric in (update, before, after))
    error_total = update_norms["difference_l2_squared"]
    reference_total = update_norms["reference_l2_squared"]
    gradient_total = gradient_norms["difference_l2_squared"]
    budget = gate ** 2 * reference_total
    if not 0 < budget < error_total or gradient_total <= 0:
        raise ValueError("archived update must have a nonzero failing error")

    groups = {key: defaultdict(list) for key in (
        "factor", "block", "attention", "projection", "attention_projection",
        "block_attention_projection", "attention_module",
    )}
    for name in names:
        match = NAME.fullmatch(name)
        if not match:
            raise ValueError(f"unknown trainable SDXL parameter layout: {name}")
        block, attention, projection, factor = match.groups()
        keys = (factor, block, attention, projection, f"{attention}.{projection}", f"{block}.{attention}.{projection}",
                name[:match.start("attention")] + attention)
        row = (name, update["per_tensor"][name]["difference_l2_squared"],
               after["per_tensor"][name]["difference_l2_squared"])
        for kind, key in zip(groups, keys):
            groups[kind][key].append(row)

    def amount(rows, index=1):
        return math.fsum(row[index] for row in rows)

    def projection(name, rows, remaining):
        return {
            "name": name, "tensors": len(rows), "nonzero_update_tensors": sum(row[1] > 0 for row in rows),
            "update_error_share": amount(rows) / error_total,
            "gradient_error_share": amount(rows, 2) / gradient_total,
            "perfect_replacement_remaining_update_relative_l2": math.sqrt(remaining / reference_total),
            "perfect_replacement_meets_gate": remaining <= budget,
        }

    def summarize(partition):
        total_update = math.fsum(amount(rows) for rows in partition.values())
        total_gradient = math.fsum(amount(rows, 2) for rows in partition.values())
        if (sum(map(len, partition.values())) != TENSOR_COUNT
                or not math.isclose(total_update, error_total, rel_tol=1e-10)
                or not math.isclose(total_gradient, gradient_total, rel_tol=1e-10)):
            raise ValueError("grouped sums do not close")
        order = sorted(partition, key=lambda name: (-amount(partition[name]), name))
        leading = [
            projection(name, partition[name], math.fsum(amount(partition[k]) for k in order if k != name))
            for name in order[:12]
        ]
        squared_errors = [amount(partition[name]) for name in order]
        minimum = next(n for n in range(1, len(order) + 1) if math.fsum(squared_errors[n:]) <= budget)
        return {
            "groups": len(order), "nonzero_groups": sum(value > 0 for value in squared_errors),
            "minimum_perfect_groups_for_gate": minimum,
            "leading_groups": leading,
            "gate_crossing_group": order[minimum - 1],
            "best_prefix_remaining_update_relative_l2": math.sqrt(
                math.fsum(squared_errors[minimum:]) / reference_total
            ),
        }

    summary = {kind: summarize(partition) for kind, partition in groups.items()}
    tensors = {row[0]: [row] for rows in groups["factor"].values() for row in rows}
    summary["individual_tensor"] = summarize(tensors)
    return {
        "schema": "bw1000.training.sdxl.original596.update_budget.v1",
        "original_GPU_step": "1087656.596",
        "original_comparator_step": "1087656.598",
        "original_comparison_SHA256": COMPARISON,
        "original_policy_SHA256": POLICY,
        "tensors": TENSOR_COUNT,
        "original_update_relative_l2": update["relative_l2"],
        "original_postclip_gradient_relative_l2": after["relative_l2"],
        "reference_update_squared_l2": reference_total,
        "update_error_squared_l2": error_total,
        "update_relative_l2_gate": gate,
        "minimum_update_squared_error_share_to_remove": 1 - budget / error_total,
        "group_scopes": summary,
        "scope": "Archived scalar norms only. Perfect group replacement holds all other coordinates fixed. "
                 "Group locations identify affected trainables, not the upstream operator that caused the error.",
        "new_CPU_steps": 0,
        "new_GPU_steps": 0,
        "training_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=RUN / "comparison.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "training-t06-sdxl-first-reference-r2/policy.json")
    args = parser.parse_args()
    comparison = pinned_json(args.comparison, 1783928, COMPARISON)
    policy = pinned_json(args.policy, 2677, POLICY)
    print(json.dumps(reduce(comparison, policy), sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
