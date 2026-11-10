import json
from pathlib import Path
import sys
import torch
from hygon_sf3d_compare_first_step import FIELDS, compare, verify_receipt
from hygon_sf3d_first_step import read_reference

root = Path("/reference")
reference, _ = read_reference(torch, root)
policy = json.loads((root / "policy.json").read_bytes())
actual = {k: reference[k] for k in FIELDS}
positive = compare(torch, actual, reference, policy)
assert positive["passed"] and len(positive["checks"]) == 12
assert all(positive["metrics"][name]["relative_l2"] == 0.0 for name in (
    "prediction", "gradients_before_clip", "gradients_after_clip", "parameter_update",
    "optimizer_exp_avg", "optimizer_exp_avg_sq",
))
rejected = []
def with_field(key, value):
    return {**actual, key: value}
def rejected_value(label, value, check):
    result = compare(torch, value, reference, policy)
    assert not result["passed"] and not result["checks"][check]
    rejected.append(label)
def rejected_structure(label, value, fragment):
    try:
        compare(torch, value, reference, policy)
    except ValueError as error:
        assert fragment in str(error)
        rejected.append(label)
    else:
        raise AssertionError(label)

rejected_value("loss_above_policy", with_field("loss", actual["loss"] * 1.02), "loss")
rejected_value("all_updates_zero", with_field("final_trainable", actual["initial_trainable"]), "nonzero_gradient_and_update")
rejected_value("wrong_gradient_direction", with_field("gradients_before_clip",
    {key: -value for key, value in actual["gradients_before_clip"].items()}), "gradient_direction")
rejected_value("optimizer_first_moment_above_policy", with_field("final_optimizer", {
    **actual["final_optimizer"], "state": {
        i: {**row, "exp_avg": row["exp_avg"] * 1.10}
        for i, row in actual["final_optimizer"]["state"].items()
    },
}), "optimizer_exp_avg")
rejected_structure("changed_fixed_input", with_field("inputs", {**actual["inputs"], "roughness": actual["inputs"]["roughness"] + 0.1}), "tensor values differ")
rejected_structure("nonfinite_material_output", with_field("outputs", {**actual["outputs"], "roughness": actual["outputs"]["roughness"] * float("nan")}), "nonfinite tensor")
rejected_structure("wrong_adamw_step", with_field("final_optimizer", {
    **actual["final_optimizer"], "state": {
        i: {**row, "step": row["step"] + 1}
        for i, row in actual["final_optimizer"]["state"].items()
    },
}), "/step")
try:
    verify_receipt({}, "a" * 64, "b" * 64)
except ValueError as error:
    assert "frozen binding" in str(error)
    rejected.append("missing_hip_and_provenance_receipt")
else:
    raise AssertionError("missing receipt accepted")
print(json.dumps({"mode":"A100 artifact comparator qualification; CPU only", "checks":positive["checks"], "expected_rejections":rejected}))
