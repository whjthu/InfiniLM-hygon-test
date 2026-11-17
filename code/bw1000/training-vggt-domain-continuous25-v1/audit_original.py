"""Independent stdlib audit of the one original VGGT 25-step/recovery run."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PINS = {
    "intent.json": "def360414498b07e46da29011a42d44bcb940c8b279d7b4709549ddac2b4f4b2",
    "outcome.json": "737d40d179c888caa74c55901fa667cdbfbb34667116ed83ae51a4a0ab1bfc18",
    "native-result.json": "3a1f3e188789a3f85181c5cf3fbb4a9855795cbb64428d8d77ed4695310d80fa",
    "before.json": "1f10ba8423c6cca74220dc476a8a4087795443148c06ee4fb0c47eb4cafaddfe",
    "after.json": "96e6ea6be8c9c0f5d4ce8e6e190a85786821e8f59be3b3f13e6510da62a5d535",
    "arms/continuous/receipt.json": "7ebdc568ad98d6b3d7316d2cf495255f6b0d6fe272bdd719b35f6bb17930c0af",
    "arms/resume10/receipt.json": "aa2ba8ffdc08e2fd691bd5358b01acd0543a1f08956a06f56806783d05898b82",
    "arms/resume20/receipt.json": "1e8fa62186b5743a3c016fa982b0e902069eed83d54337b10168ff0335649a06",
}


def read(root, name, pin=None):
    path = root / name
    if path.is_symlink() or path.stat().st_size > 256 * 1024:
        raise ValueError("Unexpected original evidence file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if pin is not None and digest != pin:
        raise ValueError("Original evidence changed: " + name)
    return json.loads(raw), digest


def audit(root):
    documents = {name: read(root, name, pin)[0] for name, pin in PINS.items()}
    intent, outcome = documents["intent.json"], documents["outcome.json"]
    if (intent["request"] != "fd7afad15b594969b8e9273b939f1fd6"
            or intent["publication"] != "218d6e6f9ff6baebfb7f46d43b8db71c9d4feb5c"
            or intent["deadline"] - intent["epoch"] != 1800):
        raise ValueError("Original request, publication or epoch changed")
    rows = {}
    for suffix, number in (("pre", 320), ("gpu", 321), ("post", 322)):
        accounting, _ = read(root, suffix + "-accounting.json")
        row = accounting["row"]
        if (row[0] != "1087656." + str(number) or row[2:4] != ["COMPLETED", "0:0"]
                or row[5] != "f11r4n19" or accounting["terminal_visible"] is not True
                or outcome["stages"][suffix] != accounting or accounting["rc"] != 0):
            raise ValueError("Original actual accounting is not successful terminal")
        rows[suffix] = row
    if "gres/dcu=1" not in rows["gpu"][6].split(","):
        raise ValueError("Actual one-device assignment changed")
    before, after = documents["before.json"], documents["after.json"]
    identity_keys = ("boot_id", "node", "pci", "holder_inventory_basis", "mode")
    if (any(before[k] != after[k] for k in identity_keys)
            or before["monotonic_ns"] >= after["monotonic_ns"]
            or before["step"] != "1087656.320" or after["step"] != "1087656.322"
            or before["kfd_holder_count"] != 0 or after["kfd_holder_count"] != 0
            or before["full_fd_inventory_performed"] is not True
            or after["full_fd_inventory_performed"] is not True):
        raise ValueError("Independent cleanup or device identity differs")
    receipts = {arm: documents["arms/" + arm + "/receipt.json"]
                for arm in ("continuous", "resume10", "resume20")}
    baseline = receipts["continuous"]
    processes, native_pins, comparisons = set(), {}, {}
    for arm, start in (("continuous", 0), ("resume10", 10), ("resume20", 20)):
        receipt = receipts[arm]
        process = receipt["process"]
        identity = (process["boot_id"], process["pid"], process["startticks"])
        if identity in processes or receipt["arm"] != arm:
            raise ValueError("Training branches are not distinct processes")
        processes.add(identity)
        if (receipt["executed_optimizer_steps"] != 25 - start
                or receipt["epoch"] != intent["epoch"]
                or receipt["deadline"] != intent["epoch"] + 1700):
            raise ValueError("Incomplete updates or reset epoch")
        for key in ("bundle_SHA256", "backend_SHA256", "checkpoint_identity", "scheduler", "gradient_accumulation"):
            if receipt[key] != baseline[key]:
                raise ValueError("Branch configuration differs")
        steps = receipt["steps"]
        log = [json.loads(line) for line in (root / "arms" / arm / "train/steps.ndjson").read_bytes().splitlines()]
        if (steps != log or [r["global_optimizer_step"] for r in steps] != list(range(start + 1, 26))
                or any(r["batch_cursor"] != (r["global_optimizer_step"] - 1) % 2 for r in steps)):
            raise ValueError("Actual update log or complete input cursor differs")
        if any(not math.isfinite(r[k]) for r in steps for k in ("loss", "grad_norm", "camera_head_grad_norm_before_clip")):
            raise ValueError("Nonfinite update evidence")
        native, native_sha = read(root, "arms/" + arm + "/native-result.json", receipt["result_SHA256"])
        native_pins[arm] = native_sha
        for key in ("camera_head_weights_changed", "frozen_aggregator_unchanged", "main_forward_device",
                    "nonzero_finite_camera_head_gradients", "resume_boundary"):
            if native["checks"][key] is not True:
                raise ValueError("Native training check failed: " + key)
        if native["metrics"]["executed_optimizer_steps"] != 25 - start:
            raise ValueError("Native result denominator differs")
        expected = {str(n) for n in (10, 20, 25) if n > start}
        if set(receipt["checkpoints"]) != expected:
            raise ValueError("Missing checkpoint boundary")
        if start:
            if receipt["resume_source_SHA256"] != baseline["checkpoints"][str(start)]["file_SHA256"]:
                raise ValueError("Resume did not use original checkpoint")
            loss_error = max(abs(r["loss"] - b["loss"]) for r, b in zip(steps, baseline["steps"][start:], strict=True))
            grad_error = max(abs(r["grad_norm"] - b["grad_norm"]) for r, b in zip(steps, baseline["steps"][start:], strict=True))
            exact = all(receipt["checkpoints"][n] == baseline["checkpoints"][n] for n in expected)
            if loss_error > 1e-6 or grad_error != 0 or not exact or receipt["final_state_SHA256"] != baseline["final_state_SHA256"]:
                raise ValueError("New-process checkpoint recovery differs")
            comparisons[arm] = dict(maximum_loss_error=loss_error, maximum_gradient_norm_error=grad_error,
                                    checkpoint_files_and_components_exact=exact)
    if documents["native-result.json"]["comparison"]["comparison_passed"] is not True:
        raise ValueError("Original collector recorded recovery failure")
    metrics = read(root, "arms/continuous/native-result.json")[0]["metrics"]
    return dict(independent_root_audit_passed=True, original_pins=PINS, accounting=rows,
                original_request=intent["request"], original_publication=intent["publication"],
                actual_update_invocations=45, continuous_updates=25, resume10_updates=15, resume20_updates=5,
                independent_processes=3, training_image_pairs=2, comparisons=comparisons,
                native_result_SHA256=native_pins, cleanup_KFD_holders=0,
                final_boundary_SHA256=baseline["final_state_SHA256"],
                training_loss_final=metrics["loss_final"],
                formal_CO3D_acceptance=False, formal_performance_acceptance=False,
                domain_heldout_effect_qualified=False, current_fleet_busy_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True, allow_nan=False))
