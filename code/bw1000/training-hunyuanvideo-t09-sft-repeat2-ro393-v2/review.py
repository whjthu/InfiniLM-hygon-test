"""Read native stored metadata, preserving every attempted step on failures."""
import math
import re
from pathlib import Path
from contract import CACHE_SHA, COMPUTE_MODEL_STAMP, LABEL, NODE, PARENT, REQUEST, RUN, load, need, read
from slurm_accounting import is_terminal
from flash_compat import BACKING, CPU_RECEIPT_SHA, ORIGINAL_SHA, PROVIDER_SHA, RELATIVE, descriptor
from contract import NATIVE

COLUMNS = ["JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]


def native(root):
    result = {"planned_updates": 25, "completed_updates": 0, "missing_updates": 25,
              "stored_records_valid": False, "native_evidence_complete": False,
              "planned_checkpoints": [10, 20, 25], "checkpoint_files": {},
              "original_numerics_qualified": False, "recovery_qualified": False,
              "domain_effect_qualified": False, "performance_qualified": False}
    phase = "steps_read"
    try:
        raw, result["steps_pin"] = read(root / "steps.ndjson", 65536)
        import json
        phase = "steps"
        for line in raw.splitlines():
            row = json.loads(line)
            need(type(row["global_step"]) is int and row["global_step"] == result["completed_updates"] + 1
                 and row["global_step"] <= 25, "Native step sequence differs")
            for field in ("loss", "grad_norm", "step_seconds", "forward_seconds", "backward_update_seconds"):
                need(type(row[field]) in (int, float) and math.isfinite(row[field]), "Nonfinite native record")
            need(row["grad_norm"] > 0 and row["step_seconds"] > 0, "Zero native gradient or duration")
            result["completed_updates"] += 1
            result["missing_updates"] = 25 - result["completed_updates"]
        need(result["completed_updates"] == 25, "Stored native update denominator incomplete")
        result["stored_records_valid"] = True
        phase = "summary_read"
        summary, result["summary_pin"] = load(root / "summary.json")
        phase = "summary_checks"
        need(summary["optimizer"] == "Muon" and summary["trainable_parameters"] == 1769472
             and summary["lora_targets"] == ["img_attn_q", "img_attn_v"] and summary["lora_rank"] == 4
             and summary["cache_sha256"] == CACHE_SHA and summary["configured_warmup_steps"] == 5
             and summary["configured_measured_steps"] == 20 and summary["completed_global_step_at_start"] == 0
             and summary["external_resume"] is False, "Original native recipe summary differs")
        need(summary["invocation"] == {"executed_global_steps": 25, "executed_warmup_steps": 5,
             "executed_measured_steps": 20, "first_global_step": 1}, "Native invocation counts differ")
        checks = summary["checks"]
        for key in ("finite_tensors", "nonzero_finite_gradients", "current_adapter", "main_forward_device",
                    "attention_cuda_sdpa_fallback", "peak_reserved_below_75gb", "muon_optimizer"):
            need(checks[key] is True, "Original backend check not passed")
        need(checks["checkpoint_resume"] is None, "Unexpected extra recovery calls")
        need(summary["adapter_sha_before"] != summary["adapter_sha_after"], "No actual adapter change")
        need(all(isinstance(summary[key], str) and re.fullmatch(r"[0-9a-f]{64}", summary[key])
                 for key in ("adapter_sha_before", "adapter_sha_after")), "Adapter value hashes malformed")
        result["adapter_before_SHA256"] = summary["adapter_sha_before"]
        result["adapter_after_SHA256"] = summary["adapter_sha_after"]
        phase = "manifest"
        manifest, result["manifest_pin"] = load(root / "manifest.json", 256 * 1024)
        need(manifest["status"] == "completed" and manifest["execution"]["status"] == "completed",
             "Native manifest did not complete")
        need(manifest["runtime"]["accelerator"] == "hip", "Native runtime is not actual HIP")
        phase = "artifacts"
        for step in (10, 20, 25):
            _, result["checkpoint_files"][str(step)] = read(root / f"checkpoint-{step}/state.pt", 64 * 1024**2)
        _, result["adapter_final_pin"] = read(root / "adapter-final.safetensors", 16 * 1024**2)
        for pin in [result["summary_pin"], result["adapter_final_pin"], *result["checkpoint_files"].values()]:
            declared = [item for item in manifest["artifacts"] if item["path"] == pin["path"]]
            need(len(declared) == 1 and declared[0]["sha256"] == pin["SHA256"],
                 "Actual stored artifact differs from native manifest")
        result["native_evidence_complete"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error_type"] = type(error).__name__
        result["failure_phase"] = phase
    return result


def terminal(receipts, all_success=True):
    need(set(receipts) == {"pre", "gpu", "post"}, "Missing phase accounting")
    rows = []
    for phase in ("pre", "gpu", "post"):
        item = receipts[phase]
        row = item.get("row")
        need(item.get("child_reaped") is True and row is not None and is_terminal(row)
             and len(row) == len(COLUMNS) and row[1] == LABEL + "-" + phase
             and re.fullmatch(re.escape(PARENT) + r"\.[0-9]+", row[0]) and row[5] == NODE,
             "Actual terminal phase is incomplete")
        if all_success or phase != "gpu":
            need(item.get("passed") is True and row[2:4] == ["COMPLETED", "0:0"],
                 "Required phase not COMPLETED 0:0")
        rows.append(int(row[0].split(".")[1]))
    need(rows == sorted(set(rows)), "Original phases are not separate ordered steps")


from framework_mount import post_mount


def post(workspace, receipts):
    terminal(receipts, all_success=False)
    pre, _ = load(workspace / "pre" / (NODE + ".json"))
    after, afterpin = load(workspace / "post" / (NODE + ".json"))
    device, devicepin = load(workspace / "assigned-device.json")
    need(pre["pci"] == after["pci"] and len(set(pre["pci"])) == 8
         and pre["boot_id"] == after["boot_id"] == device["boot_id"]
         and pre["node"] == after["node"] == NODE
         and pre["step"] == receipts["pre"]["row"][0]
         and device["request_id"] == REQUEST and device["step"] == receipts["gpu"]["row"][0]
         and device["assets"]["execution_mount_checked"] is True
         and device["assets"]["weight_file_identity"] == COMPUTE_MODEL_STAMP
         and after["step"] == receipts["post"]["row"][0]
         and device["device"]["pci"] in pre["pci"]
         and pre["kfd_holder_count"] == after["kfd_holder_count"] == 0,
         "Independent post or device identity differs")
    mount, mount_pin = load(workspace / "framework-mount-ingress.json")
    post_mount(mount, REQUEST, device)
    compatibility, compatibility_pin = load(workspace / "compatibility-ingress.json")
    need(compatibility == device["compatibility"] and compatibility["provider_CPU_receipt"]["SHA256"] == CPU_RECEIPT_SHA
         and compatibility["provider_current_source"]["SHA256"] == PROVIDER_SHA
         and compatibility["read_only_bind_verified"] is True and compatibility["container_compile_passed"] is True
         and compatibility["actual_provider_function_replayed"] is False and compatibility["attention_math_changed"] is False,
         "Frozen compatibility/provider ingress differs")
    spec = descriptor()
    _, backing = read(BACKING, spec["patched"]["bytes"])
    _, original = read(NATIVE / RELATIVE, 3909)
    need(backing["SHA256"] == spec["patched"]["SHA256"] == compatibility["overlay"]["SHA256"]
         and original["SHA256"] == ORIGINAL_SHA, "Saved overlay or canonical original changed")
    return {"post_device_verified": True, "post_pin": afterpin, "assigned_device_pin": devicepin,
            "compatibility_ingress_pin": compatibility_pin, "framework_mount_ingress_pin": mount_pin, "post_overlay_backing_pin": backing,
            "post_original_canonical_pin": original, "readonly_bind_runtime_attested": True,
            "coverage": "Audited passive KFD+sameUID FD helper; unavailable inventory fails closed"}
