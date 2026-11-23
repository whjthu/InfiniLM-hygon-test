"""Independently compare the private, one-update DP2 replay artifacts."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath

from compare_training_dp2_first_rank_metadata import hex_digest, need, pinned, recipe


PLAN_SHA256 = "e67398ba3b99eeb14b46d65475666b75380dcb1d2ac649b5470bced4765edd3e"
BACKEND_SHA256 = "d6dcba6f9f495e843d51598e748b578f7583f9ecab7059990a23b97366349656"
PHASES = ("before", "forward", "preclip", "postclip", "updated")


def read_binding_file(path, digest):
    need(not path.is_symlink(), "original binding file must not be a symbolic link")
    with path.open("rb") as stream:
        raw = stream.read(16385)
    need(0 < len(raw) <= 16384 and sha256(raw).hexdigest() == digest, "original binding bytes differ")
    return json.loads(raw)


def read_run(root, evidence_sha, plan, original_recipe):
    evidence = json.loads(pinned(root, "dp2-evidence.json", evidence_sha))
    need(evidence["schema"] == "hunyuanvideo-dp2-post-torchrun-evidence-v1"
         and evidence["torchrun_exit_code"] == 0 and evidence["backend_SHA256"] == BACKEND_SHA256
         and evidence["rank_window_metrics"]["measurement_updates_per_second"] is None
         and evidence["rank_window_metrics"]["complete_5_warmup_20_measured_window"] is False,
         "expected the frozen, untimed first-update backend evidence")
    config = json.loads(pinned(root, "dp2-request.json", evidence["request_SHA256"]))["config"]
    binding = config["first_flow_replay"]
    need(recipe(config) == original_recipe and config.get("resume_from_checkpoint") is None
         and binding["plan_SHA256"] == PLAN_SHA256 and hex_digest(binding["manifest_SHA256"]),
         "diagnostic recipe or sealed input binding differs")
    result = json.loads(pinned(root, "dp2-result.json", evidence["result_SHA256"]))
    metrics = result["metrics"]
    need(metrics["world_size"] == 2 and metrics["parallel_mode"] == "data_parallel"
         and metrics["total_global_steps"] == metrics["steps"] == 1 and metrics["warmup_steps"] == 0
         and metrics["performance_accepted"] is False
         and result["checks"]["replicated_state_agreement"] is True, "unexpected DP2 diagnostic result")
    logs = {}
    for leaf in result["logs"]:
        name = "/".join(PurePosixPath(leaf["path"]).parts[-2:])
        need(name not in logs, f"duplicate rank log binding: {name}")
        logs[name] = leaf["sha256"]
    need(set(logs) == {f"rank-{rank}/{name}" for rank in (0, 1) for name in ("summary.json", "steps.ndjson")},
         "rank log denominator differs")
    ranks = []
    for rank in (0, 1):
        directory = f"rank-{rank}"
        summary_name, step_name = f"{directory}/summary.json", f"{directory}/steps.ndjson"
        summary = json.loads(pinned(root, summary_name, logs[summary_name]))
        [step] = [json.loads(line) for line in pinned(root, step_name, logs[step_name]).splitlines()]
        need(type(summary["rank"]) is type(step["rank"]) is int and rank == summary["rank"] == step["rank"]
             and summary["completed_global_step_at_start"] == 0 and step["global_step"] == 1
             and summary["formal_performance_eligible"] is False and summary["measurement_steps"] == 0
             and summary["adapter_before"] == plan["initial_adapter_SHA256"]
             and summary["adapter_before"] != summary["adapter_after"]
             and step["flow_input_sha256"] == plan["expected_flow_SHA256"][str(rank)]
             and all(hex_digest(step[key]) for key in ("replica_sha256", "rank_state_sha256", "rng_after_sha256"))
             and hex_digest(summary["process_identity_SHA256"]),
             f"first-update provenance differs for rank {rank}")
        artifacts = summary["first_update_artifacts"]
        names = [PurePosixPath(item["path"]).parts[-2:] for item in artifacts]
        need(names == [(directory, f"first-update-{phase}.pt") for phase in PHASES],
             f"phase denominator differs for rank {rank}")
        ranks.append((summary, step, dict(zip(PHASES, artifacts, strict=True))))
    need(len({item[0]["process_identity_SHA256"] for item in ranks}) == 2
         and len({item[1]["replica_sha256"] for item in ranks}) == 1,
         "rank process identities or replicated update differ")
    return binding["manifest_SHA256"], ranks


def load_phase(torch, root, rank, phase, pin):
    directory = root / f"rank-{rank}"
    path = directory / f"first-update-{phase}.pt"
    need(not directory.is_symlink() and not path.is_symlink() and path.resolve(strict=True).is_relative_to(root),
         "phase path escaped its run")
    size = pin["bytes"]
    need(type(size) is int and 0 < size <= 64 << 20 and hex_digest(pin["sha256"]), "invalid phase extent or SHA")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    need(len(raw) == size and sha256(raw).hexdigest() == pin["sha256"], "phase bytes differ from rank summary")
    return torch.load(BytesIO(raw), map_location="cpu", weights_only=True)


def target_scope(ranks):
    first, second = [row[0]["slurm_device"] for row in ranks]
    need(all(isinstance(item, dict) and set(item) == {"step", "node", "pci"} for item in (first, second))
         and first["step"] == second["step"] and first["node"] == second["node"]
         and first["pci"] != second["pci"], "target rank device assignment differs")
    return {"step": first["step"], "node": first["node"], "PCI_count": 2}


def compare_tree(torch, np, reference, target):
    totals = {"tensor_leaves": 0, "tensor_elements": 0, "scalar_leaves": 0, "different_elements": 0,
              "different_scalars": 0, "max_absolute": 0.0, "sum_squared_delta": 0.0, "sum_squared_reference": 0.0}

    def visit(a, b, location):
        need(type(a) is type(b), f"tree types differ at {location}")
        if torch.is_tensor(a):
            need(a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == "cpu",
                 f"tensor metadata differs at {location}")
            need(not a.is_complex(), f"tensor must be real at {location}")
            xa, xb = (item.detach().to(dtype=torch.float64).numpy() for item in (a, b))
            need(np.isfinite(xa).all() and np.isfinite(xb).all(), f"tensor must be finite at {location}")
            delta = np.abs(xa - xb)
            totals["tensor_leaves"] += 1
            totals["tensor_elements"] += xa.size
            totals["different_elements"] += int(torch.count_nonzero(a != b).item())
        elif isinstance(a, dict):
            need(a.keys() == b.keys(), f"mapping leaves differ at {location}")
            for key in a:
                visit(a[key], b[key], f"{location}/{key}")
            return
        elif isinstance(a, (list, tuple)):
            need(len(a) == len(b), f"sequence lengths differ at {location}")
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                visit(left, right, f"{location}/{index}")
            return
        elif type(a) in (float, int):
            xa, xb = np.float64(a), np.float64(b)
            need(np.isfinite(xa) and np.isfinite(xb), f"numeric scalar must be finite at {location}")
            delta = np.float64(abs(a - b))
            totals["scalar_leaves"] += 1
            totals["different_scalars"] += int(a != b)
        else:
            need(type(a) in (str, bool, type(None)) and a == b, f"structural state differs at {location}")
            return
        totals["max_absolute"] = max(totals["max_absolute"], float(np.max(delta, initial=0.0)))
        totals["sum_squared_delta"] += float(np.sum(delta * delta, dtype=np.float64))
        totals["sum_squared_reference"] += float(np.sum(xa * xa, dtype=np.float64))

    visit(reference, target, "root")
    numerator = totals.pop("sum_squared_delta")
    denominator = totals.pop("sum_squared_reference")
    need(np.isfinite(numerator) and np.isfinite(denominator), "comparison norm overflow")
    totals["absolute_l2"] = numerator**0.5
    totals["relative_l2"] = (numerator / denominator)**0.5 if denominator else None
    return totals


def updates(torch, before, after):
    a, b = before["trainable"], after["trainable"]
    need(a and a.keys() == b.keys(), "adapter update denominator differs")
    result = {}
    for key, old in a.items():
        new = b[key]
        need(torch.is_tensor(old) and torch.is_tensor(new) and old.shape == new.shape
             and old.dtype == new.dtype and old.is_floating_point(), "adapter metadata changed within update")
        result[key] = new.to(dtype=torch.float64) - old.to(dtype=torch.float64)
    return result


def compare(torch, np, reference, target, ref_ranks, target_ranks):
    cases = []
    for rank, (ref, obs) in enumerate(zip(ref_ranks, target_ranks, strict=True)):
        measurements, adapters = {}, {}
        for phase in PHASES:
            a, b = (load_phase(torch, root, rank, phase, row[2][phase])
                    for root, row in ((reference, ref), (target, obs)))
            if phase in ("before", "updated"):
                adapters[phase] = (a, b)
                need(all(isinstance(state.get("optimizer", {}).get("state"), dict)
                         and isinstance(state.get("trainable"), dict) and state["trainable"]
                         for state in (a, b)), "incomplete adapter or optimizer state")
                if phase == "updated":
                    need(a["optimizer"]["state"] and b["optimizer"]["state"], "missing updated optimizer state")
            if phase == "forward":
                need(set(a) == set(b) == {"prediction", "loss"} and torch.is_tensor(a["prediction"])
                     and a["prediction"].numel() > 0 and a["loss"].numel() == b["loss"].numel() == 1
                     and a["loss"].item() == ref[1]["loss"] and b["loss"].item() == obs[1]["loss"],
                     "forward artifact does not match the recorded first training step")
                for key in ("prediction", "loss"):
                    measurements[key] = compare_tree(torch, np, a[key], b[key])
            elif phase == "postclip":
                need(set(a) == set(b) == {"norm", "gradients"}
                     and set(a["gradients"]) == set(b["gradients"]) == set(adapters["before"][0]["trainable"])
                     and all(torch.is_tensor(tensor) for value in (a, b) for tensor in value["gradients"].values())
                     and a["norm"].item() == ref[1]["grad_norm_before_clip"]
                     and b["norm"].item() == obs[1]["grad_norm_before_clip"],
                     "clipping artifact does not match the recorded first training step")
                measurements["gradient_norm_before_clip"] = compare_tree(torch, np, a["norm"], b["norm"])
                measurements["gradient_after_clip"] = compare_tree(torch, np, a["gradients"], b["gradients"])
            else:
                if phase == "preclip":
                    need(set(a) == set(b) == set(adapters["before"][0]["trainable"])
                         and all(torch.is_tensor(tensor) for value in (a, b) for tensor in value.values()),
                         "incomplete preclip gradients")
                measurements[{"before": "initial_state", "preclip": "gradient_before_clip",
                              "updated": "state_after_update"}[phase]] = compare_tree(torch, np, a, b)
        before, after = adapters["before"], adapters["updated"]
        measurements["optimizer_state"] = compare_tree(torch, np, after[0]["optimizer"], after[1]["optimizer"])
        changes = [updates(torch, before[index], after[index]) for index in (0, 1)]
        need(all(any(torch.count_nonzero(tensor).item() for tensor in update.values()) for update in changes),
             "expected a real nonzero adapter update on both platforms")
        measurements["parameter_update"] = compare_tree(torch, np, *changes)
        cases.append({"rank": rank, "flow_SHA256": ref[1]["flow_input_sha256"], "comparisons": measurements})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "original-launch", "reference", "target", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("reference-evidence-sha256", "target-evidence-sha256"):
        parser.add_argument("--" + name, required=True)
    arguments = parser.parse_args()
    plan = read_binding_file(arguments.plan, PLAN_SHA256)
    original = recipe(read_binding_file(arguments.original_launch, plan["launch_SHA256"])["config"])
    ref_root, obs_root = (path.resolve(strict=True) for path in (arguments.reference, arguments.target))
    need(ref_root != obs_root and arguments.reference_evidence_sha256 != arguments.target_evidence_sha256,
         "expected separate platform receipts")
    ref_manifest, ref_ranks = read_run(ref_root, arguments.reference_evidence_sha256, plan, original)
    obs_manifest, obs_ranks = read_run(obs_root, arguments.target_evidence_sha256, plan, original)
    need(ref_manifest == obs_manifest, "captures are not the same sealed input manifest")
    import numpy as np
    import torch

    report = {"case": "T09.dp2", "plan_SHA256": PLAN_SHA256, "input_manifest_SHA256": ref_manifest,
              "reference_evidence_SHA256": arguments.reference_evidence_sha256,
              "target_evidence_SHA256": arguments.target_evidence_sha256,
              "target_reported_device_scope": target_scope(obs_ranks),
              "ranks": compare(torch, np, ref_root, obs_root, ref_ranks, obs_ranks),
              "scope": "two ranks and one untimed optimizer update per run",
              "formal_numerics_qualified": False, "hardware_accounting_independently_verified": False}
    with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
