"""Verify the training acceptance matrix and the evidence pinned by it."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median

from training_formal_metrics import (
    complete_window, sha, stage_three, verify_hunyuan_a100_recorded_updates, verify_hunyuan_dp2_a100_windows,
)
from training_A100_engineering_timing import VGGT_CASE, check_generation, check_vggt, check_vlm
from training_framework_identity import verify_overlay_source, verify_registry_source
from training_function import verify_function
from training_numerics import (
    ZERO_LR_VLM_REQUESTS, comparison_plan as numerics_plan, original_vlm_update_schedule, verify_numerics,
)
from training_recovery import recovery_plan, verify_recovery, verify_registered_zimage_recovery
from training_sample_artifacts import verifier as sample_artifact_verifier


TRAINING_GROUPS = {f"T{i:02d}" for i in range(1, 13)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_zimage_engineering_windows(report, reference, a100_median, read_json):
    """Bind the measured Z-Image windows to their original claim and A100 clock audit."""
    manifest_item = report["source_manifest"]
    manifest = read_json(manifest_item)
    claim_pin = manifest["files"]["CLAIM.json"]
    claim_path = PurePosixPath(manifest_item["path"]).with_name("CLAIM.json")
    claim = read_json(dict(claim_pin, path=claim_path.as_posix()))
    seal = read_json(report["source_seal"])["terminal"]
    require(report["case"] == claim["case"] == "T07.sft"
            and report["request"] == claim["request"] and report["label"] == claim["label"]
            and report["claim_SHA256"] == claim_pin["SHA256"]
            and report["framework_commit"] == claim["common_candidate"]["candidate_commit"] == seal["framework"]
            and seal["manifest_SHA256"] == claim["common_candidate"]["manifest_sha256"]
            and seal["SHA256"] == claim["common_candidate"]["source_stage_sha256"]
            and seal["case_count"] == 17, "Z-Image engineering source identity differs")
    baseline = report["accounting"]["baseline"]
    require(report["accounting_columns"] == ["JobIDRaw", "JobName%100", "State", "ExitCode",
                                             "ElapsedRaw", "NodeList", "AllocTRES"]
            and set(report["accounting"]) == {"baseline", "control", "pre", "post"}
            and all(isinstance(row, list) and len(row) == 7 and all(isinstance(field, str) for field in row)
                    for row in report["accounting"].values())
            and len({row[0] for row in report["accounting"].values()}) == 4
            and baseline[0] == report["actual_gpu_step"] and report["same_PCI_verified"] is True
            and report["independent_post_clean"] is True and report["original_controller_passed"] is True
            and report["failure"] is None, "Z-Image engineering run or cleanup is unresolved")
    for role, row in report["accounting"].items():
        node = claim["control_node"] if role == "control" else claim["node"]
        gres = [part for part in row[6].split(",") if part.startswith("gres/")]
        require(row[0].startswith(claim["parent"] + ".") and row[0].rsplit(".", 1)[1].isdigit()
                and row[1:4] == [claim["label"] + "-" + role, "COMPLETED", "0:0"] and row[5] == node
                and gres == (["gres/dcu=1"] if role == "baseline" else []), "Z-Image engineering Slurm step differs")
    repeats = report["repeats"]
    require(set(repeats) == {"repeat01", "repeat02", "repeat03"}
            and report["qualified_repeats"] == report["planned_repeats"] == report["independent_process_count"] == 3
            and report["completed_optimizer_updates"] == claim["actual_optimizer_updates_planned"] == 75
            and report["rate_units"] == "updates_per_second"
            and claim["checkpoint_ordinals"] == [10, 20, 25], "Z-Image engineering repeat scope differs")
    rates = []
    for row in repeats.values():
        clock = row["clock"]
        seconds = complete_window(clock)
        start, end = clock["start_ns"], clock["end_ns"]
        require(row["passed"] is True and clock["clock"] == "monotonic_ns" and clock["warmup_updates"] == 5
                and clock["includes_input_fetch_and_durable_checkpoint_io"] is True
                and clock["profile"] is False and clock["first_update_capture"] is False
                and clock["checkpoint_ordinals"] == [10, 20, 25]
                and type(start) is int and type(end) is int and end > start >= 0
                and math.isclose(seconds, (end - start) / 10**9, rel_tol=1e-12)
                and math.isclose(clock["updates_per_second"], 20 / seconds, rel_tol=1e-12)
                and [cp["global_step"] for cp in row["checkpoints"]] == [10, 20, 25]
                and all(type(cp["bytes"]) is int and cp["bytes"] > 0 and sha(cp["SHA256"])
                        for cp in row["checkpoints"]), "Z-Image engineering clock or checkpoints differ")
        rates.append(20 / seconds)
    hygon_median = median(rates)
    ratio = hygon_median / a100_median
    require(report["A100_timing_evidence"]["path"] == reference["path"]
            and report["A100_timing_evidence"]["SHA256"] == report["a100_source_SHA256"] == reference["SHA256"]
            and report["a100_median"] == a100_median
            and math.isclose(report["hygon_median"], hygon_median, rel_tol=1e-12)
            and math.isclose(report["engineering_rate_ratio"], ratio, rel_tol=1e-12)
            and report["formal_A100_numerics"] is report["heldout_effect_verified"] is report["stage_three_accepted"] is False,
            "Z-Image engineering summary or qualification differs")
    return {"actual_gpu_step": baseline[0], "framework_commit": report["framework_commit"], "repeats": len(rates),
            "hygon_median_updates_per_second": hygon_median, "a100_median_updates_per_second": a100_median,
            "engineering_rate_ratio": ratio, "stage_three_accepted": False}


def verify(root, *, private_roots=()):
    status = json.loads((root / "status/bw1000/latest.json").read_bytes())
    index = status["unified_training_acceptance_matrix"]

    def pinned(path, expected, max_bytes=None):
        file = (root / path).resolve(strict=True)
        require(file.is_relative_to(root), f"evidence outside repository: {path}")
        with file.open("rb") as stream:
            raw = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        require(max_bytes is None or len(raw) <= max_bytes, f"evidence exceeds byte limit: {path}")
        require(hashlib.sha256(raw).hexdigest() == expected, f"hash mismatch: {path}")
        return raw

    matrix = json.loads(pinned(index["matrix"], index["matrix_SHA256"]))
    cases = matrix["cases"]
    ids = [case["case"] for case in cases]
    require(len(ids) == len(set(ids)) == matrix["case_count"] == index["cases"] == 17,
            "unexpected training case denominator")
    require(matrix["framework_commit"] == index["framework_commit"], "framework pin differs")
    require(all(type(case["stage3_accepted"]) is bool for case in cases), "invalid acceptance flag")
    require(sum(case["stage3_accepted"] for case in cases) == index["stage3_passed"],
            "stage-three count differs")
    require(matrix["stage3"] == {"minimum_A100_throughput_ratio": 0.9,
                                 "maximum_relative_effect_regression": 0.05},
            "unexpected stage-three thresholds")
    ledger = matrix["node_ledger"]
    groups = json.loads(pinned(ledger["path"], ledger["SHA256"]))["groups"]
    require({group for group in groups if group.startswith("T")} == TRAINING_GROUPS,
            "training task groups differ")
    assigned = [case for group in sorted(TRAINING_GROUPS) for case in groups[group]["cases"]]
    require(ids == assigned, "matrix and resource ledger training case identities differ")
    pins = dict(matrix["evidence_SHA256"])
    verify_sample_artifacts = sample_artifact_verifier(root, private_roots)

    def pinned_attachment(item, max_bytes=None):
        path, digest = item["path"], item["SHA256"]
        require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
        return pinned(path, digest, max_bytes)

    def verify_formal_attachment(item):
        # Attachments are small summaries or manifests for private experiment receipts.
        return json.loads(pinned_attachment(item))

    registry = matrix.get("live_registry_evidence")
    observed = None
    verified_registry = None
    require(registry == index.get("live_registry_evidence"), "compute registry evidence index differs")
    if registry is not None:
        path, digest = registry["path"], registry["SHA256"]
        require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
        report = json.loads(pinned(path, digest))
        observed = report["result"]
        case_bindings = {case["case"]: [case["launch_SHA256"], case["original_cards"]] for case in cases}
        require(observed["registry_cases"] == ids and observed["entry_registry_matches_matrix"] is True
                and observed["registry_order_matches"] is True and not observed["mismatched_launch_or_card_bindings"]
                and report["verified_registry_bindings"] == case_bindings
                and report["matrix_framework_commit"] == registry["recipe_registration_framework_commit"]
                and registry["deployed_framework_commit"] == matrix["framework_commit"]
                and observed["explicit_GPU_allocation"] == observed["training_updates"] == 0,
                "compute registry evidence differs from the indexed cases or source")
        require(len(report["rows"]) == 1 and report["rows"][0][0] == observed["step"]
                and report["rows"][0][2:4] == ["COMPLETED", "0:0"],
                "compute registry evidence has no successful terminal step")
        verified_registry = verify_registry_source(
            matrix["framework_commit"], matrix["entry"], registry, observed,
            verify_formal_attachment(registry["source_manifest"]),
            pinned_attachment(registry["source_entry"]),
        )

    vlm_schedules = {}
    for case in cases:
        if "recovery_plan" in case:
            recovery_plan(case, verify_formal_attachment)
        if "numerics_plan" in case:
            numerics_plan(case, verify_formal_attachment)
        if case["case"] in ZERO_LR_VLM_REQUESTS and "capture_value_evidence" in case:
            vlm_schedules[case["case"]] = original_vlm_update_schedule(case, verify_formal_attachment)
        for evidence in case.get("supplemental_evidence", {}).values():
            path, sha = evidence["path"], evidence["SHA256"]
            require(pins.setdefault(path, sha) == sha, f"conflicting evidence hashes: {path}")
        for field in ("reference_evidence", "hygon_evidence"):
            path = case[field]
            require(path is None or path in pins, f"{case['case']}: unpinned {field}")
        formal = case.get("formal_acceptance")
        if case["stage3_accepted"]:
            require(isinstance(formal, dict), f"{case['case']}: missing formal acceptance evidence")
        if formal is not None:
            gate_paths = set()
            gate_results = {}
            for gate in ("function", "numerics", "recovery"):
                item = formal["gate_evidence"][gate]
                path, digest = item["path"], item["SHA256"]
                require(path not in gate_paths, f"{case['case']}: duplicate gate evidence")
                gate_paths.add(path)
                require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
                result = json.loads(pinned(path, digest))
                require(result["case"] == case["case"] and result["framework_commit"] == matrix["framework_commit"]
                        and result["gate"] == gate and result["passed"] is True,
                        f"{case['case']}: wrong or failed {gate} gate")
                gate_results[gate] = result
                if gate == "recovery":
                    verify_recovery(case, matrix["framework_commit"], result, verify_formal_attachment)
                elif gate == "numerics":
                    verify_numerics(case, matrix["framework_commit"], result, verify_formal_attachment)
            verify_function(case, matrix["framework_commit"], matrix.get("entry"), observed,
                            gate_results["function"], gate_results["numerics"]["Hygon"], verify_formal_attachment)
            item = formal["metrics_evidence"]
            path, digest = item["path"], item["SHA256"]
            require(path not in gate_paths, f"{case['case']}: duplicate metrics evidence")
            require(pins.setdefault(path, digest) == digest, f"conflicting evidence hashes: {path}")
            metrics = stage_three(case, matrix["framework_commit"], json.loads(pinned(path, digest)),
                                  verify_formal_attachment, read_binary=pinned_attachment,
                                  verify_artifacts=verify_sample_artifacts)
            require(metrics["passed"] is case["stage3_accepted"],
                    f"{case['case']}: formal measurements and acceptance flag disagree")

    a100_windows = {}
    timing_audits = matrix['original_A100_engineering_timing_audits']
    require(set(timing_audits) == {'generation', 'vlm', 'vggt'}, 'unexpected original timing audit bindings')
    case_by_id = {case['case']: case for case in cases}
    verified_zimage_recovery = None
    zimage_refs = case_by_id['T07.sft'].get('supplemental_evidence', {})
    zimage_source = zimage_refs.get('registered_native_HIP_recovery_source_scope')
    if zimage_source is not None:
        zimage_terminal = zimage_refs['registered_native_HIP_continuous25_and_fresh_resume20_exact']
        source_root = PurePosixPath(zimage_source['path']).parent / 'framework-delta'
        verified_zimage_recovery = verify_registered_zimage_recovery(
            verify_formal_attachment(zimage_terminal), verify_formal_attachment(zimage_source),
            source_root, ids, pinned_attachment,
        )
    overlay_index = case_by_id["T09.dp2"].get("qualified_framework_overlay")
    verified_overlay = None
    if overlay_index is not None:
        overlay = verify_formal_attachment(overlay_index)
        require(overlay["case"] == "T09.dp2" and overlay["base_framework_commit"] == matrix["framework_commit"],
                "verified overlay refers to a different case or base framework")
        verified_overlay = verify_overlay_source(overlay, pinned)
    portable = case_by_id["T09.dp2"].get("portable_framework_bundle")
    if portable is not None:
        require(verified_overlay is not None, "portable source bundle must identify a verified overlay")
        result = verify_formal_attachment(portable)
        imported = result["independent_fresh_import"]
        staged = result["private_target_bundle_reopen"]["receiver"]
        require(imported["verified_by_fresh_bundle_import"] is True
                and imported["source_files"] == verified_overlay["framework_files"]
                and imported["git_tree"] == staged["git_tree"] == verified_overlay["framework_tree"]
                and imported["candidate_commit"] == staged["git_commit"]
                and imported["bundle_sha256"] == staged["bundle_sha256"]
                and imported["bundle_bytes"] == staged["bytes"]
                and imported["registered_cases"] == ids
                and imported["new_case_acceptance"] == imported["executed_GPU_steps"] == 0,
                "portable source bundle, imported registry and verified overlay differ")
        verified_overlay["portable_candidate_commit"] = imported["candidate_commit"]
    verified_integrated_candidate = None
    integrated = status.get("unified_training_portable_candidate")
    if integrated is not None:
        result = verify_formal_attachment(integrated["evidence"])
        local = result["independent_local_import"]
        remote = result["target_materialized_source"]
        bundle = result["target_stored_bundle"]
        help_check = result["target_registered_diagnostic_help"]
        require(verified_overlay is not None
                and local["parent_commit"] == verified_overlay.get("portable_candidate_commit")
                and local["candidate_commit"] == remote["candidate_commit"] == integrated["candidate_commit"]
                and local["tree"] == remote["git_tree"] == integrated["tree"]
                and local["files"] == remote["target_source_files"] == verified_overlay["framework_files"] + 1
                and local["registry"] == remote["target_registry"] == ids
                and local["bundle_sha256"] == bundle["bundle_sha256"]
                and local["bundle_bytes"] == bundle["bytes"]
                and local["fresh_import_verified"] and remote["source_checkout_verified"]
                and remote["registry_executed_in_place"]
                and help_check["passed"] and help_check["registered_case"] == "T07.sft"
                and result["executed_hardware_training_steps_on_candidate"] == 0
                and integrated["device_acceptance"] is False,
                "integrated candidate, target source and original DP2 parent differ")
        changes = local["replaced_members"]
        introduced = "generation/src/gencuda/zimage_fixed_reference.py"
        require(set(changes) == {
            introduced, "generation/src/gencuda/zimage_first_update.py",
            "generation/src/gencuda/zimage_native_record.py", "tools/hygon_training_recipe.py",
            "tools/hygon_zimage_native_record.py",
        } and changes["tools/hygon_training_recipe.py"]["SHA256"] == help_check["entry_sha256"],
                "integrated candidate changed an unexpected provider")
        for name, pin in changes.items():
            overlay_path = ("training-zimage-coco20-recorded-reference-v2" if name == introduced
                            else "training-common-zimage-fixed-first-overlay-v1")
            raw = pinned_attachment({"path": f"code/bw1000/{overlay_path}/framework-delta/{name}",
                                     "SHA256": pin["SHA256"]}, 128 << 10)
            require(len(raw) == pin["bytes"], "integrated source member length differs")
        verified_integrated_candidate = {
            "candidate_commit": local["candidate_commit"], "tree": local["tree"], "source_files": local["files"],
            "case_count": len(ids), "target_registry_verified": True, "device_acceptance": False,
        }
    generation_timing = verify_formal_attachment(timing_audits['generation'])
    a100_windows.update(check_generation(case_by_id, generation_timing))
    zimage_windows = zimage_refs.get("registered_native_HIP_three_complete_engineering_windows")
    verified_zimage_windows = None
    if zimage_windows is not None:
        verified_zimage_windows = verify_zimage_engineering_windows(
            verify_formal_attachment(zimage_windows), timing_audits["generation"],
            a100_windows["T07.sft"], verify_formal_attachment,
        )
    no_boundary_description = [name for name, audit in generation_timing['cases'].items()
                               if audit['repeats'][0]['reported_timing_boundary'] is None]
    recovered_boundaries = verify_formal_attachment(matrix['original_A100_recovered_RL_clock_boundaries'])
    require(set(recovered_boundaries['cases']) == set(no_boundary_description)
            and recovered_boundaries['original_result_JSON_files'] == 6
            and recovered_boundaries['formal_acceptance'] is False,
            'recovered A100 clock boundary coverage differs')
    for name, audit in recovered_boundaries['cases'].items():
        original_runs = generation_timing['cases'][name]['repeats']
        recovered_runs = audit['runs']
        require(recovered_boundaries['original_manifest_SHA256'] == generation_timing['cases'][name]['source_manifest_SHA256']
                and len(recovered_runs) == len(original_runs) == 3
                and all(new['repeat'] == old['repeat']
                        and new['archived_result_SHA256'] == old['original_result_SHA256']
                        and new['archived_step_log_SHA256'] == old['raw_step_log_SHA256']
                        and new['complete_window_seconds'] == old['complete_window_seconds']
                        for new, old in zip(recovered_runs, original_runs)),
                f'{name}: recovered A100 clock boundary refers to other runs')
    a100_windows.update(check_vlm(case_by_id, verify_formal_attachment(timing_audits['vlm'])))
    a100_windows.update(check_vggt(case_by_id[VGGT_CASE], verify_formal_attachment(timing_audits['vggt'])))
    sage_reference_path = next(case['reference_evidence'] for case in cases if case['case'] == 'T09.sage-grpo')
    require(sage_reference_path in pins, 'missing original Hunyuan A100 timing reference')
    hunyuan_a100 = json.loads(pinned(sage_reference_path, pins[sage_reference_path]))
    sage_step_index = status['t09_hunyuan_SAGE_A100_original_step_audit']
    sage_steps = verify_formal_attachment({'path': sage_step_index['evidence'], 'SHA256': sage_step_index['evidence_SHA256']})
    for case in cases:
        if case['case'] in ('T09.sft', 'T09.sage-grpo'):
            a100_windows[case['case']] = verify_hunyuan_a100_recorded_updates(case, hunyuan_a100, sage_steps)
        if "a100_rank_window_audit" in case:
            path = case["a100_rank_window_audit"]
            require(path == case["reference_evidence"] and path in pins,
                    "missing original DP2 rank-window audit")
            a100_windows[case["case"]] = verify_hunyuan_dp2_a100_windows(
                case, json.loads(pinned(path, pins[path])), verify_formal_attachment)

    for path, sha in pins.items():
        pinned(path, sha)
    require(set(a100_windows) == set(ids), 'original engineering timing does not cover all 17 cases')
    human = pinned(index["human_readable"], index["human_readable_SHA256"]).decode("utf-8")
    markdown_ids = [line.split("|")[1].strip() for line in human.splitlines() if line.startswith("| T")
                    and line.split("|")[1].strip() != "用例"]
    require(markdown_ids == ids, "Markdown and JSON training case order differs")
    return {"cases": len(ids), "stage3_passed": index["stage3_passed"], "evidence_files": len(pins),
            "verified_deployed_registry_source": verified_registry,
            "verified_DP2_framework_overlay_source": verified_overlay,
            "verified_integrated_portable_candidate": verified_integrated_candidate,
            "verified_zimage_same_platform_engineering_recovery": verified_zimage_recovery,
            "verified_zimage_engineering_windows": verified_zimage_windows,
            "verified_A100_engineering_medians_updates_per_second": a100_windows,
            "A100_engineering_cases_without_archived_timing_boundary_description": no_boundary_description,
            "A100_engineering_clock_boundaries_verified_from_original_source": sorted(recovered_boundaries['cases']),
            "verified_A100_VLM_optimizer_step2_learning_rates_by_rank": vlm_schedules}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--private-root", action="append", type=Path, default=[],
                        help="additional local or mounted experiment directory holding original samples")
    args = parser.parse_args()
    print(json.dumps(verify(args.repo.resolve(strict=True), private_roots=args.private_root)))


if __name__ == "__main__":
    main()
