"""Render the one-score-object Kimi input-retention candidate from frozen r461 bytes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / "output-residual-wave-compile-CPU-v1"
ORIGINAL = HERE.parent / "output-residual-fused-scores-v1"
BASE_PIN = "e55041e1f3b5b48c776ce0a3983fce550559dbdc0daa4a50e4dacd4319b13efe"
DIAGNOSTIC = "results/bw1000/2026-11-20/Kimi461-original337-residual-stage-diagnostic-terminal.json"
DIAGNOSTIC_PIN = "1d656f3da9182fb90a7c4e5a2c4aec98db846a769f86e7d0ac2b06a649692075"
LABEL = "Kimi461-output-residual-retained28-score-gfx936-CPU-v1"
ARM = "retained28"


def sha(raw):
    return sha256(raw).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("frozen original differs: " + path.name)
    return raw


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("frozen source transform anchor differs")
    return text.replace(before, after)


def materialize(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    pins = json.loads(checked(BASE / "SOURCE_FILES.json", BASE_PIN))["files_SHA256"]
    plan = json.loads(checked(BASE / "BUILD_PLAN.json", pins["BUILD_PLAN.json"]))
    worker = checked(BASE / "compile_once.py", pins["compile_once.py"]).decode()
    for name, pin in plan["repo_source_SHA256"].items():
        checked(ROOT / name, pin)
    diagnostic = json.loads(checked(ROOT / DIAGNOSTIC, DIAGNOSTIC_PIN))
    body = (HERE / "retained_values.inc").read_text(encoding="utf-8")
    old_label = plan["label"]
    old_rendered = plan["rendered_files_SHA256"]["full-tree"]
    old_id = plan["variant_ids"]["full-tree"]
    variant = dict(parameter="retain_mapped_bf16_values_across_rms_and_dot", private_bf16_words_per_lane=28,
                   baseline_variant_id=old_id, fixed_reduction="full-tree", fixed_threads=256,
                   fixed_blocks=4608, shape=[512, 9, 7168])
    identity = sha(json.dumps(variant, sort_keys=True, separators=(",", ":")).encode())
    with TemporaryDirectory(dir=output.parent, prefix="kimi-retained-score-") as folder:
        stage = Path(folder) / "source"
        stage.mkdir(mode=0o700)
        case = stage / ARM
        produced = runpy.run_path(str(ORIGINAL / "render.py"))["materialize"](case)
        if produced != old_rendered:
            raise ValueError("full-tree source rendering differs")
        kernel = case / "fused_scores.cu"
        source = kernel.read_bytes().decode()
        patched = once(source, "__global__ void scores_kernel", body + "\n__global__ void scores_kernel")
        patched = once(patched, "    scratch[lane] = square_lane(row, lane);",
                       "    RetainedValues held;\n    scratch[lane] = retain_and_square(row, lane, held);")
        patched = once(patched, "dot_lane(row, norm, projection, inv_rms, lane)",
                       "dot_retained(held, norm, projection, inv_rms, lane)")
        kernel.write_bytes(patched.encode())
        rendered = {p.name: sha(p.read_bytes()) for p in case.iterdir()}
        if set(rendered) != set(old_rendered) or {
                name for name in rendered if rendered[name] != old_rendered[name]} != {"fused_scores.cu"}:
            raise ValueError("candidate changed files outside the frozen score translation unit")
        plan.update(label=LABEL, variant_ids={ARM: identity}, variant=variant, artifact_expected={
            ARM: ARM + "/fused_scores.o"}, rendered_files_SHA256={ARM: rendered}, parameter=variant["parameter"],
            frozen_compiler_template_SHA256=pins["compile_once.py"], inherited_wave_package_SHA256=BASE_PIN,
            synchronization_source_count={ARM: plan["synchronization_source_count"]["full-tree"]},
            original_fixed_geometry_provider="1103606.340", cold_diagnostic=dict(
                source=DIAGNOSTIC, SHA256=DIAGNOSTIC_PIN, seconds=diagnostic["stage_seconds"]),
            input_values_logical_bytes_per_removed_pass=512 * 9 * 7168 * 2,
            original_gfx936_compiler_retains_across_barrier=None,
            logical_private_bf16_bytes_per_block=256 * 28 * 2, measured_gfx936_spill_bytes=None,
            target_validation=dict(
                compile_evidence=["both_gfx936_objects", "VGPR_count", "scratch_bytes_per_thread",
                                  "input_loads_before_and_after_barriers", "spill_instructions"],
                require_eliminated_actual_input_reload=True, require_no_added_scratch_or_spill=True,
                full_internal_BF16_score_exact_and_full_output_exact=True,
                warmed_timing=dict(pairs=3, order=["AB", "BA", "AB"], warmups_per_arm=2, timed_per_arm=8,
                                   report_each_pair_and_median=True,
                                   initial_screen="candidate synchronized dispatch median < baseline in every pair",
                                   count_all_planned_attempts=True)))
        worker = once(worker, f'LABEL = "{old_label}"', f'LABEL = "{LABEL}"')
        worker = once(worker, 'ARMS = ("full-tree", "wave32")', f'ARMS = ("{ARM}",)')
        worker = once(worker, """        require(report["arms"][ARMS[0]]["object_SHA256"] != report["arms"][ARMS[1]]["object_SHA256"],
                "CANDIDATE_OBJECTS_MUST_DIFFER")
""", "")
        (stage / "compile_once.py").write_bytes(worker.encode())
        (stage / "BUILD_PLAN.json").write_bytes((json.dumps(plan, sort_keys=True, indent=2) + "\n").encode())
        files = {p.relative_to(stage).as_posix(): sha(p.read_bytes()) for p in stage.rglob("*") if p.is_file()}
        manifest = (json.dumps({"files_SHA256": files}, sort_keys=True, indent=2) + "\n").encode()
        (stage / "SOURCE_FILES.json").write_bytes(manifest)
        stage.rename(output)
    return dict(target_manifest_SHA256=sha(manifest), candidate_id=identity, source_SHA256=rendered["fused_scores.cu"],
                worker_SHA256=files["compile_once.py"], target_files=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(materialize(parser.parse_args().output), sort_keys=True, indent=2))
