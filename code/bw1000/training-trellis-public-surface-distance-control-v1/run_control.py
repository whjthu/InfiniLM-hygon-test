"""Measure an independently sampled surface diagnostic on the frozen BoomBox export."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import time

import numpy as np

from surface_distance import metrics


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = REPO / "local-only/training-trellis-public-mesh-quality-input-v1-original"
CONTROL = REPO / "local-only/training-trellis-public-BoomBox-native-GLB-registration-control-v1"
OUT = REPO / "local-only/training-trellis-public-surface-distance-control-v1-original"
BRIDGE = REPO / "tools/prepare_trellis_export_prediction.py"
BRIDGE_SHA = "8fe604ed390a226d21398f7f751c64a0151569f190a5066b6e34bd5be4c2c427"
GLB_SHA = "c8da6defeaee8fc0ea94939c51e72ec1aff469961f2b7f9ffd58e4ebe2ca0957"
PLAN_SHA = "a10f8adea636b8786637abc43d43ac559c704852ab775b577719705fa0f51428"
REGISTRATION_SHA = "6e67b6d15ae43dc1ed40f354d5d4ec3f479330d661e458103bf4582fa7ee943c"
CONTROL_OUTCOME = REPO / "results/bw1000/2026-11-21/training-trellis-boombox-frozen-GLB-scoring-route-control-local.json"
CONTROL_OUTCOME_SHA = "dca4adb3a1b53494c44e1fc1b0043232dad94f19643c1b04e59e6b1c77cc8bd6"
SHIFT = (0.025, 0.0, 0.0)


def calculate():
    started = time.monotonic()
    if sha256(BRIDGE.read_bytes()).hexdigest() != BRIDGE_SHA:
        raise ValueError("original geometry bridge source differs")
    source = runpy.run_path(str(BRIDGE))
    read = source["exact"]
    score, old_metric = source["quality_modules"]()
    registry = json.loads(read(source["QUALITY"] / "REFERENCE_INPUTS.json", 16384, source["REGISTRY_SHA"]))
    frozen = registry["objects"]["BoomBox"]
    reference = score.reference(INPUTS, frozen)
    calibration = json.loads(read(CONTROL / "cli-fixture/registration.json", 410, REGISTRATION_SHA))
    if calibration["object"] != "BoomBox" or calibration["reference_SHA256"] != frozen["file"]["SHA256"]:
        raise ValueError("original registration identity differs")
    source_GL = read(CONTROL / "control.glb", 653576, GLB_SHA)
    reader = source["original_export_functions"]()["read_glb"]
    arrays, _ = reader(source["PinnedGLB"](source_GL), dict(max_bytes=653576, max_faces=6036, texture_size=32))
    registered = source["normalize"](arrays[0], reference, calibration["transform"])
    pred_faces = arrays[3].reshape(-1, 3)
    if len(pred_faces) != frozen["faces"]:
        raise ValueError("original registered face count differs")
    plan = json.loads(read(CONTROL / "control-predictions.json", 534, PLAN_SHA))
    point_pin = plan["objects"]["BoomBox"]["points"]
    points_raw = read(CONTROL / "points.npz", point_pin["bytes"], point_pin["SHA256"])
    predicted = score.arrays(points_raw, ("points",))["points"]
    ref_points = reference["reference_points"]
    old = old_metric.point_metrics(predicted, ref_points)
    prior = json.loads(read(CONTROL_OUTCOME, 3285, CONTROL_OUTCOME_SHA))
    if old != prior["geometry_control"]["registered_score_control"]:
        raise ValueError("independent point baseline no longer matches its original result")
    ref_mesh = (reference["vertices"], reference["faces"], ref_points)
    registered_result = metrics(ref_mesh, (registered, pred_faces, predicted))
    # This fixed shift tests geometric sensitivity and uses the same sampled points.
    shift = np.asarray(SHIFT, dtype=np.float64)
    shifted = metrics(ref_mesh, (registered + shift, pred_faces, predicted + shift))
    if (registered_result["sampled_surface_F_at_1percent"] != 1.0
            or max(registered_result[k] for k in ("max_predicted_distance", "max_reference_distance")) > 1e-6
            or shifted["sampled_surface_F_at_1percent"] >= registered_result["sampled_surface_F_at_1percent"]):
        raise ValueError("surface diagnostic integration or geometry control differs")
    return dict(case="T10.domain-public-mesh-quality", object="BoomBox", split="dev", blind_objects=0,
                source=dict(reference=frozen["file"], GLB_SHA256=GLB_SHA, independent_points_SHA256=point_pin["SHA256"],
                            calibration_SHA256=REGISTRATION_SHA, prior_point_result_SHA256=CONTROL_OUTCOME_SHA),
                fixed_frame=registry["frame"], faces_per_mesh=len(pred_faces), old_independent_point_metrics=old,
                actual_registered_surface_control=registered_result, translated_surface_control=shifted,
                fixed_translation_in_reference_extents=SHIFT, measured_seconds=time.monotonic()-started,
                actual_model_predictions=0, actual_GPU_steps=0, actual_optimizer_updates=0,
                actual_Slurm_submissions=0, formal_T10_qualified=False, training_READY_added=0)


def main():
    OUT.mkdir(mode=0o700)
    report = calculate()
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with (OUT / "result.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(report, private_result_SHA256=sha256(raw).hexdigest()), allow_nan=False))


if __name__ == "__main__":
    main()
