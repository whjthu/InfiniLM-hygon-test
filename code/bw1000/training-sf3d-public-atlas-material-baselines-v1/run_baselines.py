"""Measure two fixed comparators on the existing real SF3D atlas development object."""

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import tarfile
import time

import numpy as np
from PIL import Image

from material_mae import FIELDS, score, vector


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "code/bw1000"
ARCHIVE = BASE / "training-framework-bd24ec3-v1/FRAMEWORK.tar.gz"
ARCHIVE_PIN = (738802, "85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f")
BACKEND = "framework/generation/src/gencuda/backends/sf3d_training.py"
BACKEND_PIN = (58458, "dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802")
JOIN = BASE / "training-sf3d-public-atlas3-resume-target-join-v1/target_join3.py"
JOIN_PIN = (2983, "151af51e11692127e00ca6521901ed0c5503c48ecf8959d7ed5f654fb327f4f3")
PBR = BASE / "training-sf3d-public-pbr-atlas2-v1/pbr.py"
PBR_MANIFEST = BASE / "training-sf3d-public-pbr-atlas2-v1/SOURCE_FILES.json"
PBR_MANIFEST_PIN = (986, "9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f")
TRAIN = ("WaterBottle", "Corset")
DEV = ("BoomBox",)
OUT = REPO / "local-only/training-sf3d-public-atlas-material-baselines-v1-original"


def exact_bytes(path, pin):
    with path.open("rb") as handle:
        raw = handle.read(pin[0] + 1)
    if (len(raw), hashlib.sha256(raw).hexdigest()) != pin:
        raise ValueError("frozen source differs: " + path.name)
    return raw


def foreground_function():
    archive_bytes = exact_bytes(ARCHIVE, ARCHIVE_PIN)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        source = archive.extractfile(BACKEND).read(BACKEND_PIN[0] + 1)
    if (len(source), hashlib.sha256(source).hexdigest()) != BACKEND_PIN:
        raise ValueError("SF3D native backend differs")
    function, = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "foreground_material_targets"]
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), BACKEND, "exec"), scope)
    return scope[function.name]


def normalized(raw):
    return vector(dict(albedo=[float(x) for x in raw["albedo"]],
                       roughness=[float(raw["roughness"])], metallic=[float(raw["metallic"])]))


def calculate():
    started = time.monotonic()
    join_raw = exact_bytes(JOIN, JOIN_PIN)
    join = {"__file__": str(JOIN), "__name__": "frozen_atlas_target_join"}
    exec(compile(join_raw, str(JOIN), "exec"), join)
    frozen = join["binding"]()
    if tuple(row["group"] for row in frozen["train"]) != TRAIN or tuple(
            row["group"] for row in frozen["development"]) != DEV or frozen["blind"]:
        raise ValueError("frozen atlas splits differ")
    training = [normalized(join["targets_for_group"](name, purpose="train")[0]) for name in TRAIN]
    mean = {name: tuple(math.fsum(row[name][channel] for row in training) / len(TRAIN)
                        for channel in range(width)) for name, width in FIELDS.items()}
    # Both policies are fixed before reading the held-out target.
    image = frozen["development"][0]
    pin = image["input_pin"]
    raw = exact_bytes(join["local_path"](image["input"]), (pin["bytes"], pin["SHA256"]))
    with Image.open(io.BytesIO(raw)) as decoded:
        if decoded.format != "PNG" or decoded.mode != "RGBA" or max(decoded.size) > 4096:
            raise ValueError("frozen RGBA atlas payload differs")
        size = list(decoded.size)
        heuristic = foreground_function()(np.asarray(decoded))
    manifest = json.loads(exact_bytes(PBR_MANIFEST, PBR_MANIFEST_PIN))
    transfer_pin = manifest["files"][PBR.name]
    exact_bytes(PBR, (transfer_pin["bytes"], transfer_pin["SHA256"]))
    to_linear = runpy.run_path(str(PBR))["srgb_linear"]
    heuristic["albedo"] = to_linear(np.asarray(heuristic["albedo"], dtype=np.float64))
    predictors = {
        "training_object_mean": {DEV[0]: mean},
        "native_RGB_heuristic_mean_to_linear": {DEV[0]: normalized(heuristic)},
    }
    target, target_pin = join["targets_for_group"](DEV[0], purpose="eval-dev")
    reference = {DEV[0]: normalized(target)}
    results = {name: score(DEV, reference, rows) for name, rows in predictors.items()}
    return dict(recipe=join["RECIPE"], binding=join["PIN"], backend_SHA256=BACKEND_PIN[1],
                training_objects=list(TRAIN), development_objects=list(DEV), blind_objects=0,
                development_image_dimensions=size, development_target_binding=target_pin,
                baselines=results, exact_material_vectors=dict(reference=reference, predictions=predictors),
                wall_seconds=time.monotonic() - started, numpy_version=np.__version__,
                actual_model_outputs=0, actual_optimizer_updates=0, actual_GPU_steps=0, added_training_READY=0)


def main():
    OUT.mkdir(mode=0o700)
    result = calculate()
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    with (OUT / "result-private.json").open("xb") as handle:
        handle.write(raw)
    public = {key: value for key, value in result.items() if key != "exact_material_vectors"}
    public["private_result"] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
    print(json.dumps(public, allow_nan=False))


if __name__ == "__main__":
    main()
