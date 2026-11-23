"""Score paired TRELLIS predictions against the four frozen public reference objects."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[2] / "results/bw1000/2026-11-24/training-trellis-khronos-four-public-heldout-geometry-local.json"
RESULT_SHA = "0fa69cf9b6dea974486f6765cb7d954b5840f6863755465c9dad09a0156c6a91"
PREVIOUS = HERE.with_name("training-trellis-public-mesh-quality-input-v1")
METRIC_SHA = "8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a"
HEX = re.compile("[0-9a-f]{64}")
NAMES = ("Avocado", "Lantern", "FlightHelmet", "ToyCar")
ARMS = ("initial", "hygon", "a100")
FRAME = "reference-normalized-glTF-Yup-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, maximum, expected):
    with path.open("rb") as file:
        data = file.read(maximum + 1)
    require(0 < len(data) <= maximum and sha256(data).hexdigest() == expected, "Input size or SHA differs")
    return data


def points(data, key, members):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(members) and {item.filename for item in entries} == members,
                "NPZ membership differs")
        item = archive.getinfo(key + ".npy")
        require(item.file_size <= 32768 and not item.flag_bits & 1, "Point buffer exceeds its byte bound")
        buffer = io.BytesIO(archive.read(item))
    version = np.lib.format.read_magic(buffer)
    require(version == (1, 0), "Unexpected NPY format")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer, max_header_size=4096)
    require(shape == (2048, 3) and not fortran and dtype == np.dtype("<f4")
            and len(buffer.getbuffer()) == buffer.tell() + 2048 * 3 * 4, "Full contiguous float32 point set required")
    data = np.frombuffer(buffer.read(), dtype="<f4").reshape(shape)
    require(np.isfinite(data).all(), "Nonfinite point set")
    return data


def digest_fields(row, names):
    require(isinstance(row, dict) and all(isinstance(row.get(key), str) and HEX.fullmatch(row[key]) for key in names),
            "Missing input identity")


def score(manifest, reference_root, prediction_root, metric):
    require(manifest["schema"] == "TRELLIS_PUBLIC_HELDOUT4_PAIRED_PREDICTIONS_V1"
            and manifest["frame"] == FRAME and not set(manifest["objects"]) - set(NAMES)
            and set(manifest["checkpoints"]) == set(ARMS), "Scoring roster or coordinate frame differs")
    checkpoints = manifest["checkpoints"]
    for arm in ARMS:
        digest_fields(checkpoints[arm], ("SHA256",))
        step = checkpoints[arm]["effective_updates"]
        require(type(step) is int and step >= 0, "Missing checkpoint update count")
    require(checkpoints["initial"]["effective_updates"] == 0
            < checkpoints["hygon"]["effective_updates"] == checkpoints["a100"]["effective_updates"],
            "Checkpoint update counts differ")
    references = json.loads(pinned(RESULT, 32768, RESULT_SHA))["objects"]
    report = {"planned_objects": 4, "planned_predictions": 12, "scored_predictions": 0,
              "frame": FRAME, "reference_result_SHA256": RESULT_SHA, "objects": {},
              "prediction_provenance_independently_validated": False, "formal_T10_accepted": False}
    prediction_members = {"points.npy"}
    reference_members = {"reference_points.npy", "vertices.npy", "faces.npy", "center_m.npy", "longest_extent_m.npy"}
    for name in NAMES:
        rows = {}
        report["objects"][name] = rows
        source = references[name]["artifact"]
        reference_error = None
        try:
            r = points(pinned(reference_root / f"{name}-reference.npz", source["bytes"], source["SHA256"]),
                       "reference_points", reference_members)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
            reference_error = error
        for arm in ARMS:
            row = {"passed": False, "Chamfer_L1": None, "Fscore_at_1percent": None}
            rows[arm] = row
            try:
                if reference_error is not None:
                    raise reference_error
                item = manifest["objects"][name]
                digest_fields(item, ("conditioning_SHA256", "registration_SHA256"))
                require(type(item["seed"]) is int and 0 <= item["seed"] < 2**64, "Missing frozen inference seed")
                pred = item["predictions"][arm]
                digest_fields(pred, ("checkpoint_SHA256", "conditioning_SHA256", "registration_SHA256",
                                     "receipt_SHA256", "points_SHA256"))
                require(pred["frame"] == FRAME and type(pred["seed"]) is int and pred["seed"] == item["seed"]
                        and pred["checkpoint_SHA256"] == checkpoints[arm]["SHA256"]
                        and all(pred[k] == item[k] for k in ("conditioning_SHA256", "registration_SHA256")),
                        "Prediction differs from the paired condition, registration, seed or checkpoint")
                path = prediction_root / f"{name}-{arm}.npz"
                q = points(pinned(path, 32768, pred["points_SHA256"]), "points", prediction_members)
                result = metric(q, r)
                require(result["passed"], "Original metric rejected prediction")
                row.update(result, prediction_points_SHA256=pred["points_SHA256"],
                           predictor_receipt_SHA256=pred["receipt_SHA256"])
                report["scored_predictions"] += 1
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                row["error"] = {"type": type(error).__name__, "message": str(error)[:160]}
    report["missing_or_failed_predictions"] = report["planned_predictions"] - report["scored_predictions"]
    return report


def load_metric():
    path = PREVIOUS / "mesh_quality.py"
    raw = pinned(path, 32768, METRIC_SHA)
    scope = {}
    exec(compile(raw, str(path), "exec"), scope)
    return scope["point_metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "prediction-root", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-SHA256", required=True)
    args = parser.parse_args()
    raw = pinned(args.manifest, 32768, args.manifest_SHA256)
    result = score(json.loads(raw), args.reference_root, args.prediction_root, load_metric())
    result["prediction_manifest_SHA256"] = args.manifest_SHA256
    with args.output.open("x", encoding="utf8") as file:
        json.dump(result, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({k: result[k] for k in ("planned_predictions", "scored_predictions", "missing_or_failed_predictions")}))
    return int(result["missing_or_failed_predictions"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
