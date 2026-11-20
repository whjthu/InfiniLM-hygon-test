"""Assemble the fixed Freiburg development cohort from verified scene manifests."""

import argparse
import json
from pathlib import Path

from bind_vggt_tum_dev_rgb import CALIBRATION, read_bounded, sha
from bind_vggt_tum_cross_camera_heads import is_sha


COMPLETE = "RGB224_AND_RELATIVE_POSE_BOUND"
FAILED = "RGB224_PRECONDITION_FAILED"


def read_scene(sequence, manifest, expected):
    raw = read_bounded(manifest, 65536)
    if not is_sha(expected) or sha(raw) != expected:
        raise ValueError("scene manifest SHA differs")
    doc = json.loads(raw)
    rows = doc["windows"]
    if (not is_sha(doc["binding_SHA256"])
            or [row["window"] for row in rows] != list(range(1, 13))
            or any(row["sequence"] != sequence for row in rows)):
        raise ValueError("expected the exact twelve ordered scene windows")
    files, original = {}, set()
    for row in rows:
        if row["state"] == FAILED and row.get("reason"):
            continue
        if row["state"] != COMPLETE:
            raise ValueError("unexpected scene preparation state")
        name = f"{sequence}-window-{row['window']:02d}-rgb224.npy"
        if row["rgb_file"] != name:
            raise ValueError("unexpected RGB224 artifact path")
        path = manifest.parent.resolve() / name
        if path.resolve(strict=True) != path:
            raise ValueError("RGB224 artifact path aliases another file")
        payload = read_bounded(path, 310000)
        if (len(payload), sha(payload)) != (row["rgb_file_bytes"], row["rgb_file_SHA256"]):
            raise ValueError("original prepared RGB224 artifact differs")
        pins = [entry["SHA256"] for entry in row["original_frames"]]
        if (len(pins) != 2 or pins[0] == pins[1] or any(not is_sha(digest) for digest in pins)
                or original.intersection(pins)):
            raise ValueError("missing or duplicated original frame identity")
        original.update(pins)
        files[name] = payload
    if not files:
        raise ValueError("each scene must have admitted RGB pairs")
    return doc["binding_SHA256"], rows, files, original


def join(scenes, output):
    if set(scenes) != set(CALIBRATION):
        raise ValueError("both official scene manifests are required")
    rows, files, frame_hashes, evidence = [], {}, set(), {}
    for sequence in CALIBRATION:
        manifest, expected = scenes[sequence]
        binding, windows, artifacts, originals = read_scene(sequence, manifest, expected)
        if originals & frame_hashes:
            raise ValueError("the official scenes share an original RGB frame")
        frame_hashes.update(originals)
        rows.extend(windows)
        files.update(artifacts)
        evidence[sequence] = dict(manifest_SHA256=expected, binding_SHA256=binding, pairs=len(artifacts), failures=12-len(artifacts))

    manifest = dict(windows=rows, scenes=evidence, scope="VGGT_TUM_cross_camera_development")
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    output.mkdir(mode=0o700)
    for name, payload in files.items():
        with (output / name).open("xb") as destination:
            destination.write(payload)
    with (output / "inputs-private.json").open("xb") as destination:
        destination.write(raw)
    return dict(cohort_manifest_SHA256=sha(raw), planned_windows=len(rows), input_pairs=len(files),
                failed_windows=len(rows)-len(files), original_frame_SHA_count=len(frame_hashes), sequences=evidence,
                actual_model_predictions=0, formal_CO3D=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", type=Path, required=True)
    parser.add_argument("--plant-sha", required=True)
    parser.add_argument("--dishes", type=Path, required=True)
    parser.add_argument("--dishes-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = {"freiburg1_plant": (args.plant, args.plant_sha),
              "freiburg2_dishes": (args.dishes, args.dishes_sha)}
    print(json.dumps(join(scenes, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
