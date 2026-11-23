"""Run the frozen framework's TRELLIS image preparation on four held-out PNGs."""

import argparse
import ast
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import PIL
from PIL import Image

from score import NAMES, pinned, require


ARCHIVE = Path(__file__).resolve().parent.with_name("training-framework-bd24ec3-v1") / "FRAMEWORK.tar.gz"
ARCHIVE_SHA = "85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f"
SOURCE = "framework/generation/src/gencuda/backends/trellis2_generation.py"
SOURCE_SHA = "79bf62335e12e80adf9244c160de93a414440de8caf561b06784416ff2129b5d"
CONDITIONS_SHA = "3357f7d00582878d4eea52bed81f87ef44aa007a2207b1901dd1c8b6b9251504"


def native_prepare():
    with tarfile.open(fileobj=io.BytesIO(pinned(ARCHIVE, 1024**2, ARCHIVE_SHA)), mode="r:gz") as archive:
        source = archive.extractfile(SOURCE).read()
    require(sha256(source).hexdigest() == SOURCE_SHA, "Native generation backend differs")
    (function,) = [node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_prepare_rgba"]
    scope = {"np": np, "Image": Image}
    exec(compile(ast.Module(body=[function], type_ignores=[]), SOURCE, "exec"), scope)
    return scope["_prepare_rgba"]


def prepare_one(path, entry, prepare):
    record = entry["condition"]
    raw = pinned(path, 32 * 1024, record["SHA256"])
    with Image.open(io.BytesIO(raw)) as image:
        require(len(raw) == record["bytes"] and image.format == "PNG" and
                image.mode == "RGBA" and image.size == (256, 256), "Condition PNG identity differs")
        rgba = np.asarray(image)
        require(sha256(rgba.tobytes()).hexdigest() == entry["rgba_SHA256"], "Decoded pixels differ")
        foreground = rgba[:, :, 3] > 0.8 * 255
        ys, xs = np.nonzero(foreground)
        bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        require(bounds == entry["bbox_inclusive_px"] and len(xs) == entry["opaque_pixels"], "Foreground differs")
        rgb = prepare(image)

    x0, y0, x1, y1 = bounds
    size = max(x1 - x0, y1 - y0, 1)
    left, top = int((x0 + x1 - size) / 2), int((y0 + y1 - size) / 2)
    right, bottom = int((x0 + x1 + size) / 2), int((y0 + y1 + size) / 2)
    retained = (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)
    require(rgb.mode == "RGB" and rgb.size == (right - left, bottom - top)
            and rgb.width > 0 and rgb.height > 0 and retained.any(), "Invalid native crop")
    with io.BytesIO() as png:
        rgb.save(png, format="PNG")
        prepared = png.getvalue()
    return prepared, {
        "condition_SHA256": record["SHA256"],
        "native_RGB_SHA256": sha256(rgb.tobytes()).hexdigest(),
        "native_PNG_SHA256": sha256(prepared).hexdigest(),
        "native_crop_box_exclusive": [left, top, right, bottom],
        "native_size": list(rgb.size),
        "input_foreground_pixels": len(xs),
        "foreground_pixels_outside_crop": int(np.count_nonzero(~retained)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(pinned(args.conditions / "CONDITIONS.json", 16 * 1024, CONDITIONS_SHA))
    require(set(manifest["objects"]) == set(NAMES), "Frozen object set differs")
    prepare = native_prepare()
    results = {name: prepare_one(args.conditions / f"{name}-condition.png", manifest["objects"][name], prepare)
               for name in NAMES}
    args.output.mkdir()
    for name, (png, _) in results.items():
        with (args.output / f"{name}-prepared.png").open("xb") as stream:
            stream.write(png)
    receipt = {
        "schema": "TRELLIS_HELDOUT4_NATIVE_PREPARATION_V1",
        "condition_manifest_SHA256": CONDITIONS_SHA,
        "generation_backend_SHA256": SOURCE_SHA,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "actual_native_preparations": len(results),
        "actual_model_forwards": 0,
        "objects": {name: row for name, (_, row) in results.items()},
    }
    with (args.output / "NATIVE_CONDITIONS.json").open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
