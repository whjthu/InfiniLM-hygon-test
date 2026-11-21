"""Validate the opt-in two-object training input before model initialization."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np


RECIPE = "sf3d-public-pbr-UV-atlas-v1"
GROUPS = ("WaterBottle", "Corset")
DATA_KEYS = {"albedo": (1, 3), "roughness": (1,), "metallic": (1,)}


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def member(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("expected one plain train input basename")
    path = root / name
    if not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("expected one regular in-place train file")
    return path


def read_pinned(path, expected, cap):
    size = expected.get("bytes")
    if type(size) is not int or not 0 < size <= cap or path.stat().st_size != size:
        raise ValueError("train input size differs")
    with path.open("rb") as f:
        raw = f.read(size + 1)
    if len(raw) != size or digest(raw) != expected.get("SHA256"):
        raise ValueError("train input digest differs")
    return raw


def load_atlas(config):
    route = config.get("sf3d_atlas")
    if not isinstance(route, dict) or route.keys() != {"manifest", "bytes", "SHA256"}:
        raise ValueError("atlas training requires an explicitly pinned train-only manifest")
    path = Path(route["manifest"])
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("atlas training manifest must be a canonical absolute file")
    raw_manifest = read_pinned(path, route, 16384)
    plan = json.loads(raw_manifest)
    if plan.get("recipe") != RECIPE or [row["group"] for row in plan["train"]] != list(GROUPS):
        raise ValueError("expected the fixed WaterBottle/Corset training order")
    if plan.keys() != {"recipe", "train"}:
        raise ValueError("atlas training manifest accepts only the predeclared train split")
    sources, targets, expected_paths, identity = [], [], [], []
    for row in plan["train"]:
        if row.keys() != {"group", "image", "image_pin", "target", "target_pin"}:
            raise ValueError("unexpected atlas train record")
        image = member(path.parent, row["image"])
        label = member(path.parent, row["target"])
        image_bytes = read_pinned(image, row["image_pin"], 8 * 1024**2)
        label_bytes = read_pinned(label, row["target_pin"], 16384)
        typed = {}
        with ZipFile(BytesIO(label_bytes)) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or any(
                not 0 < archive.getinfo("native_batch_" + key + ".npy").file_size <= 1024 for key in DATA_KEYS
            ):
                raise ValueError("atlas label array exceeds the material field size budget")
        with np.load(BytesIO(label_bytes), allow_pickle=False) as values:
            for key, shape in DATA_KEYS.items():
                value = values["native_batch_" + key]
                if (value.shape != shape or value.dtype != np.float32
                        or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()):
                    raise ValueError("atlas label is incompatible with the original material loss")
                typed[key] = value[0].copy() if key == "albedo" else value.item()
        sources.append(image_bytes)
        targets.append(typed)
        expected_paths.append(image)
        identity.append(dict(group=row["group"], input=row["image_pin"], target=row["target_pin"]))
    actual = [Path(value) for value in config.get("input_images", [])]
    if actual != expected_paths:
        raise ValueError("configured native input order differs from the train-only manifest")
    training_labels = dict(schema="SF3D_PUBLIC_ATLAS_TRAIN_LABEL_INPUT_ID_V1", recipe=RECIPE,
                           train_ordered=identity)
    return dict(images=sources, targets=targets,
                labels_sha256=digest(canonical(training_labels)),
                input_sha256=digest(canonical([{"sha256": row["input"]["SHA256"]} for row in identity])),
                manifest_sha256=digest(raw_manifest))


def atlas_config_sha256(base_config_sha256, atlas):
    return digest(canonical(dict(schema="SF3D_ATLAS_NATIVE_CONFIG_V1", native_config_sha256=base_config_sha256,
                                 training_label_input_SHA256=atlas["labels_sha256"],
                                 input_helper_SHA256=digest(Path(__file__).read_bytes()))))
