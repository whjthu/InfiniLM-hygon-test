"""Bind one verified TUM pair to the unchanged VGGT camera-head update loop."""

import argparse
from hashlib import sha256
from pathlib import Path


BACKEND_SHA256 = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
ADAPTER_SHA256 = "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962"


def checked_config(config):
    expected = {
        "training_scope": "camera_head",
        "cache_frozen_aggregator_features": False,
        "validate_recovery": False,
        "image_size": 224,
        "frames_per_step": 2,
        "cached_batches": 1,
        "max_steps": 1,
        "warmup_steps": 0,
        "checkpoint_steps": [1],
        "cards": [0],
    }
    for key, value in expected.items():
        actual = config.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"unexpected VGGT indoor qualification setting: {key}")
    if any(key in config for key in ("resume_from_checkpoint", "first_update_capture")):
        raise ValueError("VGGT indoor qualification needs a fresh owner")
    return Path(config["dataset_root"]), Path(config["dataset_manifest"]), config["dataset_sha"]


def prepare_native_batch(torch, config, make_batch, valid_depth):
    import json

    from .tum_vggt_pair import prepare_pair, read_pinned

    root, manifest, expected_sha = checked_config(config)
    raw, receipt = prepare_pair(root, json.loads(read_pinned(manifest, expected_sha)))
    batch = make_batch(torch, raw)
    shapes = {
        "images": (1, 2, 3, 224, 224), "depths": (1, 2, 224, 224),
        "extrinsics": (1, 2, 3, 4), "intrinsics": (1, 2, 3, 3),
        "point_masks": (1, 2, 224, 224),
    }
    if any(tuple(batch[key].shape) != shape for key, shape in shapes.items()) or not valid_depth(batch["point_masks"]):
        raise ValueError("VGGT native RGB-D input shape or valid pixels differ")
    return batch, receipt


def render_backend(raw):
    """Copy the pinned backend for the independent domain namespace."""
    if sha256(raw).hexdigest() != BACKEND_SHA256:
        raise ValueError("expected canonical VGGT a68952c backend bytes")
    source = raw.decode("utf8")
    replacements = {
        "    from data.datasets.co3d import Co3dDataset\n": "    from ..vggt_tum_native_overlay import prepare_native_batch\n",
        '        "sequence_count": len(dataset.sequence_list),\n': (
            '        "sequence_count": 1,\n'
            '        "data_case": "vggt-tum-indoor-first-update",\n'
            '        "tum_rgbd_geometry": tum_geometry,\n'
            '        "formal_CO3D_eligible": False,\n'
            '        "formal_performance_eligible": False,\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("canonical VGGT source boundary differs")
        source = source.replace(old, new)
    before = "    dataset = Co3dDataset(\n"
    after = "    data_seconds = perf_counter() - data_started\n"
    if source.count(before) != 1 or source.count(after) != 1:
        raise ValueError("canonical VGGT data collector differs")
    start, end = source.index(before), source.index(after)
    replacement = (
        "    batch, tum_geometry = prepare_native_batch(torch, config, _tensor_batch, valid_depth_batch)\n"
        "    batches = [batch]\n"
        "    rejected_batches = 0\n"
    )
    return (source[:start] + replacement + source[end:]).encode("utf8")


def stage_backend(source, target):
    overlay = render_backend(source.read_bytes())
    adapter = Path(__file__).with_name("tum_vggt_pair.py").read_bytes()
    if sha256(adapter).hexdigest() != ADAPTER_SHA256:
        raise ValueError("expected qualified TUM pair adapter bytes")
    target.mkdir()
    for name, data in (
        ("backends/vggt_training.py", overlay), ("tum_vggt_pair.py", adapter),
        ("vggt_tum_native_overlay.py", Path(__file__).read_bytes()),
    ):
        path = target / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as output:
            output.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_backend", type=Path)
    parser.add_argument("private_gencuda_delta", type=Path)
    args = parser.parse_args()
    stage_backend(args.original_backend, args.private_gencuda_delta)
