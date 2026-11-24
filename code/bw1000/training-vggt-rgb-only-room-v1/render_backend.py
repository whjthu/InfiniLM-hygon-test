"""Render an RGB-only opt-in overlay against the exact frozen VGGT backend."""

from __future__ import annotations

import argparse
import ast
import hashlib
import textwrap
from pathlib import Path


BASE_SHA256 = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected a unique frozen backend anchor")
    return source.replace(old, new, 1)


def render(raw):
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("canonical VGGT backend SHA-256 differs")
    source = raw.decode("utf-8").replace("\r\n", "\n")
    source = _replace(source, "def _vggt_batch_content_sha256(batches: list[dict[str, Any]]) -> str:\n",
        """def _vggt_batch_content_sha256(batches: list[dict[str, Any]]) -> str:
    if any("pose_valid" in batch for batch in batches):
        if not all("pose_valid" in batch and "depths" not in batch and "point_masks" not in batch for batch in batches):
            raise ValueError("cannot hash mixed RGB-only and RGB-D batches")
        return resume_common.tree_sha256({
            "mode": "tum_room_rgb_only",
            "batches": [
                {key: batch[key] for key in ("seq_name", "ids", "images", "extrinsics", "intrinsics", "pose_valid")}
                for batch in batches
            ],
        })
""")
    source = _replace(source, '    return {key: resume_common.json_sha256(value) for key, value in evidence.items()}, evidence\n',
        """    if any("pose_valid" in batch for batch in batches):
        from . import vggt_rgb_only_room

        evidence["sources"]["rgb_only_room_reader"] = _vggt_source_sha256(vggt_rgb_only_room)
    return {key: resume_common.json_sha256(value) for key, value in evidence.items()}, evidence
""")
    source = _replace(source, "def _tensor_batch(torch: Any, raw: dict[str, Any]) -> dict[str, Any]:\n",
        "def _tensor_batch(torch: Any, raw: dict[str, Any], *, rgb_only: bool = False) -> dict[str, Any]:\n")
    source = _replace(source, '    return {\n        "images": images,\n        "depths":',
        """    if rgb_only:
        return {
            "images": images,
            "extrinsics": torch.from_numpy(raw["extrinsics"]).unsqueeze(0),
            "intrinsics": torch.from_numpy(raw["intrinsics"]).unsqueeze(0),
            "pose_valid": torch.from_numpy(raw["pose_valid"]).unsqueeze(0),
            "seq_name": raw["seq_name"],
            "ids": [int(item) for item in raw["ids"]],
        }
    return {
        "images": images,
        "depths":""")
    source = _replace(source, '    valid_frames = batch["point_masks"].sum(dim=(-1, -2)) > 100\n',
        """    if "pose_valid" in batch:
        valid_frames = batch["pose_valid"]
    else:
        valid_frames = batch["point_masks"].sum(dim=(-1, -2)) > 100
""")
    source = _replace(source, "    from data.datasets.co3d import Co3dDataset\n",
        """    input_mode = config.get("vggt_input_mode", "co3d_rgbd")
    if input_mode not in ("co3d_rgbd", "tum_room_rgb_only"):
        raise ValueError("unsupported VGGT input mode")
    rgb_only = input_mode == "tum_room_rgb_only"
    if rgb_only:
        from . import vggt_rgb_only_room

        config = dict(config)
        config.setdefault("cached_batches", len(vggt_rgb_only_room.WINDOWS))
    else:
        from data.datasets.co3d import Co3dDataset
""")
    start = source.index('    dataset_root = Path(config["dataset_root"])\n')
    end = source.index('    data_seconds = perf_counter() - data_started\n', start)
    original_data = source[start:end]
    rgb_data = """    if rgb_only:
        room_root = Path(config["rgb_only_room_root"])
        if (image_size != 224 or frames_per_step != 2
                or int(config["cached_batches"]) != len(vggt_rgb_only_room.WINDOWS)):
            raise ValueError("RGB-only room input requires RGB224 and all seven ordered two-frame pairs")
        dataset_root = room_root / "local-only"
        annotation_root = room_root / vggt_rgb_only_room.PUBLIC_ROOT
        data_started = perf_counter()
        batches = [
            _tensor_batch(torch, raw, rgb_only=True)
            for raw in vggt_rgb_only_room.read_rgb_only_room(room_root)
        ]
        rejected_batches = 0
    else:
"""
    source = _replace(source, original_data, rgb_data + textwrap.indent(original_data, "    "))
    source = _replace(source, '        "sequence_count": len(dataset.sequence_list),\n',
                      '        "sequence_count": 1 if rgb_only else len(dataset.sequence_list),\n')
    source = _replace(source, '    return result\n',
        """    if rgb_only:
        metrics.update(input_mode=input_mode, rgb_only_pairs=len(batches), frozen_window_denominator=12)
    return result
""")
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rendered = render(args.canonical.read_bytes())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)


if __name__ == "__main__":
    main()
