"""Verify the rendered VGGT overlay and pinned original room rows without Torch."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import render_backend
import vggt_rgb_only_room as room


CANONICAL = Path(
    "code/bw1000/training-qwen-A100-first-updates-v1/deltas/generation/src/"
    "gencuda/backends/vggt_training.py"
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _test_rgbd_projection(original, rendered):
    """Remove only the opt-in branches and compare the whole default module AST."""
    new = copy.deepcopy(rendered)
    _function(new, "_vggt_batch_content_sha256").body.pop(0)
    identity = _function(new, "_vggt_resume_identity")
    identity.body.pop(-2)
    tensor = _function(new, "_tensor_batch")
    tensor.args.kwonlyargs.clear()
    tensor.args.kw_defaults.clear()
    tensor.body.pop(2)
    loss = _function(new, "_camera_loss")
    loss.body[1:2] = loss.body[1].orelse

    run = _function(new, "run_vggt_training")
    text = lambda node: ast.unparse(node)
    mode_start = next(i for i, node in enumerate(run.body) if text(node).startswith("input_mode = "))
    input_branch = run.body[mode_start + 3]
    run.body[mode_start:mode_start + 4] = input_branch.orelse
    collect_index = next(i for i, node in enumerate(run.body)
                         if text(node).startswith("if rgb_only:\n    room_root"))
    default_source = run.body[collect_index].orelse
    run.body[collect_index:collect_index + 1] = default_source
    metrics = next(node.value for node in run.body
                   if isinstance(node, ast.Assign) and text(node.targets[0]) == "metrics")
    count = next(i for i, key in enumerate(metrics.keys) if key.value == "sequence_count")
    metrics.values[count] = metrics.values[count].orelse
    run.body.pop(-2)
    assert ast.dump(new) == ast.dump(original), "RGB-D AST projection differs"


def _expect_error(callback, expected):
    try:
        callback()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("invalid input unexpectedly passed")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "bytes_SHA256": _sha(value.tobytes())}
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _numpy_test_hash(value):
    return _sha(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode())


def _test_digest(tree, batches):
    """Execute actual backend content projection with a NumPy-only test hash."""
    environment = {"resume_common": SimpleNamespace(tree_sha256=_numpy_test_hash), "Any": object}
    function = _function(tree, "_vggt_batch_content_sha256")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<VGGT content projection>", "exec"), environment)
    project = environment[function.name]
    baseline = project(batches)
    assert baseline == project(batches)
    assert baseline != project(list(reversed(batches)))
    changes = {
        "ids": list(reversed(batches[0]["ids"])),
        "images": batches[0]["images"][::-1].copy(),
        "extrinsics": batches[0]["extrinsics"][::-1].copy(),
        "intrinsics": batches[0]["intrinsics"] * 2,
        "pose_valid": np.array([True, False]),
        "seq_name": "changed_for_test",
    }
    for field, changed in changes.items():
        assert baseline != project([dict(batches[0], **{field: changed}), *batches[1:]]), field
    missing_validity = {key: value for key, value in batches[0].items() if key != "pose_valid"}
    _expect_error(lambda: project([missing_validity, *batches[1:]]), "mixed")
    return baseline


def _test_reader_rejections(root, raw):
    _expect_error(lambda: render_backend.render(raw + b"\n"), "SHA-256 differs")
    group, _, suffix, _, bom, _, crop, pose, *keys = room._GROUPS[0]
    stem = room._STEM + group
    source = root / "local-only" / (stem + "-v1")
    output = root / "local-only" / (stem + "-rgb224-v1")
    public = json.loads((root / room.PUBLIC_ROOT / (stem + suffix)).read_bytes())["actual"]
    original = json.loads((source / bom).read_bytes())
    target = json.loads((output / pose).read_bytes())
    images = np.load(output / crop, allow_pickle=False)
    _expect_error(lambda: room._checked_pair(images[::-1].copy(), public, original, target, *keys), "frame order")
    _expect_error(lambda: room._checked_pair(images.astype(np.float32), public, original, target, *keys), "storage")
    reordered = dict(original, frames=original["frames"][::-1])
    _expect_error(lambda: room._checked_pair(images, public, reordered, target, *keys), "frame order")
    wrong_geometry = dict(target, extrinsics=[target["extrinsics"][0]] * 2)
    _expect_error(lambda: room._checked_pair(images, public, original, wrong_geometry, *keys), "extrinsics")
    wrong_validity = dict(target, pose_valid=[True, False])
    _expect_error(lambda: room._checked_pair(images, public, original, wrong_validity, *keys), "validity")
    data = (output / crop).read_bytes()
    with mock.patch.object(Path, "open", return_value=io.BytesIO(data[:-1] + bytes([data[-1] ^ 1]))):
        _expect_error(lambda: room._verified(output / crop, public["private_output_pins"][crop], 1_000_000), "pin differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    raw = (args.workspace / CANONICAL).read_bytes()
    rendered = render_backend.render(raw)
    original_tree, rendered_tree = ast.parse(raw), ast.parse(rendered)
    for source in (rendered, Path(room.__file__).read_bytes(), Path(render_backend.__file__).read_bytes()):
        compile(source, "<VGGT audit>", "exec")
    _test_rgbd_projection(original_tree, rendered_tree)
    batches = room.read_rgb_only_room(args.workspace)
    assert len(batches) == len(room.WINDOWS) == 7
    expected = {"images", "extrinsics", "intrinsics", "pose_valid", "seq_name", "ids"}
    for batch in batches:
        assert set(batch) == expected
        assert batch["pose_valid"].dtype == np.bool_ and batch["pose_valid"].tolist() == [True, True]
        assert batch["extrinsics"].shape == (2, 3, 4) and batch["extrinsics"].dtype == np.float32
        assert batch["intrinsics"].shape == (2, 3, 3) and batch["intrinsics"].dtype == np.float32
    assert len({item for batch in batches for item in batch["ids"]}) == 14
    _test_reader_rejections(args.workspace, raw)
    local_hash = _test_digest(rendered_tree, batches)
    receipt = {
        "schema": "vggt-room-rgb-only-overlay-local-v1",
        "observed_at_China": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "frozen_backend_SHA256": _sha(raw),
        "rendered_backend_SHA256": _sha(rendered.encode()),
        "reader_SHA256": _sha(Path(room.__file__).read_bytes()),
        "renderer_SHA256": _sha(Path(render_backend.__file__).read_bytes()),
        "verification_SHA256": _sha(Path(__file__).read_bytes()),
        "frozen_window_denominator": 12, "local_pairs": 7, "local_frames": 14,
        "pose_valid_frames": 14, "independent_heldout_sites": 0,
        "rows": [{"window": window, "passed": True, "pose_valid_frames": 2}
                 for window in room.WINDOWS],
        "checks": {
            "whole_rgbd_ast_default_projection_unchanged": True,
            "canonical_sha_mismatch_rejected": True,
            "crop_hash_mismatch_rejected": True,
            "crop_and_original_order_mismatch_rejected": True,
            "non_rgb8_rejected": True,
            "incorrect_pose_or_validity_rejected": True,
            "digest_is_deterministic_and_binds_mode_order_rgb_camera_and_validity": True,
            "mixed_modes_rejected": True,
            "default_digest_projection_ast_unchanged": True,
            "syntax": True,
        },
        "numpy_content_projection_test_SHA256": local_hash, "numpy_version": np.__version__,
        "native_torch_executed": False, "native_input_hashes": 0, "native_updates": 0,
        "gpu": 0, "network_reads": 0,
    }
    encoded = (json.dumps(receipt, indent=2) + "\n").encode()
    with args.receipt.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"receipt": str(args.receipt), "bytes": len(encoded), "SHA256": _sha(encoded),
                      "local_pairs": len(batches), "native_updates": 0}))


if __name__ == "__main__":
    main()
