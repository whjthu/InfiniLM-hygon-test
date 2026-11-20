"""Stage the unchanged directional native loss with a four-view training collector."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


BASE_FILES = {
    "backends/vggt_training.py": "888064324c11ce0e2df5ce11f27b7e26a9d8b400aa8cb96f0924059fd1256d32",
    "domain_overlay.py": '78a24b978f5e7f9a063f265d435cd95183e7e996aa55a0cba4c5adb9befa58d7',
    "bidirectional.py": '02a64ac7586a52493c90b393228a3fe429a4e5d163ca7ef7414c1e08f7bd261b',
    "data_contract.py": "4329a619f5d2220c04417346a537cb8ffd994d83c3f27c35e96de29ef08349c7",
    "tum_vggt_pair.py": "c453a9c561c6411b8c2f25652f5c004c928db423c03ca0a45eba7d47ed9e3962",
}


def render(backend, helper):
    backend_name = "backends/vggt_training.py"
    if sha256(backend).hexdigest() != BASE_FILES[backend_name]:
        raise ValueError("Expected the original two-scene TUM backend")
    helper_sha = sha256(helper).hexdigest()
    edits = {
        "    from ..domain_overlay import batches as domain_batches\n": (
            "    from ..domain_overlay import batches as domain_batches\n"
            "    from .. import relative_pose_loss\n"
            f"    direction_loss_source = {helper_sha!r}\n"
            "    if sha256_path(Path(relative_pose_loss.__file__)) != direction_loss_source:\n"
            "        raise ValueError('VGGT direction-loss helper differs from the staged backend')\n"
            "    direction_pose_decoder = relative_pose_loss.checked_decoder(torch, source_root)\n"
        ),
        '            loss = _camera_loss(torch, predictions["pose_enc_list"], batch, observer=first_update_recorder)\n': (
            '            loss = _camera_loss(torch, predictions["pose_enc_list"], batch, observer=first_update_recorder)\n'
            '            direction_penalty = relative_pose_loss.training_penalty(\n'
            '                torch, direction_pose_decoder, predictions["pose_enc_list"][-1], batch\n'
            '            )\n'
            '            loss = loss + relative_pose_loss.WEIGHT * direction_penalty\n'
        ),
        '            "loss": value,\n': (
            '            "loss": value,\n'
            '            "relative_translation_direction_cost": float(direction_penalty.detach().cpu()),\n'
        ),
        '        "domain_geometry": domain_geometry,\n': (
            '        "domain_geometry": domain_geometry,\n'
            '        "relative_translation_loss": {\n'
            '            "source_SHA256": direction_loss_source,\n'
            '            "weight": relative_pose_loss.WEIGHT,\n'
            '            "minimum_target_baseline": relative_pose_loss.MINIMUM_BASELINE,\n'
            '            "predicted_norm_floor": relative_pose_loss.PREDICTED_NORM_FLOOR,\n'
            '            "pose_source_SHA256": relative_pose_loss.PROVIDER_SHA256,\n'
            '        },\n'
        ),
    }
    source = backend.decode("utf-8")
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError("Native VGGT source boundary changed")
        source = source.replace(old, new)
    ast.parse(source)
    compile(helper, "relative_pose_loss.py", "exec")
    return source.encode("utf-8")


def stage(base_delta, target):
    members = {}
    for name, wanted in BASE_FILES.items():
        raw = (base_delta / name).read_bytes()
        if sha256(raw).hexdigest() != wanted:
            raise ValueError("Baseline TUM delta changed: " + name)
        members[name] = raw
    helper = Path(__file__).with_name("relative_pose_loss.py").read_bytes()
    backend = "backends/vggt_training.py"
    members[backend] = render(members[backend], helper)
    members["relative_pose_loss.py"] = helper
    target.mkdir()
    for name, raw in members.items():
        path = target / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {name: sha256(raw).hexdigest() for name, raw in members.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_delta", type=Path)
    parser.add_argument("new_gencuda_delta", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.base_delta, args.new_gencuda_delta), sort_keys=True))
