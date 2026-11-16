"""Reuse the verified single-device controller for a separate VGGT development pair."""

from hashlib import sha256
import re


PARENT_CONTROLLER_SHA = "1fbea4e33fe3059455a6b6d6c08d58041cc185be2c9bf9f3170e8cc8528695c8"
TRAIN_PAIR_SHA = "b98b2473335cecd21c65be9dfb64d908b177802adfa74ce5750a5e4c4efeac50"
CANONICAL_BACKEND_SHA = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
LABEL = "training-vggt-tum-indoor-paired-dev-v1"


def render(raw, pair_sha, sequence):
    if sha256(raw).hexdigest() != PARENT_CONTROLLER_SHA:
        raise ValueError("original device controller differs")
    if not re.fullmatch(r"[0-9a-f]{64}", pair_sha) or pair_sha == TRAIN_PAIR_SHA:
        raise ValueError("expected a separately pinned development pair")
    if not re.fullmatch(r"freiburg3_[a-z0-9_]+", sequence) or sequence in (
            "freiburg3_structure_texture_near", "freiburg3_long_office_household"):
        raise ValueError("expected an independent development sequence")
    controller = raw.decode("utf8")
    edits = {
        'LABEL = "training-vggt-tum-indoor-first-update-v2"': f'LABEL = "{LABEL}"',
        f'PAIR_SHA = "{TRAIN_PAIR_SHA}"': f'PAIR_SHA = "{pair_sha}"\nSEQUENCE = "{sequence}"',
        'OVERLAY_SHA = "41b3b4ae14321bdaf7eda2f2e983fcfdbe935c9a7a37fc2ff9d441b96f05cd8e"':
            f'CANONICAL_BACKEND_SHA = "{CANONICAL_BACKEND_SHA}"',
        "    from vggt_tum_native_overlay import stage_backend\n": "",
        'f"{UPSTREAM}:/vggt-source:ro",': (
            'f"{UPSTREAM}:/vggt-source:ro", '
            'f"{RESULTS/\'training-vggt-tum-indoor-first-update-v2\'/\'train-output\'}:/trained:ro",'
        ),
    }
    for old, new in edits.items():
        if controller.count(old) != 1:
            raise ValueError("original controller anchor differs")
        controller = controller.replace(old, new, 1)
    start = controller.index("    stage_backend(root")
    end = controller.index('    work = root / "src/run_tum_native_once.py"', start)
    controller = controller[:start] + '    pinned(root / "framework" / BACKEND_REL, CANONICAL_BACKEND_SHA)\n' + controller[end:]
    start = controller.index("    import torch\n", controller.index("def native("))
    end = controller.index("\n\ndef controller(payload):", start)
    worker = '''    import torch
    from gencuda.backends import vggt_training
    from score_vggt_tum_pair import score_native
    from tum_vggt_pair import prepare_pair

    pinned(Path(vggt_training.__file__), CANONICAL_BACKEND_SHA)
    require(torch.version.hip == "6.3.26113" and importlib.metadata.version("torch") ==
            "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "the qualified torch/HIP provider differs")
    manifest = json.loads(pinned(root / "data/pair.json", PAIR_SHA))
    require(manifest["sequence"] == SEQUENCE, "wrong development sequence")
    pair, geometry = prepare_pair(root / "data", manifest)
    weight_identity(Path("/weights/model.pt"))
    result = score_native(pair, sequence=SEQUENCE, pair_sha=PAIR_SHA, source=Path("/vggt-source"),
                          weights=Path("/weights/model.pt"), trained_head=Path("/trained/camera_head.pt"))
    weight_identity(Path("/weights/model.pt"))
    save(root, "native-result.json", dict(paired_pose=result, data_geometry=geometry, actual_DCU=1))
'''
    return (controller[:start] + worker + controller[end:]).encode("utf8")
