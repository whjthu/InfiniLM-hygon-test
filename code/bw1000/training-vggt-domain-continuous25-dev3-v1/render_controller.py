"""Render the existing resident three-pair evaluator with the actual continuous25 head."""
import hashlib
import json
from pathlib import Path
import re

from training_guard import check_binding, HEAD_FILE_SHA256, HEAD_MODULE_SHA256, HEAD_STAT

LABEL = "training-vggt-tum-continuous25-dev3-v1"
DEV_SET_SHA256 = "6a0490db52ea7120766b3741b9dc788533abb81b8d2006ec7e8a2535fbd49a88"
SCORER_SHA256 = "6454c9a5f10459e5f2eb8b268918d1303d3f04c2038a12cca22c20ac87b58e22"
CANONICAL_BACKEND_SHA256 = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"


def scorer(raw):
    if hashlib.sha256(raw).hexdigest()!=SCORER_SHA256:
        raise ValueError("Original development scorer changed")
    source=raw.decode()
    changes={
        'UPDATED_HEAD = "fb84a5c4a5678cf724f29fc2a9f7a2d8b0f4d10bfd5a04f2e2d8f684454dec11"':
            f'UPDATED_HEAD = "{HEAD_MODULE_SHA256}"',
        'HEAD_STAT = (864719552, 23356225900, 1794863955000000000)':
            f'HEAD_STAT = {HEAD_STAT!r}\nHEAD_FILE_SHA256 = "{HEAD_FILE_SHA256}"',
        '    if stat_identity(trained_head) != HEAD_STAT:\n        raise ValueError("original trained head file identity differs")':
            '    if stat_identity(trained_head) != HEAD_STAT or stream_sha256(trained_head) != HEAD_FILE_SHA256:\n'
            '        raise ValueError("original continuous25 head file identity differs")',
    }
    for before,after in changes.items():
        if source.count(before)!=1:
            raise ValueError("Original scorer identity boundary differs")
        source=source.replace(before,after)
    return source.encode()


def controller(original_single_controller, binding):
    check_binding(binding)
    from render_tum_development_set import render as original_renderer, LABEL as original_label

    source=original_renderer(original_single_controller,DEV_SET_SHA256,"freiburg3_cabinet").decode()
    binding_raw=(json.dumps(binding,sort_keys=True,indent=2)+"\n").encode()
    binding_sha=hashlib.sha256(binding_raw).hexdigest()
    guard_sha=hashlib.sha256(Path(__file__).with_name("training_guard.py").read_bytes()).hexdigest()
    replacements={
        f'LABEL = "{original_label}"':f'LABEL = "{LABEL}"\nTRAIN_BINDING_SHA256 = "{binding_sha}"\nTRAIN_GUARD_SHA256 = "{guard_sha}"',
        '    save(root, "intent.json", dict(publication=payload["publication"], request=payload["request"],':
            '    save(root, "training-authority.json", training_authority)\n'
            '    save(root, "intent.json", dict(publication=payload["publication"], request=payload["request"],',
        '    pinned(Path(vggt_training.__file__), CANONICAL_BACKEND_SHA)\n':
            '    pinned(Path(vggt_training.__file__), CANONICAL_BACKEND_SHA)\n'
            '    authority = json.loads((root / "training-authority.json").read_bytes())\n',
        '                                          formal_CO3D_eligible=False, formal_performance_eligible=False))':
            '                                          formal_CO3D_eligible=False, formal_performance_eligible=False,\n'
            '                                          training_authority=authority))',
        "f\"{RESULTS/'training-vggt-tum-indoor-first-update-v2'/'train-output'}:/trained:ro\",":
            "f\"{RESULTS/'training-vggt-tum-continuous25-recovery-v1'/'arms/continuous/train'}:/trained:ro\",",
    }
    # The source binding is in the authenticated controller payload, before any new GPU scope.
    replacements['def controller(payload):\n    os.umask(0o077)\n'] = (
        'def controller(payload):\n    os.umask(0o077)\n'
        '    payload_binding_raw = base64.b64decode(payload["files"]["src/training-binding.json"]["b64"], validate=True)\n'
        '    require(sha(payload_binding_raw) == TRAIN_BINDING_SHA256, "training binding differs")\n'
        '    guard_raw = base64.b64decode(payload["files"]["src/training_guard.py"]["b64"], validate=True)\n'
        '    require(sha(guard_raw) == TRAIN_GUARD_SHA256, "training guard differs")\n'
        '    guard = {"__name__": "training_guard"}\n'
        '    exec(compile(guard_raw, "training_guard.py", "exec"), guard)\n'
        '    training_binding = json.loads(payload_binding_raw)\n'
        '    training_authority = guard["preflight"](training_binding)\n'
    )
    for before,after in replacements.items():
        if source.count(before)!=1:
            raise ValueError("Original dev-set controller source boundary differs")
        source=source.replace(before,after)
    return source.encode(),binding_raw
