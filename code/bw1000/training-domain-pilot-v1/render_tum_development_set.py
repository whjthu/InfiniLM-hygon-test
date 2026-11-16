"""Render a one-card resident VGGT scorer for a frozen development set."""

from render_tum_dev_controller import LABEL as SINGLE_LABEL, render as render_single


LABEL = "training-vggt-tum-indoor-development-set-v1"


def render(raw, manifest_sha, sequence):
    source = render_single(raw, manifest_sha, sequence).decode("utf8")
    replacements = {
        f'LABEL = "{SINGLE_LABEL}"': f'LABEL = "{LABEL}"',
        '    from score_vggt_tum_pair import score_native\n    from tum_vggt_pair import prepare_pair\n': (
            '    from score_vggt_tum_pair import score_native_pairs, summarize_development\n'
            '    from tum_development_set import load_development_set\n'
        ),
        '    manifest = json.loads(pinned(root / "data/pair.json", PAIR_SHA))\n'
        '    require(manifest["sequence"] == SEQUENCE, "wrong development sequence")\n'
        '    pair, geometry = prepare_pair(root / "data", manifest)\n': (
            '    cases, geometry = load_development_set(root / "data", root / "data/pair.json", PAIR_SHA)\n'
        ),
        '    result = score_native(pair, sequence=SEQUENCE, pair_sha=PAIR_SHA, source=Path("/vggt-source"),\n'
        '                          weights=Path("/weights/model.pt"), trained_head=Path("/trained/camera_head.pt"))\n': (
            '    scored = score_native_pairs(cases, source=Path("/vggt-source"), weights=Path("/weights/model.pt"),\n'
            '                                trained_head=Path("/trained/camera_head.pt"))\n'
        ),
        '    save(root, "native-result.json", dict(paired_pose=result, data_geometry=geometry, actual_DCU=1))': (
            '    save(root, "native-result.json", dict(paired_pose_rows=scored, data_geometry=geometry, actual_DCU=1,\n'
            '                                          summary=summarize_development(scored), optimizer_updates=0,\n'
            '                                          formal_CO3D_eligible=False, formal_performance_eligible=False))'
        ),
        '    for name, digest in json.loads((root / "data/pair.json").read_bytes())["files_SHA256"].items():\n'
        '        pinned(root / "data" / name, digest)\n': '',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("single-pair controller source boundary differs")
        source = source.replace(old, new, 1)
    return source.encode("utf8")
