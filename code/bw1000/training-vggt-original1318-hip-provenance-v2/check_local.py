"""Check the true render and Linux mapping parser without loading HIP or dispatching."""

import ast
import ctypes
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import hip_provenance as provenance
import render


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "training-vggt-original1318-dev24-paired-v1"
AMD = "/opt/dtk/lib/libamdhip64.so.5"
GALAXY = "/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0"


def denied(run, expected):
    try:
        run()
    except ValueError as exc:
        assert str(exc) == expected
        return True
    raise AssertionError(expected)


def maps(source, second=None, permission="r-xp"):
    other = f"2200-2300 r-xp 00000100 08:02 42 {second}\n" if second else ""
    return f"1000-2000 {permission} 00000000 08:02 42 {source}\n{other}"


def check():
    sys.path.insert(0, str(ORIGINAL))
    try:
        import launch
    finally:
        sys.path.pop(0)
    scope = dict(request="e"*32, parent="1103606", node="f13r1n09", only_exclusive_node=True,
                 approved_operational_scope=True)
    previous, old_equivalence, old_label = launch.render(scope)
    candidate, equivalence, label = render.render(launch, scope)
    index = launch.auth.LEGACY_INDEX
    changed = {key for key in candidate if candidate[key] != previous[key]}
    assert set(candidate) == set(previous) and changed == {render.DSO, launch.CONTROLLER, index}
    assert equivalence == old_equivalence and label == old_label
    assert candidate[launch.CONTROLLER].replace(b"hip_dso(hip)", b"hip_dso()") == previous[launch.CONTROLLER]
    assert all(render.digest(candidate[key]) == sha for key, sha in json.loads(candidate[index]).items())
    assert candidate[launch.auth.SCORER] == previous[launch.auth.SCORER]
    for key in (render.DSO, launch.CONTROLLER):
        ast.parse(candidate[key], filename=key)
    ast.parse(Path(__file__).read_bytes(), filename=str(__file__))
    function = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)
    hip = SimpleNamespace(**{name:function for name in provenance.SYMBOLS})
    addr, again = provenance.symbols(hip)
    assert addr == again and addr > 0
    assert provenance.mapped_identity(maps(AMD), (0x1100, 0x1200)) == (AMD, (8, 2), 42)
    assert provenance.mapped_identity(maps(GALAXY), (0x1100, 0x1200)) == (GALAXY, (8, 2), 42)
    assert provenance.mapped_identity(maps(GALAXY, AMD), (0x1100, 0x1200))[0] == GALAXY
    one = "both HIP symbols must share exactly one file-backed executable object"
    name = "HIP symbol mapping has no supported live provider filename"
    negatives = [
        denied(lambda: provenance.mapped_identity(maps(AMD), (0, 1)),
               "expected the exact existing HIP handle symbol addresses"),
        denied(lambda: provenance.mapped_identity(maps(AMD, GALAXY), (0x1100, 0x2210)), one),
        denied(lambda: provenance.mapped_identity(maps(AMD, permission="r--p"), (0x1100, 0x1200)), one),
        denied(lambda: provenance.mapped_identity(maps(AMD+" (deleted)"), (0x1100, 0x1200)), name),
        denied(lambda: provenance.mapped_identity(maps("/tmp/unrelated.so"), (0x1100, 0x1200)), name),
        denied(lambda: provenance.mapped_identity(maps(AMD).replace(" 42 ", " 0 "), (0x1100, 0x1200)),
               "HIP symbol mapping has no stable file identity"),
        denied(lambda: render.render(launch, dict(scope, request=render.OLD_REQUEST)),
               "original .3188 request is closed"),
    ]
    return {"state":"ORIGINAL_RENDER_AND_HIP_MAP_FIXTURES_CHECKED", "changed_render_members":sorted(changed),
            "scorer_unchanged":True, "fixture_supported_provider_names":2, "rejections":sum(negatives),
            "actual_HIP_loaded_locally":False, "target_provider_identity_observed":False, "replay_allowed":False}


if __name__ == "__main__":
    print(json.dumps(check(),sort_keys=True))
