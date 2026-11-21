"""Run or reconcile the exact one-object retained28 score compile on its O01 slot."""

from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "output-residual-wave-host-compile-v1/host_once.py"
TEMPLATE_SHA256 = "ad70c04a2279a2eebf7fd51e07b714b890bb430d259b60aa114ca55c2d557a08"
LABEL = "Kimi461-output-residual-retained28-score-gfx936-CPU-v2"
PARENT, NODE, ZONE = "1103606", "f12r2n10", "O01"
MANIFEST = "9d3efa3a83f6733a479b23f95bbf4dcadb94a88c7b1a01216f2b889b2d6caf04"
SOURCE = "output-residual-retain-score-input-v1"
SOURCE_PIN = "27a0c112c5eec86a964bdcf8e9b4d1567335e531f8f5dcc050fb1becafbfa2b9"
RENDER_SHA256 = "5b9b43a66c9f6b4f9dae6cdd1c2f59141e05c976b5d34a1113a34e8bdbe73e35"
REQUEST = sha256(f"{LABEL}:{PARENT}:{NODE}:{MANIFEST}".encode()).hexdigest()[:32]


def controller():
    raw = TEMPLATE.read_bytes()
    if sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("frozen two-arm controller template differs")
    code = raw.decode()
    edits = (
        ('LABEL = "Kimi461-output-residual-wave-score-gfx936-CPU-v1"', f'LABEL = "{LABEL}"'),
        ('PARENT, NODE, ZONE = "1103606", "f12r2n11", "O01"', f'PARENT, NODE, ZONE = "{PARENT}", "{NODE}", "{ZONE}"'),
        ('PACKAGE = HERE.parent / "output-residual-wave-compile-CPU-v1"', f'PACKAGE = HERE.parent / "{SOURCE}"'),
        ('e55041e1f3b5b48c776ce0a3983fce550559dbdc0daa4a50e4dacd4319b13efe', SOURCE_PIN),
        ('d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6', MANIFEST),
        ('ARMS = ("full-tree", "wave32")', 'ARMS = ("retained28",)'),
        ('planned_device_TUs=2, rendered_manifest_SHA256=MANIFEST', 'planned_device_TUs=1, rendered_manifest_SHA256=MANIFEST'),
        ('planned_device_TUs=2, actual_device_TUs=2,', 'planned_device_TUs=1, actual_device_TUs=1,'),
        ('len(files) == 13', 'len(files) == 7'),
        ('files=13, target_compiles=0', 'files=7, target_compiles=0'),
        ('    return member, terminal, groups, runpy.run_path(str(PACKAGE / "prepare.py"))["prepare"]',
         f'    read(HERE / "render_target.py", "{RENDER_SHA256}")\n'
         '    return member, terminal, groups, runpy.run_path(str(HERE / "render_target.py"))["render"]'),
        ('    require(shares[0] == shares[1] and verified[ARMS[0]]["object_SHA256"] != verified[ARMS[1]]["object_SHA256"],\n'
         '            "score arms do not have a common DTK closure and two distinct objects")',
         '    require(len(shares) == 1 and shares[0], "single score arm lacks its original dependency set")'),
    )
    for origin, replacement in edits:
        if code.count(origin) != 1:
            raise ValueError("frozen controller edit anchor differs")
        code = code.replace(origin, replacement)
    scope = {"__file__": str(HERE / "host_once.py"), "__name__": "retained28_target_controller"}
    exec(compile(code, str(TEMPLATE), "exec"), scope)
    if scope["REQUEST"] != REQUEST:
        raise ValueError("derived retained28 request identity differs")
    return scope


if __name__ == "__main__":
    raise SystemExit(controller()["main"]())
