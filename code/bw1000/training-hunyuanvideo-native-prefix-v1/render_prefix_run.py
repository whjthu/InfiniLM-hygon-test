"""Render a single guarded candidate using the proven native Slurm and asset launcher."""

from hashlib import sha256
import json


BASE_RENDERER_SHA = "cc1f4e3d05675726acbb301267d520020e81e6a8eebe07374651297b94f0e7b2"
BASE_STAGER_SHA = "1122afed03db69c25b04db36bfeefe69003ba8cc12978fe6c5e99b76451af024"
OPERATOR_SHA = "d72efcb8f77a02cb9161f56497a327a5c848430819f0d839d43bdbd45ea8a98f"
NATIVE_BACKEND_SHA = "31b052ebbb04e58c4034e003717d4f3049400b7d67fd2c11b2da60b05a544b88"
OBSERVER_SHA = "520eb4d0f6457176ed4e4eb98ff0a2bcaea1284f2ad273f3efe894ffd7085e37"
MODULE = "hunyuanvideo_native_prefix.py"


def verified_renderer(raw):
    if sha256(raw).hexdigest() != BASE_RENDERER_SHA:
        raise ValueError("The original launcher renderer differs")
    module = {}
    exec(compile(raw, "frozen_launcher_renderer", "exec"), module)
    return module


def observer(operator, session):
    if sha256(operator).hexdigest() != OPERATOR_SHA:
        raise ValueError("The qualified prefix SDPA implementation differs")
    merged = operator + b"\n\n" + session
    if sha256(merged).hexdigest() != OBSERVER_SHA:
        raise ValueError("The native prefix observer identity differs")
    return merged


def render_stager(patch, old_stager):
    if sha256(old_stager).hexdigest() != BASE_STAGER_SHA:
        raise ValueError("Source stager identity differs")
    for old, new in (
        ("from render_t09_phase_profile import render", "from render_native_prefix import render"),
        ('OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_phase_profile.py"',
         'OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_native_prefix.py"'),
        ("a237fb3930ae5b60c03e576082ccfbb52720e95d89421bc203dfb434f229576a", OBSERVER_SHA),
        ("614f4d8598ef8b5d913183242d4760d6a56ae20520974f6c8073a54d104439e8", NATIVE_BACKEND_SHA),
    ):
        old_stager = patch(old_stager, old, new)
    return old_stager


def render_package(prior_renderer, base_manifest, base, prior_extra, additions, scope):
    prior = verified_renderer(prior_renderer)
    if (set(additions) != {MODULE, "render_native_prefix.py", "review_native_prefix.py"}
            or sha256(additions[MODULE]).hexdigest() != OBSERVER_SHA):
        raise ValueError("Delivered native candidate sources differ")
    files = prior["render"](base_manifest, base, prior_extra, scope)
    for name in ("hunyuanvideo_phase_profile.py", "render_t09_phase_profile.py",
                 "profile_run_review.py", "SOURCE_FILES.json"):
        del files[name]
    files.update(additions)
    patch = prior["patch"]
    files["stage_t09_phase_framework.py"] = render_stager(patch, prior_extra["stage_t09_phase_framework.py"])
    files["native_worker.py"] = patch(files["native_worker.py"],
                                      'RUN / "src/hunyuanvideo_phase_profile.py"',
                                      'RUN / "src/hunyuanvideo_native_prefix.py"')
    files["commands.py"] = patch(files["commands.py"], "export HUNYUAN_T09_PHASE_PROFILE=6,7",
                                 "export HUNYUAN_T09_NATIVE_PREFIX=5326:4785")
    control = files["controller.py"]
    for old, new in (
        ("from profile_run_review import read_phases", "from review_native_prefix import read_prefix_update"),
        ('result["phase_profile"] = read_phases(output)', 'result["native_prefix"] = read_prefix_update(output)'),
        ('and result["phase_profile"]["passed"]', 'and result["native_prefix"]["passed"]'),
    ):
        control = patch(control, old, new)
    files["controller.py"] = control
    claim = json.loads(files["CLAIM.json"])
    claim.update(state="SCOPED_NATIVE_PREFIX_FIRST_UPDATE",
                 recipe_scope="One matched native first-update pair, then 25 recorded guarded updates")
    files["CLAIM.json"] = json.dumps(claim, indent=2).encode()
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    files["SOURCE_FILES.json"] = json.dumps({"files": {n: prior["pin"](raw) for n, raw in files.items()}},
                                             indent=2).encode()
    return files
