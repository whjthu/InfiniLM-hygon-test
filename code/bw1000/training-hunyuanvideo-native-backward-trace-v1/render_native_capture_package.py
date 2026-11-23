"""Rebind the qualified single-card launcher to the exact native backward observer."""

from hashlib import sha256
import json


PRIOR_RENDERER_SHA = "cc1f4e3d05675726acbb301267d520020e81e6a8eebe07374651297b94f0e7b2"
PRIOR_STAGER_SHA = "1122afed03db69c25b04db36bfeefe69003ba8cc12978fe6c5e99b76451af024"
BACKWARD_MODULE = "hunyuanvideo_backward_trace.py"
BACKWARD_RENDERER = "render_hunyuanvideo_backward_trace.py"
BACKWARD_REVIEW = "review_backward_capture.py"
OBSERVER_SHA = "3176cbec0e421ce84cf495360aee11d09b6c7332068977c565341ff1f9af11fc"
RENDERER_SHA = "0cbb490097d890cda4966ebad554581611a914def032dea08bfed4d68fe513f8"
PATCHED_BACKEND_SHA = "e84dc8d8074aca0ea8c47d6cf1d38442d1bc518cf79b883738c79f48bccab9ce"


def patch(raw, original, replacement):
    token = original.encode()
    if raw.count(token) != 1:
        raise ValueError("Previously qualified package source anchor differs")
    return raw.replace(token, replacement.encode(), 1)


def render_stager(old_stager):
    if sha256(old_stager).hexdigest() != PRIOR_STAGER_SHA:
        raise ValueError("Unexpected source stager")
    for old, new in (
        ("from render_t09_phase_profile import render", "from render_hunyuanvideo_backward_trace import render"),
        ('OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_phase_profile.py"',
         'OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_backward_trace.py"'),
        ("a237fb3930ae5b60c03e576082ccfbb52720e95d89421bc203dfb434f229576a", OBSERVER_SHA),
        ("614f4d8598ef8b5d913183242d4760d6a56ae20520974f6c8073a54d104439e8", PATCHED_BACKEND_SHA),
    ):
        old_stager = patch(old_stager, old, new)
    return old_stager


def render_package(prior_renderer, base_manifest, base, prior_extra, additions, scope):
    if sha256(prior_renderer).hexdigest() != PRIOR_RENDERER_SHA:
        raise ValueError("Previously qualified launcher renderer differs")
    if (set(additions) != {BACKWARD_MODULE, BACKWARD_RENDERER, BACKWARD_REVIEW}
            or sha256(additions[BACKWARD_MODULE]).hexdigest() != OBSERVER_SHA
            or sha256(additions[BACKWARD_RENDERER]).hexdigest() != RENDERER_SHA):
        raise ValueError("Device-tested observer or exact renderer differs")
    qualified = {}
    exec(compile(prior_renderer, "qualified_profile_renderer", "exec"), qualified)
    result = qualified["render"](base_manifest, base, prior_extra, scope)
    for unused in ("hunyuanvideo_phase_profile.py", "render_t09_phase_profile.py",
                   "profile_run_review.py", "SOURCE_FILES.json"):
        del result[unused]
    result.update(additions)
    result["stage_t09_phase_framework.py"] = render_stager(prior_extra["stage_t09_phase_framework.py"])
    result["native_worker.py"] = patch(result["native_worker.py"],
                                      'RUN / "src/hunyuanvideo_phase_profile.py"',
                                      'RUN / "src/hunyuanvideo_backward_trace.py"')
    result["commands.py"] = patch(result["commands.py"], "export HUNYUAN_T09_PHASE_PROFILE=6,7",
                                  "export HUNYUAN_T09_BACKWARD_TRACE=6")
    controller = result["controller.py"]
    for old, new in (
        ("from profile_run_review import read_phases", "from review_backward_capture import read_backward_capture"),
        ('result["phase_profile"] = read_phases(output)', 'result["backward_capture"] = read_backward_capture(output)'),
        ('and result["phase_profile"]["passed"]', 'and result["backward_capture"]["passed"]'),
    ):
        controller = patch(controller, old, new)
    result["controller.py"] = controller
    claim = json.loads(result["CLAIM.json"])
    claim.update(state="SCOPED_NATIVE_BACKWARD_CAPTURE",
                 recipe_scope="Original 25 updates with complete backward tracing on global step 6")
    result["CLAIM.json"] = json.dumps(claim, indent=2).encode()
    for name, raw in result.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    result["SOURCE_FILES.json"] = json.dumps({"files": {name: qualified["pin"](raw)
                                                       for name, raw in result.items()}}, indent=2).encode()
    return result
