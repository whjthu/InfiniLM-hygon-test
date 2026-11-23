"""Bind forward tracing to the previously qualified native training launcher."""

from hashlib import sha256
import json


PRIOR_BUILD_SHA = "e48a934b994f6a312c9847a16d08f96575190d3e7faf324eb21ccb00da61993c"
FORWARD_BACKEND_SHA = "d8fe370c10af1f004623ce3e6bdde1f56ba16585fd9e52c57c4c48d4ea586e12"
FORWARD_OBSERVER_SHA = "78af3854fc43b50944be884bbfd1641b40689bf426bb710fa5752bddc51bbd04"
FORWARD_MODULE = "hunyuanvideo_forward_trace.py"
FORWARD_RENDERER = "render_forward_capture.py"
FORWARD_REVIEW = "review_forward_capture.py"


def load_previous(raw):
    if sha256(raw).hexdigest() != PRIOR_BUILD_SHA:
        raise ValueError("Prior tested native launcher binding differs")
    env = {}
    exec(compile(raw, "previous_native_launcher", "exec"), env)
    return env


def render_stager(prior, old_stager):
    staged = prior["render_stager"](old_stager)
    for old, new in (
        ("from render_hunyuanvideo_backward_trace import render", "from render_forward_capture import render"),
        ('OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_backward_trace.py"',
         'OBSERVER = "generation/src/gencuda/backends/hunyuanvideo_forward_trace.py"'),
        (prior["OBSERVER_SHA"], FORWARD_OBSERVER_SHA),
        (prior["PATCHED_BACKEND_SHA"], FORWARD_BACKEND_SHA),
    ):
        staged = prior["patch"](staged, old, new)
    return staged


def render_package(prior_source, prior_renderer, base_manifest, base, prior_extra, backward, forward, scope):
    prior = load_previous(prior_source)
    if (set(forward) != {FORWARD_MODULE, FORWARD_RENDERER, FORWARD_REVIEW}
            or sha256(forward[FORWARD_MODULE]).hexdigest() != FORWARD_OBSERVER_SHA):
        raise ValueError("Forward observer or package source differs")
    result = prior["render_package"](prior_renderer, base_manifest, base, prior_extra, backward, scope)
    for old in (prior["BACKWARD_MODULE"], prior["BACKWARD_RENDERER"], prior["BACKWARD_REVIEW"], "SOURCE_FILES.json"):
        del result[old]
    result.update(forward)
    result["stage_t09_phase_framework.py"] = render_stager(prior, prior_extra["stage_t09_phase_framework.py"])
    patch = prior["patch"]
    result["native_worker.py"] = patch(result["native_worker.py"],
                                      'RUN / "src/hunyuanvideo_backward_trace.py"',
                                      'RUN / "src/hunyuanvideo_forward_trace.py"')
    result["commands.py"] = patch(result["commands.py"], "export HUNYUAN_T09_BACKWARD_TRACE=6",
                                  "export HUNYUAN_T09_FORWARD_TRACE=6")
    controller = result["controller.py"]
    for old, new in (
        ("from review_backward_capture import read_backward_capture", "from review_forward_capture import read_forward_capture"),
        ('result["backward_capture"] = read_backward_capture(output)', 'result["forward_capture"] = read_forward_capture(output)'),
        ('and result["backward_capture"]["passed"]', 'and result["forward_capture"]["passed"]'),
    ):
        controller = patch(controller, old, new)
    result["controller.py"] = controller
    claim = json.loads(result["CLAIM.json"])
    claim.update(state="SCOPED_NATIVE_FORWARD_CAPTURE",
                 recipe_scope="Original 25 updates with the whole forward and host-only SDPA descriptors on step 6")
    result["CLAIM.json"] = json.dumps(claim, indent=2).encode()
    for name, raw in result.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    pins = {name: {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()} for name, raw in result.items()}
    result["SOURCE_FILES.json"] = json.dumps({"files": pins}, indent=2).encode()
    return result
