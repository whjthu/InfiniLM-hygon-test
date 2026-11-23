"""Rebind the native runner for the baseline repeat and key/value-prefix comparison."""

from hashlib import sha256
import json

import derive_causal_observer as derive


PRIOR_PACKAGE_SHA = "9747787738558e7fb40420299128a98380f944d3bdee36826a6f4b79928c1467"
PRIOR_REVIEW_SHA = "34ba857c9998315d1ffbb30c7c10388e9a4219dc7b1c8d5e14e0c19117b4390f"
OBSERVER_SHA = "b5540ddc10a58b26160f34b0d49b7c8bfdf666017266611e8ba97c067dcde0be"
BACKEND_SHA = "5ad1a2b49470138bbd61ff39240b9e17c1cb57dbc60e1bcd28973fcbba48c46f"
GENERATOR_SHA = "65d96e99bd4fb0784f3ca9e930a9f3abcaa2f052fe653419e92a93140c8ab7db"
MODULE = "hunyuanvideo_native_kv_prefix.py"


def previous(source):
    if sha256(source).hexdigest() != PRIOR_PACKAGE_SHA:
        raise ValueError("Native prefix runner identity differs")
    env = {}
    exec(compile(source, "prior_native_prefix_runner", "exec"), env)
    return env


def sources(prior_sources):
    control = derive.observer(prior_sources["hunyuanvideo_native_prefix.py"])
    generator = derive.backend_renderer(prior_sources["render_native_prefix.py"])
    reviewer = prior_sources["review_native_prefix.py"]
    if (sha256(control).hexdigest() != OBSERVER_SHA or sha256(generator).hexdigest() != GENERATOR_SHA
            or sha256(reviewer).hexdigest() != PRIOR_REVIEW_SHA):
        raise ValueError("Native control source proof differs")
    before = '        pair, pair_pin = load(root / "first-update.json", 16384)'
    checked = (
        '        aa, aa_pin = load(root / "baseline-control.json", 16384)\n'
        '        result["baseline_control"] = aa\n'
        '        result["baseline_control_SHA256"] = aa_pin["SHA256"]\n'
        '        if (aa["actual_global_step"] != 1 or aa["passed"] is not True or aa["baseline_reverted"] is not True\n'
        '                or aa["tolerance"] != {"relative_l2": 0.0001, "max_tensor_relative_l2": 0.001}\n'
        '                or set(aa["groups"]) != {"forward", "unclipped_gradients", "clipped_gradients",\n'
        '                                          "optimizer_tensors", "parameter_updates"}\n'
        '                or any(row["passed"] is not True or row["tensors"] <= 0\n'
        '                       or not 0 <= row["relative_l2"] <= 0.0001\n'
        '                       or not 0 <= row["maximum_per_tensor_relative_l2"] <= 0.001\n'
        '                       for row in aa["groups"].values())):\n'
        '            raise ValueError("Native baseline repeat did not pass")\n'
        + before
    )
    reviewer = derive.patch(reviewer.decode(), before, checked)
    reviewer = derive.patch(reviewer,
        '        result["first_update"] = pair',
        '        if pair["identical_inputs"] != aa["identical_inputs"]:\n'
        '            raise ValueError("Baseline repeat and candidate actual input hashes differ")\n'
        '        result["first_update"] = pair')
    return {MODULE: control, "render_native_prefix.py": generator, "review_native_prefix.py": reviewer.encode()}


def render_stager(old_package, old_renderer, old_stager):
    initial = old_package["verified_renderer"](old_renderer)
    patch = initial["patch"]
    staged = old_package["render_stager"](patch, old_stager)
    for a, b in (("hunyuanvideo_native_prefix.py", MODULE), (old_package["OBSERVER_SHA"], OBSERVER_SHA),
                 (old_package["NATIVE_BACKEND_SHA"], BACKEND_SHA)):
        staged = patch(staged, a, b)
    return staged


def render_package(old_package, old_renderer, base_manifest, base, extras, prior_sources, scope):
    result = old_package["render_package"](old_renderer, base_manifest, base, extras, prior_sources, scope)
    generated = sources(prior_sources)
    del result["SOURCE_FILES.json"]
    del result["hunyuanvideo_native_prefix.py"]
    result.update(generated)
    result["stage_t09_phase_framework.py"] = render_stager(old_package, old_renderer, extras["stage_t09_phase_framework.py"])
    patch = old_package["verified_renderer"](old_renderer)["patch"]
    result["native_worker.py"] = patch(result["native_worker.py"], "hunyuanvideo_native_prefix.py", MODULE)
    result["commands.py"] = patch(result["commands.py"], "HUNYUAN_T09_NATIVE_PREFIX", "HUNYUAN_T09_NATIVE_KV_PREFIX")
    claim = json.loads(result["CLAIM.json"])
    claim.update(state="SCOPED_NATIVE_KV_PREFIX_BASELINE_REPEAT",
                 recipe_scope="Restore between two original first-update arms and a K/V-only candidate")
    result["CLAIM.json"] = json.dumps(claim, indent=2).encode()
    for name, raw in result.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    pins = {n: {"SHA256": sha256(raw).hexdigest(), "bytes": len(raw)} for n, raw in result.items()}
    result["SOURCE_FILES.json"] = json.dumps({"files": pins}, indent=2).encode()
    return result
