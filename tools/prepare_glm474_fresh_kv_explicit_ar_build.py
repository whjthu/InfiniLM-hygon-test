"""Bind explicit decode to the original CPU308 link and the qualified gfx936 pack object."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMPLATE = HERE / "prepare_glm474_fresh_kv_native_build.py"
TEMPLATE_SHA256 = "12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd"
CALLER = REPO / "code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-explicit-ar-source-v1"
CALLER_SHA256 = "60b7b3b3d1a74bed37d948e9397a8b0035d1a438c411b194cf8c023b0f59f2ac"
CALLER_PUBLICATION = "a139ccf33119e6765897ed966fd53e99b8b9f2fa"
LABEL = "GLM474-fresh-kv-explicit-ar-Native-CPU-v1"
DECODE_EXPORT = "modern_glm_global_fresh_kv_decode_r474"
HOST = {
    "u00": "full_bridge_observer.cpp",
    "u15": "NoV_tp16.cpp",
    "u17": "prepare_tp16.cpp",
}
ADDED = {
    "u00": ("forward_explicit_ar.inc", "fresh_kv_explicit_ar.hpp"),
    "u15": ("fresh_kv_route.hpp", "fresh_kv_explicit_ar.hpp"),
    "u17": ("fresh_kv_route.hpp", "fresh_kv_state.inc", "fresh_kv_materialize.inc"),
}
PLANNED = dict(fresh_M=3, HOST_compile=3, GPU_compile=0, reused_objects=29, link_objects=33, links=1)


def caller_sources():
    path = CALLER / "SOURCE_FILES.json"
    manifest = path.read_bytes()
    if sha256(manifest).hexdigest() != CALLER_SHA256:
        raise ValueError("explicit decode source manifest differs")
    declaration = json.loads(manifest)
    seen = set()
    for item in declaration["files"]:
        path = REPO / item["path"]
        if path.parent != CALLER or path.resolve(strict=True) != path or path.name in seen:
            raise ValueError("unexpected explicit decode source location")
        seen.add(path.name)
        with path.open("rb") as handle:
            raw = handle.read(item["bytes"] + 1)
        if (len(raw), sha256(raw).hexdigest()) != (item["bytes"], item["SHA256"]):
            raise ValueError("explicit decode source bytes differ: " + path.name)
    if "render.py" not in seen:
        raise ValueError("explicit decode renderer is unbound")
    generated = runpy.run_path(str(CALLER / "render.py"))["sources"]()
    expected = declaration["generated_sources_SHA256"]
    if {name: sha256(raw).hexdigest() for name, raw in generated.items()} != expected:
        raise ValueError("explicit decode generated source differs")
    return generated, expected


def backend():
    raw = TEMPLATE.read_bytes()
    if sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("original Native recipe builder differs")
    code = raw.decode()
    edits = (
        ("'de7d18d6ce837171ebd01726adc0077602bada6f'", repr(CALLER_PUBLICATION)),
        ("'/refactor/candidates/GLM474-fresh-kv-pack-Native-CPU-v1'", repr(f"/refactor/candidates/{LABEL}")),
        ("CHANGED = {'u15': 'NoV_tp16.cpp', 'u17': 'prepare_tp16.cpp'}", f"CHANGED = {HOST!r}"),
        ("SOURCE_DELTA = {'u15': ('fresh_kv_route.hpp',),\n"
         "                'u17': ('fresh_kv_route.hpp', 'fresh_kv_state.inc', 'fresh_kv_materialize.inc')}",
         f"SOURCE_DELTA = {ADDED!r}"),
        ("planned=dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1)",
         f"planned={PLANNED!r}"),
        ("if name != 'glm_moe_dsa_attention_prepare.hpp':",
         "if name != 'glm_moe_dsa_attention_prepare.hpp' and not name.endswith('.py'):"),
        ("set(EXPORTS) & reference_symbols.keys()", "set(LINK_EXPORTS) & reference_symbols.keys()"),
        ("dict.fromkeys(EXPORTS, 'T')", "dict.fromkeys(LINK_EXPORTS, 'T')"),
    )
    for old, new in edits:
        if code.count(old) != 1:
            raise ValueError("original Native recipe edit anchor differs")
        code = code.replace(old, new)
    scope = {"__name__": "explicit_decode_native_builder", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, str(TEMPLATE), "exec"), scope)
    scope.update(CALLER=CALLER, CALLER_MANIFEST=CALLER_SHA256, caller_sources=caller_sources,
                 LINK_EXPORTS=[*scope["EXPORTS"], DECODE_EXPORT])
    return scope


if __name__ == "__main__":
    backend()["main"]()
