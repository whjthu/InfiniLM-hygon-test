"""Instrument the already validated native inverse RoPE routes without changing dispatch."""

from hashlib import sha256
from pathlib import PurePosixPath


WIDTH_CALLER_SHA256 = "8ef228c7918329339bfc0477c5887f6c6055bd7bd55f9470038946909c9f4e7f"
MODEL_OWNER_SHA256 = "db5d66d85ea5922a5f6f3959ad36f9d5318d17883f0597afaab1e29e24e16704"


def checked_replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Unexpected native RoPE caller source anchor")
    return text.replace(before, after, 1)


def render(caller):
    if sha256(caller).hexdigest() != WIDTH_CALLER_SHA256:
        raise ValueError("Original width-selection source differs")
    source = caller.decode()
    entry = "rt::Tensor AttentionPrepare::inverse_output_rope(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&pos)const{"
    legacy = "    Work work{r};auto result=work.alloc(x->shape(),x->dtype());result->copy_from(r,x);rotate(r,result,std::nullopt,pos,true);r.syncStream();return result;"
    cases = (
        ('#include "inverse_output_width_policy.hpp"',
         '#include "inverse_output_width_policy.hpp"\n#include "inverse_output_route_counts.hpp"'),
        (entry, 'extern "C" int deep_inverse_output_rope_route_counts_v1(std::uint64_t* values, std::size_t count) {\n'
         '    return inverse_output::read_route_counts(values, count);\n}\n'
         + entry + "\n    inverse_output::RouteSample route_sample;"),
        ("    const size_t tokens = x->size(0);\n    if (requested",
         "    auto fallback = requested == inverse_output::Arm::legacy ? inverse_output::RouteIndex::legacy\n"
         "        : inverse_output::RouteIndex::fallback_geometry;\n"
         "    const size_t tokens = x->size(0);\n    if (requested"),
        ("            r.syncStream();\n            return result;\n        }\n    }\n" + legacy,
         "            r.syncStream();\n"
         "            route_sample.complete(requested == inverse_output::Arm::pack32\n"
         "                ? inverse_output::RouteIndex::pack32 : inverse_output::RouteIndex::pack64);\n"
         "            return result;\n"
         "        }\n        fallback = inverse_output::RouteIndex::fallback_no_stream;\n    }\n"
         + legacy.removesuffix("return result;") + "route_sample.complete(fallback);return result;"),
    )
    rendered = source
    for before, after in cases:
        rendered = checked_replace(rendered, before, after)
    reverted = rendered
    for before, after in reversed(cases):
        reverted = checked_replace(reverted, after, before)
    if reverted != source:
        raise ValueError("Native math or dispatch outside route accounting changed")
    return rendered.encode()


def render_model_owner(owner, native_library):
    if sha256(owner).hexdigest() != MODEL_OWNER_SHA256:
        raise ValueError("The verified TP8 model owner differs")
    library = PurePosixPath(native_library)
    if (PurePosixPath("/refactor/results") not in library.parents or ".." in library.parts
            or library.name != "libDeep_inverse_rope_width_routes.so"):
        raise ValueError("Unexpected instrumented Native artifact name or namespace")
    source = owner.decode()
    cases = (
        (
            'NATIVE = Path("/refactor/results/deep-rope-width-caller-tu15-native-cpu-v1/artifacts/libDeep_inverse_rope_width_caller.so")',
            f"NATIVE = Path({str(library)!r})",
        ),
        ('import time\n\n', 'import time\n\nfrom observe_routes import RouteCounts, verified_delta\n\n'),
        ('ROPE = ("deep_inverse_rope_h1_pack32_v1", "deep_inverse_rope_h1_pack64_v1")',
         'ROPE = ("deep_inverse_rope_h1_pack32_v1", "deep_inverse_rope_h1_pack64_v1",\n'
         '        "deep_inverse_output_rope_route_counts_v1")'),
        ('    visible_provider(lib)\n    need(all(fn() == 0',
         '    visible_provider(lib)\n    routes = RouteCounts(lib)\n    routes.snapshot()\n    need(all(fn() == 0'),
        ('        start = time.monotonic()\n\n        def finite(logits):',
         '        route_before = routes.snapshot()\n        start = time.monotonic()\n\n'
         '        def finite(logits):'),
        ('            binding.check(lib.modern_native_engine_rank_logits_equal(ctx))',
         '            route_after = routes.snapshot()\n'
         '            width_routes = verified_delta(route_before, route_after, config["arm"], require_selected=True)\n'
         '            binding.check(lib.modern_native_engine_rank_logits_equal(ctx))'),
        ('             EOS=natural, cap=not natural, rows=1, context=32768, owner_decode_tokens_s=rate,',
         '             inverse_rope_request_routes=width_routes, EOS=natural, cap=not natural, rows=1,\n'
         '             context=32768, owner_decode_tokens_s=rate,'),
    )
    for before, after in cases:
        source = checked_replace(source, before, after)
    compile(source, "instrumented_tp8_model_owner", "exec")
    return source.encode()
