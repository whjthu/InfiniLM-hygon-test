"""Run or reconcile the original three-unit explicit-AR Native CPU step."""

from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "fresh-kv-pack-native-host-v1/host_once.py"
TEMPLATE_SHA256 = "b7845bd133a642877b43b803780df6ae8e19f3d9ca5598c028f791a34911bc7b"


def controller():
    raw = TEMPLATE.read_bytes()
    if sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("original Native HOST controller differs")
    source = raw.decode()
    bindings = (
        ('"GLM474-fresh-kv-pack-Native-CPU-v1"', '"GLM474-fresh-kv-explicit-ar-Native-CPU-v1"', 1),
        ('"e5ead062820a840906c74fce7294023b"', '"eca8f3dcb6d3d3938ce50868dfe670fd"', 1),
        ('dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1)',
         'dict(fresh_M=3, HOST_compile=3, GPU_compile=0, reused_objects=29, link_objects=33, links=1)', 1),
        ('"34da8adfefc5f08d3ceef01b462d546e7d2b9c69"', '"ec92243f2c9fa4573f8029197448334ef396a84e"', 1),
        ('glm-r474-fresh-kv-pack-native-cpu-v1.json', 'glm-r474-fresh-kv-explicit-ar-native-cpu-v1.json', 1),
        ('2691b803a2b8d2b54f5f52e7013f58aa2c15749ed0f02a70dc1baf400595089f',
         '008c9a412b060c21f8e6c60ed41ebcc7e1f4ba0f6eedf7905a503132f5b8a8fe', 1),
        ('Path("tools/run_glm474_fresh_kv_native_once.py")', 'Path("tools/run_glm474_fresh_kv_explicit_ar_once.py")', 1),
        ('Path("tools/prepare_glm474_fresh_kv_native_build.py")',
         'Path("tools/prepare_glm474_fresh_kv_explicit_ar_build.py")', 1),
        ('WORKER: "2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b",',
         'WORKER: "c1fd88cc12d62835c88038fb9c6f7de893e9e809ad8b608ec028b942b8c7708c",\n'
         '    Path("tools/run_glm474_fresh_kv_native_once.py"): '
         '"2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b",', 1),
        ('RECIPE: "12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd",',
         'RECIPE: "e30caa47c2677aa2f3995abc2ffda5b51ae807f16f4e3ddcb1246a7b85db7c13",\n'
         '    Path("tools/prepare_glm474_fresh_kv_native_build.py"): '
         '"12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd",', 1),
        ('("u15", "u17")', '("u00", "u15", "u17")', 2),
        ('{"u15", "u17"}', '{"u00", "u15", "u17"}', 2),
        ('linked["strong_exports"] == 283', 'linked["strong_exports"] == 284', 1),
    )
    for old, new, count in bindings:
        if source.count(old) != count:
            raise ValueError("original Native HOST controller binding differs")
        source = source.replace(old, new)
    scope = {"__name__": "explicit_ar_host", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(TEMPLATE), "exec"), scope)
    return scope


if __name__ == "__main__":
    raise SystemExit(controller()["main"]())
