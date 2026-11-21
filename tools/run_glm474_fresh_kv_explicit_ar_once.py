"""Execute the three-unit explicit decode Native build within the frozen zero-GRES step."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from prepare_glm474_fresh_kv_explicit_ar_build import LABEL, PLANNED, backend


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "run_glm474_fresh_kv_native_once.py"
TEMPLATE_SHA256 = "2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b"
CLAIM_FILE = "coordination/tasks/glm-r474-fresh-kv-explicit-ar-native-cpu-v1.json"


def runner():
    raw = TEMPLATE.read_bytes()
    if sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("original Native phase runner differs")
    code = raw.decode()
    edits = (
        ("import prepare_glm474_fresh_kv_native_build as build", ""),
        ("'GLM474-fresh-kv-pack-Native-CPU-v1'", repr(LABEL)),
        ("'coordination/tasks/glm-r474-fresh-kv-pack-native-cpu-v1.json'", repr(CLAIM_FILE)),
        ("'tools/prepare_glm474_fresh_kv_native_build.py'", "'tools/prepare_glm474_fresh_kv_explicit_ar_build.py'"),
        ("dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1)",
         repr(PLANNED)),
        ("len(recipe['reuse']) == 30, 'frozen two-unit recipe differs'",
         "len(recipe['reuse']) == PLANNED['reused_objects'], 'frozen explicit-decode recipe differs'"),
        ("report['actual']['reused_objects'] = 30",
         "report['actual']['reused_objects'] = PLANNED['reused_objects']"),
    )
    for old, new in edits:
        if code.count(old) != 1:
            raise ValueError("original phase runner edit anchor differs")
        code = code.replace(old, new)
    scope = {"__name__": "explicit_decode_phase_runner", "__file__": str(Path(__file__).resolve()),
             "build": SimpleNamespace(**backend())}
    exec(compile(code, str(TEMPLATE), "exec"), scope)
    return scope


if __name__ == "__main__":
    raise SystemExit(runner()["main"]())
