"""Adopt only the .855 object in the frozen explicit-AR three-TU build."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import kernel_855


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/refactor")
LABEL = "GLM474-fresh-kv-explicit-ar-Native-CPU-v2"
REQUEST = sha256(f"{LABEL}:1103606:f12r2n13:{kernel_855.OBJECT_SHA}".encode()).hexdigest()[:32]
PLANNED = dict(fresh_M=3, HOST_compile=3, GPU_compile=0, reused_objects=29, link_objects=33, links=1)
CLAIM = HERE / "SCOPE.json"
CALLER = HERE.parent / "fresh-kv-explicit-ar-source-v1"
SOURCE_MANIFEST_SHA256 = "60b7b3b3d1a74bed37d948e9397a8b0035d1a438c411b194cf8c023b0f59f2ac"
EXPLICIT_PREPARE = "tools/prepare_glm474_fresh_kv_explicit_ar_build.py"
EXPLICIT_WORKER = "tools/run_glm474_fresh_kv_explicit_ar_once.py"
EXPLICIT_READER = "tools/read_glm474_fresh_kv_explicit_ar_native_terminal.py"
SOURCE_PINS = {
    EXPLICIT_PREPARE: "e30caa47c2677aa2f3995abc2ffda5b51ae807f16f4e3ddcb1246a7b85db7c13",
    EXPLICIT_WORKER: "c1fd88cc12d62835c88038fb9c6f7de893e9e809ad8b608ec028b942b8c7708c",
    EXPLICIT_READER: "6523fce41d08066f63fa08e970e48723288fe875b866be29a67493a69ca1322a",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def old_script(name):
    script = REPO / name
    kernel_855.read(script, SOURCE_PINS[name])
    return runpy.run_path(str(script))


def caller_sources():
    manifest = json.loads(kernel_855.read(CALLER / "SOURCE_FILES.json", SOURCE_MANIFEST_SHA256))
    pins = manifest["generated_sources_SHA256"]
    require(len(pins) == 12 and all(Path(name).name == name for name in pins),
            "frozen explicit-AR rendered sources differ")
    directory = HERE / "generated"
    require({path.name for path in directory.iterdir()} == pins.keys(), "rendered caller inventory differs")
    return {name: kernel_855.read(directory / name, pin) for name, pin in pins.items()}, pins


def backend():
    recipes = old_script(EXPLICIT_PREPARE)["backend"]()
    kernel = recipes["KERNEL"].with_name(kernel_855.LABEL)
    recipes.update(KERNEL=kernel, KERNEL_LABEL=kernel_855.LABEL, KERNEL_MANIFEST=kernel_855.STAGE_SHA,
                   KERNEL_REQUEST=kernel_855.REQUEST, caller_sources=caller_sources,
                   NATIVE=recipes["NATIVE"].with_name(LABEL))
    compile_check = recipes["compiled_kernel"]

    def compiled_kernel(proof, raw_worker):
        require(proof["row"] == kernel_855.ROW and proof["request"] == kernel_855.REQUEST
                and proof["public_result_SHA256"] == kernel_855.PUBLIC_SHA
                and proof["intent_SHA256"] == kernel_855.INTENT_SHA
                and all(proof[key] == value for key, value in kernel_855.PROOF.items()),
                "exact adopted kernel proof differs")
        return compile_check(proof, raw_worker)

    recipes["compiled_kernel"] = compiled_kernel
    return recipes


def native_worker():
    old = old_script(EXPLICIT_WORKER)["runner"]()
    recipes = backend()
    raw_prepare = recipes["prepare"]

    def prepare(args):
        proof = json.loads(kernel_855.read(args.kernel_reconciliation, args.kernel_reconciliation_sha256, limit=16384))
        kernel_855.verify_compute(ROOT, proof)
        return raw_prepare(args)

    recipes["prepare"] = prepare
    old.update(__file__=str(Path(__file__).resolve()), build=SimpleNamespace(**recipes), LABEL=LABEL,
               CLAIM=CLAIM, RECIPE_TOOL=Path(__file__).resolve(), KERNEL_START=kernel_855.REQUEST)
    return old


def independent_reader(claim_sha):
    """Adopt the original reader on new explicitly pinned artifacts, using no compiler."""
    kernel_855.read(CLAIM, claim_sha, limit=16384)
    reader = old_script(EXPLICIT_READER)
    recipes = backend()
    explicit = SimpleNamespace(LABEL=LABEL, PLANNED=PLANNED,
                               HOST=recipes["CHANGED"], CALLER=CALLER, backend=backend, caller_sources=caller_sources)
    env = reader["audit"].__globals__
    env.update(REQUEST=REQUEST, KERNEL_REQUEST=kernel_855.REQUEST, CLAIM=CLAIM, CLAIM_PIN=claim_sha,
               explicit=explicit)
    return reader


def main():
    sys.path.insert(0, str(REPO / "tools"))
    if len(sys.argv) > 2 and sys.argv[1] == "--audit-compute":
        claim_sha = sys.argv[2]
        del sys.argv[1:3]
        independent_reader(claim_sha)["main"]()
        return 0
    return native_worker()["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
