"""Build and verify the initial T512 prefill branch on the original DTK mount."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AREA = HERE.parent.relative_to(REPO)
PRIOR_DIR = REPO / AREA / "fresh-kv-explicit-ar-native-host-v2"
ROOT = PurePosixPath("/refactor")
LABEL = "GLM474-fresh-kv-prefill512-Native-CPU-v1"
REQUEST = "d843277fdaf047749665c22190000b6a"
PLANNED = dict(fresh_M=3, HOST_compile=3, GPU_compile=0, reused_objects=29, link_objects=33, links=1)
PLACEMENT = dict(parent="1103606", node="f12r2n13", zone="O01", cpus=4, memory_MiB=16384, DCU=0)
CLAIM = HERE / "SCOPE.json"
CALLER = HERE.parent / "fresh-kv-prefill512-caller-source-v1"
CALLER_PUBLICATION = "b00171f1d33114bb651804052147f37d6976c9b6"
CALLER_MANIFEST = "151db4f53f0932b05cf8fb17abe76b3d1cd851a7e62346ad30d778faeb5c2d55"
NEW_EXPORT = "modern_glm_global_fresh_kv_prefill512_r474"
ORIGINAL_NATIVE_ROOT = ROOT / "candidates/GLM474-fresh-kv-explicit-ar-Native-CPU-v2"
ORIGINAL_NATIVE = ORIGINAL_NATIVE_ROOT / "libGLM_fresh_kv_pack.so"
ORIGINAL_NATIVE_SHA = "08cf47237d030dbf928b6b7a405556da26dd9630858d2a1cd506b61bd94a07d7"
ORIGINAL_NATIVE_BYTES = 1961800
ORIGINAL_ARCHIVE = Path("results/bw1000/2026-11-22/GLM474-explicit-AR-Native-v2-original918-terminal-and-independent-ABI.json")
ORIGINAL_ARCHIVE_SHA = "6809da6c2106c7e33886d600983c1b18dc6906478a7ab650c2b05458c61f679b"
LEGACY_WORKER_SHA = "93f4bf723ffca900159f678d9910ab3cc10753cfb614e667f7f6ccd2275531c0"
KERNEL_SHA = "a451ba33d8b2af5353563400aeb65519669da8915d515c2221eb03119cb44304"
HOST = {"u00": "full_bridge_observer.cpp", "u15": "NoV_tp16.cpp", "u17": "prepare_tp16.cpp"}
MORE_INCLUDES = {
    "u00": ("forward_explicit_prefill512.inc", "fresh_kv_prefill512.hpp", "fresh_kv_route.hpp"),
    "u15": ("fresh_kv_prefill512.hpp",),
    "u17": ("fresh_kv_prefill512.hpp", "fresh_kv_prefill512_materialize.inc"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def require_request(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) and value == REQUEST,
            "expected the canonical original 32-hex T512 request")


def checked(path, expected=None, limit=2**21):
    path = Path(path)
    require(path.resolve(strict=True) == path, "source is not a canonical original path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (expected is None or sha256(raw).hexdigest() == expected),
            "frozen bytes differ: " + path.name)
    return raw


def execute(path, text, **injections):
    namespace = dict(__file__=str(path), __name__="verified_glm_prefill_source", **injections)
    exec(compile(text, str(path), "exec"), namespace)
    return namespace


def once(source, old, new):
    require(source.count(old) == 1, "old exact transformation differs")
    return source.replace(old, new, 1)


def baseline():
    module = PRIOR_DIR / "kernel_855.py"
    kernel = SimpleNamespace(**execute(module, checked(module, KERNEL_SHA)))
    path = PRIOR_DIR / "worker.py"
    text = once(checked(path, LEGACY_WORKER_SHA).decode(), "import kernel_855\n", "")
    return kernel, execute(path, text, kernel_855=kernel)


def verify_original_archive(read_rows=None):
    raw = checked(REPO / ORIGINAL_ARCHIVE, ORIGINAL_ARCHIVE_SHA, limit=16384)
    archive = json.loads(raw)
    audit = archive["audit"]
    require(archive["state"] == "NATIVE_ABI_AND_DEPENDENCIES_VERIFIED_POST_CLEAN"
            and archive["actual_GPU_steps"] == 0 and archive["independent_post"]["clean"] is True
            and archive["independent_zero_GRES_post_step"] == "1103606.920"
            and audit["strong_exports"] == 284 and audit["original_worker_provider_count"] == 71
            and audit["library_SHA256"] == ORIGINAL_NATIVE_SHA and audit["library_bytes"] == ORIGINAL_NATIVE_BYTES
            and audit["actual"] == PLANNED
            and [row[0] for row in archive["actual_accounting"]] == ["1103606.918", "1103606.919"],
            "original .918 Native and its independent qualification differ")
    if read_rows is not None:
        rows = read_rows()
        for original in archive["actual_accounting"]:
            require([row for row in rows if row[0] == original[0]] == [original], "original .918/.919 accounting differs")
        after = [row for row in rows if row[0] == "1103606.920"]
        require(len(after) == 1 and after[0][2:4] == ["COMPLETED", "0:0"] and after[0][5] == PLACEMENT["node"]
                and all(part.endswith("=0") for part in after[0][6].split(",") if part.startswith("gres/")),
                "original .920 zero-GRES post differs")
    return archive


def caller_sources():
    manifest = json.loads(checked(CALLER / "SOURCE_FILES.json", CALLER_MANIFEST, 16384))
    pins = manifest["generated_sources_SHA256"]
    directory = HERE / "generated"
    require(len(pins) == 15 and all(Path(name).name == name for name in pins)
            and {path.name for path in directory.iterdir()} == pins.keys(), "rendered scope differs")
    return {name: checked(directory / name, pin) for name, pin in pins.items()}, pins


def additions(previous, new, prior_needed, new_needed):
    require(len(previous) == 284 and NEW_EXPORT not in previous
            and new == {**previous, NEW_EXPORT: "T"} and len(new) == 285,
            "the .918 strong map plus exactly one new prefill export is required")
    require(prior_needed == new_needed, "the new Native DSO changed original NEEDED entries")


def compare_link_paths(built, recipes, query):
    names = (str(ORIGINAL_NATIVE), str(PurePosixPath(built) / "libGLM_fresh_kv_pack.so"))
    symbols = [recipes["symbol_map"](query(["/usr/bin/nm", "-D", "--defined-only", path])) for path in names]
    needed = [recipes["needed_libraries"](query(["/usr/bin/readelf", "-d", path])) for path in names]
    additions(*symbols, *needed)


def qualifications(prior, recipes, kernel, prior_proof, *, verify_kernel=True):
    verify_original_archive()
    if verify_kernel:
        kernel.verify_compute(ROOT, prior_proof)
    kernel_worker = json.loads(recipes["checked"](recipes["KERNEL"] / "CPU-result.json",
                                                  prior_proof["receipt_SHA256"], 2**20))
    object_pin = recipes["compiled_kernel"](prior_proof, kernel_worker)
    base = json.loads(recipes["checked"](recipes["ORIGIN"] / "build-result.json", recipes["ORIGIN_BUILD_SHA"], 2**20))
    prior_manifest = json.loads(checked(prior["CALLER"] / "SOURCE_FILES.json", prior["SOURCE_MANIFEST_SHA256"], 16384))
    original = prior["backend"]()
    expected = original["make_recipe"](base, prior_manifest["generated_sources_SHA256"], object_pin)
    plan = json.loads(checked(CALLER.parent / "depth-stratified-observer-CPU-v1/PLAN.json",
                              recipes["SYSTEM_PLAN_SHA"], 2**21))
    expected.update(pinned_system_header_map=plan["system_pins"], expected_providers=plan["original169"]["providers"])
    previous_bytes = checked(ORIGINAL_NATIVE_ROOT / "BUILD_RECIPE.json", limit=2**21)
    actual = json.loads(previous_bytes)
    require(actual == expected and len(actual["reuse"]) == 29 and len(actual["expected_providers"]) == 71,
            "actual .918 recipe is not the independently identified CPU308 + same .855 / same 29 objects")
    binary = checked(ORIGINAL_NATIVE, ORIGINAL_NATIVE_SHA, limit=4*1024**2)
    require(len(binary) == ORIGINAL_NATIVE_BYTES and binary.startswith(b"\x7fELF"), "actual .918 linked predecessor differs")
    return actual, sha256(previous_bytes).hexdigest()


def backend():
    kernel, prior = baseline()
    recipes = prior["backend"]()
    require(recipes["CHANGED"] == HOST and recipes["NATIVE"] == ORIGINAL_NATIVE_ROOT, "old three-TU builder differs")
    recipes.update(CALLER=CALLER, CALLER_PUBLICATION=CALLER_PUBLICATION, CALLER_MANIFEST=CALLER_MANIFEST,
                   caller_sources=caller_sources, NATIVE=ROOT / "candidates" / LABEL,
                   SOURCE_DELTA={unit: (*names, *MORE_INCLUDES[unit]) for unit, names in recipes["SOURCE_DELTA"].items()},
                   LINK_EXPORTS=[*recipes["LINK_EXPORTS"], NEW_EXPORT])
    original_prepare, original_verify = (recipes[name] for name in ("prepare", "verify_link"))

    def prepare(args):
        raw = checked(args.kernel_reconciliation, args.kernel_reconciliation_sha256, limit=16384)
        actual, prior_sha = qualifications(prior, recipes, kernel, json.loads(raw))
        built = original_prepare(args)
        fresh = json.loads(recipes["checked"](recipes["NATIVE"] / "BUILD_RECIPE.json", built["recipe_SHA256"], 2**21))
        require(fresh["reuse"] == actual["reuse"] and fresh["kernel"] == actual["kernel"]
                and fresh["expected_providers"] == actual["expected_providers"]
                and fresh["pinned_system_header_map"] == actual["pinned_system_header_map"]
                and set(fresh["host"]) == set(actual["host"]) == set(HOST),
                "new prefill caller changed the original dependency, object or provider closure")
        return dict(built, original_918_recipe_SHA256=prior_sha)

    def verify_link(recipe_sha):
        verified = original_verify(recipe_sha)
        require(verified["strong_exports"] == 285 and verified["providers"] == 71, "expected 285 exports and 71 providers")
        compare_link_paths(recipes["NATIVE"], recipes, reader_template(prior)["passive"])
        return dict(verified, original_918_library_SHA256=ORIGINAL_NATIVE_SHA)

    recipes.update(prepare=prepare, verify_link=verify_link)
    return kernel, prior, recipes


def reader_template(prior):
    relative = prior["EXPLICIT_READER"]
    path = REPO / relative
    text = checked(path, prior["SOURCE_PINS"][relative]).decode()
    text = once(text, 'linked["strong_exports"] == 284', 'linked["strong_exports"] == 285')
    text = once(text, "strong_exports=284,", "strong_exports=285,")
    return execute(path, text)


def worker():
    kernel, prior, recipes = backend()
    scope = prior["native_worker"]()
    # Adopt the proven compiler scheduler, reaper and single absolute epoch.
    scope.update(__file__=str(HERE / "worker.py"), build=SimpleNamespace(**recipes),
                 LABEL=LABEL, CLAIM=CLAIM, RECIPE_TOOL=HERE / "worker.py", KERNEL_START=kernel.REQUEST)
    validate = scope["validate_claim"]

    def claim(pin):
        record = validate(pin)
        require_request(record["request"])
        require(record["caller_publication"] == CALLER_PUBLICATION
                and record["caller_manifest_SHA256"] == CALLER_MANIFEST and record["new_export"] == NEW_EXPORT,
                "new request, caller or added export differs")
        return record

    scope["validate_claim"] = claim
    return scope


def independent_reader(claim_sha):
    kernel, prior, recipes = backend()
    claim = json.loads(checked(CLAIM, claim_sha, 16384))
    require_request(claim["request"])
    require(claim["new_export"] == NEW_EXPORT, "independent prefill claim differs")
    reader = reader_template(prior)
    legacy_audit = reader["audit"]

    def audit(*args, **kwargs):
        report = legacy_audit(*args, **kwargs)
        if report["state"] == "NATIVE_ABI_AND_DEPENDENCIES_VERIFIED":
            proof = json.loads(reader["read"](args[2], 16384))
            _, original_recipe_sha = qualifications(prior, recipes, kernel, proof, verify_kernel=False)
            linked = recipes["verify_link"](report["recipe_SHA256"])
            require(linked["library_SHA256"] == report["library_SHA256"], "independent passive link differs")
            report.update(original_918_library_SHA256=ORIGINAL_NATIVE_SHA,
                          original_918_recipe_SHA256=original_recipe_sha, passive_provider_count=71,
                          independent_passive_dependency_resolution=True)
        return report

    reader.update(REQUEST=REQUEST, KERNEL_REQUEST=kernel.REQUEST, CLAIM=CLAIM, CLAIM_PIN=claim_sha,
                  explicit=SimpleNamespace(LABEL=LABEL, PLANNED=PLANNED, HOST=HOST, CALLER=CALLER,
                                           backend=lambda: recipes, caller_sources=caller_sources), audit=audit)
    return reader


def main():
    sys.path.insert(0, str(REPO / "tools"))
    if sys.argv[1:2] == ["--audit-compute"]:
        require(len(sys.argv) >= 3, "independent reader requires exact claim SHA")
        claim_sha = sys.argv.pop(2)
        del sys.argv[1]
        independent_reader(claim_sha)["main"]()
        return 0
    return worker()["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
