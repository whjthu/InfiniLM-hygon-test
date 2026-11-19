"""Audit the original CPU331 component against the providers it actually imports."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "Kimi461-output-residual-component-CPU-v1"
PARENT = "Kimi461-output-residual-tile64-CPU-v1"
STEP = "1103606.331"
COMPONENT = "libKimi_output_residual_component_r461.so"
ORIGINAL = "686ff08996dad428665624e34b84025dbb6dc5414c9780ff74da64ffedcb9f7d"
PARENT_SHA = "6d9421acefdb6091694ffe8429c3c87b90239e00eea9c95afb28a23844cb65e1"
MANIFEST = "35f3b6e5a3e199bcdfa78a38fe98e8b8c8099d69329f5e9d055885b6ef0e6544"
EXT = "_infinilm.cpython-310-x86_64-linux-gnu.so"
RUNTIME = "libinfinilm_runtime.so"
CORE = {
    "ReplicatedLinear constructor": (
        "_ZN8infinilm2nn6LinearC1ERNS_7runtime7RuntimeEmmbRKN6infini2rt8DataTypeESt8optionalIS7_E", EXT),
    "RMSNorm constructor": (
        "_ZN8infinilm6layers7RMSNormC1ERNS_7runtime7RuntimeEmdRKN6infini2rt8DataTypeE", EXT),
    "ReplicatedLinear forward": (
        "_ZNK8infinilm2nn10BaseLinear7forwardERNS_7runtime7RuntimeERNS2_6TensorE", EXT),
    "RMSNorm forward": (
        "_ZNK8infinilm6layers7RMSNorm7forwardERNS_7runtime7RuntimeERKNS2_6TensorE", EXT),
    "weight loading": (
        "_ZN8infinilm7weights12ModelWeights14load_parameterERNS_7runtime7RuntimeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS2_6TensorE", EXT),
    "Runtime creation": ("_ZN8infinilm7runtime7Runtime6createEN6infini2rt6DeviceE", RUNTIME),
    "Runtime synchronization": ("_ZN8infinilm7runtime7Runtime10syncStreamEv", RUNTIME),
}


def need(check, text):
    if not check:
        raise ValueError(text)


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def pin(path):
    return {"bytes": path.stat().st_size, "SHA256": digest_file(path)}


def checked_json(path, digest):
    raw = path.read_bytes()
    need(len(raw) <= 1048576 and hashlib.sha256(raw).hexdigest() == digest, "wrong original receipt")
    return json.loads(raw)


def call(*argv):
    return subprocess.check_output(argv, text=True, timeout=15)


def dependencies(path, target):
    lhs, sep, rhs = path.read_text().replace("\\\n", " ").partition(": ")
    need(sep and shlex.split(lhs) == [target], "wrong original make target")
    names = set(shlex.split(rhs))
    need(bool(names) and all(n.startswith("/") for n in names), "unbound original make dependency")
    return names


def loader_rows(text):
    table = {}
    for line in text.splitlines():
        pieces = line.split()
        if len(pieces) >= 3 and pieces[1] == "=>":
            soname, path = pieces[0], pieces[2]
            need(path.startswith("/") and soname not in table, "ambiguous original loader row")
            table[soname] = path
    need("undefined symbol" not in text.lower() and "not found" not in text.lower(), "original relocation failed")
    return table


def matrix_symbols(path, flag):
    text = call("/usr/bin/nm", "-D", "--format=posix", flag, str(path))
    return {row[0].split("@", 1)[0]: row[1] for line in text.splitlines() if len(row := line.split()) >= 2}


def main():
    cols = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
    rows = [line.split("|") for line in call("sacct", "-nP", "-j", "1103606", "-S",
            "2026-11-19T20:00:00", "-o", cols).splitlines() if line.startswith(STEP + "|")]
    row, = rows
    need(len(row) == 7 and row[1:4] == [LABEL, "FAILED", "1:0"] and "gres/dcu" not in row[-1],
         "wrong original CPU331 accounting")
    d = ROOT / "candidates" / LABEL
    old = checked_json(ROOT / "candidates" / PARENT / "CPU-result.json", PARENT_SHA)
    receipt = checked_json(d / "CPU-result.json", ORIGINAL)
    need(receipt["actual_CPU"] == STEP and receipt["manifest_SHA256"] == MANIFEST
         and receipt["completed"] is True and receipt["passed"] is False
         and receipt["reason"] == "component has no bound CPU330 model provider"
         and receipt["device_calls"] == receipt["weights_read"] == 0
         and receipt["Native_dlopen"] is False and old["passed"] is True, "original qualification differs")
    for n, record in receipt["logs"].items():
        need(pin(d / (n + ".log")) == record, "original build log changed")
    compile, = [x for x in receipt["commands"] if "-c" in x]
    link, = [x for x in receipt["commands"] if "-shared" in x]
    target = compile[compile.index("-o") + 1]
    projected = dependencies(d / "before.d", target)
    actual = dependencies(d / "actual.d", target)
    need(projected == actual and "-Wl,--no-undefined" in link
         and "-Wl,-Bsymbolic" in link and target in link, "wrong original compile/link closure")
    library = d / COMPONENT
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", call("/usr/bin/readelf", "-dW", str(library)))
    table = loader_rows((d / "passive-relocations.log").read_text())
    need(EXT in needed and RUNTIME in needed and Path(old["candidate"]).name not in needed,
         "wrong actual direct providers")
    # Check the participating project libraries against CPU330's measured
    # complete loader closure. The original model DSO itself was not a provider.
    core_paths = {}
    project = {}
    for soname, location in table.items():
        if not location.startswith("/refactor/"):
            continue
        host = ROOT / Path(location).relative_to("/refactor")
        actual_hash = digest_file(host)
        need(actual_hash == old["actual_passive_providers"].get(location),
             "component loader selected a different project library")
        project[soname] = {"path": location, "SHA256": actual_hash}
        core_paths[soname] = matrix_symbols(host, "--defined-only")

    imports = matrix_symbols(library, "--undefined-only")
    exports = matrix_symbols(library, "--defined-only")
    for sym in ("kimi_output_residual_component_r461", "kimi_output_residual_error_r461"):
        need(exports.get(sym) == "T", "component entry not exported")
    associations = {}
    for name, (symbol, expected) in CORE.items():
        owners = [so for so, defs in core_paths.items() if defs.get(symbol) in ("T", "W")]
        need(imports.get(symbol) == "U" and owners == [expected], f"unbound component symbol: {name}")
        associations[name] = expected
    print(json.dumps({
        "original_accounting": row, "original_receipt_SHA256": ORIGINAL, "original_guard_failed": True,
        "artifact_passive_qualification": True, "original_step_status_changed": False,
        "dependency_entries": len(actual), "provider_count": len(table), "project_providers": project,
        "symbol_providers": associations, "component": pin(library), "object": pin(d / "component.o"),
        "original_parent_model_DSO_imported": False, "Native_dlopen": False, "device_calls": 0, "weights_read": 0,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
