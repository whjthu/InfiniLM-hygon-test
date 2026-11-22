"""Rebuild the original TU15 and link it with the two validated width providers."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time


SCOPE = Path(__file__).resolve().parent
OUTPUT = Path("/audit-out")
ORIGINAL = Path("/refactor/candidates/Deep-C4-constant-slots-r482-Native-CPU-v1")
PARENT_RECEIPT_SHA = "1d3938d699393a85ccddc65a6aeb34a687db14447c4b5ba8861386174d029a6b"
PARENT_LIBRARY_SHA = "90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1"
SOURCE_NAME = "deepseek_v4_attention_prepare.cpp"
HEADER_NAME = "inverse_output_width_policy.hpp"
EXPORTS = {"deep_inverse_rope_h1_pack32_v1", "deep_inverse_rope_h1_pack64_v1"}
CXX = "/usr/bin/g++"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def verify(pins):
    for path, digest in pins.items():
        need(sha(path) == digest, f"Frozen input differs: {path}")


def dependencies(path, target):
    line = path.read_text(encoding="utf-8").replace("\\\n", " ").strip()
    lhs, separator, rhs = line.partition(":")
    names = shlex.split(rhs)
    need(separator and shlex.split(lhs) == [str(target)] and names and all(n.startswith("/") for n in names),
         "Generated dependency rule differs")
    return set(names)


def linked_symbols(text):
    return {symbol: kind for kind, symbol in re.findall(r"(?m)^\s*[0-9a-fA-F]+\s+([TDBR])\s+(\S+)\s*$", text)}


def build(claim, report, run):
    receipt_path = ORIGINAL / "CPU-result.json"
    need(sha(receipt_path) == PARENT_RECEIPT_SHA, "Original CPU1113 receipt differs")
    receipt = json.loads(receipt_path.read_bytes())
    need(receipt["actual_CPU"] == "910407.1113" and receipt["passed"] is True and receipt["completed"] is True
         and receipt["all19_MD_equal_fresh_M"] is True and not receipt["unmatched_dependencies"]
         and receipt["library_SHA256"] == PARENT_LIBRARY_SHA, "CPU1113 parent build is unqualified")
    parent = Path(receipt["library"])
    need(parent.parent == ORIGINAL and sha(parent) == PARENT_LIBRARY_SHA, "Parent DSO differs")
    old = claim["original_TU15"]
    objects = dict(receipt["all50_linked_objects_SHA256"])
    need(len(objects) == 50 and objects.pop(old["object"]) == old["object_SHA256"],
         "Original TU15 slot differs")
    verify({old["object"]: old["object_SHA256"], old["source"]: old["source_SHA256"]})
    source, header = SCOPE / SOURCE_NAME, SCOPE / HEADER_NAME
    deps = dict(claim["original_dependencies"])
    need(len(deps) == 986 and deps.pop(old["source"]) == old["source_SHA256"],
         "Original TU15 include set differs")
    deps.update({str(source): claim["sources"][SOURCE_NAME], str(header): claim["sources"][HEADER_NAME]})
    verify(deps)
    report["expected_dependencies"] = len(deps)
    providers = {path: digest for path, digest in claim["providers"].items()}
    need(len(providers) == 2, "Both original device objects are required")
    verify(providers)
    verify(objects)
    resident_providers = receipt["new_transitive_library_SHA256"]
    verify(resident_providers)
    report["original_objects_verified"] = len(objects)

    obj = OUTPUT / "TU15.o"
    before, after = OUTPUT / "TU15.before.d", OUTPUT / "TU15.actual.d"
    prefix = [CXX, "-iquote", old["quote_include_origin"], *old["compiler_prefix"][1:]]
    run("fresh-M", [*prefix, "-M", "-MF", str(before), "-MT", str(obj), str(source)])
    observed = dependencies(before, obj)
    if observed != set(deps):
        report["dependency_added"] = sorted(observed - set(deps))[:16]
        report["dependency_missing"] = sorted(set(deps) - observed)[:16]
        raise ValueError("Fresh M path set differs")
    verify(deps)
    run("compile", [*prefix, "-MD", "-MF", str(after), "-c", str(source), "-o", str(obj)])
    report["actual_host_compiles"] = 1
    need(dependencies(after, obj) == observed, "Actual MD path set differs")
    verify(deps)
    obj_sha = sha(obj)

    [parent_link] = [row["argv"] for row in receipt["commands"]
                     if row["rc"] == 0 and row["argv"][-2:] == ["-o", str(parent)]]
    originals = [word for word in parent_link if word.endswith(".o")]
    need(parent_link[0] == "g++" and len(originals) == len(set(originals)) == 50
         and set(originals) == set(objects) | {old["object"]} and "-Wl,--no-undefined" in parent_link
         and "-Wl,-Bsymbolic" in parent_link
         and max(parent_link.index(word) for word in originals) < next(
             i for i, word in enumerate(parent_link) if word.startswith("-l")
         ), "Original linker order or object closure differs")
    native = OUTPUT / "libDeep_inverse_rope_width_caller.so"
    slot = parent_link.index(old["object"])
    argv = [CXX, *parent_link[1:slot], str(obj), *providers, *parent_link[slot + 1:-1], str(native)]
    run("link", argv)
    report["actual_native_links"] = 1
    symbols = [linked_symbols(run(label, ["/usr/bin/nm", "-D", "--defined-only", str(path)]))
               for label, path in (("parent-nm", parent), ("candidate-nm", native))]
    old_symbols, new_symbols = symbols
    need(old_symbols and not EXPORTS & old_symbols.keys()
         and new_symbols == {**old_symbols, **dict.fromkeys(EXPORTS, "T")},
         "Candidate strong exports differ from parent plus the two width providers")
    tags = run("dynamic", ["/usr/bin/readelf", "-d", str(native)])
    needed = re.findall(r"\(NEEDED\).*Shared library: \[([^]]+)\]", tags)
    need(needed == receipt["NEEDED"] and "SYMBOLIC" in tags, "Candidate dynamic dependencies differ")
    loader = run("loader", ["/usr/bin/ldd", "-r", str(native)])
    paths = set(re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)", loader, re.M))
    need(not re.search(r"not found|undefined symbol", loader, re.I) and paths == resident_providers.keys(),
         "Candidate passive provider closure differs")
    verify({**objects, **providers, **resident_providers, str(obj): obj_sha})
    report.update(passed=True, object_SHA256=obj_sha, library_SHA256=sha(native),
                  library_bytes=native.stat().st_size, original_strong_exports=len(old_symbols),
                  new_exports=sorted(EXPORTS), retained_loader_paths=len(paths),
                  fresh_M_SHA256=sha(before), actual_MD_SHA256=sha(after))


def main():
    claim = json.loads((SCOPE / "CLAIM.json").read_bytes())
    parent, step = (os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    node = socket.gethostname().split(".")[0]
    need((parent, node) == (claim["parent"], claim["node"]) and step.isdecimal()
         and not os.environ.get("SLURM_STEP_GPUS")
         and all(os.environ.get(key) == "-1" for key in
                 ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
         "Unexpected build placement or GPU mask")
    verify(claim["target_tools"])
    deadline = float(os.environ["TUNING_CPU_DEADLINE"])
    need(0 < deadline - time.time() <= 300, "Single build deadline differs")
    report = {"request": claim["request"], "step": parent + "." + step, "node": node, "passed": False,
              "actual_host_compiles": 0, "actual_native_links": 0, "actual_GPU_steps": 0}

    def run(label, command):
        left = min(120, deadline - time.time() - 12)
        need(left > 1, "Original CPU deadline exhausted")
        child = subprocess.run(command, capture_output=True, timeout=left, check=False)
        for suffix, raw in (("stdout", child.stdout), ("stderr", child.stderr)):
            with (OUTPUT / (label + "." + suffix)).open("xb") as stream:
                stream.write(raw)
        need(child.returncode == 0, f"{label} failed with rc {child.returncode}")
        return child.stdout.decode("utf-8", errors="replace")

    try:
        build(claim, report, run)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report.update(error=type(error).__name__, error_summary=str(error)[:200])
    finally:
        with (OUTPUT / "original-result.json").open("x", encoding="utf-8") as out:
            json.dump(report, out, sort_keys=True, allow_nan=False)
            out.write("\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
