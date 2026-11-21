"""Read the original explicit-AR three-TU build without running a compiler or loading its DSO."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import subprocess

import prepare_glm474_fresh_kv_explicit_ar_build as explicit
from read_training_slurm_membership import allocated_dcu
from read_vggt_tum_bidirectional30_terminal import TERMINAL


REQUEST = "eca8f3dcb6d3d3938ce50868dfe670fd"
KERNEL_REQUEST = "b9a7f0b78c566911ed40004fc906ce44"
CLAIM = explicit.REPO / "coordination/tasks/glm-r474-fresh-kv-explicit-ar-native-cpu-v1.json"
CLAIM_PIN = "008c9a412b060c21f8e6c60ed41ebcc7e1f4ba0f6eedf7905a503132f5b8a8fe"
PARENT, NODE = "1103606", "f12r2n13"
PLACEMENT = dict(parent=PARENT, node=NODE, zone="O01", cpus=4, memory_MiB=16384, DCU=0)
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
LIMIT = 2**21
SO_LIMIT = 16 * 1024**2


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, limit=LIMIT):
    with Path(path).open("rb") as fd:
        raw = fd.read(limit + 1)
    need(0 < len(raw) <= limit, "missing or oversized original artifact: " + Path(path).name)
    return raw


def terminal_row(raw):
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    need(all(len(row) == len(COLUMNS) for row in rows), "original accounting columns differ")
    matches = [row for row in rows if row[1] == explicit.LABEL]
    need(len(matches) == 1, "expected one original explicit-AR accounting step")
    [row] = matches
    need(re.fullmatch(PARENT + r"\.[0-9]+", row[0]) and row[5] == NODE and row[2].split()
         and row[2].split()[0] in TERMINAL and re.fullmatch(r"[0-9]+:[0-9]+", row[3])
         and row[4].isdecimal() and int(row[4]) <= 1800, "original explicit-AR step is foreign or nonterminal")
    resources = dict(text.split("=", 1) for text in row[6].split(","))
    need(allocated_dcu(row[6]) == 0 and resources.get("cpu") == "4" and resources.get("node") == "1"
         and resources.get("mem") in {"16G", "16384M", "16777216K"}
         and all(value == "0" for key, value in resources.items() if key.startswith("gres/")),
         "original explicit-AR zero-GRES accounting differs")
    return row


def passive(argv):
    raw = subprocess.check_output(argv, stdin=subprocess.DEVNULL, timeout=30)
    need(0 < len(raw) <= LIMIT, "passive ELF tool exceeded output bound")
    return raw.decode("utf-8")


def verified_artifacts(worker, recipe, output, backend, deps, query):
    plan = explicit.PLANNED
    units = tuple(sorted(explicit.HOST))
    lib = output / "libGLM_fresh_kv_pack.so"
    wanted = [(f"{phase}-{unit}", recipe["host"][unit][key])
              for phase, key in (("fresh", "preprocess"), ("compile", "compile")) for unit in units]
    wanted.append(("link", recipe["link"]))
    need([(entry["phase"], entry["argv"]) for entry in worker["commands"]] == wanted
         and all(entry["returncode"] == 0 and entry["owned_child_reaped"] is True for entry in worker["commands"])
         and worker["actual"] == plan and worker["attempted"] == dict(fresh_M=3, HOST_compile=3, links=1),
         "original M/compile/link phase order, reaping or denominators differ")
    fresh = worker["fresh_dependency_evidence"]
    after = worker["compile_dependency_evidence"]
    linked = worker["linked"]
    recipe_pin = worker["recipe_SHA256"]
    need(fresh["phase"] == "fresh" and after["phase"] == linked["phase"] == "actual"
         and all(stage["recipe_SHA256"] == recipe_pin for stage in (fresh, after, linked))
         and after["hosts"] == fresh["hosts"] == linked["hosts"] and set(linked["hosts"]) == set(units)
         and deps(recipe_pin, "actual") == after, "original three independent M/MD closures differ")
    md_pin = {}
    for unit in units:
        target = recipe["host"][unit]["compile"][-1]
        before, actual = (read(output / f"{unit}.{suffix}.d") for suffix in ("before", "actual"))
        first = backend["dependencies"](before, target)
        need(first == backend["dependencies"](actual, target) and len(first) == linked["hosts"][unit]["paths"],
             "original per-unit M/MD payload differs: " + unit)
        md_pin[unit] = dict(before=digest(before), actual=digest(actual))
    binary = read(lib, SO_LIMIT)
    need(len(binary) == linked["library_bytes"] and len(binary) >= 64 and digest(binary) == linked["library_SHA256"]
         and binary[:7] == b"\x7fELF\x02\x01\x01" and struct.unpack_from("<HHI", binary, 16) == (3, 62, 1),
         "original linked x86-64 DSO bytes differ")
    reference = recipe["original_library"]["path"]
    ref_raw = backend["checked"](reference, backend["ORIGIN_LIBRARY_SHA"])
    need(digest(ref_raw) == backend["ORIGIN_LIBRARY_SHA"], "original CPU308 reference ELF differs")
    symbols = [backend["symbol_map"](query(["/usr/bin/nm", "-D", "--defined-only", path]))
               for path in (reference, str(lib))]
    needed = [backend["needed_libraries"](query(["/usr/bin/readelf", "-d", path]))
              for path in (reference, str(lib))]
    additions = dict.fromkeys(backend["LINK_EXPORTS"], "T")
    need(len(symbols[0]) == 281 and not symbols[0].keys() & additions.keys()
         and symbols[1] == {**symbols[0], **additions} and len(symbols[1]) == linked["strong_exports"] == 284
         and needed[0] == needed[1], "explicit decode Native exports or ELF NEEDED differ")
    need(linked["providers"] == len(recipe["expected_providers"]) == 71
         and linked["explicit_Runtime_calls"] == 0 and linked["implicit_ldd_initialization"] == "UNKNOWN",
         "original worker provider scope differs")
    return dict(state="NATIVE_ABI_AND_DEPENDENCIES_VERIFIED", library_SHA256=digest(binary), library_bytes=len(binary),
                dependency_file_SHA256=md_pin, added_exports=sorted(additions), strong_exports=284,
                ELF_NEEDED=needed[1], original_worker_provider_count=71, independent_loader_resolution=False)


def audit(intent_path, accounting_raw, kernel_path, output, backend, *, query=passive, deps=None):
    row = terminal_row(accounting_raw)
    intent_raw = read(intent_path, 16384)
    intent = json.loads(intent_raw)
    epoch = intent["epoch"]
    need(intent["request"] == REQUEST and intent["claim_SHA256"] == CLAIM_PIN and intent["placement"] == PLACEMENT
         and intent["replay_allowed"] is False and type(epoch) is float and math.isfinite(epoch) and epoch > 0
         and intent["deadline"] == epoch + 1800 and intent["work_deadline"] == epoch + 1680,
         "not the original single-epoch three-TU intent")
    result = dict(request=REQUEST, state="ORIGINAL_TERMINAL_UNVERIFIED", accounting_columns=COLUMNS,
                  accounting_row=row, accounting_SHA256=digest(accounting_raw), intent_SHA256=digest(intent_raw),
                  planned=explicit.PLANNED, actual=None, GPU_GO=False, Model_GO=False, replay_allowed=False)
    if row[2:4] != ["COMPLETED", "0:0"]:
        return result
    kernel_raw = read(kernel_path, 16384)
    kernel = json.loads(kernel_raw)
    need(digest(kernel_raw) == intent["kernel_reconciliation_SHA256"] and kernel["request"] == KERNEL_REQUEST
         and kernel["row"][0] == intent["kernel_step"] and kernel["row"][0] != row[0],
         "original gfx936 pack reconciliation differs")
    prior = json.loads(backend["checked"](backend["KERNEL"] / "CPU-result.json", kernel["receipt_SHA256"], 2**20))
    pack = backend["compiled_kernel"](kernel, prior)
    backend["checked"](pack["path"], pack["SHA256"])
    worker_raw = read(output / "CPU-result.json", 2**20)
    worker = json.loads(worker_raw)
    expected = dict(label=explicit.LABEL, request=REQUEST, step=row[0], node=NODE, epoch=epoch, deadline=epoch+1800,
                    work_deadline=epoch+1680, claim_SHA256=CLAIM_PIN, planned=explicit.PLANNED, passed=True,
                    completed=True, phase="complete", GPU_GO=False, Model_GO=False, replay_allowed=False)
    need(all(type(worker.get(k)) is type(v) and worker[k] == v for k, v in expected.items()),
         "original three-TU worker, epoch or phase differs")
    recipe_raw = read(output / "BUILD_RECIPE.json")
    recipe = json.loads(recipe_raw)
    need(digest(recipe_raw) == worker["recipe_SHA256"], "original three-TU recipe bytes differ")
    _, pins = explicit.caller_sources()
    origin = json.loads(backend["checked"](backend["ORIGIN"] / "build-result.json", backend["ORIGIN_BUILD_SHA"], 2**20))
    expected_recipe = backend["make_recipe"](origin, pins, pack)
    plan = json.loads(backend["checked"](explicit.CALLER.parent / "depth-stratified-observer-CPU-v1/PLAN.json",
                                           backend["SYSTEM_PLAN_SHA"]))
    expected_recipe.update(pinned_system_header_map=plan["system_pins"], expected_providers=plan["original169"]["providers"])
    need(recipe == expected_recipe, "original CPU308 recipe, reused objects or provider pins differ")
    for unit in explicit.HOST:
        raw = read(output / f"{unit}.o", SO_LIMIT)
        need(len(raw) >= 64 and raw[:7] == b"\x7fELF\x02\x01\x01" and struct.unpack_from("<HHI", raw, 16) == (1, 62, 1),
             "original HOST relocatable differs: " + unit)
        result.setdefault("fresh_host_object_SHA256", {})[unit] = digest(raw)
    result.update(verified_artifacts(worker, recipe, output, backend, deps or backend["verify_dependencies"], query))
    result.update(worker_SHA256=digest(worker_raw), recipe_SHA256=digest(recipe_raw),
                  kernel_reconciliation_SHA256=digest(kernel_raw), kernel_object_SHA256=pack["SHA256"],
                  actual=explicit.PLANNED)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("intent", "accounting", "kernel-reconciliation"):
        p.add_argument("--" + field, type=Path, required=True)
    args = p.parse_args()
    claim_raw = read(CLAIM, 16384)
    claim = json.loads(claim_raw)
    need(digest(claim_raw) == CLAIM_PIN and claim["request"] == REQUEST
         and claim["planned"] == explicit.PLANNED, "published three-TU claim differs")
    backend = explicit.backend()
    result = audit(args.intent, read(args.accounting), args.kernel_reconciliation, Path(backend["NATIVE"]), backend)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
