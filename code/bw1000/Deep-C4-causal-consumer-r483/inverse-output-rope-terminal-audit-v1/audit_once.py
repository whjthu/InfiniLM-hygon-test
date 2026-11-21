"""Independently read one existing Deep inverse RoPE compiler outcome."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import shlex
import struct


REPO = Path(__file__).resolve().parents[4]
PACKAGE = Path("code/bw1000/Deep-C4-causal-consumer-r483/inverse-output-rope-CPU-v1")
CLAIM = Path("coordination/tasks/deep-inverse-output-rope-gfx936-host-compile-v1.json")
ELF_HELPER = Path("code/bw1000/GLM-C1-layer-observer-r474/routed-i128-cpu-artifact-audit-v1/audit_once.py")
CLAIM_SHA = "96d5da415b35a3f2b615667103ebdc897bdd56a11cf27fb3315ac038388a536b"
MANIFEST = "fb0edd19695838a6df632e88ae10351569362de28ed11067848bb909b3aa7465"
ELF_SHA = "5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46"
PARENT, NODE = "1103606", "f12r2n12"
LABEL = "Deep-inverse-output-rope-gfx936-CPU-v1"
REQUEST = "ed7f8bd81af545252986bd3d25810f73"
EXPORTS = {"deep_inverse_rope_h1_v1", "deep_inverse_rope_h2_v1"}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED", "SPECIAL_EXIT"}
BASE = f"/refactor/candidates/{LABEL}/inverse_output_rope"
SOURCE = f"/refactor/incoming/{LABEL}/inverse_output_rope.cu"
LIMIT = 2 * 1024 * 1024


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit=LIMIT, empty=False):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require((empty or len(raw)) and len(raw) <= limit, "ARTIFACT_SIZE_BOUND")
    return raw


def checked(path, expected, limit=LIMIT, empty=False):
    raw = read(path, limit, empty)
    require(sha(raw) == expected, f"ARTIFACT_SHA_MISMATCH: {path.name}")
    return raw


def load_scope(repo):
    claim = json.loads(checked(repo / CLAIM, CLAIM_SHA))
    manifest = json.loads(checked(repo / PACKAGE / "SOURCE_FILES.json", MANIFEST))["files_SHA256"]
    require(claim["request"] == REQUEST and claim["label"] == LABEL and claim["package_manifest_SHA256"] == MANIFEST
            and claim["placement"] == {"parent": PARENT, "node": NODE, "zone": "O01", "cpus": 4,
                                       "memory_MiB": 8192, "DCU": 0} and claim["wall_seconds"] == 300
            and claim["replay_allowed"] is False and claim["GPU_GO"] is False, "FROZEN_SCOPE")
    for name, expected in manifest.items():
        require(Path(name).name == name, "FROZEN_SOURCE_NAME")
        checked(repo / PACKAGE / name, expected)
    plan = json.loads(read(repo / PACKAGE / "PLAN.json"))
    require(plan["CPU"] == 4 and plan["memory_MiB"] == 8192 and plan["Slurm_seconds"] == 300
            and plan["GRES"] == "none" and plan["source_SHA256"] == manifest["inverse_output_rope.cu"]
            and {row["export"] for row in plan["variants"]} == EXPORTS and len(plan["variants"]) == 2
            and plan["compiler_prefix"] == ["/opt/dtk/cuda/cuda/bin/nvcc", "-std=c++17", "-O3",
                                            "-arch=gfx936", "-Xcompiler=-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1"],
            "FROZEN_BUILD_PLAN")
    helper_path = repo / ELF_HELPER
    checked(helper_path, ELF_SHA)
    return plan, runpy.run_path(str(helper_path))


def exact_row(accounting):
    selected = []
    for line in accounting.decode("utf-8").splitlines():
        if not line:
            continue
        fields = line.removesuffix("|").split("|")
        require(len(fields) == len(COLUMNS), "ACCOUNTING_COLUMNS")
        if fields[1] == LABEL:
            selected.append(fields)
    require(len(selected) == 1, "EXACT_ORIGINAL_NUMBERED_STEP")
    row, = selected
    step, _, state, exit_code, elapsed, node, tres = row
    states = state.split()
    parts = [item.split("=", 1) for item in tres.split(",")] if tres else []
    require(all(len(item) == 2 for item in parts) and len(dict(parts)) == len(parts), "ACCOUNTING_TRES_FORMAT")
    alloc = dict(parts)
    require(states and states[0] in TERMINAL and re.fullmatch(PARENT + r"\.[0-9]+", step)
            and node == NODE and elapsed.isdecimal() and int(elapsed) <= 300 and re.fullmatch(r"\d+:\d+", exit_code)
            and alloc.get("cpu") == "4" and alloc.get("node") == "1"
            and alloc.get("mem") in {"8G", "8192M", "8388608K"} and not any(key.startswith("gres/") for key in alloc),
            "ORIGINAL_TERMINAL_RESOURCE_SCOPE")
    return row


def dependency_paths(raw, target):
    lhs, colon, rhs = raw.decode("utf-8").replace("\\\n", " ").partition(":")
    names = set(shlex.split(rhs))
    require(lhs.strip() == target and colon and names and all(name.startswith("/") for name in names),
            "EXACT_MAKE_DEPENDENCY_TARGET")
    return names


def own_host_functions(data):
    """Read ELF64 little-endian .symtab directly from the immutable object."""
    off = struct.unpack_from("<Q", data, 40)[0]
    width, count = struct.unpack_from("<HH", data, 58)
    require(width == 64 and 0 < count < 4096 and off >= 64 and off + width * count <= len(data),
            "HOST_ELF_SECTION_TABLE")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, off + width * i) for i in range(count)]

    def payload(section):
        start, size = section[4:6]
        require(start <= len(data) and size <= len(data) - start, "HOST_ELF_SECTION_PAYLOAD")
        return data[start:start + size]

    declared = set()
    for section in sections:
        if section[1] != 2:
            continue
        link = section[6]
        require(link < count and sections[link][1] == 3 and section[9] == 24 and section[5] % 24 == 0,
                "HOST_ELF_SYMBOL_LAYOUT")
        table, symbols = payload(sections[link]), payload(section)
        for start in range(0, len(symbols), 24):
            name_idx, info, visibility, idx, value, size = struct.unpack_from("<IBBHQQ", symbols, start)
            if name_idx == 0:
                continue
            end = table.find(b"\0", name_idx)
            require(name_idx < len(table) and end >= name_idx, "HOST_ELF_SYMBOL_NAME")
            name = table[name_idx:end]
            if not name.startswith(b"deep_inverse_rope_"):
                continue
            require(info == 0x12 and visibility == 0 and 0 < idx < count and sections[idx][1] == 1
                    and (sections[idx][2] & 0x4) and size > 0 and value + size <= sections[idx][5],
                    "HOST_ELF_DEFINED_STRONG_FUNCTION")
            decoded = name.decode("ascii")
            require(decoded not in declared, "HOST_ELF_DUPLICATE_FUNCTION")
            declared.add(decoded)
    require(declared == EXPORTS, "HOST_ELF_EXACT_TWO_WRAPPERS")
    return sorted(declared)


def audit(intent_path, controller_path, accounting_path, build, producer_path, repo=REPO):
    plan, helpers = load_scope(repo)
    raw_accounting = read(accounting_path)
    row = exact_row(raw_accounting)
    raw_intent, raw_ctrl = read(intent_path), read(controller_path)
    intent, controller = json.loads(raw_intent), json.loads(raw_ctrl)
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["source_manifest_SHA256"] == MANIFEST
            and intent["claim_SHA256"] == CLAIM_SHA and intent["parent"] == PARENT and intent["node"] == NODE
            and type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0 and intent["deadline"] == epoch + 300
            and intent["replay_allowed"] is False, "EXACT_ORIGINAL_INTENT")
    require(controller["request"] == REQUEST and controller["accounting_columns"] == COLUMNS
            and controller["actual_GPU_steps"] == 0, "EXACT_CONTROLLER_SCOPE")
    if controller.get("terminal_visible") is True:
        require(controller["row"] == row and controller["observations"][-1] == row, "CONTROLLER_TERMINAL_CONFLICT")
    else:
        require(controller["state"] == "UNKNOWN", "CONTROLLER_UNKNOWN_MISMATCH")

    result = dict(request=REQUEST, accounting_columns=COLUMNS, accounting_row=row,
                  accounting_SHA256=sha(raw_accounting), intent_SHA256=sha(raw_intent),
                  controller_SHA256=sha(raw_ctrl), original_controller_state=controller["state"],
                  candidate_ids=[row["id"] for row in plan["variants"]], planned_candidates=2,
                  qualified_compiler_objects=0, device_numerical_qualified=False,
                  device_ISA_independently_verified=False, passed=False)
    if row[2:4] != ["COMPLETED", "0:0"]:
        require(controller["state"] != "COMPILE_VERIFIED", "ORIGINAL_FAILURE_CONFLICT")
        return dict(result, outcome="ORIGINAL_NONZERO_TERMINAL")

    receipt_raw = read(build / "CPU-result.json")
    receipt = json.loads(receipt_raw)
    expected = dict(label=LABEL, manifest_SHA256=MANIFEST, original_step=row[0], node=NODE, epoch=epoch,
                    deadline=epoch + 300, GRES="none", planned_device_TUs=1, actual_device_TUs=1,
                    actual_GPU_steps=0, explicit_Runtime_calls=0, completed=True, passed=True,
                    device_qualified=False, exports=sorted(EXPORTS), object=BASE + ".o")
    require(all(type(receipt.get(key)) is type(value) and receipt[key] == value
                for key, value in expected.items()), "EXACT_WORKER_SCOPE")
    if controller["state"] == "COMPILE_VERIFIED":
        require(controller["receipt_SHA256"] == sha(receipt_raw)
                and controller["object_SHA256"] == receipt["object_SHA256"], "CONTROLLER_WORKER_CONFLICT")
    argv = plan["compiler_prefix"]
    commands = {
        "fresh-M": argv + ["-M", "-MF", BASE + ".before.d", "-MT", BASE + ".o", SOURCE],
        "compile": argv + ["-MD", "-MF", BASE + ".d", "-MT", BASE + ".o", "-c", SOURCE, "-o", BASE + ".o"],
        "symbols": ["/usr/bin/nm", "-g", "--defined-only", BASE + ".o"],
    }
    require([r["label"] for r in receipt["commands"]] == list(commands), "EXACT_COMPILER_COMMAND_SEQUENCE")
    logs = {}
    for record in receipt["commands"]:
        label = record["label"]
        require(record["returncode"] == 0 and record["argv"] == commands[label], "EXACT_GFX936_ARGUMENTS")
        for stream in ("stdout", "stderr"):
            logs[label, stream] = checked(build / f"{label}.{stream}", record[stream + "_SHA256"], empty=True)

    builds = receipt["dependencies_SHA256"]
    before = checked(build / "inverse_output_rope.before.d", receipt["fresh_M_SHA256"])
    actual = checked(build / "inverse_output_rope.d", receipt["actual_MD_SHA256"])
    before_set, actual_set = (dependency_paths(raw, BASE + ".o") for raw in (before, actual))
    require(before_set == actual_set == set(builds) and {SOURCE, plan["cuda_runtime_header"]} <= before_set,
            "INDEPENDENT_M_MD_EQUALITY")
    producer_raw = checked(producer_path, plan["producer_SHA256"])
    producer = json.loads(producer_raw)
    maps = producer["fresh_dependencies_SHA256"]
    require(len(producer_raw) == plan["producer_bytes"] and producer["actual_CPU"] == "910407.1113"
            and producer["completed"] is True and producer["passed"] is True and len(maps) == 19
            and producer["all19_fresh_M_before_compile"] is True and producer["all19_MD_equal_fresh_M"] is True,
            "EXACT_PROVEN_DEPENDENCY_PRODUCER")
    allowed = {SOURCE: plan["source_SHA256"]}
    for deps in maps.values():
        for name, digest in deps.items():
            require(allowed.setdefault(name, digest) == digest, "PRODUCER_SHA_CONFLICT")
    require(all(allowed.get(name) == digest for name, digest in builds.items())
            and builds[plan["cuda_runtime_header"]] == plan["cuda_runtime_header_SHA256"],
            "INDEPENDENT_PRODUCER_DEPENDENCY_CLOSURE")

    data = checked(build / "inverse_output_rope.o", receipt["object_SHA256"], 16 * 1024**2)
    helpers["host_elf"](data)
    exports = own_host_functions(data)
    nm_exports = {parts[-1] for line in logs["symbols", "stdout"].decode("utf-8").splitlines()
                  if len(parts := line.split()) >= 3 and parts[-2] == "T"
                  and parts[-1].startswith("deep_inverse_rope_")}
    require(nm_exports == set(exports), "ELF_AND_NM_EXPORTS_DISAGREE")
    images = helpers["embedded_amdgpu"](data)
    return dict(result, outcome="ORIGINAL_GFX936_COMPILER_OBJECT_VERIFIED", passed=True,
                qualified_compiler_objects=1, worker_SHA256=sha(receipt_raw), object_SHA256=sha(data),
                object_bytes=len(data), host_exports=exports, dependency_count=len(builds),
                producer_SHA256=sha(producer_raw), embedded_uncompressed_AMDGPU_images=images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intent", "controller", "accounting", "build-dir", "producer"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.intent, args.controller, args.accounting, args.build_dir, args.producer),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
