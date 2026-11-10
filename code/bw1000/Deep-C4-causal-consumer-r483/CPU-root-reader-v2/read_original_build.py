"""Verify CPU-v2 launch identity and terminal accounting before reading build outputs."""

import math
import re
import hashlib
import json
import shlex
import struct
import sys

LABEL = "Deep-C4-causal-consumer-r483-CPU-v2"
PUBLICATION = "730af6c4130270dd7e65122e3c6e5e41fd79741f"
SOURCE_MANIFEST = "cf54f255ed4c88a23a5b7402e18092e06394eb648666ed0cd004b4bd70e3ea64"
CLAIM = "0700705e6958fef6cad66cc29c2b0fd38c9f0c1ec6910d7b8224a369c4f8b0db"
COLUMNS = ("JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw")
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def verify_terminal(intent, launch, accounting, expected_job):
    """Return exact terminal evidence. Raise on missing, ambiguous or live accounting."""
    if not re.fullmatch(r"910407\.[1-9][0-9]*", expected_job):
        raise ValueError("exact observed CPU-v2 step is required")
    for key, value in {
        "publication": PUBLICATION, "source_manifest_SHA256": SOURCE_MANIFEST, "claim_SHA256": CLAIM,
        "once": True, "GPU": False, "Native_dlopen": False,
    }.items():
        if type(intent.get(key)) is not type(value) or intent[key] != value:
            raise ValueError(f"CPU-v2 intent {key} differs")
    epoch = intent.get("epoch")
    end = intent.get("deadline")
    observed = launch.get("observed_at_epoch")
    if any(type(x) not in (int, float) or not 0 < x < math.inf for x in (epoch, end, observed)):
        raise ValueError("CPU-v2 epoch fields must be finite positive numbers")
    if end - epoch != 300 or not epoch <= observed < end:
        raise ValueError("CPU-v2 epoch does not contain the launch observation")
    if (launch.get("epoch") != epoch or launch.get("deadline") != end
            or not isinstance(intent.get("HOST"), str) or not intent["HOST"]
            or launch.get("HOST") != intent["HOST"] or launch.get("GPU") is not False):
        raise ValueError("CPU-v2 launch differs from original intent")
    pid = launch.get("owned_srun_PID")
    ticks = launch.get("startticks")
    if (type(pid) is not int or pid <= 0 or type(launch.get("PGID")) is not int
            or launch["PGID"] != pid or not isinstance(ticks, str)
            or not re.fullmatch(r"[1-9][0-9]*", ticks)):
        raise ValueError("CPU-v2 owned srun identity differs")
    argv = intent.get("argv")
    required = {
        "--jobid=910407", "--job-name=" + LABEL, "-c4", "--mem=8192M", "--gres=none",
        "a14r2n20", "00:05:00", "TUNING_CPU_CLAIM_SHA=" + CLAIM,
        "TUNING_CPU_EPOCH=" + repr(epoch), "TUNING_CPU_DEADLINE=" + repr(end),
    }
    if (not isinstance(argv, list) or not argv or argv[0] != "srun"
            or any(not isinstance(arg, str) for arg in argv) or not required <= set(argv)):
        raise ValueError("CPU-v2 original resource/deadline argv differs")
    if tuple(accounting.get("columns", ())) != COLUMNS or type(accounting.get("rc")) is not int or accounting["rc"] != 0:
        raise ValueError("explicit successful Slurm accounting columns are required")
    lines = [line.split("|") for line in accounting["stdout"].splitlines() if line]
    rows = [row for row in lines if row[0] == expected_job or (len(row) > 1 and row[1] == LABEL)]
    if len(rows) != 1 or len(rows[0]) != len(COLUMNS):
        raise ValueError("CPU-v2 exact step accounting missing or ambiguous")
    handle, label, state, exit_code, elapsed = rows[0]
    if handle != expected_job or label != LABEL:
        raise ValueError("CPU-v2 accounting step or label differs")
    normalized = "CANCELLED" if re.fullmatch(r"CANCELLED by [0-9]+", state) else state
    if normalized not in TERMINAL:
        raise ValueError("CPU-v2 terminal state is unconfirmed")
    if not re.fullmatch(r"[0-9]+:[0-9]+", exit_code) or not re.fullmatch(r"[0-9]+", elapsed):
        raise ValueError("CPU-v2 accounting exit or elapsed is invalid")
    return dict(job=handle, state=state, exit_code=exit_code, elapsed_s=int(elapsed),
                original_epoch=epoch, original_deadline=end,
                can_inspect_successful_build=state == "COMPLETED" and exit_code == "0:0")


BUILD_PLAN_SHA256 = "587a51c72b7bfb3342676a90b73ffa7cab84bdbfd9c8271ff46d23048fea5082"
PACKAGE = "/refactor/incoming/Deep-C4-causal-consumer-r483/CPU-v2"
OUTPUT = "/refactor/candidates/" + LABEL


def read_build(intent, launch, accounting, expected_job, receipt_sha, read):
    """Read exact original files through a read-bytes callback after terminal accounting."""
    if sys.flags.optimize:
        raise RuntimeError("passive ELF evidence requires assertions enabled")
    terminal = verify_terminal(intent, launch, accounting, expected_job)
    if not terminal["can_inspect_successful_build"]:
        return {"accounting": terminal, "build_provenance_checked": False}
    if not isinstance(receipt_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha):
        raise ValueError("the original CPU-v2 receipt digest is required")

    def checked(path, digest):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"missing evidence digest: {path}")
        data = read(path)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"evidence bytes differ: {path}")
        return data

    q = json.loads(checked(OUTPUT + "/CPU-result.json", receipt_sha))
    expected = dict(
        actual_CPU=expected_job, completed=True, passed=True, CPU_only=True, new_HOST=1, new_GPU=0,
        Native_dlopen=False, Runtime=False, SDK_explicit_calls=0, GPU_job=False, weights=False,
        Torch_imported=False, Model_qualified=False, all_fresh_M_before_compile=True,
        all_fresh_M_equal_MD=True, unmatched_dependencies=[], project_maps=[],
        source_manifest_SHA256=SOURCE_MANIFEST, claim_SHA256=CLAIM,
        original_epoch=terminal["original_epoch"], original_deadline=terminal["original_deadline"],
    )
    for field, value in expected.items():
        if type(q.get(field)) is not type(value) or q[field] != value:
            raise ValueError(f"CPU-v2 build receipt field differs: {field}")
    elapsed = q.get("elapsed_s")
    if type(elapsed) not in (int, float) or not 0 <= elapsed <= 270:
        raise ValueError("CPU-v2 build receipt elapsed outside inside limit")

    manifest = json.loads(checked(PACKAGE + "/SOURCE_MANIFEST.json", SOURCE_MANIFEST))
    checked(PACKAGE + "/CPU_RESOURCE_CLAIM.json", CLAIM)
    plan = json.loads(checked(PACKAGE + "/BUILD_PLAN.json", BUILD_PLAN_SHA256))
    for name, digest in manifest["files_SHA256"].items():
        checked(PACKAGE + "/" + name, digest)

    compiled = q["compiled"]
    fresh = q["fresh_dependencies_SHA256"]
    if set(compiled) != {"probe"} or set(fresh) != {"probe"}:
        raise ValueError("CPU-v2 must contain exactly one probe translation unit")
    unit = compiled["probe"]
    deps = fresh["probe"]
    source = PACKAGE + "/native-src/probe.cpp"
    header = PACKAGE + "/native-src/input_guard.hpp"
    obj = OUTPUT + "/probe.o"
    md = OUTPUT + "/probe.d"
    if (unit["source"], unit["object"], unit["MD"]) != (source, obj, md):
        raise ValueError("CPU-v2 object, source or MD path differs")
    for path in (source, header):
        expected_sha = manifest["files_SHA256"][path.removeprefix(PACKAGE + "/")]
        if deps.get(path) != expected_sha:
            raise ValueError("CPU-v2 source or guarded header missing from fresh M")
    if unit["source_SHA256"] != deps[source]:
        raise ValueError("CPU-v2 compiled source digest differs")
    if (plan["class_header_SHA256"] not in deps.values()
            or plan["forbidden_old_class_header_SHA256"] in deps.values()):
        raise ValueError("CPU-v2 class header closure differs")

    def read_make_dependencies(data):
        target, separator, paths = data.decode().replace(chr(92) + chr(10), " ").partition(":")
        if not separator or shlex.split(target) != [obj]:
            raise ValueError("CPU-v2 dependency target differs")
        return set(shlex.split(paths))

    before = read(OUTPUT + "/probe.before.d")
    md_bytes = checked(md, unit["MD_SHA256"])
    if read_make_dependencies(before) != set(deps) or read_make_dependencies(md_bytes) != set(deps):
        raise ValueError("CPU-v2 fresh M and actual MD closure differ")
    checked(obj, unit["object_SHA256"])
    libraries = q["libraries"]
    if set(libraries) != {"probe"}:
        raise ValueError("CPU-v2 library receipt differs")
    reported = libraries["probe"]
    item, = plan["links"]
    library_path = OUTPUT + "/" + item["output"]
    if reported["path"] != library_path or reported["all_linked_object_SHA256"] != {obj: unit["object_SHA256"]}:
        raise ValueError("CPU-v2 linked artifact path or object bytes differ")
    probe = checked(library_path, reported["SHA256"])
    logs = []
    for stream in ("stdout", "stderr"):
        name = "probe-ldd." + stream
        log = q["logs"][name]
        path = OUTPUT + "/" + name
        if log["path"] != path:
            raise ValueError("CPU-v2 original loader log path differs")
        data = checked(path, log["SHA256"])
        if type(log.get("bytes")) is not int or len(data) != log["bytes"]:
            raise ValueError("CPU-v2 original loader log byte length differs")
        logs.append(data.decode())
    links = [command["argv"] for command in q["commands"] if command["name"] == "probe-link"]
    if len(links) != 1:
        raise ValueError("CPU-v2 exact original link invocation required")
    linkage = verify_probe_link(plan, reported, probe, chr(10).join(logs), obj, links[0])
    return dict(accounting=terminal, build_provenance_checked=True, receipt_SHA256=receipt_sha,
                original_M_first_observed_SHA256=hashlib.sha256(before).hexdigest(),
                MD_SHA256=unit["MD_SHA256"], object_SHA256=unit["object_SHA256"],
                guarded_header_SHA256=deps[header], dependency_count=len(deps),
                probe_SHA256=reported["SHA256"], **linkage, GPU_GO=False, Model_GO=False)


def elf_bytes(b):
 assert b[:6]==b'\x7fELF\x02\x01';off=struct.unpack_from('<Q',b,40)[0];stride,n=struct.unpack_from('<HH',b,58);assert stride==64;sects=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*stride)for i in range(n)]
 def string(sec,i):a=sec[4]+i;return b[a:b.index(0,a)].decode()
 symbols={};needed=[];symbolic=False
 for sec in sects:
  if sec[1]==11:
   assert sec[9]==24;strings=sects[sec[6]]
   for pos in range(sec[4],sec[4]+sec[5],24):
    name,info,other,section,value,size=struct.unpack_from('<IBBHQQ',b,pos)
    if name:symbols[string(strings,name)]={'binding':info>>4,'type':info&15,'section':section,'value':value,'size':size}
  if sec[1]==6:
   strings=sects[sec[6]]
   for pos in range(sec[4],sec[4]+sec[5],16):
    tag,value=struct.unpack_from('<qQ',b,pos)
    if tag==1:needed.append(string(strings,value))
    if tag==16 or(tag==30 and value&2):symbolic=True
 return symbols,needed,symbolic


def verify_probe_link(plan, reported, probe_bytes, ldd_text, obj, link_argv):
    item, = plan["links"]
    symbols, needed, symbolic = elf_bytes(probe_bytes)
    native = item["required_NEEDED"]
    allowed = set(item["allowed_other_NEEDED"]) | {native}
    if (not symbolic or needed != reported["NEEDED"] or needed.count(native) != 1
            or not set(needed) <= allowed or len(needed) != len(set(needed))):
        raise ValueError("independent ELF NEEDED or SYMBOLIC differs")
    expected_exports = set(item["required_exports"])
    found_exports = {
        name for name, entry in symbols.items()
        if name.startswith("deep_c4_probe_") and entry["binding"] == 1
        and entry["type"] == 2 and entry["section"] != 0
    }
    if len(expected_exports) != 13 or found_exports != expected_exports:
        raise ValueError("CPU-v2 exact 13 probe exports differ")
    if "not found" in ldd_text or "undefined symbol" in ldd_text:
        raise ValueError("original passive loader evidence reported a failure")
    observed = set(re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)", ldd_text, re.M))
    providers = reported["all_transitive_resolved_library_SHA256"]
    if len(observed) != 69 or observed != set(providers) or providers != plan["dynamic_SHA256"]:
        raise ValueError("CPU-v2 original 69-library path/hash closure differs")
    lib_options = [i for i, arg in enumerate(link_argv) if arg.startswith("-l")]
    if ([arg for arg in link_argv if arg.endswith(".o")] != [obj] or link_argv.count(native) != 1
            or not lib_options or not link_argv.index(obj) < link_argv.index(native) < min(lib_options)
            or "-Wl,--no-undefined" not in link_argv):
        raise ValueError("CPU-v2 original link object/provider order differs")
    return dict(exports=sorted(found_exports), needed=needed, transitive_count=len(observed),
                passive_ELF_verified=True)


def main():
    """Read original CPU-v2 evidence in place on the login host."""
    import argparse
    from pathlib import Path, PurePosixPath

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--expected-step", required=True)
    parser.add_argument("--receipt-sha", help="Original successful CPU-result.json SHA256")
    args = parser.parse_args()

    root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014").resolve(strict=True)
    def read_original(container_path):
        path = PurePosixPath(container_path)
        if path.parts[:2] != ("/", "refactor") or ".." in path.parts:
            raise ValueError("CPU-v2 evidence path outside the original root")
        host = root.joinpath(*path.parts[2:]).resolve(strict=True)
        if not host.is_relative_to(root):
            raise ValueError("CPU-v2 evidence target outside the original root")
        with host.open("rb") as file:
            data = file.read(32 * 1024**2 + 1)
        if len(data) > 32 * 1024**2:
            raise ValueError("CPU-v2 evidence exceeds the per-file read limit")
        return data

    with args.accounting.open("rb") as file:
        accounting_bytes = file.read(1024**2 + 1)
    if len(accounting_bytes) > 1024**2:
        raise ValueError("accounting exceeds the evidence read limit")
    intent_bytes = read_original("/refactor/results/" + LABEL + "-intent.json")
    launch_bytes = read_original("/refactor/results/" + LABEL + "-launch.json")
    result = read_build(json.loads(intent_bytes), json.loads(launch_bytes),
                        json.loads(accounting_bytes), args.expected_step, args.receipt_sha, read_original)
    result["input_evidence_SHA256"] = {
        name: hashlib.sha256(raw).hexdigest() for name, raw in
        (("intent", intent_bytes), ("launch", launch_bytes), ("accounting", accounting_bytes))
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
