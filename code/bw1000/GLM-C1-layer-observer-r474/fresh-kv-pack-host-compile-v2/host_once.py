"""Verify all reviewed bytes before admitting the distinct GLM fresh-KV v2 scope."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v2"
SOURCE = HERE.parent / "fresh-kv-pack-CPU-v1"
LEGACY = HERE.parent / "fresh-kv-pack-host-compile-v1"
STAGE = HERE / "compile-stage"
FAMILY = HERE.parent.relative_to(REPO).as_posix()
PREDECESSOR = "b9a7f0b78c566911ed40004fc906ce44"
SOURCE_INDEX_SHA = "a5e43581817e7fad431440bfa79ff3eba3708dec095dc05b76675b6d7388d259"
OLD_CLAIM_SHA = "005a011e94cbe1fa5f6f11a735d1deb1afd9ee88656632656b48e091c576a5f9"
OLD_HOST_SHA = "490c8d242b28e9a1cbe7eb8e4c15c4bdcf0b9534c953c77f43cfb877d6341db4"
OLD_COMPILER_SHA = "d598ee49072a880a0c2a07f1cc7043e8e4d28f406bfe9a5e1352bb14e8723280"
KERNEL_SHA = "2ed3e3e021546bd7fc56225c18c59cee51b621af252e36f3de06bb2a0da342c1"
PLACEMENT = dict(parent="1103606", node="f12r2n13", zone="O01", cpus=4, memory_MiB=8192, DCU=0)
SOURCE_FILES = {"LOCAL_RESULT.json", "PLAN.json", "README.zh-CN.md", "check_local.cpp", "check_local.py",
                "fresh_kv_pack.cu", "prepare_target.py"}
STAGE_FILES = {"SOURCE_FILES.json", "PLAN.json", "compile_once.py", "fresh_kv_pack.cu"}
UPSTREAM = "coordination/tasks/glm-r474-fresh-kv-pack-gfx936-cpu-v1.json"
UPSTREAM_SHA = "6c20deb38e5a726501f044588afc24047a07fd76e3701fe9f2104694f9da2723"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, wanted):
    need(path.resolve(strict=True) == path, "source must have its exact non-symlink path")
    with path.open("rb") as stream:
        raw = stream.read(256 * 1024 + 1)
    need(0 < len(raw) <= 256 * 1024 and digest(raw) == wanted, f"frozen bytes differ: {path.name}")
    return raw


def package(claim_sha):
    require_digest = re.fullmatch(r"[0-9a-f]{64}", claim_sha)
    need(require_digest is not None, "independent v2 HOST claim SHA required")
    claim = json.loads(pin(HERE / "HOST_CLAIM.json", claim_sha))
    manifest = claim["stage_manifest_SHA256"]
    expected_request = digest(f"{LABEL}:{PLACEMENT['parent']}:{PLACEMENT['node']}:{manifest}".encode())[:32]
    need(claim["label"] == LABEL and claim["request"] == expected_request != PREDECESSOR
         and claim["placement"] == PLACEMENT and claim["predecessor"] == PREDECESSOR
         and claim["planned_device_translation_units"] == 1 and claim["GPU_GO"] is False
         and claim["replay_allowed"] is False, "v2 claim scope differs")
    source_index = json.loads(pin(SOURCE / "SOURCE_FILES.json", SOURCE_INDEX_SHA))["files_SHA256"]
    need(source_index.keys() == SOURCE_FILES, "frozen source package index differs")
    original_claim = json.loads(pin(LEGACY / "CLAIM.json", OLD_CLAIM_SHA))
    original_host = pin(LEGACY / "host_once.py", OLD_HOST_SHA)
    old_plan = json.loads(pin(SOURCE / "PLAN.json", source_index["PLAN.json"]))
    expect = {f"{FAMILY}/{SOURCE.name}/{name}": sha for name, sha in source_index.items()}
    expect.update(original_claim["shared_files"])
    expect.update({f"{FAMILY}/{SOURCE.name}/SOURCE_FILES.json": SOURCE_INDEX_SHA,
                   f"{FAMILY}/{LEGACY.name}/host_once.py": OLD_HOST_SHA,
                   f"{FAMILY}/{LEGACY.name}/CLAIM.json": OLD_CLAIM_SHA,
                   UPSTREAM: UPSTREAM_SHA, old_plan["compiler_template_path"]: old_plan["compiler_template_SHA256"]})
    pinned_files = claim["files_SHA256"]
    support = {(HERE / name).relative_to(REPO).as_posix() for name in ("host_once.py", "prepare.py")}
    stage_names = {(STAGE / name).relative_to(REPO).as_posix() for name in STAGE_FILES}
    need(set(pinned_files) == set(expect) | support | stage_names
         and all(pinned_files[path] == sha for path, sha in expect.items()), "complete reviewed checkout inventory differs")
    for path, sha in pinned_files.items():
        relative = PurePosixPath(path)
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid reviewed member")
        pin(REPO.joinpath(*relative.parts), sha)
    need({path.name for path in STAGE.iterdir()} == STAGE_FILES, "generated stage directory is incomplete or has extra files")
    members = json.loads(pin(STAGE / "SOURCE_FILES.json", manifest))["files_SHA256"]
    need(members.keys() == STAGE_FILES - {"SOURCE_FILES.json"}, "generated stage members differ")
    for name, sha in members.items():
        pin(STAGE / name, sha)
    plan = json.loads(pin(STAGE / "PLAN.json", members["PLAN.json"]))
    compiler = pin(STAGE / "compile_once.py", claim["compiler_SHA256"])
    restored = compiler.replace(LABEL.encode(), original_claim["label"].encode(), 1)
    need(compiler.count(LABEL.encode()) == 1 and digest(restored) == OLD_COMPILER_SHA
         and plan == dict(old_plan, label=LABEL, compile_entry_SHA256=claim["compiler_SHA256"])
         and members["compile_once.py"] == claim["compiler_SHA256"]
         and members["fresh_kv_pack.cu"] == claim["kernel_SHA256"] == KERNEL_SHA,
         "v2 must retain the original compiler recipe and both original kernel variants")
    return claim, plan, original_host


def prepare_controller(claim, plan, original_host):
    namespace = {"__file__": str(LEGACY / "host_once.py"), "__name__": "frozen_glm_pack_v1_helpers"}
    exec(compile(original_host, namespace["__file__"], "exec"), namespace)
    namespace.update(ROOT=ROOT, REPO=REPO, LABEL=LABEL, REQUEST=claim["request"],
                     MANIFEST=claim["stage_manifest_SHA256"], COMPILER=claim["compiler_SHA256"],
                     SOURCE_LEAF=f"/refactor/incoming/{LABEL}/fresh_kv_pack.cu",
                     OBJECT=f"/refactor/candidates/{LABEL}/fresh_kv_pack.o")
    controller, waiter = namespace["shared_controller"](STAGE, plan)
    return namespace, controller, waiter


def execute(claim_sha, mode, *, query=None, observe_slot=None, runner=None, wait_terminal=None):
    need(REPO == ROOT / "incoming" / (LABEL + "-reviewed-checkout"), "v2 requires the new canonical target checkout")
    claim, plan, original = package(claim_sha)
    helpers, shared, accounting = prepare_controller(claim, plan, original)
    lookup = query if query is not None else lambda cap: helpers["account_both"](shared, cap)
    slot = observe_slot or shared["check_slot"]
    run_path = ROOT / "results" / (LABEL + "-controller")
    output = ROOT / "candidates" / LABEL
    if mode == "reconcile":
        return helpers["inspect_original"](run_path, output, claim_sha, lookup, accounting["is_terminal"], plan)
    need(all(not path.exists() and not path.is_symlink() for path in (run_path, output)),
         "v2 execution namespace is already present")
    if mode == "preflight":
        slot()
        need(lookup(8) is None, "v2 Slurm label already present")
        return dict(request=claim["request"], label=LABEL, state="V2_TARGET_PACKAGE_PRECHECKED",
                    claim_SHA256=claim_sha, stage_manifest_SHA256=claim["stage_manifest_SHA256"],
                    reviewed_members=len(claim["files_SHA256"]), actual_new_Slurm_steps=0)
    need(mode == "launch", "unknown v2 mode")
    return shared["controller"](run_path, output, claim_sha, wait_terminal or accounting["wait_terminal"],
                                query=lookup, observe_slot=slot, run=runner or subprocess.run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "launch", "reconcile"):
        modes.add_argument("--" + mode, dest="mode", action="store_const", const=mode)
    args = parser.parse_args()
    result = execute(args.claim_sha256, args.mode)
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(result["state"] not in ("V2_TARGET_PACKAGE_PRECHECKED", "COMPILE_VERIFIED"))


if __name__ == "__main__":
    raise SystemExit(main())
