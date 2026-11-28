"""Run the SDXL three-family CP25 comparison with the qualified n03 owner."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "incoming/sdxl-cc0-full25-cp20-abd5c98bc5dc-stage"
CP25 = RESULTS / "sdxl-cc0-full25-cp20-abd5c98bc5dc/durable/once/continuous/checkpoint-25"
CP25_SHA = "beb593565087406f3b9090a468fd650a18b097124d4f44f9e26acda0aa7640b4"
BASE = "686c7cc61b2b8b3862ad47a58376700cf5d2f946ba7c8fca392023335eae16ab"
R33_HOST = "d8de70e7e492b77c9ad0b504cf6179cd7f4299b2dfab4ba83a7fc8cd1a40853a"
R33_PROOF = "e5557e400bb7777d54c4e2c86d9595f946b81994847903e81e0faa7c3f5f9dfc"
RUNTIME = "ae871dd0e3f1193a63367eb7225413559e7973520f8beae96a55ea0d20a5e6af"
RUNNER = "bf1fd9e680cdb26b103cf1eb593100fec4f86a66a80f516c4656e371b67eb7e7"
PLAN = "b4287b48cff81e86c740d446fa6314514f4102bad86f323dee6d83d679299dc4"
ARMS = ("base", "cp25")
FILES = {"device.py": "native.py", "plan.py": "plan.py", "PLAN.json": "PLAN.json"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def label(request):
    need(re.fullmatch(r"[0-9a-f]{32}", request), "EXACT_NEW_REQUEST")
    return "sdxl-cc0-cp25-pairs-" + request[:12]


def paths(request):
    return ROOT / "incoming" / (label(request) + "-stage"), RESULTS / label(request)


def module(path, pin):
    raw = path.read_bytes()
    need(path.resolve(strict=True) == path and len(raw) <= 128 << 10 and sha256(raw).hexdigest() == pin,
         "EXACT_ARCHIVED_OWNER_CODE")
    provider = module_from_spec(spec_from_file_location("pinned_" + path.stem, path))
    exec(compile(raw, str(path), "exec"), provider.__dict__)
    return provider


def bind():
    continuity = module(ARCHIVE / "host.py", R33_HOST)
    base = module(ARCHIVE / "base-host.py", BASE)
    base.paths, base.label = paths, label
    base.staged = lambda common, target, request, digest=None: staged(base, continuity, common, target, request, digest)
    base.guest = lambda target, work, scratch, request, pin: guest(base, target, work, scratch, request, pin)
    base.observe = observe
    controller = inspect.getsource(base.start)
    anchor = "planned_updates=1, actual_optimizer_updates=None,"
    need(controller.count(anchor) == 1, "EXACT_MATCHED_CONTROLLER_DENOMINATOR")
    exec(compile(controller.replace(anchor, "planned_slots=12, planned_pairs=6,"),
                 "<qualified-owner-paired-denominator>", "exec"), base.__dict__)
    return base


def staged(base, continuity, common, target, request, digest=None):
    raw = common.read(target / "stage.json", digest, 16 << 10)
    doc = json.loads(raw)
    need(doc["request"] == request and doc["plan_SHA256"] == PLAN and doc["GPU_GO"] is False,
         "ONLY_NEUTRAL_PAIRED_STAGE")
    common.read(target / "host.py", doc["host_SHA256"], 128 << 10)
    pins = json.loads(common.read(target / "runner.json", RUNNER, 16 << 10))
    for local, original in FILES.items():
        need(len(common.read(target / local, pins[original]["SHA256"], 128 << 10)) == pins[original]["bytes"],
             "EXACT_PAIRED_SOURCE_MEMBER")
    common.read(target / "r33.json", R33_PROOF, 64 << 10)
    runtime = json.loads(common.read(ARCHIVE / "continuity-runtime.json", RUNTIME, 16 << 10))
    need(continuity.proof(runtime, continuity.stamp()) == doc["image_id"], "EXACT_RETAINED_DTK_IMAGE")
    base.live_runtime(common)
    return doc


def stage(args, base):
    target, work = paths(args.request)
    need(not target.exists() and not work.exists(), "EXCLUSIVE_NEW_SOURCE_STAGE")
    common = base.isolated(ARCHIVE / "common.py", "pinned_archive_common", base.COMMON)
    continuity = module(ARCHIVE / "host.py", R33_HOST)
    runtime = json.loads(common.read(ARCHIVE / "continuity-runtime.json", RUNTIME, 16 << 10))
    image_id = continuity.proof(runtime, continuity.stamp())
    base.live_runtime(common)
    base.assets(common)
    manifest = common.read(args.runner / "SOURCE_FILES.json", RUNNER, 16 << 10)
    manifest_doc = json.loads(manifest)
    sources = {name: common.read(args.runner / old, manifest_doc[old]["SHA256"], 128 << 10)
               for name, old in FILES.items()}
    proof = common.read(args.r33_proof, R33_PROOF, 64 << 10)
    host = common.read(HERE / "host.py", args.host_sha, 128 << 10)
    need(base.checkpoint_hash(CP25 / "pytorch_lora_weights.safetensors") == CP25_SHA, "ACTUAL_CP25_ADAPTER")
    support = {name: common.read(ARCHIVE / "support" / name, pin, 128 << 10)
               for name, pin in common.SUPPORT.items()}
    for name, pin in (("common.py", base.COMMON), ("pilot.py", base.PILOT), ("legacy.py", base.LEGACY)):
        sources[name] = common.read(ARCHIVE / name, pin, 128 << 10)
    target.mkdir(mode=0o700)
    (target / "support").mkdir(mode=0o700)
    for name, raw in support.items():
        common.put(target / "support" / name, raw)
    for name, raw in dict(sources, **{"host.py": host, "r33.json": proof, "runner.json": manifest}).items():
        common.put(target / name, raw)
    for argv in (["verify-plan", "--plan", str(target / "PLAN.json"), "--r33-proof", str(target / "r33.json")],
                 ["run", "--help"]):
        subprocess.run([base.HOST_PYTHON, "-B", str(target / "device.py"), *argv],
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                       cwd="/", timeout=15)
    doc = dict(request=args.request, plan_SHA256=PLAN, source_SHA256=RUNNER, host_SHA256=args.host_sha,
               image_id=image_id, planned_slots=12, planned_pairs=6, GPU_GO=False)
    stage_sha = common.put(target / "stage.json", common.wire(doc))
    base.staged(common, target, args.request, stage_sha)
    return dict(doc, stage_SHA256=stage_sha)


def guest(base, target, work, scratch, request, device_pin):
    common = base.isolated(target / "common.py", "pinned_compute_common", base.COMMON)
    grant = base.scope(json.loads(common.read(work / "auth.json")), request)
    device = json.loads(common.read(work / "owner/DEVICE.json", device_pin))
    need(device["request"] == request and device["boot"] == grant["boot"]
         and device["pci"] == grant["pci"], "EXACT_SHARED_OWNER_DEVICE_GRANT")
    authority = dict(request=request, step=device["step"], node=base.NODE, boot=device["boot"], pci=device["pci"],
                     epoch=grant["epoch"], deadline=grant["deadline"], GPU_GO=True, gpu_count=1, plan_SHA256=PLAN,
                     output="/durable/pair-once", CP25_SHA256=CP25_SHA)
    pin = common.put(work / "owner/PAIR_AUTHORITY.json", common.wire(authority))
    mounts = ((target, "/owner-code", "ro"), (base.PIPE / "source", "/first-update-source", "ro"),
              (base.PIPE / "pipeline", "/official-sdxl", "ro"), (CP25.parent, "/r33/continuous", "ro"),
              (base.RUNTIME, "/training-runtime", "ro"), (work / "owner", "/owner", "ro"),
              (work / "durable", "/durable", "rw"), (scratch, "/tmp", "rw"),
              (Path("/usr/local/hyhal"), "/opt/hyhal", "ro"))
    command = [base.APPTAINER, "exec", "--rocm", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, path, mode in mounts:
        command.extend(("--bind", f"{source}:{path}:{mode}"))
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; "
             "unset PYTHONPYCACHEPREFIX MIOPEN_SYSTEM_DB_PATH MIOPEN_LOCKFILE_DIR MIOPEN_LOCKFILE_PATH; "
             "export PYTHONPATH=/training-runtime/site PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
             "MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1; "
             'exec /usr/bin/python3 -B -u "$@"')
    return [*command, str(ROOT / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell,
            "sdxl-pair-owner", "/owner-code/device.py", "run", "--request", request,
            "--authority", "/owner/PAIR_AUTHORITY.json", "--authority-sha", pin, "--plan", "/owner-code/PLAN.json",
            "--r33-proof", "/owner-code/r33.json", "--pipeline", "/official-sdxl",
            "--pipeline-manifest", "/first-update-source/pipeline.json", "--asset-source", "/first-update-source",
            "--adapter", "/r33/continuous/checkpoint-25/pytorch_lora_weights.safetensors", "--output", "/durable/pair-once"]


def observe(request, work, common, old):
    roles = {role: common.role_row(old, request, role) for role in ("pre", "gpu", "post")}
    result = dict(request=request, planned_slots=12, planned_pairs=6, accounting=roles,
                  cleanup_verified=False, scientific_pass=False, sealed=False, formal_domain_credit=0,
                  gpu_intent=(work / "gpu-intent.json").is_file())
    if not (work / "auth.json").is_file():
        return result
    auth = json.loads(common.read(work / "auth.json"))
    folder = work / "durable/pair-once"
    if (folder / "result.json").is_file():
        raw = common.read(folder / "result.json", limit=64 << 10)
        worker = json.loads(raw)
        ledger = common.read(folder / "generations.jsonl", worker["generations_SHA256"], 64 << 10)
        rows = [json.loads(line) for line in ledger.splitlines()]
        need(worker["request"] == request and (worker["planned_slots"], worker["planned_pairs"]) == (12, 6)
             and worker["plan_SHA256"] == PLAN and worker["score_computed"] is False and
             worker["sealed"] is True and len(ledger) == worker["generations_bytes"]
             and [(row["arm"], row["pair"]) for row in rows] == [(arm, i) for arm in ARMS for i in range(6)]
             and all(row["outcome"] in ("completed", "error", "missing") for row in rows),
             "EXACT_PAIRED_WORKER_DENOMINATOR")
        counts = {arm: {key: sum(row["arm"] == arm and row["outcome"] == key for row in rows)
                        for key in ("completed", "error", "missing")} for arm in ARMS}
        need(worker["counts"] == counts, "EXACT_PAIRED_OUTCOME_COUNTS")
        result.update(sealed=True, counts=counts, worker_result_SHA256=sha256(raw).hexdigest(),
                      complete_same_noise_pairs=worker["complete_same_noise_pairs"], worker_failure=worker["failure"])
        if roles["gpu"] is not None:
            need(roles["gpu"][0] == worker["facts"]["step"], "EXACT_NUMBERED_GPU_WORKER_RESULT")
    if all(roles.values()) and roles["post"][2:4] == ["COMPLETED", "0:0"]:
        pre, gpu, post = [common.role_row(old, request, role, terminal=True) for role in ("pre", "gpu", "post")]
        try:
            before, _ = common.proof(work, request, "pre", pre)
            after, _ = common.proof(work, request, "post", post)
            intended = json.loads(common.read(work / "post-intent.json"))
        except FileNotFoundError:
            return result
        need(before["boot_id"] == after["boot_id"] == auth["boot"] and before["pci"] == after["pci"] == auth["pci"]
             and after["owned_processes_remaining"] == 0 and pre[2:4] == ["COMPLETED", "0:0"]
             and intended["gpu_step"] == gpu[0] and pre[8] <= gpu[7] <= gpu[8] <= post[7],
             "INDEPENDENT_PHYSICAL_CLEAN_POST")
        result["cleanup_verified"] = True
        result["scientific_pass"] = (result["sealed"] and worker["passed"] is True
                                     and worker["complete_same_noise_pairs"] == 6 and gpu[2:4] == ["COMPLETED", "0:0"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "start", "terminal", "compute-pre", "compute-gpu", "compute-post"))
    parser.add_argument("--request", required=True)
    for name in ("runner", "r33-proof", "auth"):
        parser.add_argument("--" + name, type=Path)
    for name in ("host-sha", "auth-sha"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard", LC_ALL="C")
    target, work = paths(args.request)
    base = bind()
    if args.mode == "stage":
        print(json.dumps(stage(args, base), sort_keys=True), flush=True)
        return 0
    common, old, snapshot, accounting, owned = base.common(target)
    if args.mode == "terminal":
        base.staged(common, target, args.request)
        print(json.dumps(observe(args.request, work, common, old), sort_keys=True), flush=True)
        return 0
    if args.mode == "start":
        return base.start(args, target, work, common, old, accounting)
    return base.compute(args, target, work, common, old, snapshot, owned)


if __name__ == "__main__":
    raise SystemExit(main())
