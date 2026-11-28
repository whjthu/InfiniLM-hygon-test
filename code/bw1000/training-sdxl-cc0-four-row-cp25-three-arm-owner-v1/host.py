"""Run six new SDXL four-row CP25 slots through the qualified single-card n03 owner."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "incoming/sdxl-cc0-full25-cp20-abd5c98bc5dc-stage"
CP25 = RESULTS / "sdxl-cc0-four-full25-cp20-a9a91b5727d8/durable/once/continuous/checkpoint-25"
ORIGINAL = RESULTS / "sdxl-cc0-cp25-pairs-fp32-94f0acaeec61/durable/pair-once"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
PARENT, NODE = "1194900", "f13r4n03"
TRAINING_PROOF = "41337046173a5f602124966c37ea866fa3d687d393382b428705f8fdf9432219"
CP25_MANIFEST = "b108ecd38a9ca05ffef58ce5454adc3db9169282d2a2f79a8a057613c4bbe14a"
OLD_RESULT = "d14e800d2744556a67b9410898522915b3938d52f79080f2f50a6d1751c8a08f"
OLD_LATENTS = "97fd295d028a65269ee428d016eefc68d7d3ba80a59be7108682159cb61b4fe8"
CP25_SHA = "7301c4f968e79b93c1ff183b03f2275789d61de06565d8b2c2ea091af27b876a"
BASE = "686c7cc61b2b8b3862ad47a58376700cf5d2f946ba7c8fca392023335eae16ab"
R33_HOST = "d8de70e7e492b77c9ad0b504cf6179cd7f4299b2dfab4ba83a7fc8cd1a40853a"
R33_PROOF = "e5557e400bb7777d54c4e2c86d9595f946b81994847903e81e0faa7c3f5f9dfc"
RUNTIME = "ae871dd0e3f1193a63367eb7225413559e7973520f8beae96a55ea0d20a5e6af"
RUNNER = "f96874ae6cd9f301a1d06b1a3f6e038dc9739e6e504372dc6c948a6263e07339"
PLAN = "b4287b48cff81e86c740d446fa6314514f4102bad86f323dee6d83d679299dc4"
NEW_ARM = "four_row_cp25"
FILES = {"device.py": "native.py", "plan.py": "plan.py", "PLAN.json": "PLAN.json", "comparison.py": "comparison.py"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def label(request):
    need(re.fullmatch(r"[0-9a-f]{32}", request) and request[:12] not in {"94f0acaeec61", "a9a91b5727d8", "abd5c98bc5dc"},
         "EXACT_NEW_REQUEST")
    return "sdxl-cc0-four-cp25-six-" + request[:12]


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
    exec(compile(controller.replace(anchor, "planned_slots=6, planned_pairs=6,"),
                 "<qualified-owner-paired-denominator>", "exec"), base.__dict__)
    return base


def staged(base, continuity, common, target, request, digest=None):
    raw = common.read(target / "stage.json", digest, 16 << 10)
    doc = json.loads(raw)
    need(doc["request"] == request and doc["plan_SHA256"] == PLAN and doc["GPU_GO"] is False
         and doc["source_SHA256"] == RUNNER and doc["CP25_manifest_SHA256"] == CP25_MANIFEST
         and doc["CP25_SHA256"] == CP25_SHA and doc["original_result_SHA256"] == OLD_RESULT
         and doc["planned_slots"] == doc["planned_pairs"] == 6,
         "ONLY_NEUTRAL_PAIRED_STAGE")
    common.read(target / "host.py", doc["host_SHA256"], 128 << 10)
    pins = json.loads(common.read(target / "runner.json", RUNNER, 16 << 10))
    for local, original in FILES.items():
        need(len(common.read(target / local, pins[original]["SHA256"], 128 << 10)) == pins[original]["bytes"],
             "EXACT_PAIRED_SOURCE_MEMBER")
    common.read(target / "r33.json", R33_PROOF, 64 << 10)
    common.read(target / "four-row-training.json", TRAINING_PROOF, 64 << 10)
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
    common.read(CP25 / "genmlu-checkpoint.json", CP25_MANIFEST, 128 << 10)
    common.read(ORIGINAL / "result.json", OLD_RESULT, 64 << 10)
    manifest = common.read(args.runner / "SOURCE_FILES.json", RUNNER, 16 << 10)
    manifest_doc = json.loads(manifest)
    sources = {name: common.read(args.runner / old, manifest_doc[old]["SHA256"], 128 << 10)
               for name, old in FILES.items()}
    proof = common.read(args.r33_proof, R33_PROOF, 64 << 10)
    training = common.read(args.training_proof, TRAINING_PROOF, 64 << 10)
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
    for name, raw in dict(sources, **{"host.py": host, "r33.json": proof, "four-row-training.json": training, "runner.json": manifest}).items():
        common.put(target / name, raw)
    doc = dict(request=args.request, plan_SHA256=PLAN, source_SHA256=RUNNER, host_SHA256=args.host_sha,
               image_id=image_id, planned_slots=6, planned_pairs=6, GPU_GO=False,
               CP25_SHA256=CP25_SHA, CP25_manifest_SHA256=CP25_MANIFEST, original_result_SHA256=OLD_RESULT)
    stage_sha = common.put(target / "stage.json", common.wire(doc))
    base.staged(common, target, args.request, stage_sha)
    return dict(doc, stage_SHA256=stage_sha)


def material_mounts(target):
    return ((target, "/owner-code", "ro"),
            (CP25.parent, "/four-row-a9a91b5727d8/continuous", "ro"), (ORIGINAL, "/original-2762", "ro"))


def material_args():
    cp = "/four-row-a9a91b5727d8/continuous/checkpoint-25"
    return ["--plan", "/owner-code/PLAN.json", "--r33-proof", "/owner-code/r33.json",
            "--training-proof", "/owner-code/four-row-training.json", "--original", "/original-2762",
            "--checkpoint-manifest", cp + "/genmlu-checkpoint.json", "--checkpoint-manifest-sha", CP25_MANIFEST,
            "--adapter", cp + "/pytorch_lora_weights.safetensors", "--adapter-sha", CP25_SHA]


def guest(base, target, work, scratch, request, device_pin):
    common = base.isolated(target / "common.py", "pinned_compute_common", base.COMMON)
    grant = base.scope(json.loads(common.read(work / "auth.json")), request)
    device = json.loads(common.read(work / "owner/DEVICE.json", device_pin))
    need(device["request"] == request and device["boot"] == grant["boot"]
         and device["pci"] == grant["pci"], "EXACT_SHARED_OWNER_DEVICE_GRANT")
    authority = dict(request=request, step=device["step"], node=base.NODE, boot=device["boot"], pci=device["pci"],
                     epoch=grant["epoch"], deadline=grant["deadline"], GPU_GO=True, gpu_count=1, plan_SHA256=PLAN,
                     output="/durable/four-row-pair-once", CP25_SHA256=CP25_SHA)
    pin = common.put(work / "owner/PAIR_AUTHORITY.json", common.wire(authority))
    mounts = (*material_mounts(target), (base.PIPE / "source", "/first-update-source", "ro"),
              (base.PIPE / "pipeline", "/official-sdxl", "ro"),
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
            "--authority", "/owner/PAIR_AUTHORITY.json", "--authority-sha", pin, *material_args(),
            "--pipeline", "/official-sdxl", "--pipeline-manifest", "/first-update-source/pipeline.json",
            "--asset-source", "/first-update-source", "--output", "/durable/four-row-pair-once"]


def verify_argv(base, target):
    argv = [base.APPTAINER, "exec", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, dest, mode in (*material_mounts(target), (base.RUNTIME, "/training-runtime", "ro")):
        argv += ["--bind", f"{source.as_posix()}:{dest}:{mode}"]
    shell = ("set -eu; export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "
             "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
             'exec /usr/bin/python3 -I -B -c \'import sys,runpy; sys.path.insert(0,"/owner-code"); '
             'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")\' "$@"')
    return [*argv, IMAGE.as_posix(), "/bin/bash", "--noprofile", "--norc", "-c", shell, "sdxl-verify",
            "/owner-code/device.py", "verify-plan", *material_args()]


def verify_name(request):
    return label(request) + "-verify"


def exact_verify_live(base, old, job, number, request):
    need(job == PARENT and number.isdecimal(), "EXACT_NUMBERED_VERIFY_ENV")
    step = job + "." + number
    rows = [old.fields(line) for line in old.query(["scontrol", "show", "step", step, "--oneliner"])]
    selected = [row for row in rows if row.get("StepId") == step]
    need(len(selected) == 1, "EXACT_ONE_CURRENT_VERIFY_CONTROLLER_RECORD")
    row, = selected
    need(row.get("State") == "RUNNING" and row.get("Name", row.get("JobName")) == verify_name(request)
         and row.get("NodeList") == NODE and old.tres(row.get("TRES", row.get("AllocTRES", "")), 0, 2, 2048),
         "EXACT_LIVE_VERIFY_NO_GRES_TRES")
    return step


def compute_verify(base, common, old, target, args):
    intent = dict(request=args.request, stage_SHA256=args.stage_sha, role="verify", GPU=False)
    need(json.loads(common.read(target / "verify-intent.json", limit=8 << 10)) == intent, "EXACT_OWNED_VERIFY_INTENT")
    job = os.environ.get("SLURM_JOB_ID", "")
    number = os.environ.get("SLURM_STEP_ID", "")
    need(os.geteuid() == base.UID and socket.gethostname().split(".")[0] == NODE
         and job == PARENT and number.isdecimal() and not os.environ.get("SLURM_STEP_GPUS")
         and re.search(rf"/job_{job}(?:\.scope)?/step_{number}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE), "EXACT_VERIFY_NODE_UID_CGROUP_ZERO_GPU")
    step = exact_verify_live(base, old, job, number, args.request)
    common.put(target / "verify-compute-intent.json", common.wire(dict(intent, step=step)))
    record = dict(intent, step=step, passed=False)
    try:
        completed = subprocess.run(verify_argv(base, target), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd="/", timeout=65, check=False)
        record.update(returncode=completed.returncode, stdout_SHA256=sha256(completed.stdout).hexdigest(),
                      stderr_SHA256=sha256(completed.stderr).hexdigest())
        payload = json.loads(completed.stdout) if completed.returncode == 0 and len(completed.stdout) <= 4096 else None
        need(payload == dict(plan_SHA256=PLAN, planned_slots=6, planned_pairs=6, actual_generations=0),
             "EXACT_UNAMBIGUOUS_ORIGINAL_MATERIAL_VERIFICATION")
        record.update(passed=True, native=payload)
    except Exception as error:
        record.update(failure_class=type(error).__name__, failure_SHA256=sha256(str(error).encode()).hexdigest())
    common.put(target / "verify-result.json", common.wire(record))
    return 0 if record["passed"] else 2


def verify_terminal(common, old, target, request, stage_sha):
    result = dict(request=request, stage_SHA256=stage_sha, passed=False, accounting=None, actual_generations=0)
    intent = dict(request=request, stage_SHA256=stage_sha, role="verify", GPU=False)
    terminal = old.matching(old.account(PARENT), name=verify_name(request))
    result["accounting"] = terminal
    path = target / "verify-result.json"
    if not path.is_file():
        return result
    raw = common.read(path, limit=8 << 10)
    value = json.loads(raw)
    step = value["step"]
    need(re.fullmatch(r"1194900\.\d+", step) and all(value[key] == item for key, item in intent.items())
         and json.loads(common.read(target / "verify-intent.json")) == intent
         and json.loads(common.read(target / "verify-compute-intent.json")) == dict(intent, step=step),
         "EXACT_STAGE_BOUND_VERIFY_RECEIPT")
    result.update(step=step, worker_SHA256=sha256(raw).hexdigest(), worker_passed=value["passed"])
    if terminal is None or terminal[2:4] != ["COMPLETED", "0:0"]:
        return result
    need(terminal[0] == step and terminal[5] == NODE and old.tres(terminal[6], 0, 2, 2048)
         and terminal[4].isdigit() and all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", text)
                                          for text in terminal[7:9])
         and terminal[7] <= terminal[8], "INDEPENDENT_VERIFY_ZERO_GRES_TERMINAL_ATTESTATION")
    result["passed"] = value["passed"] is True and value.get("returncode") == 0 and value.get("native") == dict(
        plan_SHA256=PLAN, planned_slots=6, planned_pairs=6, actual_generations=0)
    return result


def verify_plan(base, common, old, target, args):
    need(os.geteuid() == base.UID, "EXACT_VERIFY_OWNER_UID")
    with old.held_lock():
        base.scheduler_guard(target, old)
        need(old.matching(old.account(PARENT), name=verify_name(args.request)) is None, "UNIQUE_VERIFY_LABEL")
        common.put(target / "verify-intent.json", common.wire(
            dict(request=args.request, stage_SHA256=args.stage_sha, role="verify", GPU=False)))
        argv = ["srun", "--exact", "--threads=1", "--immediate=5", "--jobid=" + PARENT, "--nodes=1", "--ntasks=1",
                "--nodelist=" + NODE, "--cpus-per-task=2", "--mem=2048M", "--time=2", "--gres=none",
                "--job-name=" + verify_name(args.request), base.HOST_PYTHON, "-I", "-S", "-B",
                str(target / "host.py"), "compute-verify", "--request", args.request, "--stage-sha", args.stage_sha]
        try:
            with (target / "verify-srun.log").open("xb") as log:
                subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               timeout=105, check=False, cwd="/")
        except (OSError, subprocess.SubprocessError) as error:
            common.put(target / "verify-launch-unknown.json", common.wire(dict(
                request=args.request, type=type(error).__name__, auto_replay=False)))
            raise
    result = verify_terminal(common, old, target, args.request, args.stage_sha)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 2


def observe(request, work, common, old):
    roles = {role: common.role_row(old, request, role) for role in ("pre", "gpu", "post")}
    result = dict(request=request, planned_slots=6, planned_pairs=6, accounting=roles,
                  cleanup_verified=False, scientific_pass=False, sealed=False, formal_domain_credit=0,
                  gpu_intent=(work / "gpu-intent.json").is_file())
    if not (work / "auth.json").is_file():
        return result
    auth = json.loads(common.read(work / "auth.json"))
    folder = work / "durable/four-row-pair-once"
    if (folder / "result.json").is_file():
        raw = common.read(folder / "result.json", limit=64 << 10)
        worker = json.loads(raw)
        ledger = common.read(folder / "generations.jsonl", worker["generations_SHA256"], 64 << 10)
        rows = [json.loads(line) for line in ledger.splitlines()]
        facts = worker["facts"]
        need(worker["request"] == request and (worker["planned_slots"], worker["planned_pairs"]) == (6, 6)
             and worker["plan_SHA256"] == PLAN and worker["sealed"] is True
             and worker["only_new_arm"] == NEW_ARM and worker["original_result_SHA256"] == OLD_RESULT
             and facts["adapter_SHA256"] == CP25_SHA and facts["adapter_manifest_SHA256"] == CP25_MANIFEST
             and len(ledger) == worker["generations_bytes"]
             and [(row["arm"], row["pair"]) for row in rows] == [(NEW_ARM, i) for i in range(6)]
             and all(row["outcome"] in ("completed", "error", "missing") for row in rows),
             "EXACT_NEW_WORKER_SIX_SLOT_DENOMINATOR")
        counts = {key: sum(row["outcome"] == key for row in rows) for key in ("completed", "error", "missing")}
        need(worker["counts"] == counts, "EXACT_NEW_ARM_OUTCOME_COUNTS")
        if worker["passed"] is True:
            need(counts["completed"] == facts["actual_generation_calls"] == len(facts["vae_decode_casts"]) == 6
                 and facts["model_loads"] == 1 and facts["latent_manifest_SHA256"] == OLD_LATENTS
                 and facts["live_adapter"] == dict(active=NEW_ARM, strength=1.0, tensors=1120, SHA256=CP25_SHA),
                 "EXACT_NATIVE_BOUNDED_GENERATIONS")
        result.update(sealed=True, counts=counts, worker_result_SHA256=sha256(raw).hexdigest(),
                      generated_same_noise_slots=counts["completed"], worker_failure=worker["failure"])
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
                                     and worker["counts"]["completed"] == 6 and gpu[2:4] == ["COMPLETED", "0:0"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "verify-plan", "verify-terminal", "compute-verify", "start", "terminal", "compute-pre", "compute-gpu", "compute-post"))
    parser.add_argument("--request", required=True)
    for name in ("runner", "r33-proof", "training-proof", "auth"):
        parser.add_argument("--" + name, type=Path)
    for name in ("host-sha", "auth-sha", "stage-sha"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.mode == "stage":
        need(all((args.runner, args.r33_proof, args.training_proof))
             and re.fullmatch(r"[a-f0-9]{64}", args.host_sha or ""), "EXACT_NEUTRAL_STAGE_ARGUMENTS")
    elif args.mode == "start" or args.mode in ("compute-pre", "compute-gpu", "compute-post"):
        need(re.fullmatch(r"[a-f0-9]{64}", args.auth_sha or "")
             and (args.mode != "start" or args.auth is not None), "EXACT_PRIVATE_GPU_AUTHORITY_ARGUMENTS")
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard", LC_ALL="C")
    target, work = paths(args.request)
    base = bind()
    if args.mode == "stage":
        print(json.dumps(stage(args, base), sort_keys=True), flush=True)
        return 0
    common, old, snapshot, accounting, owned = base.common(target)
    if args.mode in ("verify-plan", "verify-terminal", "compute-verify"):
        need(re.fullmatch(r"[a-f0-9]{64}", args.stage_sha or ""), "EXACT_VERIFY_STAGE_SHA")
        base.staged(common, target, args.request, args.stage_sha)
        if args.mode == "compute-verify":
            return compute_verify(base, common, old, target, args)
        if args.mode == "verify-plan":
            return verify_plan(base, common, old, target, args)
        result = verify_terminal(common, old, target, args.request, args.stage_sha)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 2
    if args.mode == "terminal":
        base.staged(common, target, args.request)
        print(json.dumps(observe(args.request, work, common, old), sort_keys=True), flush=True)
        return 0
    if args.mode == "start":
        grant = base.scope(json.loads(common.read(args.auth, args.auth_sha, 16 << 10)), args.request)
        base.staged(common, target, args.request, grant["stage_SHA256"])
        need(verify_terminal(common, old, target, args.request, grant["stage_SHA256"])["passed"],
             "INDEPENDENT_ZERO_GRES_VERIFY_PLAN_REQUIRED")
        return base.start(args, target, work, common, old, accounting)
    return base.compute(args, target, work, common, old, snapshot, owned)


if __name__ == "__main__":
    raise SystemExit(main())
