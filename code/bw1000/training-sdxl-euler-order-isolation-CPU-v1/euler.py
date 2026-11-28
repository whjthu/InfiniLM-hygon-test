"""Isolate Euler default-list representation from the actual SDXL CPU timeline."""

import argparse
from hashlib import sha256
from importlib import metadata
from itertools import islice, permutations
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time

OLD_RESULT = "d14e800d2744556a67b9410898522915b3938d52f79080f2f50a6d1751c8a08f"
OLD_WIRE = "b9e0375a791fc3c9666c8dee997ee0e5e29d624a00920139daf4880ea154c46f"
OLD_REQUEST = "94f0acaeec614c5fa9715033ae1a755d"
PLAN = "b4287b48cff81e86c740d446fa6314514f4102bad86f323dee6d83d679299dc4"
SCHEDULER = "af3e45a949aff8b8341ab8b811429ec03fee857a700a1d9477363e4fff9666e2"
CONFIGURATION = "4a7af9be48913edfa77f3d32c375c997fa15b53c3f04efd5c371b36c5c6c1960"
EULER_PROVIDER = "ecee4a110e09b91edf76715434afa60a1b34281c2e08166114f7be0a1facfa63"
DIFFUSERS = Path("/training-runtime/site/diffusers")
DIFFUSERS_INIT = "27b79203d8e25aa693aefceed1a03838b51cfe6359317e16cb50e941a79e2ac9"
FIELD = "_use_default_values"
SEEDS = ("17", "101")
PERMUTATION_LIMIT = 100000
HEX = re.compile(r"[a-f0-9]{64}")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def read(path, limit, pin=None, *, readonly=False):
    need(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
         "EXACT_CANONICAL_REGULAR_FILE")
    if readonly:
        need(os.statvfs(path).f_flag & os.ST_RDONLY, "EXACT_READONLY_ORIGINAL_MEMBER")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= limit, "BOUNDED_ORIGINAL_MEMBER")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    need(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key) for key in keys)
         and (pin is None or digest(raw) == pin), "EXACT_STABLE_SOURCE_BYTES")
    return raw


def put(path, value):
    with path.open("xb") as stream:
        stream.write(wire(value))
        stream.flush()
        os.fsync(stream.fileno())


def original(root, *, readonly=True):
    value = json.loads(read(root / "result.json", 64 << 10, OLD_RESULT, readonly=readonly))
    need(value["request"] == OLD_REQUEST and value["facts"]["step"] == "1194900.2762"
         and value["facts"]["effective_scheduler_config_SHA256"] == OLD_WIRE and value["plan_SHA256"] == PLAN
         and value["sealed"] is value["passed"] is True
         and value["planned_pairs"] == value["complete_same_noise_pairs"] == 6 and value["planned_slots"] == 12,
         "EXACT_ORIGINAL_2762_NATIVE_SCOPE")
    raw = read(root / "generations.jsonl", 128 << 10, value["generations_SHA256"], readonly=readonly)
    rows = [json.loads(line) for line in raw.splitlines()]
    need(len(raw) == value["generations_bytes"]
         and [(row["arm"], row["pair"]) for row in rows] ==
         [(arm, j) for arm in ("base", "cp25") for j in range(6)]
         and all(row["outcome"] == "completed" and bool(HEX.fullmatch(row["timesteps_SHA256"])) for row in rows),
         "EXACT_ORIGINAL_TWELVE_TIMELINE_OBSERVATIONS")
    timelines = [row["timesteps_SHA256"] for row in rows[:6]]
    need(timelines == [row["timesteps_SHA256"] for row in rows[6:]], "ORIGINAL_PAIRED_TIMESTEPS_MATCH")
    return timelines, digest(raw)


def normalized(config):
    values = config.get(FIELD)
    need(type(values) is list and len(values) <= 64 and all(type(item) is str and 0 < len(item) <= 128 for item in values)
         and len(values) == len(set(values)),
         "BOUNDED_UNIQUE_DEFAULT_FIELD_MEMBERS")
    result = dict(config, **{FIELD: sorted(values)})
    need(len(wire(result)) <= 16 << 10, "BOUNDED_EFFECTIVE_SCHEDULER_CONFIG")
    return result


def reconstruct(config, recorded_sha, limit=PERMUTATION_LIMIT):
    need(HEX.fullmatch(recorded_sha) and type(limit) is int and 1 <= limit <= PERMUTATION_LIMIT,
         "EXACT_BOUNDED_DEFAULT_PERMUTATION_SEARCH")
    config = normalized(config)
    values = config[FIELD]
    total = math.factorial(len(values))
    start = time.monotonic()
    attempted = 0
    for option in islice(permutations(values), limit):
        if time.monotonic() - start >= 6:
            break
        attempted += 1
        if digest(wire(dict(config, **{FIELD: list(option)}))) == recorded_sha:
            return dict(found=True, attempts=attempted, total_possible=total, exhaustion=False,
                        limited_by_time=False, limit=limit)
    return dict(found=False, attempts=attempted, total_possible=total, exhaustion=attempted == total,
                limited_by_time=attempted < min(total, limit), limit=limit)


def tensor(value, torch):
    need(torch.is_tensor(value) and value.device.type == "cpu" and value.ndim == 1
         and 0 < value.numel() <= 4096 and bool(torch.isfinite(value).all()), "BOUNDED_FINITE_NATIVE_SCHEDULER_TENSOR")
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(value.dtype), shape=list(value.shape), bytes=len(raw), raw_hex=raw.hex(), SHA256=digest(raw))


def child(args):
    verify_scope()
    need(os.environ.get("PYTHONHASHSEED") == args.seed and args.seed in SEEDS, "EXACT_CHILD_HASHSEED")
    read(args.scheduler / "scheduler_config.json", 16 << 10, SCHEDULER, readonly=True)
    read(DIFFUSERS / "__init__.py", 2 << 20, DIFFUSERS_INIT, readonly=True)
    read(DIFFUSERS / "configuration_utils.py", 2 << 20, CONFIGURATION, readonly=True)
    scheduler_source = DIFFUSERS / "schedulers/scheduling_euler_discrete.py"
    scheduler_raw = read(scheduler_source, 41804, EULER_PROVIDER, readonly=True)
    need(len(scheduler_raw) == 41804, "EXACT_NATIVE_EULER_PROVIDER_EXTENT")
    scheduler_sha = digest(scheduler_raw)
    import diffusers
    from diffusers import EulerDiscreteScheduler, configuration_utils
    import torch

    need(metadata.version("diffusers") == "0.39.0" and Path(diffusers.__file__) == DIFFUSERS / "__init__.py"
         and Path(configuration_utils.__file__) == DIFFUSERS / "configuration_utils.py"
         and Path(sys.modules[EulerDiscreteScheduler.__module__].__file__) == scheduler_source
         and EulerDiscreteScheduler.__name__ == "EulerDiscreteScheduler"
         and torch.version.hip == "6.3.26113" and torch.version.cuda is None and not torch.cuda.is_initialized(),
         "EXACT_NATIVE_DTK_DIFFUSERS_EULER_AND_ZERO_CUDA")
    original_init = torch.cuda._lazy_init

    def no_accelerator(*_, **__):
        raise RuntimeError("ZERO_GRES_EULER_GPU_INITIALIZATION_FORBIDDEN")

    torch.cuda._lazy_init = no_accelerator
    try:
        scheduler = EulerDiscreteScheduler.from_pretrained(str(args.scheduler), local_files_only=True)
        need(type(scheduler) is EulerDiscreteScheduler, "EXACT_NATIVE_EULER_OBJECT")
        config = dict(scheduler.config)
        normalized(config)
        default_sigmas = tensor(scheduler.sigmas, torch)
        scheduler.set_timesteps(30, device="cpu")
        timesteps = tensor(scheduler.timesteps, torch)
        sigmas = tensor(scheduler.sigmas, torch)
        need(timesteps["shape"] == [30] and sigmas["shape"] == [31] and not torch.cuda.is_initialized(),
             "EXACT_NATIVE_30_STEP_CPU_TIMELINE")
        put(args.child_output, dict(pid=os.getpid(), seed=args.seed, python_hash=hash("sdxl-euler-order-controlled"),
            scheduler_source_SHA256=scheduler_sha, torch_version=str(torch.__version__), diffusers_version="0.39.0",
            source_config_SHA256=SCHEDULER, raw_wire_SHA256=digest(wire(config)), config=config,
            timesteps=timesteps, sigmas=sigmas, default_sigmas=default_sigmas, gpu_initialized=False))
    finally:
        torch.cuda._lazy_init = original_init
    return 0


def check_tensor(value):
    need(set(value) == {"dtype", "shape", "bytes", "raw_hex", "SHA256"}
         and type(value["bytes"]) is int and 0 < value["bytes"] <= 16 << 10
         and len(value["raw_hex"]) == 2 * value["bytes"], "BOUNDED_CHILD_TENSOR_RESULT")
    raw = bytes.fromhex(value["raw_hex"])
    need(len(raw) == value["bytes"] and digest(raw) == value["SHA256"], "EXACT_CHILD_TENSOR_BYTES")
    return dict(dtype=value["dtype"], shape=value["shape"], bytes=value["bytes"], SHA256=value["SHA256"])


def compare(children, old_timelines, ledger_sha):
    first, second = children
    need([row["seed"] for row in children] == list(SEEDS) and first["pid"] != second["pid"]
         and first["python_hash"] != second["python_hash"] and all(row["gpu_initialized"] is False for row in children)
         and first["scheduler_source_SHA256"] == second["scheduler_source_SHA256"] == EULER_PROVIDER
         and first["torch_version"] == second["torch_version"]
         and all(row["source_config_SHA256"] == SCHEDULER and row["diffusers_version"] == "0.39.0" for row in children), "INDEPENDENT_CONSISTENT_ZERO_GRES_CHILDREN")
    configs = [row["config"] for row in children]
    raw = [digest(wire(config)) for config in configs]
    need(raw == [row["raw_wire_SHA256"] for row in children], "EXACT_ORIGINAL_WIRE_ENCODING")
    canonical = [normalized(config) for config in configs]
    others = [{key: value for key, value in item.items() if key != FIELD} for item in configs]
    equal = wire(canonical[0]) == wire(canonical[1])
    members_equal = canonical[0][FIELD] == canonical[1][FIELD]
    order_changed = configs[0][FIELD] != configs[1][FIELD] and members_equal
    outcomes = [reconstruct(canonical[0], OLD_WIRE)] if equal else [reconstruct(item, OLD_WIRE) for item in canonical]
    tensors = {key: [check_tensor(row[key]) for row in children] for key in ("timesteps", "sigmas", "default_sigmas")}
    same_bytes = {key: first[key] == second[key] for key in tensors}
    need(len(old_timelines) == 6 and all(HEX.fullmatch(value) for value in old_timelines),
         "EXACT_SIX_RETAINED_ORIGINAL_TIMESTEP_HASHES")
    matches = [[value["SHA256"] == pin for pin in old_timelines] for value in tensors["timesteps"]]
    return dict(raw_config_SHA256=raw, normalized_config_SHA256=[digest(wire(item)) for item in canonical],
                original_effective_config_SHA256=OLD_WIRE, original_raw_matches=[pin == OLD_WIRE for pin in raw],
                normalized_equal=equal, other_config_fields_equal=wire(others[0]) == wire(others[1]),
                default_members_equal=members_equal, default_order_changed=order_changed,
                same_default_member_order_drift_observed=raw[0] != raw[1] and equal and order_changed,
                old_wire_reconstruction=outcomes, tensor_properties=tensors, tensor_values_equal_between_children=same_bytes,
                original_six_timestep_SHA256=old_timelines, original_six_timestep_matches=matches,
                original_ledger_SHA256=ledger_sha, original_sigma_recorded=False, original_sigma_comparison=None,
                config_provider_SHA256=CONFIGURATION, scheduler_provider_SHA256=first["scheduler_source_SHA256"],
                torch_version=first["torch_version"], hashseed=list(SEEDS), model_loads=0, generated_images=0,
                GPU_initialized=False, original_2774_cause_proven=False)


def verify_scope():
    need(sys.platform == "linux" and all(os.environ.get(key) == "-1"
         for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and all(os.environ.get(key) == "1" for key in
                 ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"))
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
         and os.environ.get("PYTHONPATH") == "/training-runtime/site", "EXACT_NUMBERED_ZERO_GRES_OFFLINE_SCOPE")


def command(current, scheduler, output, seed):
    return [sys.executable, "-B", str(current), "child", "--scheduler", str(scheduler),
            "--child-output", str(output), "--seed", seed], dict(os.environ, PYTHONHASHSEED=seed)


def run(args):
    verify_scope()
    need(re.fullmatch(r"[a-f0-9]{32}", args.request) and args.request != OLD_REQUEST
         and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and args.output.parent.stat().st_uid == os.geteuid()
         and stat.S_IMODE(args.output.parent.stat().st_mode) == 0o700, "EXACT_FRESH_PRIVATE_DIAGNOSTIC_SCOPE")
    timelines, ledger_sha = original(args.original)
    args.output.mkdir(mode=0o700)
    children = []
    failure = None
    try:
        with tempfile.TemporaryDirectory(prefix="euler-readback-", dir="/tmp") as scratch:
            root = Path(scratch)
            for seed in SEEDS:
                target = root / f"child-{seed}.json"
                argv, env = command(Path(__file__).resolve(), args.scheduler, target, seed)
                result = subprocess.run(argv, env=env, cwd="/", timeout=35, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                put(args.output / f"child-{seed}.status.json", dict(seed=seed, returncode=result.returncode,
                        stdout_SHA256=digest(result.stdout), stderr_SHA256=digest(result.stderr)))
                need(result.returncode == 0, "EXACT_INDEPENDENT_NATIVE_CHILD_COMPLETION")
                children.append(json.loads(read(target, 128 << 10)))
        projection = compare(children, timelines, ledger_sha)
    except Exception as error:
        projection = dict(model_loads=0, generated_images=0, original_2774_cause_proven=False)
        failure = dict(type=type(error).__name__, detail_SHA256=digest(str(error).encode()))
    report = dict(request=args.request, actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  native_children_completed=len(children), diagnostic_observed=failure is None, failure=failure,
                  **projection)
    put(args.output / "result.json", report)
    print(json.dumps({key: report[key] for key in ("request", "diagnostic_observed", "native_children_completed")}),
          flush=True)
    return 0 if failure is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    launch, worker = sub.add_parser("run"), sub.add_parser("child")
    for mode in (launch, worker):
        mode.add_argument("--scheduler", type=Path, required=True)
    launch.add_argument("--original", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--request", required=True)
    worker.add_argument("--child-output", type=Path, required=True)
    worker.add_argument("--seed", required=True)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else child(args)


if __name__ == "__main__":
    raise SystemExit(main())
