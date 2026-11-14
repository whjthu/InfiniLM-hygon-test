"""Explicit two-process DP/SP1 native training; no FSDP, DTensor, or restart."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import signal
import subprocess
import sys
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .. import hunyuanvideo_distributed_checkpoint as checkpoint
from ..runtime import MEMORY_LIMIT_BYTES, memory_stats, require_cuda, within_memory_limit
from . import hunyuanvideo_training as native


TIMING_BOUNDARY = (
    "before first measured update through completed final two-rank checkpoint commit and barrier; "
    "includes replicated-state checks and checkpoint IO; excludes loading and warmup"
)


def rank_window_metrics(
    output_dir: Path, logs: list[dict[str, Any]], *, warmup: int, measured: int, resumed: bool
) -> dict[str, Any]:
    """Aggregate the sealed rank windows after both workers exit."""
    starts, steps, seconds = [], [], []
    for rank in range(2):
        path = output_dir / f"rank-{rank}" / "summary.json"
        pins = [log["sha256"] for log in logs if Path(log["path"]).resolve() == path.resolve()]
        raw = path.read_bytes()
        if pins != [sha256(raw).hexdigest()]:
            raise RuntimeError(f"DP2 rank {rank} summary identity mismatch")
        summary = json.loads(raw)
        if type(summary["rank"]) is not int or summary["rank"] != rank:
            raise RuntimeError(f"DP2 rank {rank} summary rank mismatch")
        if summary["timing_boundary"] != TIMING_BOUNDARY:
            raise RuntimeError(f"DP2 rank {rank} timing boundary mismatch")
        start = summary["completed_global_step_at_start"]
        count = summary["measurement_steps"]
        elapsed = summary["measurement_wall_seconds"]
        if (type(start) is not int or not 0 <= start < warmup + measured
                or resumed != (start > 0)
                or type(count) is not int or count != warmup + measured - max(start, warmup)
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0):
            raise RuntimeError(f"DP2 rank {rank} invalid measurement window")
        starts.append(start)
        steps.append(count)
        seconds.append(elapsed)
    if starts[0] != starts[1]:
        raise RuntimeError("DP2 rank measurement starts disagree")
    slowest = max(seconds)
    return {
        "rank_measurement_wall_seconds": seconds,
        "slowest_rank_measurement_wall_seconds": slowest,
        "measurement_updates_per_second": steps[0] / slowest,
        "timing_boundary": TIMING_BOUNDARY,
        "complete_5_warmup_20_measured_window": not resumed and warmup == 5 and measured == 20,
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("parallel_mode") != "data_parallel":
        raise ValueError("DP2 requires parallel_mode: data_parallel")
    cards = config.get("cards")
    if (type(cards) not in (list, tuple) or len(cards) != 2
            or any(type(card) is not int or card < 0 for card in cards) or cards[0] == cards[1]):
        raise ValueError("DP2 requires exactly two distinct integer cards")
    for key, expected in (("world_size", 2), ("sp", 1), ("sequence_parallel_size", 1)):
        value = config.get(key, expected)
        if type(value) is not int or value != expected:
            raise ValueError(f"unsupported DP2 {key}")
    for key in ("fsdp", "elastic", "optimizer_sharding"):
        if type(config.get(key, False)) is not bool or config.get(key, False):
            raise ValueError(f"DP2 rejects {key}")
    if any(key in config for key in ("rank", "local_rank", "local_world_size")):
        raise ValueError("rank topology is assigned by the fixed two-process launcher")
    if config.get("validate_recovery", False) is not False:
        raise ValueError("DP2 uses external-process resume, not in-process recovery")
    if config.get("checkpoint_step_basis", "global-optimizer-step") != "global-optimizer-step":
        raise ValueError("DP2 requires global optimizer checkpoint steps")
    for key, default, minimum, maximum in (("seed", 20260901, 0, 2**32 - 2),
                                          ("warmup_steps", 5, 0, 1000000),
                                          ("max_steps", 20, 1, 1000000),
                                          ("process_timeout_seconds", 3600, 1, 86400),
                                          ("collective_timeout_seconds", 600, 1, 1800)):
        value = config.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid DP2 {key}")
    warmup, maximum = config.get("warmup_steps", 5), config.get("max_steps", 20)
    steps = config.get("checkpoint_steps", [10, 20])
    if (type(steps) not in (list, tuple) or not steps
            or any(type(step) is not int or not warmup < step <= warmup + maximum for step in steps)):
        raise ValueError("invalid DP2 global checkpoint steps")
    sha = config.get("training_cache_sha")
    if type(sha) is not str or len(sha) != 64 or set(sha) - set("0123456789abcdef"):
        raise ValueError("DP2 requires an exact training_cache_sha")
    for key, default in (("learning_rate", 1e-5), ("max_grad_norm", 1.0),
                         ("train_timestep_shift", 3.0)):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid DP2 {key}")
    if config.get("deterministic_algorithms", True) is not True:
        raise ValueError("DP2 requires deterministic algorithms")


def worker_rank(environ: Any) -> int:
    if environ.get("WORLD_SIZE") != "2" or environ.get("LOCAL_WORLD_SIZE") != "2":
        raise ValueError("worker requires fixed local world size two")
    rank = environ.get("RANK")
    if rank not in ("0", "1") or environ.get("LOCAL_RANK") != rank:
        raise ValueError("worker requires rank == local_rank in {0,1}")
    if environ.get("TORCHELASTIC_RESTART_COUNT", "0") != "0" or environ.get("GROUP_RANK", "0") != "0":
        raise ValueError("elastic restarts and additional nodes are unsupported")
    return int(rank)


def validate_parent_launch(config: dict[str, Any], environ: Any = None) -> None:
    validate_config(config)
    environ = os.environ if environ is None else environ
    if any(name in environ for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("DP2 parent must not run inside a distributed launcher")
    validate_visible_cards(config, environ)


def validate_visible_cards(config: dict[str, Any], environ: Any, *, require_bound: bool = False) -> None:
    requested = ",".join(str(card) for card in config["cards"])
    for name in ("CUDA_VISIBLE_DEVICES",):
        value = environ.get(name)
        if value != requested and (value is not None or require_bound):
            raise RuntimeError(f"DP2 {name} does not match admitted cards {requested}")


def bind_visible_cards(config: dict[str, Any], environ: Any = None) -> None:
    environ = os.environ if environ is None else environ
    validate_visible_cards(config, environ)
    requested = ",".join(str(card) for card in config["cards"])
    for name in ("CUDA_VISIBLE_DEVICES",):
        environ[name] = requested


def validate_source_root(source: Path, *, approved_parent: Path) -> Path:
    raw_parent = approved_parent.expanduser()
    if raw_parent.is_symlink() or getattr(raw_parent, "is_junction", lambda: False)():
        raise RuntimeError("DP2 approved source parent must not be a symlink or junction")
    parent = raw_parent.resolve(strict=True)
    selected = source.expanduser().resolve(strict=True)
    if not parent.is_dir() or not selected.is_dir() or selected == parent or not selected.is_relative_to(parent):
        raise RuntimeError("DP2 native source escapes the approved source parent")
    # Source snapshots must not contain symlinks, including untraversed directory links.
    if (source.is_symlink() or getattr(source, "is_junction", lambda: False)()
            or any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                   or not path.resolve(strict=True).is_relative_to(selected) for path in selected.rglob("*"))):
        raise RuntimeError("DP2 native source tree must not contain symlinks")
    return selected


def validate_loaded_native_modules(source: Path, modules: Any = None) -> None:
    modules = sys.modules if modules is None else modules
    root = source.resolve(strict=True)
    for name, module in list(modules.items()):
        if name != "hyvideo" and not name.startswith("hyvideo."):
            continue
        filename = getattr(module, "__file__", None)
        paths = list(getattr(module, "__path__", ()))
        origin = getattr(getattr(module, "__spec__", None), "origin", filename)
        if filename is None and (not paths or origin is not None):
            raise RuntimeError(f"DP2 native transitive import escaped source root: {name}")
        if filename is not None and not Path(filename).resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"DP2 native transitive import escaped source root: {name}")
        for path in paths:
            if not Path(path).resolve(strict=True).is_relative_to(root):
                raise RuntimeError(f"DP2 native package search path escaped source root: {name}")
        if origin is not None and not Path(origin).resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"DP2 native module spec escaped source root: {name}")


def validate_dp_checks(checks: Any, *, resumed: bool) -> None:
    required = {"nonzero_finite_gradients", "replicated_state_agreement",
                "complete_trainable_gradient_coverage", "current_adapter",
                "peak_reserved_below_75gb", "main_forward_device"}
    if type(resumed) is not bool:
        raise TypeError("DP2 resumed mode must be a boolean")
    if resumed:
        required |= {"checkpoint_resume", "external_resume_boundary"}
    if type(checks) is not dict or set(checks) != required or any(checks[name] is not True for name in required):
        raise RuntimeError("DP2 required checks must have the exact schema and all be boolean True")


def validate_dp_result(result: Any, *, resumed: bool) -> None:
    if type(result) is not dict:
        raise RuntimeError("DP2 result must be a mapping")
    validate_dp_checks(result.get("checks"), resumed=resumed)


def exchange_json(torch: Any, value: Any, device: Any) -> list[Any]:
    checkpoint.require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 2,
                       "collective requires an initialized fixed DP2 group")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("ascii")
    length = torch.tensor([len(encoded)], dtype=torch.int32, device=device)
    lengths = [torch.empty_like(length) for _ in range(2)]
    torch.distributed.all_gather(lengths, length)
    sizes = [int(item.cpu().item()) for item in lengths]
    checkpoint.require(all(0 < size <= 8192 for size in sizes), "invalid collective metadata length")
    data = torch.zeros(max(sizes), dtype=torch.int32, device=device)
    data[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.int32, device=device)
    received = [torch.empty_like(data) for _ in range(2)]
    torch.distributed.all_gather(received, data)
    return [json.loads(bytes(item.cpu().tolist()[:size]).decode("ascii"))
            for item, size in zip(received, sizes)]


def collective_action(torch: Any, device: Any, label: str, action: Callable[[], Any]) -> Any:
    result = None
    error = None
    try:
        result = action()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"[:1000]
    peers = exchange_json(torch, {"error": error}, device)
    if any(peer["error"] is not None for peer in peers):
        raise RuntimeError(f"DP2 {label} rejected on a rank: {peers}")
    return result


def agree(torch: Any, device: Any, value: Any, label: str) -> None:
    peers = exchange_json(torch, value, device)
    checkpoint.require(peers[0] == peers[1], f"DP2 {label} disagreement")


def average_gradients(torch: Any, model: Any, optimizer: Any, device: Any) -> None:
    parameters = [p for group in optimizer.param_groups for p in group["params"]]

    def preflight() -> None:
        for p in parameters:
            checkpoint.require(not checkpoint.native._is_dtensor(p) and p.grad is not None
                               and not checkpoint.native._is_dtensor(p.grad),
                               "unused or DTensor trainable parameter; no complete-training claim")
            checkpoint.require(p.grad.layout == torch.strided and p.grad.dtype == p.dtype
                               and p.grad.shape == p.shape and str(p.grad.device) == str(p.device),
                               "invalid native gradient layout or placement")
            checkpoint.require(bool(torch.isfinite(p.grad).all()), "non-finite local gradient")

    collective_action(torch, device, "gradient preflight", preflight)
    for p in parameters:
        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
        p.grad.div_(2)
    collective_action(torch, device, "averaged gradient validation", preflight)


def _replica_state(model: Any, optimizer: Any, counts: dict[str, int]) -> str:
    return checkpoint.replica_sha({"state": checkpoint.native.capture_model_state(model, optimizer),
                                   "gradient_counts": counts})


def save_transaction(torch: Any, device: Any, directory: Path, contract: dict[str, Any],
                     model: Any, optimizer: Any, rank: int, step: int, counts: dict[str, int]) -> Path:
    def save_local() -> dict[str, Any]:
        payload = checkpoint.capture_rank(contract=contract, model=model, optimizer=optimizer,
                                          rank=rank, step=step, gradient_counts=counts)
        checkpoint.validate_rank(payload, contract, model, optimizer, rank)
        path = directory / f"rank-{rank}.pt"
        checkpoint.require(not path.exists() and not (directory / "COMMITTED.json").exists(),
                           "refusing to overwrite checkpoint rank state")
        checkpoint.native.save_atomic(path, payload)
        return {"rank": rank, "sha256": checkpoint.native.file_sha256(path),
                "replica_sha256": checkpoint.replica_sha(payload)}

    receipt = collective_action(torch, device, "rank-state save", save_local)
    receipts = exchange_json(torch, receipt, device)
    collective_action(torch, device, "transaction commit", lambda: checkpoint.publish_commit(
        directory, receipts, contract, step, writer_rank=rank) if rank == 0 else None)
    collective_action(torch, device, "committed transaction verification", lambda: checkpoint.load_committed(directory))
    return directory / "COMMITTED.json"


def restore_transaction(torch: Any, device: Any, directory: Path, contract: dict[str, Any],
                        model: Any, optimizer: Any, rank: int) -> tuple[int, dict[str, int]]:
    def preflight() -> tuple[dict[str, Any], dict[str, Any]]:
        manifest, payloads = checkpoint.load_committed(directory)
        checkpoint.require(checkpoint.native.tree_sha256(manifest["contract"]) == checkpoint.native.tree_sha256(contract),
                           "external transaction contract mismatch")
        checkpoint.require(manifest["global_step"] < contract["warmup_steps"] + contract["max_steps"],
                           "checkpoint already completed the configured schedule")
        payload = payloads[rank]
        checkpoint.validate_rank(payload, contract, model, optimizer, rank)
        return manifest, payload

    manifest, payload = collective_action(torch, device, "all-rank restore preflight", preflight)
    agree(torch, device, checkpoint.native.file_sha256(directory / "COMMITTED.json"), "restore transaction")
    collective_action(torch, device, "state application", lambda: checkpoint.apply_rank_state(payload, model, optimizer))
    observed = _replica_state(model, optimizer, payload["gradient_counts"])
    agree(torch, device, observed, "applied replica state")
    checkpoint.require(observed == checkpoint.replica_sha(payload), "restored optimizer/model bytes differ")
    collective_action(torch, device, "rank RNG restoration", lambda: checkpoint.restore_rank_rng(payload, contract, rank))

    def assert_boundary() -> None:
        restored = checkpoint.capture_rank(contract=contract, model=model, optimizer=optimizer,
                                           rank=rank, step=manifest["global_step"],
                                           gradient_counts=payload["gradient_counts"])
        checkpoint.require(restored["integrity_sha256"] == payload["integrity_sha256"], "restored rank boundary differs")

    collective_action(torch, device, "restored boundary", assert_boundary)
    return manifest["global_step"], dict(payload["gradient_counts"])


def _allocator_cap(torch: Any, rank: int) -> dict[str, Any]:
    total = int(torch.cuda.get_device_properties(rank).total_memory)
    requested = min(1.0, (MEMORY_LIMIT_BYTES - 1048576) / total)
    torch.cuda.set_per_process_memory_fraction(requested, rank)
    actual = float(torch.cuda.get_per_process_memory_fraction(rank))
    checkpoint.require(math.isfinite(actual) and 0 < actual <= 1 and int(total * actual) <= MEMORY_LIMIT_BYTES,
                       "allocator cap exceeds 75GB")
    return {"requested_fraction": requested, "actual_fraction": actual, "device_total_bytes": total}


def _worker(request: dict[str, Any]) -> None:
    config = request["config"]
    validate_config(config)
    rank = worker_rank(os.environ)
    validate_visible_cards(config, os.environ, require_bound=True)
    source = validate_source_root(Path(request["source_root"]),
                                  approved_parent=Path(request["approved_source_parent"]))
    validate_loaded_native_modules(source)
    torch = require_cuda()
    checkpoint.require(torch.cuda.device_count() == 2, "exactly two visible CUDA devices required")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    cap = _allocator_cap(torch, rank)
    torch.cuda.reset_peak_memory_stats(rank)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.distributed.init_process_group("nccl", init_method="env://", rank=rank, world_size=2,
                                        timeout=timedelta(seconds=config.get("collective_timeout_seconds", 600)))
    try:
        weights = Path(request["weights"])
        output = Path(request["output_dir"])
        rank_dir = output / f"rank-{rank}"
        rank_dir.mkdir(exist_ok=False)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, str(source))
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.optim.muon import get_muon_optimizer
        from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
        from peft import LoraConfig

        def source_preflight() -> None:
            validate_loaded_native_modules(source)
            for value in (initialize_parallel_state, get_muon_optimizer, HunyuanVideo_1_5_DiffusionTransformer):
                path = Path(inspect.getsourcefile(value)).resolve()
                checkpoint.require(path.is_relative_to(source.resolve()), "native import escaped the selected source root")
            checkpoint.require(native._source_sha256(get_muon_optimizer, "Muon") == checkpoint.MUON_SOURCE_SHA256,
                               "native Muon source mismatch")

        collective_action(torch, device, "native source admission", source_preflight)
        initialize_parallel_state(sp=1)
        seed = config.get("seed", 20260901)
        native._seed_initialization(torch, seed)
        torch.cuda.set_device(rank)
        cache_path = Path(config["training_cache"])

        def load_cache() -> dict[str, Any]:
            checkpoint.require(checkpoint.native.file_sha256(cache_path) == config["training_cache_sha"], "cache SHA mismatch")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            native.validate_hunyuanvideo_cache(cache)
            checkpoint.require(all(bool(torch.isfinite(cache[key].float()).all()) for key in native.REQUIRED_CACHE_SHAPES),
                               "non-finite native cache")
            return cache

        cache = collective_action(torch, device, "native cache admission", load_cache)
        model, factory, lora_type, version = native.prepare_native_model(torch, weights, config, device=device)
        collective_action(torch, device, "prepared native import origins", lambda: validate_loaded_native_modules(source))
        optimizer = factory(model, lr=float(config.get("learning_rate", 1e-5)),
                            weight_decay=float(config.get("weight_decay", 0.01)))
        identity = native._resume_identity(config=config, transformer_path=weights / "transformer" / version,
                                          cache_sha256=config["training_cache_sha"], transformer=model,
                                          get_muon_optimizer=factory, lora_config_type=lora_type, torch=torch)
        source_files = {path.relative_to(source).as_posix(): checkpoint.native.file_sha256(path)
                        for path in sorted(source.rglob("*.py"))}
        identity["dependencies"] = checkpoint.native.json_sha256({"native": identity["dependencies"],
            "dp_worker": checkpoint.native.file_sha256(__file__),
            "dp_checkpoint": checkpoint.native.file_sha256(checkpoint.__file__),
            "upstream_python": source_files,
            "nccl_python": native._source_sha256(torch.distributed.init_process_group, "NCCL initialization")})
        contract = checkpoint.build_contract(identity=identity, model=model, optimizer=optimizer, seed=seed,
            warmup_steps=config.get("warmup_steps", 5), max_steps=config.get("max_steps", 20), devices=["cuda:0", "cuda:1"])
        agree(torch, device, checkpoint.native.tree_sha256(contract), "parameter order and native contract")
        counts = checkpoint.native.common.initial_optimizer_progress(model, optimizer)
        agree(torch, device, _replica_state(model, optimizer, counts), "initial adapter and optimizer")
        flow_values = native.prepare_native_cache(torch, cache, device=device)
        forward_devices = []

        def check_input(_module: Any, inputs: tuple[Any, ...]) -> None:
            forward_devices.append(str(getattr(inputs[0], "device", "missing")))

        def check_output(_module: Any, _inputs: tuple[Any, ...], value: Any) -> None:
            forward_devices.append(str(getattr(value[0], "device", "missing")))

        pre_handle = model.register_forward_pre_hook(check_input)
        post_handle = model.register_forward_hook(check_output)
        random.seed(seed + rank)
        native.np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(seed + rank)
        completed = 0
        resume = config.get("resume_from_checkpoint")
        if resume is not None:
            completed, counts = restore_transaction(torch, device, Path(resume).expanduser().resolve(),
                                                   contract, model, optimizer, rank)
        before = native._adapter_sha(model)
        warmup, maximum = contract["warmup_steps"], contract["max_steps"]
        total = warmup + maximum
        checkpoint_steps = set(config.get("checkpoint_steps", [10, 20])) | {total}
        trainable = [p for p in model.parameters() if p.requires_grad]
        records, artifacts = [], []
        step_log = rank_dir / "steps.ndjson"
        started = perf_counter()
        measurement_window_started = None
        with step_log.open("x", encoding="utf-8") as handle:
            for step in range(completed + 1, total + 1):
                if getattr(model, "_hunyuanvideo_resume_poisoned", False):
                    raise RuntimeError("poisoned DP2 worker")
                if step > warmup and measurement_window_started is None:
                    torch.cuda.synchronize(rank)
                    torch.distributed.barrier()
                    measurement_window_started = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.set_device(rank)
                inputs = native.sample_native_flow_inputs(torch, flow_values, config, device=device)
                loss = native.native_flow_mse(torch, model, flow_values, inputs)
                collective_action(torch, device, "loss", lambda: checkpoint.require(bool(torch.isfinite(loss)), "non-finite loss"))
                loss.backward()
                average_gradients(torch, model, optimizer, device)
                norm = torch.nn.utils.clip_grad_norm_(trainable, float(config.get("max_grad_norm", 1.0)))
                collective_action(torch, device, "pre-update norm", lambda: checkpoint.require(
                    bool(torch.isfinite(norm)) and float(norm) > 0, "non-finite or zero gradient norm"))
                collective_action(torch, device, "native optimizer step", optimizer.step)
                counts = checkpoint.native.common.advance_optimizer_progress(counts, model, optimizer)
                payload = collective_action(torch, device, "post-update state validation", lambda: checkpoint.capture_rank(
                    contract=contract, model=model, optimizer=optimizer, rank=rank, step=step, gradient_counts=counts))
                fingerprint = checkpoint.replica_sha(payload)
                agree(torch, device, fingerprint, "post-update replica")
                torch.cuda.synchronize(rank)
                collective_action(torch, device, "memory bound", lambda: checkpoint.require(
                    within_memory_limit(memory_stats()["reserved"]), "rank allocator peak exceeds 75GB"))
                record = {"rank": rank, "global_step": step, "loss": float(loss.detach().cpu()),
                          "grad_norm_before_clip": float(norm.detach().cpu()), "replica_sha256": fingerprint,
                          "rank_state_sha256": payload["integrity_sha256"],
                          "rng_after_sha256": checkpoint.native.tree_sha256(payload["rng"]),
                          "flow_input_sha256": checkpoint.native.tree_sha256(checkpoint.native._cpu_tree(inputs))}
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if step in checkpoint_steps:
                    marker = save_transaction(torch, device, output / f"checkpoint-{step}", contract,
                                              model, optimizer, rank, step, counts)
                    if rank == 0:
                        artifacts.append({"path": str(marker), "sha256": checkpoint.native.file_sha256(marker)})
        torch.cuda.synchronize(rank)
        torch.distributed.barrier()
        measurement_wall_seconds = perf_counter() - measurement_window_started
        after = native._adapter_sha(model)
        memory = memory_stats()
        checks = {"nonzero_finite_gradients": bool(records), "replicated_state_agreement": True,
                  "complete_trainable_gradient_coverage": all(value == total for value in counts.values()),
                  "current_adapter": before != after, "peak_reserved_below_75gb": within_memory_limit(memory["reserved"]),
                  "main_forward_device":
                  bool(forward_devices) and all(item == str(device) for item in forward_devices)
                  and all(str(p.device) == str(device) for p in model.parameters())}
        if resume is not None:
            checks.update(checkpoint_resume=True, external_resume_boundary=True)
        collective_action(torch, device, "final native import origins", lambda: validate_loaded_native_modules(source))
        collective_action(torch, device, "final checks", lambda: validate_dp_checks(checks, resumed=resume is not None))
        summary = {"rank": rank, "contract": contract, "allocator_cap": cap, "cuda_memory": memory, "torch_threads": torch.get_num_threads(),
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                   "adapter_before": before, "adapter_after": after, "measurement_wall_seconds": measurement_wall_seconds,
                   "measurement_steps": total - max(completed, warmup),
                   "timing_boundary": TIMING_BOUNDARY, "completed_global_step_at_start": completed,
                   "invocation": native._invocation_counts(completed, warmup, total), "checks": checks}
        checkpoint.write_json_atomic(rank_dir / "summary.json", summary)
        pre_handle.remove()
        post_handle.remove()
        summaries = exchange_json(torch, {"rank": rank, "sha256": checkpoint.native.file_sha256(rank_dir / "summary.json")}, device)
        if rank == 0:
            native._save_adapter(output / "adapter-final.safetensors", model)
            artifacts.append({"path": str(output / "adapter-final.safetensors"),
                              "sha256": checkpoint.native.file_sha256(output / "adapter-final.safetensors")})
            logs = []
            for peer in summaries:
                directory = output / f"rank-{peer['rank']}"
                checkpoint.require(checkpoint.native.file_sha256(directory / "summary.json") == peer["sha256"], "rank summary changed")
                for name in ("steps.ndjson", "summary.json"):
                    logs.append({"path": str(directory / name), "sha256": checkpoint.native.file_sha256(directory / name)})
            result = {"artifacts": artifacts, "logs": logs, "metrics": {"parallel_mode": "data_parallel", "world_size": 2,
                **native._invocation_counts(completed, warmup, total), "training_seconds": perf_counter() - started,
                "steps": total - max(completed, warmup), "warmup_steps": max(0, warmup - completed),
                "total_global_steps": total,
                "performance_accepted": False}, "checks": checks, "stage_timings": {}}
            checkpoint.write_json_atomic(output / "dp2-result.json", result)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _terminate_owned(process: Any) -> None:
    # The child is a new session; never signal unrelated launcher/process groups.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _local_torchrun_command(worker_module: str, worker_args: list[str]) -> list[str]:
    # Standalone c10d can advertise the unresolvable hostname in network-none containers.
    return [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
            "--rdzv-backend=static", "--rdzv-endpoint=127.0.0.1:29500", "--node-rank=0",
            "--master-addr=127.0.0.1", "--master-port=29500", "--local-addr=127.0.0.1",
            "--max-restarts=0", "--module", worker_module, *worker_args]


def run_hunyuanvideo_data_parallel(weights: Path, config: dict[str, Any], output_dir: Path, source_root: Path) -> dict[str, Any]:
    validate_parent_launch(config)
    bind_visible_cards(config)
    approved_parent = source_root.expanduser().parent
    source_root = validate_source_root(source_root, approved_parent=approved_parent)
    approved_parent = approved_parent.resolve(strict=True)
    validate_loaded_native_modules(source_root)
    if os.name != "posix":
        raise RuntimeError("DP2 CUDA launcher requires Linux")
    if config.get("resume_from_checkpoint") is not None:
        checkpoint.load_committed(Path(config["resume_from_checkpoint"]).expanduser().resolve())
    visible = ",".join(str(card) for card in config["cards"])
    output_dir.mkdir(parents=True, exist_ok=True)
    request = output_dir / "dp2-request.json"
    checkpoint.require(not request.exists() and not (output_dir / "dp2-result.json").exists(), "DP2 output already used")
    checkpoint.write_json_atomic(request, {"config": config, "weights": str(weights.resolve()),
        "source_root": str(source_root), "approved_source_parent": str(approved_parent),
        "output_dir": str(output_dir.resolve())})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    argv = _local_torchrun_command("gencuda.backends.hunyuanvideo_data_parallel",
            ["--worker", str(request), "--request-sha", checkpoint.native.file_sha256(request)])
    with (output_dir / "torchrun.log").open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=config.get("process_timeout_seconds", 3600))
            if code != 0:
                raise RuntimeError(f"DP2 torchrun exited {code}; see {output_dir / 'torchrun.log'}")
        except BaseException:
            _terminate_owned(process)
            raise
    path = output_dir / "dp2-result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    resumed = config.get("resume_from_checkpoint") is not None
    validate_dp_result(result, resumed=resumed)
    result["metrics"].update(rank_window_metrics(
        output_dir, result["logs"], warmup=config.get("warmup_steps", 5),
        measured=config.get("max_steps", 20), resumed=resumed,
    ))
    result["logs"].append({"path": str(output_dir / "torchrun.log"), "sha256": checkpoint.native.file_sha256(output_dir / "torchrun.log")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--request-sha", required=True)
    arguments = parser.parse_args()
    checkpoint.require(checkpoint.native.file_sha256(arguments.worker) == arguments.request_sha, "worker request changed")
    _worker(json.loads(arguments.worker.read_text(encoding="utf-8")))
