"""Fixed DP2 replicated plain-Tensor state; no optimizer sharding or math."""
from __future__ import annotations

import json
import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any

from . import hunyuanvideo_resume_checkpoint as native

SCHEMA = "hunyuanvideo-dp2-replicated-full-step-v2"
MUON_SOURCE_SHA256 = "11862efb0b26bf308d313e4e18c7fcc0dd6934a275c640862ad7769b0bdfaccd"
Error = native.CheckpointValidationError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def topology(model: Any, optimizer: Any) -> dict[str, Any]:
    import torch

    native._validate_live_muon(model, optimizer)
    require(not any(type(module).__module__.startswith("torch.distributed.fsdp") for module in model.modules()),
            "FSDP modules are unsupported")
    implementation = native._optimizer_implementation(optimizer)
    require(implementation["source_sha256"] == MUON_SOURCE_SHA256, "unpinned native Muon source")
    names = native._parameter_names(model, optimizer)
    flattened = [p for group in optimizer.param_groups for p in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(len(flattened) == len(trainable) and {id(p) for p in flattened} == {id(p) for p in trainable},
            "optimizer must cover the exact trainable range")
    groups = []
    for group_names, group in zip(names, optimizer.param_groups):
        parameters = []
        for name, p in zip(group_names, group["params"]):
            branch = optimizer.state.get(p, {}).get("use_muon")
            require(type(branch) is bool and branch == (p.ndim >= 2), "native parameter branch mismatch")
            require(p.layout == torch.strided and p.is_floating_point(), "plain floating parameters required")
            parameters.append({"name": name, "shape": list(p.shape), "dtype": str(p.dtype),
                               "use_muon": branch, "replicas": [0, 1]})
        hyperparameters = native._cpu_tree({key: value for key, value in group.items() if key != "params"})
        hyperparameters["adamw_betas"] = list(hyperparameters["adamw_betas"])
        groups.append({"parameters": parameters, "hyperparameters": hyperparameters})
    return {"implementation": implementation, "groups": groups}


def build_contract(*, identity: dict[str, str], model: Any, optimizer: Any, seed: int,
                   warmup_steps: int, max_steps: int, devices: list[str]) -> dict[str, Any]:
    require(type(identity) is dict and set(identity) == native.IDENTITY_FIELDS, "invalid identity fields")
    require(all(type(v) is str and len(v) == 64 and set(v) <= set("0123456789abcdef")
                for v in identity.values()), "invalid identity SHA")
    require(type(seed) is int and 0 <= seed < 2**32 - 1, "invalid seed")
    require(type(warmup_steps) is int and warmup_steps >= 0 and type(max_steps) is int and max_steps > 0,
            "invalid global schedule")
    require(type(devices) is list and devices in (["cpu", "cpu"], ["cuda:0", "cuda:1"]), "invalid rank devices")
    return {"schema": SCHEMA, "world_size": 2, "sp": 1, "ownership": "replicated-plain-tensor",
            "identity": dict(identity), "seed": seed, "warmup_steps": warmup_steps,
            "max_steps": max_steps, "devices": list(devices), "topology": topology(model, optimizer)}


def rank_metadata(contract: dict[str, Any], rank: int, step: int) -> dict[str, Any]:
    validate_contract(contract)
    require(type(rank) is int and rank in (0, 1), "invalid DP2 rank")
    require(type(step) is int and 1 <= step <= contract["warmup_steps"] + contract["max_steps"],
            "not a completed global optimizer step")
    return {"contract": contract, "rank": rank, "device": contract["devices"][rank],
            "global_step": step, "next_global_step": step + 1,
            "rank_seed": contract["seed"] + rank, "cache_cursor": 0}


def validate_contract(contract: Any) -> None:
    require(type(contract) is dict and set(contract) == {"schema", "world_size", "sp", "ownership", "identity", "seed",
            "warmup_steps", "max_steps", "devices", "topology"}, "invalid DP2 contract fields")
    require(contract["schema"] == SCHEMA and type(contract["world_size"]) is int and contract["world_size"] == 2
            and type(contract["sp"]) is int and contract["sp"] == 1
            and contract["ownership"] == "replicated-plain-tensor", "unsupported distributed topology")
    require(type(contract["devices"]) is list and contract["devices"] in (["cpu", "cpu"], ["cuda:0", "cuda:1"]),
            "invalid rank devices")
    require(type(contract["seed"]) is int and 0 <= contract["seed"] < 2**32 - 1, "invalid rank seed")
    require(type(contract["warmup_steps"]) is int and contract["warmup_steps"] >= 0
            and type(contract["max_steps"]) is int and contract["max_steps"] > 0, "invalid global schedule")
    require(type(contract["identity"]) is dict and set(contract["identity"]) == native.IDENTITY_FIELDS
            and all(type(v) is str and len(v) == 64 and set(v) <= set("0123456789abcdef")
                    for v in contract["identity"].values()), "invalid identity")
    layout = contract["topology"]
    require(type(layout) is dict and set(layout) == {"implementation", "groups"}, "invalid optimizer topology")
    implementation = layout["implementation"]
    require(type(implementation) is dict and set(implementation) == {"module", "qualname", "source_sha256"}
            and implementation["qualname"] == "Muon" and implementation["source_sha256"] == MUON_SOURCE_SHA256
            and type(implementation["module"]) is str and bool(implementation["module"]), "invalid native implementation")
    require(type(layout["groups"]) is list and bool(layout["groups"]), "missing optimizer groups")
    names = []
    for group in layout["groups"]:
        require(type(group) is dict and set(group) == {"parameters", "hyperparameters"}, "invalid group topology")
        require(type(group["hyperparameters"]) is dict
                and set(group["hyperparameters"]) == native._MUON_GROUP_FIELDS - {"params"}, "invalid group hyperparameters")
        require(type(group["parameters"]) is list and bool(group["parameters"]), "empty group names")
        for parameter in group["parameters"]:
            require(type(parameter) is dict and set(parameter) == {"name", "shape", "dtype", "use_muon", "replicas"},
                    "invalid parameter topology")
            require(type(parameter["name"]) is str and bool(parameter["name"]), "invalid parameter name")
            require(type(parameter["shape"]) is list and all(type(d) is int and d > 0 for d in parameter["shape"]),
                    "invalid parameter shape")
            require(parameter["dtype"] in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")
                    and type(parameter["use_muon"]) is bool and parameter["use_muon"] == (len(parameter["shape"]) >= 2),
                    "invalid parameter branch or dtype")
            require(type(parameter["replicas"]) is list and parameter["replicas"] == [0, 1]
                    and all(type(rank) is int for rank in parameter["replicas"]), "optimizer partitioning is unsupported")
            names.append(parameter["name"])
    require(len(names) == len(set(names)), "duplicate parameter/group names")


def _device_ids(contract: dict[str, Any], rank: int) -> list[int]:
    return [] if contract["devices"][rank] == "cpu" else [rank]


def _pack_rng(rng: dict[str, Any]) -> dict[str, Any]:
    result = dict(rng)
    algorithm, keys, position, has_gauss, cached = rng["numpy"]
    result["numpy"] = (algorithm, keys.tolist(), position, has_gauss, cached)
    return result


def _unpack_rng(rng: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    require(type(rng) is dict and type(rng.get("numpy")) is tuple and len(rng["numpy"]) == 5,
            "invalid packed NumPy RNG")
    algorithm, keys, position, has_gauss, cached = rng["numpy"]
    require(type(keys) is list and len(keys) == 624
            and all(type(key) is int and 0 <= key < 2**32 for key in keys), "invalid NumPy RNG keys")
    result = dict(rng)
    result["numpy"] = (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
    return result


def _counts(counts: Any, model: Any, step: int) -> None:
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    require(type(counts) is dict and set(counts) == names, "gradient-consumption names mismatch")
    require(all(type(v) is int and 0 <= v <= step for v in counts.values()), "invalid consumed-gradient count")


def validate_rank(payload: dict[str, Any], contract: dict[str, Any], model: Any, optimizer: Any,
                  rank: int, *, native_rng: bool = True) -> None:
    require(type(payload) is dict and set(payload) == {"metadata", "state", "rng", "gradient_counts", "integrity_sha256"},
            "invalid rank-state fields")
    unsigned = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    require(payload["integrity_sha256"] == native.tree_sha256(unsigned), "rank-state integrity mismatch")
    metadata = payload["metadata"]
    require(type(metadata) is dict, "invalid rank metadata")
    step = metadata.get("global_step")
    require(native.tree_sha256(metadata) == native.tree_sha256(rank_metadata(contract, rank, step)),
            "rank, device, schedule or identity mismatch")
    rebuilt = build_contract(identity=contract["identity"], model=model, optimizer=optimizer,
                             seed=contract["seed"], warmup_steps=contract["warmup_steps"],
                             max_steps=contract["max_steps"], devices=contract["devices"])
    require(native.tree_sha256(rebuilt) == native.tree_sha256(contract), "native topology contract mismatch")
    require(all(str(value.device) == contract["devices"][rank]
                for value in list(model.parameters()) + list(model.buffers())), "wrong live rank placement")
    native.validate_model_state(payload["state"], model, optimizer, step)
    saved = payload["state"]["optimizer"]
    fresh = optimizer.state_dict()
    require(all(a["params"] == b["params"] and all(type(i) is int for i in a["params"])
                for a, b in zip(saved["param_groups"], fresh["param_groups"])), "swapped or out-of-bounds parameter IDs")
    _counts(payload["gradient_counts"], model, step)
    for group, names in zip(saved["param_groups"], payload["state"]["optimizer_parameter_names"]):
        for identifier, name in zip(group["params"], names):
            slots = saved["state"][identifier]
            if not slots["use_muon"]:
                consumed = payload["gradient_counts"][name]
                require((consumed == 0 and set(slots) == {"use_muon"})
                        or (consumed > 0 and slots.get("step") == consumed),
                        "AdamW step does not match explicit consumed gradients")
    native.common.validate_rng(_unpack_rng(payload["rng"]), _device_ids(contract, rank), native=native_rng)


def capture_rank(*, contract: dict[str, Any], model: Any, optimizer: Any, rank: int,
                 step: int, gradient_counts: dict[str, int]) -> dict[str, Any]:
    import torch

    payload = {"metadata": rank_metadata(contract, rank, step),
               "state": native.capture_model_state(model, optimizer),
               "rng": _pack_rng(native.common.capture_rng(_device_ids(contract, rank))),
               "gradient_counts": dict(gradient_counts)}
    payload["integrity_sha256"] = native.tree_sha256(payload)
    validate_rank(payload, contract, model, optimizer, rank, native_rng=False)
    return payload


def replica_sha(payload: dict[str, Any]) -> str:
    # Only placement metadata and rank RNG differ; no optimizer tensor is omitted.
    state = {key: value for key, value in payload["state"].items() if key != "optimizer_placements"}
    return native.tree_sha256({"state": state, "gradient_counts": payload["gradient_counts"]})


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".dp2-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rank(directory: Path, rank: int, sha: str) -> dict[str, Any]:
    import torch

    path = directory / f"rank-{rank}.pt"
    require(path.is_file() and not path.is_symlink(), "missing or substituted rank file")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha, "corrupt rank file")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    require(type(payload) is dict and set(payload) == {"metadata", "state", "rng", "gradient_counts", "integrity_sha256"},
            "invalid committed rank fields")
    require(payload["integrity_sha256"] == native.tree_sha256(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}), "rank integrity mismatch")
    return payload


def publish_commit(directory: Path, receipts: list[dict[str, Any]], contract: dict[str, Any], step: int,
                   *, writer_rank: int) -> Path:
    require(type(writer_rank) is int and writer_rank == 0, "only rank zero may publish")
    require(not (directory / "COMMITTED.json").exists(), "checkpoint already committed")
    require(type(receipts) is list and len(receipts) == 2, "missing rank receipt")
    fingerprints = []
    for rank, receipt in enumerate(receipts):
        require(type(receipt) is dict and set(receipt) == {"rank", "sha256", "replica_sha256"}
                and type(receipt["rank"]) is int and receipt["rank"] == rank, "duplicate or swapped rank receipt")
        payload = _read_rank(directory, rank, receipt["sha256"])
        require(native.tree_sha256(payload["metadata"]) == native.tree_sha256(rank_metadata(contract, rank, step)),
                "mixed rank contracts or global steps")
        fingerprint = replica_sha(payload)
        require(fingerprint == receipt["replica_sha256"], "rank receipt state mismatch")
        fingerprints.append(fingerprint)
    require(fingerprints[0] == fingerprints[1], "replicated optimizer/model states disagree")
    marker = directory / "COMMITTED.json"
    write_json_atomic(marker, {"schema": SCHEMA, "contract": contract, "global_step": step, "ranks": receipts})
    return marker


def load_committed_snapshot(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Return the verified transaction and the digest of the marker actually parsed."""
    marker = directory / "COMMITTED.json"
    require(marker.is_file() and not marker.is_symlink(), "checkpoint has no commit marker")
    marker_raw = marker.read_bytes()
    manifest = json.loads(marker_raw.decode("utf-8"))
    require(type(manifest) is dict and set(manifest) == {"schema", "contract", "global_step", "ranks"}
            and manifest["schema"] == SCHEMA, "not a DP2 v2 transaction")
    receipts = manifest["ranks"]
    require(type(receipts) is list and len(receipts) == 2, "transaction requires both ranks")
    payloads = []
    for rank, receipt in enumerate(receipts):
        require(type(receipt) is dict and set(receipt) == {"rank", "sha256", "replica_sha256"}
                and type(receipt["rank"]) is int and receipt["rank"] == rank, "invalid committed rank ordering")
        payload = _read_rank(directory, rank, receipt["sha256"])
        require(native.tree_sha256(payload["metadata"]) == native.tree_sha256(
            rank_metadata(manifest["contract"], rank, manifest["global_step"])), "mixed transaction metadata")
        require(replica_sha(payload) == receipt["replica_sha256"], "committed state digest mismatch")
        payloads.append(payload)
    require(replica_sha(payloads[0]) == replica_sha(payloads[1]), "replica divergence")
    return manifest, payloads, hashlib.sha256(marker_raw).hexdigest()


def load_committed(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, payloads, _ = load_committed_snapshot(directory)
    return manifest, payloads


def apply_rank_state(payload: dict[str, Any], model: Any, optimizer: Any) -> None:
    # Caller must complete all-rank preflight first; RNG is a separate final stage.
    try:
        native._copy_model_state(payload["state"], model, optimizer)
    except BaseException as error:
        setattr(model, "_hunyuanvideo_resume_poisoned", True)
        raise native.ResumeApplyFatal("DP2 state application failed; terminate all ranks") from error


def restore_rank_rng(payload: dict[str, Any], contract: dict[str, Any], rank: int) -> None:
    import torch

    native.common._restore_rng(_unpack_rng(payload["rng"]), _device_ids(contract, rank))
