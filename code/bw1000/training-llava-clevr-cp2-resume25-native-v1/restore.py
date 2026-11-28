"""Observe one real native CP2 restore and compare the live state to pinned members."""

from functools import wraps
import inspect
import json
from pathlib import Path
import random
import re


def need(ok, code):
    if not ok:
        raise ValueError(code)


def bind_hooks(trainer, checkpoint, load_rng, compare_rng, allowed_provider):
    """Wrap the existing bound loaders without replacing their implementation."""
    methods = ("_load_from_checkpoint", "_load_optimizer_and_scheduler", "_load_rng_state")
    originals, counts = {}, dict.fromkeys(methods, 0)
    for name in methods:
        original = getattr(trainer, name)
        allowed_provider(Path(inspect.getsourcefile(inspect.unwrap(original))), name)
        originals[name] = original

    def wrap(name, original):
        @wraps(original)
        def observed(*args, **kwargs):
            path = args[0] if args else next((kwargs[k] for k in
                    ("resume_from_checkpoint", "checkpoint") if k in kwargs), None)
            need(path is not None and Path(path).resolve(strict=True) == checkpoint and counts[name] == 0,
                 "EXACT_ONCE_ORIGINAL_CP2_TRAINER_RESTORE")
            saved = load_rng() if name == "_load_rng_state" else None
            result = original(*args, **kwargs)
            if name == "_load_rng_state":
                compare_rng(saved)
            counts[name] += 1
            return result
        return observed

    for name, original in originals.items():
        setattr(trainer, name, wrap(name, original))
    return counts, originals


def load_rng(torch, np, path):
    provider = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    with torch.serialization.safe_globals([
        (provider._reconstruct, "numpy.core.multiarray._reconstruct"),
        (provider._reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray, np.dtype, type(np.dtype("uint32")),
    ]):
        return torch.load(path, weights_only=True, map_location="cpu")


def compare_rng(torch, np, saved, rank):
    native = np.random.get_state()
    need(torch.equal(torch.random.get_rng_state(), saved["cpu"])
         and random.getstate() == saved["python"] and native[0] == saved["numpy"][0]
         and np.array_equal(native[1], saved["numpy"][1]) and native[2:] == saved["numpy"][2:]
         and len(saved["cuda"]) == 8 and torch.equal(torch.cuda.get_rng_state(rank), saved["cuda"][rank]),
         "EXACT_RESTORED_RANK_CPU_PYTHON_NUMPY_AND_LOCAL_DCU_RNG")


def named_groups(live, named, saved):
    owners = {id(parameter): name for name, parameter in named}
    need(len(owners) == len(named) == len(set(owners.values())) == 504
         and len(live.param_groups) == len(saved["param_groups"]) > 0 and len(live.state) == 504,
         "EXACT_504_OPTIMIZER_NAMED_OWNER_COUNT")
    pairs = []
    for live_group, saved_group in zip(live.param_groups, saved["param_groups"], strict=True):
        for parameter, index in zip(live_group["params"], saved_group["params"], strict=True):
            need(type(index) is int and id(parameter) in owners and parameter in live.state,
                 "EXACT_SAVED_OPTIMIZER_INDEX_TO_NAMED_PARAMETER")
            pairs.append((index, owners[id(parameter)], parameter))
        need(all(live_group[key] == saved_group[key] for key in
                 ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")), "NATIVE_OPTIMIZER_GROUP_SETTINGS")
    need(len(pairs) == len({name for _, name, _ in pairs}) == 504
         and {index for index, _, _ in pairs} == set(saved["state"]) and len(saved["state"]) == 504,
         "BIJECTIVE_NATIVE_ADAMW_INDEX_TO_NAME")
    return pairs


def live_resume(torch, trainer, rank_parameters, root, fingerprint):
    from safetensors.torch import load_file

    source = load_file(str(root / "adapter_model.safetensors"), device="cpu")
    named = dict(rank_parameters)
    need(len(named) == len(source) == 504 and all(re.search(r"\.lora_[AB]\.weight$", name) for name in source),
         "EXACT_ORIGINAL_CP2_SAVED_ADAPTER_PARAMETERS")
    expected = {key.removesuffix(".weight") + ".default.weight": value for key, value in source.items()}
    need(set(named) == set(expected) and all(value.dtype == named[key].dtype
         and value.shape == named[key].shape and torch.equal(value, named[key].detach().cpu())
         for key, value in expected.items()), "LIVE_ADAPTER_MUST_MATCH_ACTUAL_CP2_SAFETENSORS")
    adapter = fingerprint(expected)["SHA256"]
    del expected, source
    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    need(type(optimizer) is torch.optim.AdamW, "EXACT_ORIGINAL_SWIFT_ADAMW")
    saved = torch.load(root / "optimizer.pt", weights_only=True, map_location="cpu")
    mapped = named_groups(optimizer, rank_parameters, saved)
    for index, _, parameter in mapped:
        current, old = optimizer.state[parameter], saved["state"][index]
        need(set(current) == set(old) == {"step", "exp_avg", "exp_avg_sq"} and old["step"].item() == 2
             and old["exp_avg"].shape == old["exp_avg_sq"].shape == parameter.shape
             and all(tensor.dtype == current[key].dtype and torch.equal(tensor, current[key].detach().cpu())
                     for key, tensor in old.items()), "ACTUAL_RESTORED_504_NAMED_ADAMW_MOMENTS")
    del saved
    from hashlib import sha256
    basis = [[key, name] for key, name, _ in mapped]
    prepared = sha256(json.dumps(basis, separators=(",", ":")).encode()).hexdigest()
    scheduler = torch.load(root / "scheduler.pt", weights_only=True, map_location="cpu")
    need((scheduler["last_epoch"], scheduler["_step_count"]) == (2, 3)
         and trainer.lr_scheduler.state_dict() == scheduler
         and scheduler["_last_lr"] == [group["lr"] for group in optimizer.param_groups]
         and all(0 < rate < 0.0001 for rate in scheduler["_last_lr"]), "EXACT_RESUMED_COSINE_SCHEDULER_STATE")
    return optimizer, dict(adapter_SHA256=adapter, adamw_index_to_named_SHA256=prepared,
                           native_named_adamw_states=504, scheduler_next_lrs=scheduler["_last_lr"])
