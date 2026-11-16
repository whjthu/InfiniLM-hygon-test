"""Exercise native skip on CPU using the fixed domain rows and load both Swift callbacks."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

from mid_epoch_recovery import BASE_PINS, CURSOR, WITNESS, candidate_sources
from qwen_domain_recipe import DATASET_SHA256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(framework, dataset):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "DCU assigned to CPU check")
    raw = dataset.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == DATASET_SHA256, "dataset identity changed")
    records = [hashlib.sha256(line).hexdigest() for line in raw.splitlines()]
    require(len(records) == len(set(records)) == 160, "fixed domain rows differ")

    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from swift.callbacks import callbacks_map
    import transformers.trainer as trainer_module

    candidate = candidate_sources(framework)
    importlib.import_module("vlm.runtime")
    for relative in (CURSOR, WITNESS):
        name = relative.removesuffix(".py").replace("/", ".")
        module = types.ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        module.__file__ = str(framework / relative)
        require(name not in sys.modules, "a runtime dependency loaded the original recovery module first")
        sys.modules[name] = module
        exec(compile(candidate[relative], module.__file__, "exec"), module.__dict__)

    import qwen_treasury_witness as plugin

    require(all(callbacks_map[name] is callback for name, callback in (
        ("qwen_treasury_continuous_witness", plugin.TreasuryContinuous),
        ("qwen_treasury_resume_witness", plugin.TreasuryResume))), "Swift callback registry differs")
    trainer_sha = hashlib.sha256(Path(trainer_module.__file__).read_bytes()).hexdigest()
    require(trainer_sha == plugin.TRAINER_SHA256, "actual Trainer implementation differs")

    def loader(rank):
        sampler = DistributedSampler(records, num_replicas=8, rank=rank, shuffle=True, seed=42, drop_last=False)
        return DataLoader(records, batch_size=1, sampler=sampler, num_workers=0)

    all_original, tail_sizes, tail_hashes = [], [], []
    for rank in range(8):
        identity = [row[0] for row in loader(rank)]
        resumed = trainer_module.skip_first_batches(loader(rank), 8)
        tail = [row[0] for row in resumed]
        require(len(identity) == 20 and len(resumed) == len(tail) == 12 and identity[8:] == tail,
                f"actual native CPU skip disagrees at logical rank {rank}")
        all_original.extend(identity)
        tail_sizes.append(len(tail))
        tail_hashes.append(hashlib.sha256("\n".join(tail).encode()).hexdigest())
    require(sorted(all_original) == sorted(records), "original logical rank coverage differs")
    return dict(state="NATIVE_CPU_SKIP_AND_CALLBACK_REGISTRATION_PASS", rows=len(records), actual_GPU_cards=0,
                actual_optimizer_updates=0, logical_ranks=8, tail_sizes=tail_sizes, paired_tail_rows=sum(tail_sizes),
                trainer_SHA256=trainer_sha, tail_inventory_SHA256=hashlib.sha256("\n".join(tail_hashes).encode()).hexdigest(),
                framework_base_source_SHA256=BASE_PINS,
                framework_candidate_source_SHA256={k: hashlib.sha256(v).hexdigest() for k, v in candidate.items()},
                actual_distributed_RNG_restored=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    cfg = p.parse_args()
    print(json.dumps(check(cfg.framework, cfg.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
