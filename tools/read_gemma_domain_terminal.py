"""Consume saved Slurm, checkpoint and rank evidence for one Gemma domain pair."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_gemma_domain_recovery import compare, read_receipt, require
from read_training_slurm_membership import allocated_dcu, completed_numbered_step


LABEL = "training-gemma-usgs-render72-exposure25-resume20-v1"
PARENT = "1103606"
NODE = "f13r1n12"
DATASET_SHA = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"
FRAMEWORK = "a68952cb2285d6593b665147249d14ec1f8cecbc"
PHASES = {"pre": 0, "gpu": 8, "post": 0}
REQUIRED_CHECKPOINT_FILES = {
    "adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "args.json",
    "optimizer.pt", "scheduler.pt", *(f"rng_state_{rank}.pth" for rank in range(8)),
}


def token(value, size):
    return isinstance(value, str) and len(value) == size and re.fullmatch(r"[0-9a-f]+", value) is not None



def numbered_step(value):
    return isinstance(value, str) and re.fullmatch(fr"{PARENT}\.[0-9]+", value) is not None


def bound_step(receipt, request, intent_sha):
    require(receipt["request_id"] == request and receipt["intent_SHA256"] == intent_sha
            and receipt["node"] == NODE and numbered_step(receipt["step"]),
            "Slurm receipt does not belong to the original request")
    return receipt["step"]


def bind_original_steps(binding, intent_sha, before, gpu, after, evidence):
    request = binding["request_id"]
    ids = {phase: bound_step(receipt, request, intent_sha)
           for phase, receipt in (("pre", before), ("gpu", gpu), ("post", after))}
    require(len(set(ids.values())) == 3 and gpu["pre_step"] == ids["pre"]
            and gpu["before_SHA256"] == evidence["before.json"]
            and after["gpu_step"] == ids["gpu"]
            and after["gpu_step_binding_SHA256"] == evidence["gpu-step-binding.json"],
            "preflight, native worker and post step do not form one original request")
    return dict(binding, steps=ids)


def terminal_phases(binding, accounting):
    columns, rows = accounting["columns"], accounting["rows"]
    ids = binding["steps"]
    require(set(ids) == set(PHASES) == set(rows) and len(set(ids.values())) == len(PHASES),
            "expected three distinct original Slurm steps")
    require({"JobName", "NodeList", "ElapsedRaw"} <= set(k.split("%", 1)[0] for k in columns),
            "accounting is missing time or allocation identity")
    verified = {}
    for phase, cards in PHASES.items():
        expected_id = ids[phase]
        require(numbered_step(expected_id),
                "Slurm step is outside the frozen parent")
        result = completed_numbered_step(columns, rows[phase], expected_id, scope=phase)
        require(result["JobName"] == f"{LABEL}-{phase}" and result["NodeList"] == NODE
                and allocated_dcu(result["AllocTRES"]) == cards
                and "node=1" in result["AllocTRES"].split(",")
                and isinstance(result["ElapsedRaw"], str) and result["ElapsedRaw"].isdecimal(),
                "accounting resource or label differs")
        verified[phase] = result
    return verified


def valid_post(binding, before, after):
    pci = before["pci"]
    require(len(pci) == len(set(pci)) == 8
            and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", p) for p in pci),
            "expected eight distinct original PCI devices")
    require(before["pci"] == after["pci"] and before["boot_id"] == after["boot_id"]
            and isinstance(before["boot_id"], str) and len(before["boot_id"]) == 36,
            "post snapshot does not describe the original node")
    for phase, receipt in (("pre", before), ("post", after)):
        require(receipt["node"] == NODE and receipt["step"] == binding["steps"][phase]
                and receipt["full_fd_inventory_performed"] is True
                and receipt["kfd_holder_count"] == 0 and receipt["master_port"] == binding["master_port"]
                and receipt["master_port_available"] is True,
                "original node snapshot or process/port admission differs")
    require(after["owned_registry_present"] is True and after["owned_processes_remaining"] == []
            and type(before["monotonic_ns"]) is int and type(after["monotonic_ns"]) is int
            and 0 < before["monotonic_ns"] < after["monotonic_ns"], "independent post-cleanup evidence is incomplete")


def audit_checkpoint(root, arm, step, evidence):
    inventory_path = Path(f"{arm}-checkpoint-{step}-files.json")
    checkpoint = Path(arm) / "train" / f"checkpoint-{step}"
    inventory = read_receipt(root, inventory_path, evidence)
    files = inventory["files"]
    require(inventory["global_step"] == step and set(files) == REQUIRED_CHECKPOINT_FILES,
            "checkpoint inventory must cover adapter, optimizer, scheduler and all rank RNG files")
    for name, pin in files.items():
        path = root / checkpoint / name
        require(type(pin["bytes"]) is int and pin["bytes"] > 0 and token(pin["SHA256"], 64)
                and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                and path.stat().st_size == pin["bytes"], "checkpoint member is missing or differs")
    state_path = checkpoint / "trainer_state.json"
    trainer = read_receipt(root, state_path, evidence)
    require(trainer["global_step"] == step and trainer["max_steps"] == 25
            and trainer["train_batch_size"] == 1
            and evidence[state_path.as_posix()] == files["trainer_state.json"]["SHA256"],
            "actual Trainer state differs from the inventoried checkpoint")
    args_path = checkpoint / "args.json"
    args = read_receipt(root, args_path, evidence)
    require(args["max_steps"] == 25 and args["gradient_accumulation_steps"] == 4
            and args["per_device_train_batch_size"] == 1
            and evidence[args_path.as_posix()] == files["args.json"]["SHA256"],
            "actual Swift arguments differ from the inventoried checkpoint")


def bind_rank_outcome(root, binding, arm, evidence, rank_pins):
    outcome = read_receipt(root, Path(arm) / "native-outcome.json", evidence)
    require(outcome["request_id"] == binding["request_id"] and outcome["step"] == binding["steps"]["gpu"]
            and outcome["passed"] is True and outcome["rc"] == 0 and outcome["owned_processes_remaining"] == 0,
            "native controller did not finish the bound arm")
    rows = outcome["ranks"]
    require(len(rows) == 8 and {row["rank"] for row in rows} == set(range(8)),
            "native controller lacks the eight original ranks")
    kind = "continuous-witness" if arm == "continuous" else "resume-witness"
    for row in rows:
        base = f"{arm}/train/{kind}/rank-{row['rank']:02d}"
        for field, file in (("result_SHA256", "result.json"), ("owner_SHA256", "owner.json")):
            require(token(row[field], 64) and rank_pins.get(f"{base}/{file}") == row[field],
                    "native controller and independently read rank receipt differ")


def read_terminal(root, pair_manifest, *, request, publication, intent_sha):
    root = root.resolve(strict=True)
    require(token(request, 32) and token(publication, 40) and token(intent_sha, 64), "invalid original identities")
    evidence = {}
    binding = read_receipt(root, Path("intent.json"), evidence)
    require(evidence["intent.json"] == intent_sha and binding["request_id"] == request
            and binding["publication"] == publication and binding["framework_base"] == FRAMEWORK
            and binding["dataset_SHA256"] == DATASET_SHA and binding["label"] == LABEL
            and "steps" not in binding
            and type(binding["master_port"]) is int and 1024 <= binding["master_port"] <= 65535,
            "original Gemma request binding differs")
    manifest = json.loads(pair_manifest)
    gates = manifest["required_state"]
    require(hashlib.sha256(pair_manifest).hexdigest() == binding["pair_manifest_SHA256"]
            and manifest["registered_case"] == "T02.sft" and manifest["base_framework_commit"] == FRAMEWORK
            and (gates["world_size"], gates["continuous_updates"], gates["resume_from"],
                 gates["resumed_updates"]) == (8, 25, 20, 5), "paired command manifest differs")
    before, after = [read_receipt(root, Path(name), evidence) for name in ("before.json", "after.json")]
    gpu = read_receipt(root, Path("gpu-step-binding.json"), evidence)
    binding = bind_original_steps(binding, intent_sha, before, gpu, after, evidence)
    accounting = read_receipt(root, Path("terminal-accounting.json"), evidence)
    require(accounting["request_id"] == request and accounting["intent_SHA256"] == intent_sha
            and accounting["gpu_step_binding_SHA256"] == evidence["gpu-step-binding.json"]
            and accounting["after_SHA256"] == evidence["after.json"],
            "accounting belongs to a different original request or cleanup evidence")
    phases = terminal_phases(binding, accounting)
    valid_post(binding, before, after)
    for arm, steps in (("continuous", (10, 20, 25)), ("resumed", (25,))):
        for step in steps:
            audit_checkpoint(root, arm, step, evidence)
    paired = compare(root)
    if paired["exact_paired_recovery"]:
        for arm in ("continuous", "resumed"):
            bind_rank_outcome(root, binding, arm, evidence, paired["evidence_SHA256"])
    return dict(request=request, publication=publication, steps=phases,
                metadata_pins=evidence, rank_comparison=paired,
                checkpoint_tensor_payloads_independently_hashed=False,
                scope="Gemma domain engineering recovery only, with saved original accounting and post snapshot",
                passed=paired["exact_paired_recovery"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    for name in ("request", "publication", "intent-sha"):
        parser.add_argument("--" + name, required=True)
    opts = parser.parse_args()
    with opts.pair_manifest.open("rb") as stream:
        manifest = stream.read(65537)
    require(0 < len(manifest) <= 65536, "paired command manifest is outside its size bound")
    outcome = read_terminal(opts.root, manifest, request=opts.request, publication=opts.publication,
                            intent_sha=opts.intent_sha)
    print(json.dumps(outcome, sort_keys=True))
    raise SystemExit(0 if outcome["passed"] else 1)


if __name__ == "__main__":
    main()
