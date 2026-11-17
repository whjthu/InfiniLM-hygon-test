"""Independent stdlib consumption of the one explicit Treasury five-update run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = ROOT / "training-qwen-treasury-explicit-sft-v1"
LABEL = "training-qwen-treasury-explicit-continuous-hip-v1"
REQUEST = "df40e4bcd2dd4c92994a2a362e875729"
PUBLICATION = "29631a9f1d8a316800ac824fe5783ec7fa0c66da"
CLAIM_SHA = "f94798a6535625ee9eadcbd1b0911366143c918c0937c1321da41b9bb7c0f649"
MANIFEST_SHA = "d6da1f93ff555a109761a332fee8e8ef8492e14bb35d34b49d9095652c8542f4"
DATA_SHA = "5a6a8f02dbfc77aa46487a0180000275771aefd56e3cc933ffdcf7c97496162e"
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
INITIAL = dict(file_SHA256="2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f",
               file_bytes=174821899, named_parameters_SHA256="086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137",
               parameter_tensors=504, parameter_numel=43646976, global_step=0,
               optimizer_state_entries=0, RNG_restored=False)
PLANNED = dict(optimizer_updates=5, ranks=8, microbatches_per_rank=20,
               rank_microbatch_occurrences=160, checkpoints=[2, 4, 5], checkpoint_state_files=42)
EPOCH = 1794878306.5448935
HANDLES = dict(pre="1087656.327", gpu="1087656.328", post="1087656.329")
ORIGINAL_PINS = {
    "intent.json": "b3b98ea16f4aec17b098726288e565a3e54dc49b471ceb9e0e3b449dac0e23e0",
    "launch-receipt.json": "95b718d78855d2c03ef63aff87d9cc8e8683b0ae5c1d94782cceab84b71d7405",
    "controller-outcome.json": "8e2fbfde544a1105096e25245ab68f9ef1a57571aeb4fbf6cb16b13fac9a80a8",
    "launch-outcome.json": "ae5265e8fc3c451c53f4401d132baa5a4dd852a4e50647d6cd452687024a7474",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path, limit=262144):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit,
            "missing, aliased or oversized original: " + path.name)
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def file_pin(path, limit):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "checkpoint file bound differs")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(identity(before) == identity(after), "checkpoint changed during streaming hash")
    return dict(bytes=before.st_size, SHA256=digest.hexdigest())


def fingerprint(value):
    require(isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{64}", value.get("SHA256", ""))
            and value.get("finite") is True, "invalid full tensor fingerprint")
    for name in ("tensors", "elements", "floating_elements"):
        require(type(value.get(name)) is int and value[name] >= 0, "invalid fingerprint count")
    require(type(value.get("floating_l2")) in (int, float) and math.isfinite(value["floating_l2"])
            and value["floating_l2"] >= 0, "invalid fingerprint norm")


def verify_cleanup(docs, steps):
    receipt = docs["post_accounting"]
    require(steps["post"][2:4] == ["COMPLETED", "0:0"] and receipt["row"] == steps["post"]
            and receipt["terminal_visible"] is True and receipt["passed"] is True and receipt["rc"] == 0,
            "independent post is not actual successful terminal")
    before, assigned, after = (docs[name] for name in ("before", "assigned", "after"))
    require(before["step"] == steps["pre"][0] and assigned["step"] == steps["gpu"][0]
            and after["step"] == steps["post"][0], "snapshot step identity differs")
    require(before["boot_id"] == assigned["boot_id"] == after["boot_id"]
            and len(set(before["pci"])) == 8
            and sorted(assigned["visible_pci"]) == before["pci"] == after["pci"]
            and before["kfd_holder_count"] == after["kfd_holder_count"] == 0
            and before["full_fd_inventory_performed"] is True
            and after["full_fd_inventory_performed"] is True and after["owned_remaining"] == 0,
            "independent device identity or post cleanup differs")
    return dict(clean=True, KFD_holders=0, owned_remaining=0)


def verify_records(docs, steps, checkpoint_pins):
    intent, launch = docs["intent"], docs["launch"]
    require(intent["request_id"] == launch["request_id"] == REQUEST
            and intent["publication"] == launch["publication"] == PUBLICATION
            and intent["claim_SHA256"] == launch["claim_SHA256"] == CLAIM_SHA
            and intent["parent"] == launch["parent"] == "1087656"
            and intent["node"] == launch["node"] == "a01r4n16", "original request or resource identity differs")
    require(type(intent["epoch"]) in (int, float) and intent["epoch"] == EPOCH
            and intent["deadline"] == intent["epoch"] + 1620
            and launch["epoch"] == intent["epoch"] and launch["deadline"] == intent["deadline"],
            "original single epoch differs")
    require(set(steps) == {"pre", "gpu", "post"}, "three-step denominator incomplete")
    for phase, row in steps.items():
        require(len(row) == 7 and row[0] == HANDLES[phase]
                and row[1] == LABEL + "-" + phase and row[2:4] == ["COMPLETED", "0:0"]
                and row[5] == "a01r4n16", "actual terminal accounting differs")
        tres = set(row[6].split(","))
        require("gres/dcu=8" in tres if phase == "gpu" else not any(x.startswith("gres/dcu=") for x in tres),
                "actual device allocation differs")
        receipt = docs[phase + "_accounting"]
        require(receipt == docs["controller"]["stages"][phase] and receipt["row"] == row
                and receipt["terminal_visible"] is True and receipt["passed"] is True and receipt["rc"] == 0,
                "stage receipt and actual accounting disagree")
    require(type(docs["controller"]["passed"]) is bool and docs["controller"]["failure"] is None,
            "original collector recorded a separate failure")
    native = docs["native"]
    require(native["passed"] is True and native["rc"] == 0 and native["error"] is None
            and native["cleanup"] == dict(owned_processes_remaining=0, termination_needed=False),
            "native run or owned-process cleanup failed")
    plan = docs["plan"]
    require(plan["dataset_SHA256"] == DATA_SHA and plan["native_preflight_step"] == "1087656.326"
            and plan["planned_updates"] == 5 and plan["domain_arm"] == "continuous"
            and plan["resume_from_checkpoint"] is None, "fresh native plan differs")
    ranks, losses, order, states = [], [[] for _ in range(5)], {}, []
    for rank in range(8):
        witness, initial, rng = (docs[name][rank] for name in ("witnesses", "initializations", "rng"))
        require(all(initial.get(key) == value for key, value in INITIAL.items()), "initial adapter or fresh optimizer differs")
        require(witness["rank"] == rank and witness["world_size"] == 8
                and witness["completed_global_step"] == 5 and witness["saved_checkpoints"] == [2, 4, 5]
                and len(witness["inputs"]) == len(witness["outputs"]) == 20
                and [s["global_step"] for s in witness["steps"]] == [1, 2, 3, 4, 5],
                "rank update or microbatch denominator differs")
        require(set(witness["frozen"]) == {"model.visual"}
                and witness["frozen"]["model.visual"]["tensors"] > 0
                and witness["frozen"]["model.visual"]["parameters"] > 0,
                "declared frozen visual scope differs")
        require(rng["rank"] == rank, "checkpoint-2 RNG rank differs")
        fingerprint(rng["RNG"])
        order[rank] = []
        for ordinal, (inp, output) in enumerate(zip(witness["inputs"], witness["outputs"]), 1):
            require(inp["ordinal"] == output["ordinal"] == ordinal and inp["global_step"] == (ordinal - 1) // 4,
                    "per-rank accumulation input order differs")
            fingerprint(inp["full_input"]); fingerprint(output["loss"])
            value = output["loss_scalar"]
            require(type(value) in (int, float) and math.isfinite(value), "nonfinite native loss")
            losses[(ordinal - 1) // 4].append(value)
            order[rank].append(inp["full_input"]["SHA256"])
        for step in witness["steps"]:
            require(set(step["state"]) == {"adapter", "optimizer", "scheduler"}, "update-state scope differs")
            for value in step["state"].values():
                fingerprint(value)
        hashes = [step["state"]["adapter"]["SHA256"] for step in witness["steps"]]
        require(hashes[0] != hashes[-1], "adapter did not change between recorded updates1 and5")
        states.append([step["state"] for step in witness["steps"]])
        ranks.append(dict(rank=rank, SHA256=docs["witness_SHA256"][rank]))
    require(all(value == states[0] for value in states) and native["finished"] == ranks,
            "rank update-state or native digest inventory differs")
    expected = {"args.json", "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", *(f"rng_state_{i}.pth" for i in range(8))}
    require(set(checkpoint_pins) == {2, 4, 5}, "checkpoint boundary denominator differs")
    for number in (2, 4, 5):
        state = docs["checkpoint_states"][number]
        require(state["global_step"] == number and state["max_steps"] == 5
                and abs(state["epoch"] - number / 5) < 1e-10, "checkpoint cursor differs")
        require(set(checkpoint_pins[number]) == expected
                and sum(x["bytes"] for x in checkpoint_pins[number].values()) <= 768 * 1024**2,
                "complete checkpoint file denominator or byte bound differs")
    return dict(actual_optimizer_updates=5, rank_microbatch_occurrences=160, all_rank_update_states_equal=True,
                adapter_change_between_recorded_updates1_and5=True, initial_adapter_verified=True,
                original_controller_passed=docs["controller"]["passed"], per_rank_input_order_SHA256=order,
                mean_rank_microbatch_loss=[math.fsum(x) / len(x) for x in losses], checkpoint_global_steps=[2, 4, 5],
                data_order_matches_old_input_control=None, row_ID_alignment_independently_proved=False,
                frozen_visual_requires_grad_boundary_observed=True, frozen_visual_parameter_byte_invariance=None,
                resume_executed=False, recovery_qualified=False, formal_T01_accepted=False,
                formal_performance=False, domain_effect_qualified=False)


def audit(root):
    require(root == RUN and not root.is_symlink(), "reader only consumes the original exact RUN")
    observed = subprocess.check_output(["sacct", "-n", "-P", "-j", "1087656", "-o", COLS], text=True, timeout=14)
    rows = [line.split("|") for line in observed.splitlines() if line]
    selected = [r for r in rows if len(r) == 7 and r[1] in {LABEL + "-" + x for x in ("pre", "gpu", "post")}]
    result = dict(original_request=REQUEST, publication=PUBLICATION, planned_denominator=PLANNED,
                  accounting_columns=COLS.split(","), actual_accounting=selected, original_artifact_pins={},
                  collected_rank_microbatches={}, missing_or_bad_originals=[], model_run_verified=False,
                  independent_post_verified=False, formal_T01_accepted=False, recovery_qualified=False)
    pins = result["original_artifact_pins"]
    def load(name, path=None):
        path = path or root / name
        value, digest = read_json(path); pins[str(path)] = digest
        if path.parent == root and path.name in ORIGINAL_PINS:
            require(digest == ORIGINAL_PINS[path.name], "actual original artifact pin differs: " + path.name)
        return value
    try:
        require(len(selected) == 3 and len({r[1] for r in selected}) == 3, "missing or duplicate actual steps")
        steps = {r[1].rsplit("-", 1)[-1]: r for r in selected}
        docs = {key: load(name) for key, name in dict(intent="intent.json", launch="launch-receipt.json",
            controller="controller-outcome.json", before="before.json", assigned="continuous/assigned.json",
            after="after.json").items()}
        for phase in ("pre", "gpu", "post"):
            docs[phase + "_accounting"] = load(phase + "-accounting.json")
        # Cleanup qualification is independent of model/witness validation.
        try:
            result["independent_post"] = verify_cleanup(docs, steps)
            result["independent_post_verified"] = True
        except (ValueError, KeyError, TypeError) as error:
            result["independent_post_error"] = str(error)
        require(all(steps[phase][0] == handle and steps[phase][2:4] == ["COMPLETED", "0:0"]
                    for phase, handle in HANDLES.items()), "actual327/328/329 terminal accounting differs")
        launch_outcome = load("launch-outcome.json")
        require(launch_outcome["request_id"] == REQUEST and launch_outcome["epoch"] == EPOCH
                and launch_outcome["rc"] == 0 and launch_outcome["client_timeout"] is False,
                "original controller invocation outcome differs")
        require(not (root / "src").is_symlink()
                and file_pin(root / "inputs.jsonl", 100000)["SHA256"] == DATA_SHA,
                "actual frozen training input differs")
        require(hashlib.sha256((root / "src/CLAIM.json").read_bytes()).hexdigest() == CLAIM_SHA
                and hashlib.sha256((root / "src/SOURCE_FILES.json").read_bytes()).hexdigest() == MANIFEST_SHA,
                "frozen training source claim/manifest changed")
        for name, info in load("src/SOURCE_FILES.json")["files"].items():
            require(Path(name).name == name and file_pin(root / "src" / name, 1024**2) == info,
                    "frozen source member differs")
        cpu_root = ROOT / "training-qwen-treasury-explicit-native-CPU-v2"
        for name, digest in {"outcome.json": "fd5dc40cca2033261bbd96debd6b98614c6adf490a029cf790c4a882eb955b89",
            "intent.json": "6437c5a8ae7525ff7f8dc03fa7beb18d179c0fd90cd08b63608794e78b692834",
            "step.log": "fe3ef60ded088bf977b8526da195c102e68f5b3696c1cd14976d504112171a90"}.items():
            require(file_pin(cpu_root / name, 4 * 1024**2)["SHA256"] == digest, "original CPU326 artifact differs")
        cpu = load("CPU326-outcome", cpu_root / "outcome.json")
        require(cpu == load("src/native-input-CPU.json") and cpu["accounting"]["row"] in rows
                and cpu["passed"] is True and cpu["metrics"]["dataset_SHA256"] == DATA_SHA,
                "actual CPU326 prerequisite no longer matches")
        docs["native"] = load("continuous/native-outcome.json")
        docs["plan"] = load("native-plan.json")
        docs.update(witnesses={}, initializations={}, rng={}, witness_SHA256={}, checkpoint_states={})
        for rank in range(8):
            folder = f"continuous/train/treasury-witness/rank-{rank:02d}/"
            for target, name in (("witnesses", "result.json"), ("initializations", "initialization.json"), ("rng", "checkpoint-2-rng.json")):
                try:
                    docs[target][rank] = load(folder + name)
                except (OSError, ValueError) as error:
                    result["missing_or_bad_originals"].append(dict(rank=rank, file=name, error=str(error)))
            if rank in docs["witnesses"]:
                value = docs["witnesses"][rank]
                result["collected_rank_microbatches"][rank] = dict(inputs=len(value.get("inputs", [])), outputs=len(value.get("outputs", [])), updates=len(value.get("steps", [])))
                docs["witness_SHA256"][rank] = pins[str(root / folder / "result.json")]
        checkpoint_pins = {}
        for number in (2, 4, 5):
            folder = f"continuous/train/checkpoint-{number}/"
            require((root / folder).is_dir() and not (root / folder).is_symlink(), "checkpoint directory aliased")
            docs["checkpoint_states"][number] = load(folder + "trainer_state.json")
            names = {"args.json", "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                     "optimizer.pt", "scheduler.pt", *(f"rng_state_{i}.pth" for i in range(8))}
            checkpoint_pins[number] = {name: file_pin(root / folder / name, (512 if name == "optimizer.pt" else 256) * 1024**2) for name in sorted(names)}
        result["checkpoint_files"] = checkpoint_pins
        require(not result["missing_or_bad_originals"], "rank evidence incomplete; full denominators retained")
        result["model_run"] = verify_records(docs, steps, checkpoint_pins)
        result["model_run_verified"] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["model_error"] = type(error).__name__ + ": " + str(error)
    result["passed"] = result["model_run_verified"] and result["independent_post_verified"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.root)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=value["passed"], output=str(args.output))))
