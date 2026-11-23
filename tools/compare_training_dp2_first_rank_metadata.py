"""Compare sealed rank-zero and rank-one first-update metadata for HunyuanVideo DP2."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re


CASE = "T09.dp2"
A100_RUN = "hunyuan-cuda-dp2-baseline-r1-repeat1"
A100_AUDIT = "1a815d5c8262ace65cd53e97fec57c70be8123c26adc043025febf3c7f10a873"
HY_STEP = "1103606.1612"
HY_EVIDENCE = "c738f699686bc32cc65c7b21a90e20d562764f19cdf84c31dd95dbc5737f681c"
AUDIT_FILES_SHA256 = {
    "a100": "5e099b38b295e0c9af79568c5246b8e3bfd537ba4fc79ce5dd0d446baf77db4a",
    "hygon": "715248822167207b824007a1c76237ebfd0bfe0e39bc203c608b2b90ed3e4aac",
}
RECIPE_FIELDS = (
    "world_size", "sp", "seed", "training_cache_sha", "transformer_version", "lora_target_modules",
    "lora_rank", "lora_alpha", "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps",
    "max_steps", "checkpoint_steps", "deterministic_algorithms", "gradient_checkpointing",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def hex_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(root, name, digest, size=None):
    supported = {"launch.json", "dp2-evidence.json", "dp2-result.json", "dp2-request.json"}
    need(name in supported or re.fullmatch(r"rank-[01]/(?:summary\.json|steps\.ndjson)", name), "unexpected file")
    path = root / name
    need(path.resolve(strict=True).is_relative_to(root) and not path.is_symlink(), "run file escaped root")
    with path.open("rb") as handle:
        raw = handle.read((2 << 20) + 1)
    need(0 < len(raw) <= 2 << 20 and (size is None or len(raw) == size)
         and hex_digest(digest) and sha256(raw).hexdigest() == digest, f"sealed bytes differ: {name}")
    return raw


def recipe(config):
    result = {key: config[key] for key in RECIPE_FIELDS}
    need(config["parallel_mode"] == "data_parallel" and result["world_size"] == 2 and result["sp"] == 1
         and result["warmup_steps"] == 5 and result["max_steps"] == 20
         and result["checkpoint_steps"] == [10, 20, 25] and hex_digest(result["training_cache_sha"]),
         "expected original DP2 recipe")
    return result


def rank_metadata(rank, summary_bytes, step_bytes):
    summary = json.loads(summary_bytes)
    steps = [json.loads(line) for line in step_bytes.splitlines()]
    need(summary["rank"] == rank and summary["completed_global_step_at_start"] == 0
         and len(steps) == 25 and [row["global_step"] for row in steps] == list(range(1, 26))
         and all(type(row["rank"]) is int and row["rank"] == rank for row in steps),
         "original continuous rank phase or update sequence differs")
    first = steps[0]
    need(all(hex_digest(value) for value in (summary["adapter_before"], first["flow_input_sha256"],
                                             first["replica_sha256"], first["rng_after_sha256"])),
         "missing first-update fingerprint")
    need(all(type(first[key]) in (int, float) and math.isfinite(first[key])
             for key in ("loss", "grad_norm_before_clip")) and first["grad_norm_before_clip"] > 0,
         "first-update metrics are not finite and nonzero")
    return {"rank": rank, "initial_adapter_SHA256": summary["adapter_before"],
            "flow_input_SHA256": first["flow_input_sha256"], "first_replica_SHA256": first["replica_sha256"],
            "loss": first["loss"], "gradient_norm_before_clip": first["grad_norm_before_clip"],
            "summary_SHA256": sha256(summary_bytes).hexdigest(), "steps_SHA256": sha256(step_bytes).hexdigest()}


def original_a100(root, audit):
    need(root.name == A100_RUN and audit["case"] == CASE and audit["original_receipt_SHA256"] == A100_AUDIT
         and audit["audit"]["passed"] is True, "unexpected A100 run or audit")
    files = audit["audit"]["runs"][A100_RUN]["original_artifacts"]

    def read(name):
        item = files[name]
        need(item["path"] == f"node/results/{A100_RUN}/{name}", "A100 original file substituted")
        return pinned(root, name, item["sha256"], item["bytes"])

    result = {"case": CASE, "run": A100_RUN, "A100_receipt_SHA256": A100_AUDIT,
              "recipe": recipe(json.loads(read("launch.json"))["config"])}
    result["rank_metadata"] = [
        rank_metadata(rank, read(f"rank-{rank}/summary.json"), read(f"rank-{rank}/steps.ndjson")) for rank in (0, 1)
    ]
    return result


def original_hygon(root, audit):
    independent = audit["independent_read_only_audit"]["audit"]
    evidence_pin = independent["continuous_evidence_SHA256"]
    need(root.name == "continuous" and audit["case"] == CASE and audit["scope"]["same_platform_trajectory"] is True
         and evidence_pin == HY_EVIDENCE
         and independent["slurm"]["runs"]["0"]["step"] == HY_STEP, "unexpected Hygon continuous run or audit")
    evidence = json.loads(pinned(root, "dp2-evidence.json", evidence_pin))
    need(evidence["schema"] == "hunyuanvideo-dp2-post-torchrun-evidence-v1"
         and evidence["torchrun_exit_code"] == 0 and evidence["backend_SHA256"] == independent["backend_SHA256"],
         "sealed Hygon run differs")
    request = json.loads(pinned(root, "dp2-request.json", evidence["request_SHA256"]))
    result = json.loads(pinned(root, "dp2-result.json", evidence["result_SHA256"]))
    need(result["checks"]["replicated_state_agreement"] is True, "Hygon replica check failed")

    def read(name):
        expected = ("continuous", *PurePosixPath(name).parts)
        matches = [row["sha256"] for row in result["logs"] if PurePosixPath(row["path"]).parts[-3:] == expected]
        need(len(matches) == 1, f"missing or ambiguous Hygon original pin: {name}")
        return pinned(root, name, matches[0])

    return {"case": CASE, "run": HY_STEP, "Hygon_evidence_SHA256": evidence_pin, "recipe": recipe(request["config"]),
            "rank_metadata": [
                rank_metadata(rank, read(f"rank-{rank}/summary.json"), read(f"rank-{rank}/steps.ndjson")) for rank in (0, 1)
            ]}


def compare(a100, hygon):
    need(a100["case"] == hygon["case"] == CASE and a100["run"] == A100_RUN and hygon["run"] == HY_STEP
         and a100["A100_receipt_SHA256"] == A100_AUDIT and hygon["Hygon_evidence_SHA256"] == HY_EVIDENCE,
         "expected the original A100 and Hygon scopes")
    different_recipe = [key for key in RECIPE_FIELDS if a100["recipe"][key] != hygon["recipe"][key]]
    rows = []
    need([row["rank"] for row in a100["rank_metadata"]] == [0, 1]
         and [row["rank"] for row in hygon["rank_metadata"]] == [0, 1], "both ordered ranks are required")
    for ref, obs in zip(a100["rank_metadata"], hygon["rank_metadata"], strict=True):
        for row in (ref, obs):
            need(all(hex_digest(row[key]) for key in ("initial_adapter_SHA256", "flow_input_SHA256",
                                                     "first_replica_SHA256", "steps_SHA256", "summary_SHA256"))
                 and all(type(row[key]) in (int, float) and math.isfinite(row[key])
                         for key in ("loss", "gradient_norm_before_clip"))
                 and row["gradient_norm_before_clip"] > 0, "invalid first-update metadata")
        same_adapter = ref["initial_adapter_SHA256"] == obs["initial_adapter_SHA256"]
        same_input = ref["flow_input_SHA256"] == obs["flow_input_SHA256"]
        comparable = not different_recipe and same_adapter and same_input
        rows.append({"rank": ref["rank"], "initial_adapter_matches": same_adapter, "flow_input_matches": same_input,
                     "logged_first_loss_absolute_delta": abs(obs["loss"] - ref["loss"]) if comparable else None,
                     "logged_preclip_norm_absolute_delta": abs(obs["gradient_norm_before_clip"]
                                                              - ref["gradient_norm_before_clip"]) if comparable else None})
    return {"case": CASE, "a100_run": A100_RUN, "hygon_step": HY_STEP, "differing_recipe_fields": different_recipe,
            "ranks": rows, "raw_tensors_exported": 0, "new_gpu_steps": 0, "formal_first_update_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("a100", "hygon", "compare"))
    parser.add_argument("first", type=Path, help="original run root, or A100 metadata file for compare")
    parser.add_argument("second", type=Path, help="published audit, or Hygon metadata file for compare")
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare(json.loads(args.first.read_bytes()), json.loads(args.second.read_bytes()))
    else:
        audit = args.second.read_bytes()
        need(sha256(audit).hexdigest() == AUDIT_FILES_SHA256[args.mode], "original independent audit differs")
        reader = original_a100 if args.mode == "a100" else original_hygon
        result = reader(args.first.resolve(strict=True), json.loads(audit))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
