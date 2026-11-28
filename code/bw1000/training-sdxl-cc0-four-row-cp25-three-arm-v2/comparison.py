"""Check the original two arms and assemble the new four-row arm without copying images."""

import argparse
import json
from pathlib import Path
import re

from scheduler_bridge import verify as verify_bridge
from plan import CONFIG, HEX, digest, frozen, need, put, read, same_noise, wire

NEW_ARM = "four_row_cp25"
OLD_REQUEST = "94f0acaeec614c5fa9715033ae1a755d"
NEW_TRAIN_REQUEST = "a9a91b5727d8462cb7f3dc4eb483875f"
OLD_RESULT = "d14e800d2744556a67b9410898522915b3938d52f79080f2f50a6d1751c8a08f"
OLD_LATENTS = "97fd295d028a65269ee428d016eefc68d7d3ba80a59be7108682159cb61b4fe8"
TRAIN_PROOF = "41337046173a5f602124966c37ea866fa3d687d393382b428705f8fdf9432219"
INPUT_PROOF = "c1a5ed37c352ccf03f5a4ee9b4a3073e6b70dc8a63cdde1b960322b28efab472"
ADAPTER_SHA = "7301c4f968e79b93c1ff183b03f2275789d61de06565d8b2c2ea091af27b876a"
ADAPTER_BYTES = 11778368
CHECKPOINT_SHA = "b108ecd38a9ca05ffef58ce5454adc3db9169282d2a2f79a8a057613c4bbe14a"
CHECKPOINT_BYTES = 462247
CHECKPOINT_STATE = "ee103bf1a4a1b2a43e45b444d132b42545d7a15111b8913bcf234077737609f1"
TRAIN_STATE = "2d92bb833a6854c5b054a8824b2e52d3f5cd6e4a11a25747486f92d132e08dc7"
CHECKPOINT_ROOT = Path("/four-row-a9a91b5727d8/continuous/checkpoint-25")
MEMBER = "pytorch_lora_weights.safetensors"
MANIFEST = "genmlu-checkpoint.json"
KEYS = ("pair", "family", "replicate", "seed", "prompt_SHA256")


def checkpoint(args):
    status = json.loads(read(args.training_proof, 32 << 10, TRAIN_PROOF))["terminal"]
    need(status["accounting"]["gpu"][:4] == ["1194900.2769", "sdxl-cc0-four-full25-cp20-a9a91b5727d8-gpu",
                                           "COMPLETED", "0:0"]
         and status["accounting"]["post"][0] == "1194900.2770" and status["request"] == NEW_TRAIN_REQUEST
         and status["completed_continuous"] == 25 and status["completed_resume"] == 5
         and status["scientific_result_SHA256"] == TRAIN_STATE and status["verified_input_rows"] == 4
         and status["verified_input_2766_SHA256"] == INPUT_PROOF and status["cleanup_verified"] is True,
         "EXACT_FOUR_ROW_TRAINING_TERMINAL")
    need(args.adapter_sha == ADAPTER_SHA and args.checkpoint_manifest_sha == CHECKPOINT_SHA
         and args.adapter == CHECKPOINT_ROOT / MEMBER and args.checkpoint_manifest == CHECKPOINT_ROOT / MANIFEST,
         "EXACT_2769_CONTINUOUS_CP25_AUTHORITY")
    raw_manifest = read(args.checkpoint_manifest, CHECKPOINT_BYTES, args.checkpoint_manifest_sha)
    need(len(raw_manifest) == CHECKPOINT_BYTES, "EXACT_2769_CP25_MANIFEST_EXTENT")
    manifest = json.loads(raw_manifest)
    keys = {"schema", "boundary", "step", "next_step", "contract", "files", "state_sha256", "integrity_sha256"}
    need(set(manifest) == keys and manifest["schema"] == "sdxl-native-lora-full-step-hip-v1"
         and (manifest["step"], manifest["next_step"]) == (25, 26)
         and manifest["boundary"] == "after-complete-optimizer-scheduler-update"
         and manifest["state_sha256"] == CHECKPOINT_STATE and manifest["integrity_sha256"] ==
         digest(wire({key: value for key, value in manifest.items() if key != "integrity_sha256"}).rstrip(b"\n")),
         "EXACT_NATIVE_CP25_CHECKPOINT_MANIFEST")
    data = manifest["contract"]["data"]
    records = data["records"]
    need(manifest["contract"]["hip"] is True and manifest["contract"]["plan"]["max_train_steps"] == 25
         and [row["id"] for row in records] == list(range(4))
         and [row["path"] for row in records] == ["metal_tool_chest_original.png", "metal_tool_chest_side.png",
                                                  "painted_wooden_stool_original.png", "painted_wooden_stool_side.png"]
         and data["ordered_sha256"] == digest(wire(records).rstrip(b"\n")), "EXACT_FOUR_TRAINING_ROW_CONTRACT")
    pin = manifest["files"][MEMBER]
    need(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and pin["bytes"] == ADAPTER_BYTES
         and pin["sha256"] == args.adapter_sha, "CP25_MANIFEST_ADAPTER_PIN")
    raw = read(args.adapter, pin["bytes"], args.adapter_sha)
    need(len(raw) == pin["bytes"], "CP25_ADAPTER_EXTENT")
    return raw


def results(root, plan, plan_sha, arms, result_sha, *, images):
    raw = read(root / "result.json", 64 << 10, result_sha)
    status = json.loads(raw)
    need(status["sealed"] is True and status["planned_slots"] == len(arms) * 6
         and status["plan_SHA256"] == plan_sha, "EXACT_SEALED_ARM_SCOPE")
    ledger = read(root / "generations.jsonl", 128 << 10, status["generations_SHA256"])
    rows = [json.loads(line) for line in ledger.splitlines()]
    need(len(ledger) == status["generations_bytes"] and len(rows) == len(arms) * 6, "EXACT_SEALED_ARM_DENOMINATOR")
    result = {}
    for (arm, sample), row in zip(((arm, sample) for arm in arms for sample in plan["samples"]), rows, strict=True):
        pair = sample["pair"]
        need(row["arm"] == arm and all(row[key] == sample[key] for key in KEYS)
             and row["outcome"] in ("completed", "error", "missing"), "EXACT_FROZEN_SLOT_IDENTITY")
        if row["outcome"] == "completed":
            need(row["image"] == f"{arm}-{pair:02d}.private.png" and type(row["image_bytes"]) is int
                 and 0 < row["image_bytes"] <= 2 << 20 and bool(HEX.fullmatch(row["image_SHA256"])),
                 "EXACT_SEALED_SLOT_IMAGE_IDENTITY")
            if images:
                image = read(root / row["image"], row["image_bytes"], row["image_SHA256"])
                need(len(image) == row["image_bytes"] and image.startswith(b"\x89PNG\r\n\x1a\n"),
                     "EXACT_ORIGINAL_PNG_EXTENT")
        result[arm, pair] = row
    return status, result


def original(root, plan, plan_sha, *, images=False):
    status, rows = results(root, plan, plan_sha, ("base", "cp25"), OLD_RESULT, images=images)
    need(status["request"] == OLD_REQUEST and status["facts"]["step"] == "1194900.2762" and status["passed"] is True
         and status["facts"]["latent_manifest_SHA256"] == OLD_LATENTS and status["complete_same_noise_pairs"] == 6
         and status["facts"]["actual_generation_calls"] == 12
         and HEX.fullmatch(status["facts"]["effective_scheduler_config_SHA256"])
         and all(rows["base", i]["outcome"] == rows["cp25", i]["outcome"] == "completed"
                 and same_noise(rows["base", i], rows["cp25", i]) for i in range(6)),
         "EXACT_ORIGINAL_2762_TWO_ARM_GENERATIONS")
    return status, rows


def old_noise(root, plan, original_rows):
    folder = root / "latents"
    pins = json.loads(read(folder / "LATENTS.json", 16 << 10, OLD_LATENTS))
    need(len(pins) == 6, "EXACT_SIX_FROZEN_NOISE_MEMBERS")
    result = {}
    for pin, item in zip(pins, plan["samples"], strict=True):
        pair = item["pair"]
        initial = original_rows["base", pair]["raw_latent_SHA256"]
        name = f"pair-{pair:02d}.bf16.private"
        need(pin == dict(pair=pair, file=name, seed=item["seed"], dtype=CONFIG["latent_dtype"],
                         shape=CONFIG["latent_shape"], bytes=8192, SHA256=initial), "EXACT_ORIGINAL_LATENT_PIN")
        raw = read(folder / name, 8192, initial)
        need(len(raw) == 8192, "EXACT_ORIGINAL_LATENT_EXTENT")
        result[pair] = (raw, initial)
    return result


def seal(folder, plan, plan_sha, request, records, facts, failure):
    need(set(records) <= {(NEW_ARM, i) for i in range(6)}, "ONLY_NEW_ARM_SLOTS")
    rows = [records.get((NEW_ARM, item["pair"]), dict(arm=NEW_ARM, outcome="missing",
            **{key: item[key] for key in KEYS})) for item in plan["samples"]]
    raw = b"".join(map(wire, rows))
    put(folder / "generations.jsonl", raw)
    counts = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in ("completed", "error", "missing")}
    result = dict(request=request, sealed=True, passed=failure is None and counts["completed"] == 6
                  and facts["actual_generation_calls"] == 6 and facts["model_loads"] == 1,
                  planned_slots=6, planned_pairs=6, only_new_arm=NEW_ARM, counts=counts,
                  generations_SHA256=digest(raw), generations_bytes=len(raw), plan_SHA256=plan_sha,
                  original_result_SHA256=OLD_RESULT, facts=facts, failure=failure, formal_domain_credit=0)
    put(folder / "result.json", wire(result))
    return result


def compose(args):
    plan, plan_sha = frozen(args.plan)
    old_status, old_rows = original(args.original, plan, plan_sha, images=True)
    terminal = json.loads(read(args.terminal, 16 << 10, args.terminal_sha))
    need(set(terminal) == {"request", "numbered_gpu_step", "worker_result_SHA256", "independent_post_clean"}
         and terminal["independent_post_clean"] is True and re.fullmatch(r"1194900\.\d+", terminal["numbered_gpu_step"])
         and terminal["numbered_gpu_step"] not in {"1194900.2762", "1194900.2769", "1194900.2774"}, "INDEPENDENT_NEW_ARM_POST")
    current, new_rows = results(args.new, plan, plan_sha, (NEW_ARM,), terminal["worker_result_SHA256"], images=True)
    need(current["request"] == terminal["request"] and current["facts"]["step"] == terminal["numbered_gpu_step"]
         and current["request"] not in {OLD_REQUEST, NEW_TRAIN_REQUEST, "94b0d2c0017a4a0ca686d04ce081796d"} and re.fullmatch(r"[a-f0-9]{32}", current["request"])
         and current["only_new_arm"] == NEW_ARM and current["original_result_SHA256"] == OLD_RESULT
         and current["facts"]["adapter_manifest_SHA256"] == CHECKPOINT_SHA
         and current["facts"]["adapter_SHA256"] == ADAPTER_SHA,
         "EXACT_NEW_ARM_TERMINAL_AND_LINEAGE")
    scheduler = None
    if current["passed"] is True or any(row["outcome"] == "completed" for row in new_rows.values()):
        need(current["facts"]["latent_manifest_SHA256"] == OLD_LATENTS, "EXACT_ORIGINAL_BF16_NOISE_MANIFEST")
        scheduler = verify_bridge(current["facts"], list(new_rows.values()),
                                  old_status["facts"]["effective_scheduler_config_SHA256"],
                                  complete=current["passed"] is True)
    counts = {outcome: sum(row["outcome"] == outcome for row in new_rows.values())
              for outcome in ("completed", "error", "missing")}
    need(current["counts"] == counts, "EXACT_NEW_ARM_OUTCOME_COUNTS")
    if current["passed"] is True:
        live = current["facts"]["live_adapter"]
        need(counts["completed"] == 6 and current["facts"]["actual_generation_calls"] == 6
             and current["facts"]["model_loads"] == 1
             and live == dict(active=NEW_ARM, strength=1.0, tensors=1120, SHA256=ADAPTER_SHA),
             "ACTUAL_SOLE_NEW_ADAPTER_AND_GENERATIONS")
    main = []
    all_rows = old_rows | new_rows
    for i in range(6):
        previous, new = old_rows["cp25", i], new_rows[NEW_ARM, i]
        paired = new["outcome"] == "completed" and same_noise(previous, new)
        need(new["outcome"] != "completed" or paired, "THREE_ARM_NOISE_OR_TIMESTEP_MISMATCH")
        main.append(dict(pair=i, old_arm="cp25", new_arm=NEW_ARM, exact_same_noise_and_timesteps=paired))
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not any(args.output.is_relative_to(root) or root.is_relative_to(args.output)
                     for root in (args.original, args.new)), "DISTINCT_PRIVATE_COMPOSITION_OUTPUT")
    args.output.mkdir(mode=0o700)
    raw = b"".join(wire(dict(row, artifact_scope=("new" if arm == NEW_ARM else "original")))
                   for (arm, _), row in all_rows.items())
    put(args.output / "three-arm-index.private.jsonl", raw)
    summary = dict(planned_pairs=6, planned_slots=18, reused_original_images=12, new_slots=6,
                   completed_primary_pairs=sum(item["exact_same_noise_and_timesteps"] for item in main), primary=main,
                   original_worker_result_SHA256=OLD_RESULT, four_row_worker_result_SHA256=terminal["worker_result_SHA256"],
                   index_SHA256=digest(raw), new_counts=counts, independently_recomputed_scheduler_bridge=scheduler,
                   blind_review_ready=current["passed"] is True, score_computed=False, formal_domain_credit=0)
    put(args.output / "result.json", wire(summary))
    return summary


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "original", "new", "terminal", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--terminal-sha", required=True)
    args = p.parse_args()
    print(json.dumps(compose(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
