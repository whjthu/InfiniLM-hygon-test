"""Frozen SDXL CC0 holdout roster, prompts, paired noise and ledger boundary."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat


ARMS = ("base", "cp25")
SCHEMA = "sdxl-r33-cc0-three-family-paired-v1"
TEMPLATE = "Studio product photograph of a single {object}, fully visible and centered against a plain neutral background."
FAMILIES = {
    "cassette_player": ("cassette player", "03a51325260af37daddd595ee3653eaaa7e34a144e5423f04e35d825f5364aa9"),
    "vintage_electric_kettle": ("vintage electric kettle", "84f6c10c359242a0cb94f654489f77022de405081c79e78c6f01c79899718ac5"),
    "vintage_telephone_wall_clock": ("vintage telephone-shaped wall clock",
                                    "a30b9bc196fb4a402c7a5b7d741916a46ee6a1fe0623dfabb13b2e7341fe40ba"),
}
CP25_SHA = "beb593565087406f3b9090a468fd650a18b097124d4f44f9e26acda0aa7640b4"
CP25_BYTES = 11778368
R33_SHA = "e5557e400bb7777d54c4e2c86d9595f946b81994847903e81e0faa7c3f5f9dfc"
FAMILY_SHA = "78790d76af2cac18e4fa69cd79a475f89e193f7275be19d9bfb2c8470d6eaf07"
CLOCK_SHA = "0c8376a1fcf55a211f54fe5613bea7945277cc53db73505bac5c7323547534fd"
PIPE_SHA = "b0a2515ad9a6422d83991788f6af7749bc50812ad0b4e028e52b86350c519175"
SCHEDULER_SHA = "af3e45a949aff8b8341ab8b811429ec03fee857a700a1d9477363e4fff9666e2"
INDEX_SHA = "6d7b93508390ab91ac5bfbe4aeb4dc2d83f7bb1b05fb069d714b5b0c75f70d44"
CONFIG = dict(width=256, height=256, steps=30, cfg=5.0, negative_prompt="", scheduler="EulerDiscreteScheduler",
              unet_dtype="torch.bfloat16", text_dtype="torch.bfloat16", vae_dtype="torch.float32",
              latent_shape=[1, 4, 32, 32], latent_dtype="torch.bfloat16",
              latent_rng="native-cpu-float32-randn-cast-bfloat16",
              denoise_rng="separate-fresh-native-cpu-generator-at-same-seed-per-arm")
HEX = re.compile(r"[0-9a-f]{64}")


def need(condition, code):
    if not condition:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def roster():
    rows = []
    for family, (name, asset) in FAMILIES.items():
        prompt = TEMPLATE.format(object=name)
        for replicate in (0, 1):
            pair = digest(f"{SCHEMA}/{family}/{asset}/{replicate}".encode())
            rows.append(dict(pair=len(rows), family=family, alias=name, cc0_bundle_SHA256=asset,
                             replicate=replicate, seed=int(pair[:15], 16),
                             prompt=prompt, prompt_SHA256=digest(prompt.encode())))
    return dict(schema=SCHEMA, arms=list(ARMS), planned_pairs=6, planned_slots=12, samples=rows, config=CONFIG,
                asset_license="CC0-1.0", trained_family_exclusions=["metal_tool_chest", "painted_wooden_stool"],
                provenance=dict(r33_terminal_SHA256=R33_SHA, public_families_SHA256=FAMILY_SHA,
                                public_clock_SHA256=CLOCK_SHA, pipeline_manifest_SHA256=PIPE_SHA,
                                original_scheduler_SHA256=SCHEDULER_SHA, original_model_index_SHA256=INDEX_SHA),
                checkpoint=dict(step="1194900.2675", source_arm="continuous", bytes=CP25_BYTES, SHA256=CP25_SHA),
                scorer=dict(primary=["single_complete_target_object", "plain_clean_background"],
                            arm_identity_hidden=True, secondary="OpenAI CLIP ViT-B/32 normalized image-text cosine",
                            incomplete_pair_retained=True, formal_FID=False, retention_evaluated=False))


def read(path, limit, expected=None):
    need(path.is_absolute() and path.resolve(strict=True) == path, "CANONICAL_PINNED_MEMBER")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "BOUNDED_PINNED_MEMBER")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and
         (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
         and (expected is None or HEX.fullmatch(expected) and digest(raw) == expected), "EXACT_PINNED_MEMBER")
    return raw


def put(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def frozen(path):
    raw = read(path, 24 << 10)
    value = roster()
    need(raw == wire(value) and len({row["seed"] for row in value["samples"]}) == 6
         and not set(FAMILIES).intersection(value["trained_family_exclusions"]), "EXACT_PRE_OUTCOME_PLAN")
    return value, digest(raw)


def original_run(path):
    raw = read(path, 32 << 10, R33_SHA)
    data = json.loads(raw)
    status = data["owner_result"]
    adapter = data["verified_adapter_extents"]["continuous_25"]
    need(status["accounting"]["gpu"][0] == "1194900.2675" and status["accounting"]["post"][0] == "1194900.2676"
         and status["accounting"]["gpu"][2:4] == status["accounting"]["post"][2:4] == ["COMPLETED", "0:0"]
         and status["cleanup_verified"] is status["scientific_pass"] is True and status["actual_updates"] == 30
         and adapter == dict(bytes=CP25_BYTES, SHA256=CP25_SHA,
                            relative_artifact="durable/once/continuous/checkpoint-25/pytorch_lora_weights.safetensors"),
         "ORIGINAL_CONTINUOUS_CP25_LINEAGE")


def seal(folder, plan, plan_sha, request, records, facts, failure):
    need(set(records).issubset({(arm, sample["pair"]) for arm in ARMS for sample in plan["samples"]}),
         "EXACT_PLANNED_SLOT_KEYS")
    rows = []
    for arm in ARMS:
        for item in plan["samples"]:
            pair = item["pair"]
            identity = {key:item[key] for key in ("pair", "family", "replicate", "seed", "prompt_SHA256")}
            row = records.get((arm, pair), dict(arm=arm, outcome="missing", **identity))
            need(row["arm"] == arm and all(row[key] == value for key, value in identity.items())
                 and row["outcome"] in {"completed", "error", "missing"}, "EXACT_PLANNED_SLOT_IDENTITY")
            rows.append(row)
    encoded = b"".join(wire(row) for row in rows)
    put(folder / "generations.jsonl", encoded)
    counts = {arm: {state: sum(row["arm"] == arm and row["outcome"] == state for row in rows)
                    for state in ("completed", "error", "missing")} for arm in ARMS}
    paired = sum(all(records.get((arm, pair), {}).get("outcome") == "completed" for arm in ARMS)
                 and same_noise(records["base", pair], records["cp25", pair])
                 for pair in range(6))
    summary = dict(request=request, planned_slots=12, planned_pairs=6, complete_same_noise_pairs=paired, counts=counts,
                   generations_SHA256=digest(encoded), generations_bytes=len(encoded), plan_SHA256=plan_sha,
                   facts=facts, failure=failure, sealed=True, passed=failure is None and paired == 6,
                   score_computed=False, formal_FID=False, formal_domain_credit=0)
    put(folder / "result.json", wire(summary))
    return summary


def same_noise(left, right):
    return all(type(left[key]) is str and HEX.fullmatch(left[key]) and left[key] == right[key]
               for key in ("raw_latent_SHA256", "prepared_latent_SHA256", "timesteps_SHA256"))
