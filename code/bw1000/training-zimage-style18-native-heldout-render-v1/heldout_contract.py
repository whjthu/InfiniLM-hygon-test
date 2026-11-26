"""Render and reconcile the 32 frozen Style18 slots in private arm custody."""

from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import time


PLAN = {"bytes": 18039, "SHA256": "f71913a6a78e1e6f109baf99d23cea2e1ada9ed0b40a2684c30c629303ecef7a"}
PACKAGE_PLAN = Path("/heldout-input/source-slots.json")
OLD_REQUEST = "cfc4c90de1f1457589f755829bdbdaae"
CHECKPOINT = {"bytes": 60423969, "checkpoint_SHA256": "dd2b830dc6472b2947e5585f30cb9e42d90bdd9e328fac123bb5ec1cccfd8f1c",
              "adapter_SHA256": "0df3ee60d1ad39c250404f29ddb4dda59c98caa52789ff79a508ddae398c0698",
              "global_step": 25, "next_data_cursor": 1}
INITIAL = {"bytes": 56897, "SHA256": "a27c2877ed6205dc2859f73d0d70a13959989ee4d0fadaacfe3e1b363f1f08e5"}
CHECKPOINT_MODULES = {
    "hy_resume_checkpoint.py": {"bytes": 27542, "SHA256": "aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7"},
    "zimage_training_resume_checkpoint.py": {"bytes": 20709, "SHA256": "8a394d733970ed558ca0a3c85f27dbbf955849ce6fb8746d35abb9c0d2d6a5c5"}}
SETTINGS = dict(width=256, height=256, denoising_steps=8, guidance_scale=1.0, exact_paired_starting_noise_required=True)
MARKS = tuple(f"{kind}{i}" for kind in ("D", "C") for i in range(1, 5))
ARMS = ("before", "after")
MAX_IMAGE = 512 * 1024
MAX_LEDGER = 128 * 1024


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def save_exclusive(path, value):
    raw = encode(value)
    with path.open("xb") as stream:
        durable(stream, raw)
    return pin(raw)


def durable(stream, raw):
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def load_claim_plan(claim, path=PACKAGE_PLAN):
    request = claim["request"]
    need(re.fullmatch(r"[a-f0-9]{32}", request) and request != OLD_REQUEST
         and claim["heldout_plan"] == PLAN and claim["planned_generations"] == 32 and claim["planned_pairs"] == 16
         and claim["actual_optimizer_updates_planned"] == 0
         and claim["initial_training_hashes"] == INITIAL
         and claim["framework_checkpoint_modules"] == CHECKPOINT_MODULES
         and all(claim["continuous25_checkpoint"][k] == v for k, v in CHECKPOINT.items()), "EXACT_FRESH_HELDOUT_CLAIM")
    with path.open("rb") as stream:
        raw = stream.read(PLAN["bytes"] + 1)
    need(pin(raw) == PLAN, "EXACT_PRIVATE_HELDOUT_PACKET")
    return samples(json.loads(raw))


def samples(packet):
    entries = packet["prompts"]
    seeds = packet["seeds"]
    need(packet["planned_images"] == 32 and packet["planned_pairs"] == 16
         and packet["generation_settings"] == SETTINGS and packet["generation_GO"] is False
         and tuple(p["slot"] for p in entries) == MARKS
         and len(seeds) == len({row["seed"] for row in seeds}) == 16, "EXACT_HELDOUT_PLAN_LAYOUT")
    rows = []
    for index, seed in enumerate(seeds):
        item = entries[index // 2]
        prompt = item["prompt"]
        slot = MARKS[index // 2]
        rep = index % 2 + 1
        pair = f"{slot}-{rep}"
        expected_seed = int(sha256(("Style18-heldout-prelabel-v1-seed/" + pair).encode()).hexdigest()[:8], 16)
        need(item["source_attested"] is True and 0 < len(prompt) <= 1200
             and sha256(prompt.encode()).hexdigest() == item["prompt_SHA256"]
             and seed == dict(pair_slot=pair, prompt_slot=slot, replicate=rep, seed=expected_seed),
             "FROZEN_PROMPT_SEED_IDENTITY")
        rows.append(dict(sample_id=pair, ordinal=index, prompt_slot=slot, replicate=rep, seed=expected_seed,
                         prompt=prompt, prompt_SHA256=item["prompt_SHA256"]))
    return rows


def planned(rows):
    return { (row["ordinal"], arm): {k: v for k, v in row.items() if k != "prompt"} | {"arm": arm}
             for row in rows for arm in ARMS }


def image_name(index, arm):
    return f"{index:02d}-{arm}.private.png"


def image_pin(path):
    if not path.is_file() or path.is_symlink():
        return None
    with path.open("rb") as f:
        raw = f.read(MAX_IMAGE + 1)
    if not 0 < len(raw) <= MAX_IMAGE:
        return None
    return pin(raw)


def verify_image(path, row):
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as f:
        body = f.read(MAX_IMAGE + 1)
    if not 0 < len(body) <= MAX_IMAGE or pin(body) != {"bytes": row.get("image_bytes"), "SHA256": row.get("image_SHA256")}:
        return False
    from PIL import Image
    try:
        with Image.open(io.BytesIO(body), formats=("PNG",)) as image:
            return (image.size == (256, 256) and image.mode == "RGB"
                    and pin(image.tobytes())["SHA256"] == row["RGB_SHA256"])
    except (OSError, ValueError):
        return False


def reconcile(output, rows):
    """Read only after worker termination. All missing slots stay in the denominator."""
    obligations = planned(rows)
    actual = {}
    duplicate = set()
    ledger = output / "generations.jsonl"
    malformed = 0
    ledger_pin = None
    if ledger.is_file() and not ledger.is_symlink():
        with ledger.open("rb") as stream:
            raw = stream.read(MAX_LEDGER + 1)
        if len(raw) <= MAX_LEDGER:
            ledger_pin = pin(raw)
            for line in raw.splitlines(keepends=True):
                try:
                    row = json.loads(line)
                    key = (row["ordinal"], row["arm"])
                    expected = obligations[key]
                    need(line.endswith(b"\n") and all(row.get(k) == v for k, v in expected.items())
                         and type(row["completed"]) is bool, "SLOT_IDENTITY")
                    if key in actual:
                        duplicate.add(key)
                    actual[key] = row
                except (TypeError, KeyError, ValueError):
                    malformed += 1
        else:
            malformed += 1
    elif ledger.exists() or ledger.is_symlink():
        malformed += 1
    slots = []
    for key, expected in obligations.items():
        row = actual.get(key)
        if row is None:
            slots.append(expected | dict(status="MISSING"))
            continue
        status = "FAILED"
        if key in duplicate:
            reason = "DUPLICATE_SLOT"
        elif not row["completed"]:
            reason = row.get("error_type", "NATIVE_FAILURE")
            if not isinstance(reason, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", reason):
                reason = "NATIVE_FAILURE"
        else:
            name = image_name(*key)
            valid = (row.get("image_relative") == name and isinstance(row.get("initial_noise_SHA256"), str)
                     and re.fullmatch(r"[0-9a-f]{64}", row["initial_noise_SHA256"])
                     and isinstance(row.get("RGB_SHA256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["RGB_SHA256"])
                     and verify_image(output / name, row))
            reason = "IMAGE_OR_NOISE_IDENTITY" if not valid else None
            if valid:
                status = "COMPLETE"
        slots.append(expected | dict(status=status, reason=reason) |
                     ({k: row[k] for k in ("initial_noise_SHA256", "image_relative", "image_bytes", "image_SHA256", "RGB_SHA256")}
                      if status == "COMPLETE" else {}))
    counts = Counter(s["status"] for s in slots)
    missing_or_failed = 0
    noise_mismatch = 0
    matches = 0
    for before, after in zip(slots[::2], slots[1::2], strict=True):
        if before["status"] != "COMPLETE" or after["status"] != "COMPLETE":
            missing_or_failed += 1
        elif before["initial_noise_SHA256"] == after["initial_noise_SHA256"]:
            matches += 1
        else:
            noise_mismatch += 1
    return dict(planned_generations=32, planned_pairs=16, attempted=len(actual), completed=counts["COMPLETE"],
                failed=counts["FAILED"], missing=counts["MISSING"], complete_same_noise_pairs=matches,
                complete_different_noise_pairs=noise_mismatch, missing_or_failed_pairs=missing_or_failed,
                duplicate_slots=len(duplicate), malformed_ledger_rows=malformed, ledger=ledger_pin,
                optimizer_updates=0, arm_allocation_read=False, blind_scored=False,
                passed=counts["COMPLETE"] == 32 and matches == 16 and malformed == 0 and not duplicate, slots=slots)


def generate_one(model, torch, named, state, row, capture):
    pipe = model.pipe
    original = pipe.generate_noise
    noises = []

    def observed(*args, **kwargs):
        noise = original(*args, **kwargs)
        need(tuple(noise.shape) == (1, 16, 32, 32) and noise.dtype == torch.bfloat16 and noise.device.type == "cuda",
             "NATIVE_INITIAL_NOISE_METADATA")
        noises.append(capture(torch, noise))
        return noise

    try:
        with torch.no_grad():
            for name, value in named.items():
                source = state[name]
                need(source.dtype == value.dtype and source.shape == value.shape, "ADAPTER_TENSOR_FORMAT")
                value.copy_(source)
            pipe.generate_noise = observed
            image = pipe(prompt=row["prompt"], seed=row["seed"], rand_device="cuda", height=256, width=256,
                         num_inference_steps=8, cfg_scale=1.0)
        torch.cuda.synchronize()
        from PIL import Image
        need(len(noises) == 1 and isinstance(image, Image.Image) and image.size == (256, 256),
             "NATIVE_GENERATION_OUTPUT")
        return image.convert("RGB"), noises[0]
    finally:
        pipe.generate_noise = original


def execute(model, torch, output, rows, named, states, deadline, capture):
    obligations = planned(rows)
    model.eval()
    try:
        with (output / "generations.jsonl").open("xb") as stream:
            for index, row in enumerate(rows):
                order = ARMS if index % 2 == 0 else ARMS[::-1]
                for arm in order:
                    deadline()
                    start = time.monotonic()
                    result = obligations[index, arm] | dict(completed=False)
                    try:
                        image, noise = generate_one(model, torch, named, states[arm], row, capture)
                        target = output / image_name(index, arm)
                        with target.open("xb") as image_file:
                            image.save(image_file, format="PNG")
                            image_file.flush()
                            os.fsync(image_file.fileno())
                        identity = image_pin(target)
                        need(identity is not None, "IMAGE_OUTPUT_BOUND")
                        result.update(completed=True, initial_noise_SHA256=noise, image_relative=target.name,
                                      image_bytes=identity["bytes"], image_SHA256=identity["SHA256"],
                                      RGB_SHA256=sha256(image.tobytes()).hexdigest(), seconds=time.monotonic()-start)
                    except Exception as error:
                        result["error_type"] = type(error).__name__
                        durable(stream, encode(result))
                        if isinstance(error, TimeoutError):
                            raise
                    else:
                        durable(stream, encode(result))
    finally:
        summary = reconcile(output, rows)
        save_exclusive(output / "summary.json", summary)
    return {k: v for k, v in summary.items() if k != "slots"} | {"initial_and_trained_adapter_identities_verified": True}
