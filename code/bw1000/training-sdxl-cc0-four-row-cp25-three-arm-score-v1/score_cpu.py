"""Score six new SDXL outputs against the retained .2765 CLIP controls."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import fmean
import sys

from plan import HEX, digest, frozen, need, put, read, same_noise, wire

ARMS = ("base", "cp25", "four_row_cp25")
NEW = ARMS[-1]
KEYS = ("pair", "family", "replicate", "seed", "prompt_SHA256")
NEW_REQUEST = "a6b938dcee0e4b98b72a062412b787b2"
OLD_REQUEST = "94f0acaeec614c5fa9715033ae1a755d"
ORIGINAL_WORKER = "d14e800d2744556a67b9410898522915b3938d52f79080f2f50a6d1751c8a08f"
NEW_WORKER = "8bc0fa0d1b65891ea1c4d28c0498e9e0e2f606c636fe727fa74104ef7ea81c30"
NEW_ARCHIVE = "a35b75b2eba1883a19c5a3492645b6620de178bf873f70f68ab3acb78c83a58a"
COMPOSITION = "3648cc4458711f150d7a6a843b6478e497e4a0c4d110255ee5c5083a2b575289"
INDEX = "0cc85b1e5e3181c4dcf003cd250e6f8eddb4b7c3fb4935285d28cea41cd19eeb"
PRIOR_ARCHIVE = "22d5ebc76b081e38b5ffc3539e57ca8ab745c7c106d49b44c428713189a95ac5"
PRIOR_RESULT = "c1ccbdc9ef0aaac2abaf35439d1d9f398094bedc7bc21550b03327a1ae39510b"
SCORER = "59feca66365c73431a963ad8500bba5c0a42a4579139d9ee40f8bc5663552772"
WEIGHT = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
PROVIDER = "77202606488902993d5fea75b321f1e9348298112e6ce42f40d52d5df7471196"
METRIC = "OpenAI CLIP ViT-B/32 normalized cosine"


def valid_cosine(value):
    return type(value) is float and math.isfinite(value) and -1.00001 <= value <= 1.00001


def readonly(path):
    need(path.is_absolute() and path.resolve(strict=True) == path and os.statvfs(path).f_flag & os.ST_RDONLY,
         "EXACT_READONLY_SCORER_INGRESS")


def original_controls(path, plan, plan_sha):
    archive = json.loads(read(path, 32 << 10, PRIOR_ARCHIVE))
    fresh = archive["fresh_score"]
    prior = fresh["scores"]
    need(fresh["accounting"][:4] == ["1194900.2765", "sdxl-product-clip-CP25-v2-837e2da238aa", "COMPLETED", "0:0"]
         and fresh["score_SHA256"] == digest(wire(prior)) == PRIOR_RESULT
         and prior["source_plan_SHA256"] == plan_sha and prior["weights_SHA256"] == WEIGHT
         and prior["generation_result_SHA256"] == ORIGINAL_WORKER and prior["metric"] == METRIC
         and prior["planned_pairs"] == prior["evaluated_pairs"] == 6 and prior["incomplete_pairs"] == 0
         and prior["real_accelerator_calls"] == 0 and len(prior["results"]) == 6,
         "EXACT_RETAINED_2765_CLIP_REPORT")
    controls = []
    for row, frozen_slot in zip(prior["results"], plan["samples"], strict=True):
        need(row["pair"] == frozen_slot["pair"] and row["family"] == frozen_slot["family"]
             and all(valid_cosine(row[arm]) for arm in ARMS[:2]) and row["delta"] == row["cp25"] - row["base"],
             "EXACT_RETAINED_PER_SLOT_CONTROL_SCORES")
        controls.append(row)
    return controls


def original_terminal(path):
    archive = json.loads(read(path, 64 << 10, NEW_ARCHIVE))
    capture = archive["terminal_observed_response"]["response"]["terminal"]
    composition = archive["composition_observed_response"]["response"]
    roles = capture["accounting"]
    need(capture["request"] == NEW_REQUEST and capture["worker_result_SHA256"] == NEW_WORKER
         and capture["cleanup_verified"] is capture["scientific_pass"] is capture["sealed"] is True
         and roles["gpu"][0] == "1194900.2821" and roles["post"][0] == "1194900.2822"
         and roles["gpu"][2:4] == roles["post"][2:4] == ["COMPLETED", "0:0"]
         and composition["composition_result_SHA256"] == COMPOSITION and composition["composition"]["index_SHA256"] == INDEX,
         "EXACT_2821_2822_AND_INDEPENDENT_COMPOSITION")
    return capture


def native_ledger(folder, pin, request, step, arms, plan_sha):
    scoped = json.loads(read(folder / "result.json", 64 << 10, pin))
    need(scoped["request"] == request and scoped["facts"]["step"] == step and scoped["plan_SHA256"] == plan_sha
         and scoped["planned_slots"] == len(arms) * 6 and scoped["planned_pairs"] == 6 and scoped["sealed"] is True,
         "EXACT_PINNED_GENERATION_SCOPE")
    raw = read(folder / "generations.jsonl", 128 << 10, scoped["generations_SHA256"])
    rows = [json.loads(line) for line in raw.splitlines()]
    need(len(raw) == scoped["generations_bytes"] and
         [(row["arm"], row["pair"]) for row in rows] == [(arm, i) for arm in arms for i in range(6)],
         "EXACT_NATIVE_GENERATION_LEDGER")
    return rows


def index_slots(composition, native, plan):
    status = json.loads(read(composition / "result.json", 64 << 10, COMPOSITION))
    need(status["index_SHA256"] == INDEX and status["planned_slots"] == 18 and status["planned_pairs"] == 6
         and status["reused_original_images"] == 12 and status["new_slots"] == 6
         and status["original_worker_result_SHA256"] == ORIGINAL_WORKER
         and status["four_row_worker_result_SHA256"] == NEW_WORKER and status["score_computed"] is False,
         "EXACT_COMPOSITION_RESULT")
    rows = [json.loads(line) for line in read(composition / "three-arm-index.private.jsonl", 128 << 10, INDEX).splitlines()]
    need(len(rows) == len(native) == 18, "EXACT_CONTRAST_18_SLOT_DENOMINATOR")
    for item, original in zip(rows, native, strict=True):
        arm, pair = item["arm"], item["pair"]
        need(arm in ARMS and type(pair) is int and 0 <= pair < 6
             and item == dict(original, artifact_scope="new" if arm == NEW else "original")
             and all(item[key] == plan["samples"][pair][key] for key in KEYS)
             and item["outcome"] in ("completed", "error", "missing"), "EXACT_SLOT_PROMPT_SEED_AND_IMAGE_LINEAGE")
        if item["outcome"] == "completed":
            need(item["image"] == f"{arm}-{pair:02d}.private.png"
                 and type(item["image_bytes"]) is int and 0 < item["image_bytes"] <= 2 << 20
                 and bool(HEX.fullmatch(item["image_SHA256"]))
                 and all(HEX.fullmatch(item[key]) for key in
                         ("raw_latent_SHA256", "prepared_latent_SHA256", "timesteps_SHA256")),
                 "EXACT_SLOT_IMAGE_AND_NOISE_PINS")
    for i in range(6):
        base, old, new = (rows[i + offset] for offset in (0, 6, 12))
        need(base["outcome"] == old["outcome"] == "completed" and same_noise(base, old)
             and (new["outcome"] != "completed" or same_noise(new, old)),
             "EXACT_SAME_NOISE_BOTH_CONTRASTS")
    return rows[12:]


def original_scorer(path):
    raw = read(path, 32 << 10, SCORER)
    spec = spec_from_file_location("sdxl_pinned_2765_scorer", path)
    module = module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    need(module.WEIGHT == WEIGHT and module.PROVIDER == PROVIDER, "EXACT_OFFICIAL_2765_SCORER_PINS")
    return module


def input_bytes(folder, row):
    path = folder / row["image"]
    readonly(path)
    raw = read(path, 2 << 20, row["image_SHA256"])
    need(len(raw) == row["image_bytes"], "EXACT_NEW_PNG_SIZE")
    return raw


def score(args, samples, slots, reports, activity):
    prior = original_scorer(args.prior_scorer)
    prior.weight(args.weights)
    prior.verify_provider(args.provider)
    sys.path.insert(0, str(args.provider))
    import clip
    from PIL import Image
    import torch

    need(Path(clip.__file__) == args.provider / "clip/__init__.py" and not torch.cuda.is_initialized(),
         "ACTUAL_CPU_CLIP_PROVIDER")
    model, transform = clip.load(str(args.weights), device="cpu", jit=False)
    model.eval()
    with torch.inference_mode():
        tokens = clip.tokenize([item["prompt"] for item in samples], truncate=False)
        features = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
        for start in range(0, 6, 2):
            pairs, values = [], []
            for index in (start, start + 1):
                row = slots[index]
                if row["outcome"] != "completed":
                    continue
                try:
                    with Image.open(io.BytesIO(input_bytes(args.new, row))) as image:
                        need(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256),
                             "EXACT_GENERATED_IMAGE_TYPE")
                        values.append(transform(image))
                    pairs.append(index)
                except (OSError, ValueError, TypeError) as error:
                    reports[index].update(status="error", error_type=type(error).__name__,
                                          error_SHA256=digest(str(error).encode()))
            if not pairs:
                continue
            activity["new_image_encoding_attempts"] += len(pairs)
            vectors = model.encode_image(torch.stack(values))
            need(len(vectors) == len(pairs), "EXACT_NATIVE_NEW_CLIP_MINIBATCH")
            activity["new_image_encodings"] += len(vectors)
            encoded = torch.nn.functional.normalize(vectors.float(), dim=-1)
            for index, feature in zip(pairs, encoded, strict=True):
                value = float(torch.dot(feature, features[index]))
                if valid_cosine(value):
                    reports[index].update(status="scored", four_row_cp25=value)
                else:
                    reports[index].update(status="error", error_type="NonfiniteOrOutOfRangeCosine")
    need(not torch.cuda.is_initialized(), "ACTUAL_CLIP_NO_ACCELERATOR_AFTER_SCORE")


def summarize(rows):
    families = sorted({row["family"] for row in rows})
    valid = [row for row in rows if row["status"] == "scored"]
    for row in valid:
        row["primary_delta_vs_two_row_cp25"] = row["four_row_cp25"] - row["two_row_cp25"]
        row["secondary_delta_vs_base"] = row["four_row_cp25"] - row["base"]
    by_family = {}
    for name in families:
        matched = [row for row in valid if row["family"] == name]
        by_family[name] = dict(planned=2, evaluated=len(matched),
            primary_mean=fmean(row["primary_delta_vs_two_row_cp25"] for row in matched) if matched else None,
            secondary_mean=fmean(row["secondary_delta_vs_base"] for row in matched) if matched else None)
    complete_families = all(entry["evaluated"] for entry in by_family.values())
    return dict(planned_pairs=6, evaluated_pairs=len(valid), incomplete_pairs=6-len(valid), family=by_family,
                counts={key: sum(row["status"] == key for row in rows) for key in ("scored", "error", "missing")},
                primary_family_equal_mean=fmean(row["primary_mean"] for row in by_family.values()) if complete_families else None,
                secondary_family_equal_mean=fmean(row["secondary_mean"] for row in by_family.values()) if complete_families else None)


def run(args):
    need(all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
         and re.fullmatch(r"[a-f0-9]{32}", args.request) and args.request not in {NEW_REQUEST, OLD_REQUEST},
         "EXACT_FRESH_ZERO_GRES_CPU_SCORER")
    parent = args.output.parent
    need(args.output.is_absolute() and parent.resolve(strict=True) == parent
         and parent.stat().st_uid == os.geteuid() and stat.S_IMODE(parent.stat().st_mode) == 0o700
         and not args.output.exists(), "EXACT_PRIVATE_SCORE_OUTPUT")
    for path in (args.plan, args.terminal, args.composition, args.original, args.new, args.prior_report,
                 args.prior_scorer, args.weights, args.provider):
        readonly(path)
    plan, plan_sha = frozen(args.plan)
    original_terminal(args.terminal)
    controls = original_controls(args.prior_report, plan, plan_sha)
    old = native_ledger(args.original, ORIGINAL_WORKER, OLD_REQUEST, "1194900.2762", ARMS[:2], plan_sha)
    new = native_ledger(args.new, NEW_WORKER, NEW_REQUEST, "1194900.2821", ARMS[2:], plan_sha)
    slots = index_slots(args.composition, old + new, plan)
    rows = [dict(pair=i, family=control["family"], base=control["base"], two_row_cp25=control["cp25"],
                 four_row_cp25=None, image_SHA256=slots[i].get("image_SHA256"), status="missing" if
                 slots[i]["outcome"] == "completed" else slots[i]["outcome"])
            for i, control in enumerate(controls)]
    failure = None
    activity = dict(new_image_encoding_attempts=0, new_image_encodings=0)
    if any(slot["outcome"] == "completed" for slot in slots):
        try:
            score(args, plan["samples"], slots, rows, activity)
        except Exception as error:
            failure = dict(type=type(error).__name__, detail_SHA256=digest(str(error).encode()))
    stats = summarize(rows)
    result = dict(request=args.request, actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  metric=METRIC, generation_slots=18, **activity,
                  original_image_encodings=0, retained_control_scores=12,
                  results=rows, failure=failure, **stats, primary_human_review_completed=False,
                  formal_FID=False, formal_domain_credit=0, formal_training_credit=0,
                  weights_SHA256=WEIGHT, provider_SHA256=PROVIDER, score_controls_SHA256=PRIOR_RESULT,
                  composition_SHA256=COMPOSITION, index_SHA256=INDEX, source_plan_SHA256=plan_sha)
    put(args.output, wire(result))
    print(json.dumps(dict(request=args.request, evaluated=stats["evaluated_pairs"], planned=6)), flush=True)
    return 0 if failure is None and stats["evaluated_pairs"] == 6 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "terminal", "composition", "original", "new", "prior-report", "prior-scorer",
                 "weights", "provider", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
