"""Run the existing pinned OpenAI ViT-B/32 cosine method after independent return."""

import argparse
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys

from plan import ARMS, HEX, digest, frozen, need, put, read, same_noise, wire


WEIGHT = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
WEIGHT_BYTES = 353976522
PROVIDER = "77202606488902993d5fea75b321f1e9348298112e6ce42f40d52d5df7471196"


def final_pairs(folder, plan, plan_sha, terminal):
    raw = read(folder / "result.json", 64 << 10, terminal["worker_result_SHA256"])
    worker = json.loads(raw)
    need(terminal["request"] == worker["request"] and terminal["independent_post_clean"] is True
         and terminal["numbered_gpu_step"] == worker["facts"]["step"] and worker["sealed"] is True
         and worker["plan_SHA256"] == plan_sha and worker["planned_slots"] == 12 and worker["planned_pairs"] == 6,
         "EXACT_INDEPENDENT_TERMINAL_AND_SEAL")
    logs = read(folder / "generations.jsonl", 128 << 10, worker["generations_SHA256"])
    rows = [json.loads(line) for line in logs.splitlines()]
    need(len(logs) == worker["generations_bytes"] and len(rows) == 12 and
         [(row["arm"], row["pair"]) for row in rows] == [(arm, i) for arm in ARMS for i in range(6)],
         "COMPLETE_PAIRED_LEDGER")
    need(worker["counts"] == {arm: {state: sum(row["arm"] == arm and row["outcome"] == state for row in rows)
                                   for state in ("completed", "error", "missing")} for arm in ARMS},
         "COMPLETE_PAIRED_FAILURE_COUNTS")
    result = []
    for index, item in enumerate(plan["samples"]):
        left, right = rows[index], rows[index+6]
        for row in (left, right):
            need(all(row[key] == item[key] for key in ("pair", "family", "replicate", "seed", "prompt_SHA256"))
                 and row["outcome"] in ("error", "missing", "completed"), "EXACT_PAIRED_LEDGER_IDENTITY")
        if left["outcome"] == right["outcome"] == "completed" and same_noise(left, right):
            result.append((item, left, right))
    need(len(result) == worker["complete_same_noise_pairs"], "EXACT_COMPLETE_PAIR_COUNT")
    return worker, result


def weight(path):
    need(path.resolve(strict=True) == path and path.stat().st_size == WEIGHT_BYTES, "EXACT_OFFICIAL_CLIP_EXTENT")
    digest = sha256()
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        for chunk in iter(lambda: file.read(4 << 20), b""):
            digest.update(chunk)
        after = os.fstat(file.fileno())
    need((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
         and digest.hexdigest() == WEIGHT, "EXACT_OFFICIAL_CLIP_WEIGHTS")


def verify_provider(provider):
    manifest = json.loads(read(provider / "PROVIDER_FILES.json", 64 << 10, PROVIDER))
    for filename, pin in manifest["files"].items():
        path = provider / filename
        need(path.is_relative_to(provider), "EXACT_OFFICIAL_CLIP_PROVIDER")
        if pin["bytes"] == 0:
            need(filename == "ftfy/py.typed" and pin["SHA256"] == sha256(b"").hexdigest()
                 and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode)
                 and path.lstat().st_size == 0, "EXACT_EMPTY_FTFY_PY_TYPED")
        else:
            need(len(read(path, 2 << 20, pin["SHA256"])) == pin["bytes"], "EXACT_OFFICIAL_CLIP_PROVIDER")
    return len(manifest["files"])


def run(args):
    need(all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "NO_GPU_CPU_SCORER")
    plan, plan_sha = frozen(args.plan)
    terminal = json.loads(read(args.terminal, 48 << 10, args.terminal_sha))
    worker, valid = final_pairs(args.generation, plan, plan_sha, terminal)
    results = []
    if valid:
        weight(args.weights)
        verify_provider(args.provider)
        sys.path.insert(0, str(args.provider))
        import clip
        from PIL import Image
        import torch

        need(Path(clip.__file__) == args.provider / "clip/__init__.py" and not torch.cuda.is_initialized(),
             "ACTUAL_CPU_CLIP_PROVIDER")
        model, transform = clip.load(str(args.weights), device="cpu", jit=False)
        model.eval()
        with torch.inference_mode():
            tokens = clip.tokenize([item["prompt"] for item, _, _ in valid], truncate=False)
            features = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
            for i, (item, left, right) in enumerate(valid):
                values = []
                for row in (left, right):
                    name = f"{row['arm']}-{row['pair']:02d}.private.png"
                    need(row["image"] == name, "EXACT_GENERATED_IMAGE_NAME")
                    raw = read(args.generation / name, 2 << 20, row["image_SHA256"])
                    need(len(raw) == row["image_bytes"], "EXACT_GENERATED_IMAGE_EXTENT")
                    with Image.open(io.BytesIO(raw)) as image:
                        need(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256),
                             "EXACT_GENERATED_IMAGE_TYPE")
                        values.append(transform(image))
                embedded = torch.nn.functional.normalize(model.encode_image(torch.stack(values)).float(), dim=-1)
                a, b = [float(torch.dot(value, features[i])) for value in embedded]
                need(all(math.isfinite(value) and -1.00001 <= value <= 1.00001 for value in (a, b)), "FINITE_CLIP_COSINE")
                results.append(dict(pair=item["pair"], family=item["family"], base=a, cp25=b, delta=b-a))
        need(not torch.cuda.is_initialized(), "SCORER_GPU_WAS_INITIALIZED")
    per_family = {family: [row["delta"] for row in results if row["family"] == family]
                  for family in {row["family"] for row in plan["samples"]}}
    report = dict(planned_slots=12, planned_pairs=6, evaluated_pairs=len(results), incomplete_pairs=6-len(results),
                  missing_and_error_counts=worker["counts"], results=results, metric="OpenAI CLIP ViT-B/32 normalized cosine",
                  family_counts={name: dict(planned=2, evaluated=len(rows)) for name, rows in sorted(per_family.items())},
                  family_equal_mean=(statistics.fmean(statistics.fmean(rows) for rows in per_family.values())
                                     if all(per_family.values()) else None),
                  primary_blind_rubric_completed=False, formal_FID=False, source_plan_SHA256=plan_sha,
                  generation_result_SHA256=terminal["worker_result_SHA256"], weights_SHA256=WEIGHT,
                  real_accelerator_calls=0)
    put(args.output, wire(report))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "terminal", "generation", "weights", "provider", "output"):
        cli.add_argument("--"+name, type=Path, required=True)
    cli.add_argument("--terminal-sha", required=True)
    args = cli.parse_args()
    need(HEX.fullmatch(args.terminal_sha), "INDEPENDENT_TERMINAL_HASH")
    run(args)


if __name__ == "__main__":
    main()
