"""Compare original and explicit answer instructions using the same native base model."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from check_qwen_native_inputs import MAX_PIXELS, SOURCE_VERSIONS, read_pinned, require
from dev_pair_generation import ENGINE_PINS, run_arm
from prepare_qwen_inputs import digest, pinned_bytes
from treasury_holdout import TABLE_ID


PROMPTS_SHA256 = "c77d5355a3adc8418f868c367c01b79c1cf0e1ce2e4ab49223bd97955bb4470d"
IDS_SHA256 = "641cfb6e1efa9bfe53b4910405dd2235e183bde949eac6e97c9fa703e1f7a156"
PREPARATION_RECEIPT_SHA256 = "1b98e20fabe453d2317372a18ebeac25598ae59eb190fd6818143caab677052b"
ARMS = ("original", "explicit")


def load_pairs(root, receipt):
    proof = json.loads(pinned_bytes(receipt, PREPARATION_RECEIPT_SHA256))
    expected_names = {"prompts.jsonl", *("media/" + n for n in proof["media_SHA256"])}
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(present == expected_names and all(not p.is_symlink() for p in root.rglob("*")),
            "calibration asset set differs")
    prompts = pinned_bytes(root / "prompts.jsonl", PROMPTS_SHA256)
    rows = [json.loads(line) for line in prompts.splitlines()]
    ids = [r["id"] for r in rows]
    require(len(rows) == len(set(ids)) == 19 and
            digest(("\n".join(ids) + "\n").encode("ascii")) == IDS_SHA256,
            "wrong arithmetic identifiers")
    for row in rows:
        table = row["id"].removesuffix("_signed_difference")
        require(set(row) == {"id", "image", *ARMS} and TABLE_ID.fullmatch(table)
                and row["id"] == table + "_signed_difference"
                and row["image"] == f"media/{table}.png"
                and row["original"].startswith("<image>\n") and row["original"].count("<image>") == 1
                and row["explicit"] == row["original"] + proof["format_instruction"], "prompt arm differs")
        media = pinned_bytes(root / row["image"], proof["media_SHA256"][table + ".png"])
        require(media.startswith(b"\x89PNG\r\n\x1a\n"), "wrong arithmetic image signature")
    return rows


def generate(args):
    inputs = args.inputs.resolve(strict=True)
    rows = load_pairs(inputs, args.receipt)
    versions = {name: importlib.metadata.version(name) for name in SOURCE_VERSIONS}
    require(versions == SOURCE_VERSIONS, "Swift inference environment differs")
    for name, sha in ENGINE_PINS.items():
        read_pinned(Path("/swift-runtime/site/swift") / name, sha)

    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one DCU must be visible")
    load_start = time.monotonic()
    model, processor = get_model_processor(str(args.model), model_type="qwen3_vl", load_model=True,
                                           download_model=False, device_map={"": 0},
                                           torch_dtype=torch.bfloat16, attn_impl="sdpa")
    require(model.device == torch.device("cuda:0") and model.dtype == torch.bfloat16
            and model.config._attn_implementation == "sdpa" and not hasattr(model, "peft_config"),
            "base model binding differs")
    template = get_template(processor, template_type="qwen3_vl", max_length=4096,
                            max_pixels=MAX_PIXELS, padding_side="right",
                            padding_free=False, truncation_strategy="raise")
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    config = RequestConfig(max_tokens=64, temperature=0, stream=False, num_beams=1, n=1)
    effective = engine._prepare_generation_config(config)
    require(not effective.do_sample and effective.max_new_tokens == 64
            and effective.num_beams == 1 and effective.eos_token_id is not None, "greedy settings differ")
    outcome = dict(actual_DCU=1, load_seconds=time.monotonic() - load_start, versions=versions, split="train",
                   actual_updates=0, max_new_tokens=64, prompts_SHA256=PROMPTS_SHA256, shared_model_load=True)
    args.output.mkdir(mode=0o700)
    for arm in ARMS:
        requests = [(row["id"], InferRequest(messages=[{"role": "user", "content": row[arm]}],
                    images=[str(inputs / row["image"])])) for row in rows]
        outcome[arm] = run_arm(engine, requests, config, None, args.output, arm, args.deadline)
        require(engine.model is model, "prompt arm replaced the base model")
    with (args.output / "generation-outcome.json").open("x", encoding="utf8") as stream:
        json.dump(outcome, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "receipt", "model", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    generate(parser.parse_args())
