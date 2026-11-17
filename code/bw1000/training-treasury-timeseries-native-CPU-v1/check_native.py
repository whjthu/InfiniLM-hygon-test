"""Qualify frozen Treasury time-series inputs with the native CPU Qwen template."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import time

MANIFEST_SHA256 = "6d0ce57a02fbfb5174d61ee3548c3b43bdcae1ddb4e5e6d589a7706dfb83ca5f"
TEMPLATE_SHA256 = "d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3"
CONFIG_SHA256 = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
VERSIONS = {"ms-swift": "4.4.2", "transformers": "5.12.1", "qwen-vl-utils": "0.0.14"}
SPLITS = ("train", "dev", "blind")
TOPICS = ("lookup", "signed_difference", "maximum_date")
MAX_LENGTH, MAX_PIXELS = 4096, 262144


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    require(not path.is_symlink(), "symlink input")
    raw = path.read_bytes()
    require(digest(raw) == expected, "input SHA mismatch")
    return raw


def jsonlines(raw):
    return [json.loads(line) for line in raw.splitlines()]


def png_size(raw):
    require(len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n"
            and raw[8:16] == b"\x00\x00\x00\rIHDR", "PNG header differs")
    width, height = struct.unpack(">II", raw[16:24])
    require(width > 0 and height > 0, "empty PNG")
    return [width, height]


def bind(root):
    """Open split inputs/media and train labels only; never dev/blind labels/windows."""
    root = root.resolve(strict=True)
    manifest = json.loads(pinned(root / "manifest.json", MANIFEST_SHA256))
    require(manifest["schema"] == "treasury-timeseries-120-rows-v1", "dataset schema differs")
    pins = manifest["files_SHA256"]
    rows, media, windows = {}, {}, {}
    seen_ids, seen_media_sha, seen_windows = set(), set(), set()
    for split in SPLITS:
        proof = manifest["splits"][split]
        require((proof["examples"], proof["media"], proof["windows"], proof["raw_dates"])
                == (144, 16, 8, 40), "planned split denominator differs")
        rows[split] = jsonlines(pinned(root / split / "inputs.jsonl", pins[split + "/inputs.jsonl"]))
        require(len(rows[split]) == 144, "split input denominator differs")
        split_media, split_windows = set(), set()
        counts = Counter()
        for row in rows[split]:
            require(set(row) == {"id", "window_id", "modality", "topic", "images", "messages"},
                    "dataset interface keys differ")
            require(row["topic"] in TOPICS and row["modality"] in ("table", "chart"), "unknown topic/media")
            require(row["id"] not in seen_ids, "duplicate question identity")
            seen_ids.add(row["id"])
            require(isinstance(row["window_id"], str) and len(row["window_id"]) == 64,
                    "window identity differs")
            require(len(row["images"]) == 1, "image count differs")
            name = row["images"][0]
            require(isinstance(name, str) and PurePosixPath(name).parts ==
                    (split, row["window_id"] + "-" + row["modality"] + ".png")
                    and name in pins, "media identity escaped split")
            messages = row["messages"]
            roles = [message.get("role") for message in messages]
            require(roles == (["user", "assistant"] if split == "train" else ["user"]),
                    "supervision leaked or absent")
            require(all(set(message) == {"role", "content"} and isinstance(message["content"], str)
                        and message["content"] for message in messages), "message interface differs")
            require(messages[0]["content"].startswith("<image>\n")
                    and messages[0]["content"].count("<image>") == 1, "image marker differs")
            split_media.add(name)
            split_windows.add(row["window_id"])
            counts[row["topic"]] += 1
        require(counts == {topic: 48 for topic in TOPICS} and len(split_media) == 16
                and len(split_windows) == 8 and not split_windows & seen_windows,
                "topic/media/window denominator or split isolation differs")
        seen_windows.update(split_windows)
        windows[split] = split_windows
        for name in sorted(split_media):
            raw = pinned(root / name, pins[name])
            require(pins[name] not in seen_media_sha, "shared media bytes across windows/splits")
            seen_media_sha.add(pins[name])
            media[name] = dict(SHA256=pins[name], bytes=len(raw), original_size=png_size(raw))
    train_labels = jsonlines(pinned(root / "train/labels.jsonl", pins["train/labels.jsonl"]))
    require(len(train_labels) == 144 and
            [(r["id"], r["messages"][1]["content"]) for r in rows["train"]] ==
            [(r["id"], r["answer"]) for r in train_labels], "train labels and supervised answers differ")
    require(len(media) == 48 and len(seen_ids) == 432 and len(seen_windows) == 24,
            "whole dataset denominator differs")
    return root, manifest, rows, media


def tokens_contract(encoded, image_token, train, prompt_ids=None):
    ids, labels = encoded["input_ids"], encoded.get("labels")
    require(isinstance(ids, list) and 0 < len(ids) <= MAX_LENGTH
            and all(type(token) is int for token in ids), "untruncated token length/type differs")
    images = [i for i, token in enumerate(ids) if token == image_token]
    require(images, "missing native image tokens")
    supervised = []
    if train:
        require(isinstance(labels, list) and len(labels) == len(ids)
                and all(type(label) is int for label in labels), "training labels length/type differs")
        require(prompt_ids and ids[:len(prompt_ids)] == prompt_ids, "native user-prefix identity differs")
        supervised = [i for i, label in enumerate(labels) if label != -100]
        require(supervised and min(supervised) >= len(prompt_ids) and min(supervised) > max(images),
                "user/image tokens supervised or answer absent")
        require(all(labels[i] == ids[i] for i in supervised)
                and all(labels[i] == -100 for i in images), "training mask differs")
    else:
        require(labels is None, "inference holdout contains labels")
    return [len(ids), len(images), len(supervised)]


def batch_contract(labels, attention, expected):
    lengths = [len(row) for row in expected]
    require(len(labels) == len(attention) == len(expected) == 2, "collator batch denominator differs")
    width = max(lengths)
    for target, mask, original in zip(labels, attention, expected):
        require(len(target) == len(mask) == width and mask == [1] * len(original) + [0] * (width-len(original)),
                "collator right padding/attention differs")
        require(target == original + [-100] * (width-len(original)), "collator labels or padding differ")


def check(root, model_dir, output, deadline, monotonic_end):
    output.mkdir()
    state = dict(state="NATIVE_CPU_QUALIFICATION_STARTED", planned_rows={s:144 for s in SPLITS},
                 completed_rows={s:0 for s in SPLITS}, planned_images=48, actual_updates=0,
                 dev_blind_label_content_opened=False, actual_native_encode_calls=0)
    def remaining():
        require(min(deadline-time.time(), monotonic_end-time.monotonic()) > 5, "original CPU deadline exhausted")
    def save():
        (output / "progress.json").write_text(json.dumps(state, sort_keys=True), encoding="utf8")
    save()
    try:
        remaining()
        require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "GPU assigned to CPU scope")
        root, manifest, rows, media = bind(root)
        state["media_SHA_verified"] = len(media)
        state["label_identity_checks"] = {s:dict(manifest_SHA256=manifest["files_SHA256"][s+"/labels.jsonl"],
            actual_bytes_SHA256_verified=(s=="train"), content_parsed=(s=="train")) for s in SPLITS}
        versions = {name:importlib.metadata.version(name) for name in VERSIONS}
        require(versions == VERSIONS, "native processor environment changed")
        config = json.loads(pinned(model_dir / "config.json", CONFIG_SHA256))
        import torch
        from swift.model import get_model_processor
        from swift.template import get_template
        from swift.template.templates import qwen
        require(not torch.cuda.is_available(), "device runtime available in CPU scope")
        pinned(Path(qwen.__file__), TEMPLATE_SHA256)
        model, processor = get_model_processor(str(model_dir), model_type="qwen3_vl", load_model=False,
                                               download_model=False, device_map="cpu")
        require(model is None, "processor materialized a model")
        template = get_template(processor, template_type="qwen3_vl", max_length=MAX_LENGTH,
                                max_pixels=MAX_PIXELS, padding_side="right", padding_free=False,
                                truncation_strategy="raise")
        counts, samples, media_grids = {}, [], {}
        for split in SPLITS:
            counts[split] = []
            for row in rows[split]:
                remaining()
                request = dict(messages=copy.deepcopy(row["messages"]), images=[str(root/row["images"][0])])
                prompt_ids = None
                if split == "train":
                    template.set_mode("transformers")
                    prompt = template.encode(dict(messages=copy.deepcopy(row["messages"][:1]), images=request["images"]))
                    state["actual_native_encode_calls"] += 1
                    tokens_contract(prompt, config["image_token_id"], False)
                    prompt_ids = prompt["input_ids"]
                template.set_mode("train" if split == "train" else "transformers")
                encoded = template.encode(request)
                state["actual_native_encode_calls"] += 1
                count = tokens_contract(encoded, config["image_token_id"], split=="train", prompt_ids)
                pixels, grid = encoded["pixel_values"], encoded["image_grid_thw"]
                require(pixels.device.type == grid.device.type == "cpu" and pixels.numel() > 0
                        and tuple(grid.shape) == (1,3) and bool(torch.isfinite(pixels).all()),
                        "native image tensor/grid invalid")
                dimensions = grid.tolist()[0]
                require(all(type(x) is int and x > 0 for x in dimensions), "invalid patch grid")
                patch = processor.image_processor.patch_size
                merge = processor.image_processor.merge_size
                require(count[1] == dimensions[0]*dimensions[1]*dimensions[2]//(merge*merge),
                        "native image grid/token count differs")
                image_name = row["images"][0]
                processed = dict(grid=dimensions, spatial_size=[dimensions[2]*patch,dimensions[1]*patch],
                                 pixel_values_shape=list(pixels.shape), image_tokens=count[1])
                require(image_name not in media_grids or media_grids[image_name] == processed,
                        "same image processing identity changed")
                media_grids[image_name] = processed
                counts[split].append(count)
                if split == "train" and (not samples or (len(samples)==1 and len(encoded["input_ids"]) != len(samples[0]["input_ids"]))):
                    samples.append(copy.deepcopy(encoded))
                state["completed_rows"][split] += 1
                save()
        require(len(samples)==2, "no two train lengths for padding qualification")
        remaining()
        template.set_mode("train")
        with torch.device("meta"):
            template.dummy_model, _ = get_model_processor(str(model_dir), model_type="qwen3_vl",
                return_dummy_model=True, download_model=False)
        params = list(template.dummy_model.parameters())
        require(params and all(p.device.type == "meta" for p in params), "RoPE dummy model allocated real parameters")
        expected_labels = [list(sample["labels"]) for sample in samples]
        batch = template.data_collator(samples)
        batch_contract(batch["labels"].tolist(), batch["attention_mask"].tolist(), expected_labels)
        require(tuple(batch["image_grid_thw"].shape)==(2,3), "collator lost image grid")
        require(all(p.device.type == "meta" for p in template.dummy_model.parameters()), "collation materialized model")
        require(state["actual_native_encode_calls"] == 576 and len(media_grids) == 48,
                "native encode/media completion denominator differs")
        state.update(state="NATIVE_CPU_INPUT_AND_TRAIN_COLLATOR_PASS", manifest_SHA256=MANIFEST_SHA256,
            versions=versions, template_SHA256=TEMPLATE_SHA256, config_SHA256=CONFIG_SHA256,
            actual_GPU_cards=0, actual_updates=0, native_rows=432, actual_native_encode_calls=576,
            model_weights_loaded=False, rope_dummy_model_on_meta=True, collated_train_rows=2,
            max_length=MAX_LENGTH, max_pixels=MAX_PIXELS, truncation="raise", semantic_readability="UNKNOWN",
            formal_training_ready=False, labels_usable_for_training_split_only=True,
            processed_media={name:dict(**media[name],**media_grids[name]) for name in sorted(media)},
            splits={s:dict(rows=len(v), input_tokens_range=[min(x[0] for x in v),max(x[0] for x in v)],
                image_tokens_range=[min(x[1] for x in v),max(x[1] for x in v)],
                supervised_tokens=sum(x[2] for x in v), counts_SHA256=digest(json.dumps(v).encode())) for s,v in counts.items()},
            blind_effect=False, model_or_numerical_qualification=False)
        save()
        return state
    except Exception as error:
        state.update(state="NATIVE_CPU_QUALIFICATION_FAILED", failure_type=type(error).__name__,
                     remaining_rows={s:144-state["completed_rows"][s] for s in SPLITS}, training_ready=False)
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    require(math.isfinite(args.deadline), "nonfinite original deadline")
    budget = min(450.0, args.deadline-time.time())
    require(budget > 5, "expired original epoch")
    result = check(args.root, args.model_dir, args.output, args.deadline, time.monotonic()+budget)
    print(json.dumps({k:result[k] for k in ("state","completed_rows","actual_GPU_cards","actual_updates")}))
