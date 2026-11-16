"""Run native Swift Qwen3-VL processing on every bound domain input, on CPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


DATASET_SHA256 = "e0219347307a94259b826b929a88f3fe991b688026677c86c25075c6941bc19b"
TEMPLATE_SHA256 = "d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3"
CONFIG_SHA256 = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
SOURCE_VERSIONS = {"ms-swift": "4.4.2", "transformers": "5.12.1", "qwen-vl-utils": "0.0.14"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f"SHA-256 mismatch: {path.name}")
    return raw


def check(dataset, model_dir):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "GPU assigned to CPU check")
    versions = {name: importlib.metadata.version(name) for name in SOURCE_VERSIONS}
    require(versions == SOURCE_VERSIONS, "Swift processor environment changed")
    raw = read_pinned(dataset, DATASET_SHA256)
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == 160, "incorrect training denominator")
    config = json.loads(read_pinned(model_dir / "config.json", CONFIG_SHA256))
    image_token_id = config["image_token_id"]

    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.templates import qwen

    read_pinned(Path(qwen.__file__), TEMPLATE_SHA256)
    loaded_model, processor = get_model_processor(str(model_dir), load_model=False,
                                                  download_model=False, device_map="cpu")
    require(loaded_model is None, "processor unexpectedly returned a model")
    template = get_template(processor, template_type="qwen3_vl", max_length=4096,
                            truncation_strategy="raise")
    template.set_mode("train")
    evidence = []
    samples = []
    for index, row in enumerate(rows):
        encoded = template.encode(row)
        ids = encoded["input_ids"]
        labels = encoded["labels"]
        length = len(ids)
        require(0 < length == len(labels) <= 4096, f"sequence length on row {index}")
        image_positions = [i for i, token in enumerate(ids) if token == image_token_id]
        supervised = [i for i, label in enumerate(labels) if label != -100]
        require(image_positions and supervised and supervised[0] > image_positions[-1],
                f"missing image or answer supervision on row {index}")
        require(all(labels[i] == -100 for i in image_positions)
                and all(labels[i] == ids[i] for i in supervised), f"label mask on row {index}")
        pixels = encoded["pixel_values"]
        grid = encoded["image_grid_thw"]
        require(pixels.device.type == "cpu" and grid.device.type == "cpu"
                and pixels.numel() > 0 and tuple(grid.shape) == (1, 3)
                and bool(torch.isfinite(pixels).all()), f"processed image on row {index}")
        evidence.append([length, len(image_positions), len(supervised), grid.tolist()])
        if not samples or (len(samples) == 1 and length != len(samples[0]["input_ids"])):
            samples.append(encoded)

    require(len(samples) == 2, "no distinct lengths for checking padding")
    lengths = [len(sample["input_ids"]) for sample in samples]
    batch = template.data_collator(samples)
    labels = batch["labels"]
    attention = batch["attention_mask"].bool()
    require(tuple(labels.shape) == tuple(attention.shape) == (2, max(lengths))
            and attention.sum(dim=1).tolist() == lengths
            and bool((labels[~attention] == -100).all())
            and tuple(batch["image_grid_thw"].shape) == (2, 3), "batched padding or vision grid")
    return {
        "state": "NATIVE_CPU_PREPROCESSING_AND_COLLATION_PASS",
        "rows": len(evidence),
        "dataset_SHA256": DATASET_SHA256,
        "template_SHA256": TEMPLATE_SHA256,
        "versions": versions,
        "input_tokens_range": [min(x[0] for x in evidence), max(x[0] for x in evidence)],
        "image_tokens_range": [min(x[1] for x in evidence), max(x[1] for x in evidence)],
        "supervised_tokens_range": [min(x[2] for x in evidence), max(x[2] for x in evidence)],
        "per_row_counts_SHA256": hashlib.sha256(json.dumps(evidence).encode()).hexdigest(),
        "collated_lengths": lengths,
        "actual_GPU_cards": 0,
        "actual_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.dataset, args.model_dir), sort_keys=True))


if __name__ == "__main__":
    main()
