"""Check answer-free Treasury development inputs through native Qwen3-VL inference preparation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from check_qwen_native_inputs import (
    CONFIG_SHA256, MAX_PIXELS, SOURCE_VERSIONS, TEMPLATE_SHA256, read_pinned, require,
)


DEV_PINS_SHA256 = "07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"


def bound_inputs(root, inventory):
    pins = json.loads(read_pinned(inventory, DEV_PINS_SHA256))
    expected_files = {"manifest.json", "visual.jsonl", "text-only.jsonl"}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(len(pins) == 39 and actual_files == set(pins) and expected_files <= set(pins),
            "development input package differs")
    require(all(not path.is_symlink() for path in root.rglob("*")), "linked development input")
    raw = {name: read_pinned(root / name, sha) for name, sha in pins.items()}
    manifest = json.loads(raw["manifest.json"])
    require(manifest["split"] == "dev" and manifest["requests"] == 108 and manifest["tables"] == 36
            and manifest["visual_SHA256"] == pins["visual.jsonl"]
            and manifest["text_only_SHA256"] == pins["text-only.jsonl"], "development manifest differs")
    visual, text = ([json.loads(line) for line in raw[name].splitlines()]
                    for name in ("visual.jsonl", "text-only.jsonl"))
    require(len(visual) == len(text) == 108 and len({row["id"] for row in visual}) == 108,
            "development request IDs or denominator differ")
    used = set()
    for image_row, text_row in zip(visual, text):
        require(set(image_row) == {"id", "messages", "images"}
                and set(text_row) == {"id", "messages"} and image_row["id"] == text_row["id"],
                "development arm identity differs")
        messages = image_row["messages"]
        require(len(messages) == 1 and set(messages[0]) == {"role", "content"}
                and messages[0]["role"] == "user" and messages[0]["content"].startswith("<image>\n")
                and messages[0]["content"].count("<image>") == 1
                and text_row["messages"] == [{"role": "user", "content": messages[0]["content"][8:]}],
                "requests must differ only by the image")
        require(len(image_row["images"]) == 1, "each visual request must use one image")
        name = image_row["images"][0]
        require(name in pins and name.startswith("media/") and name.endswith(".png")
                and raw[name].startswith(b"\x89PNG\r\n\x1a\n"), "wrong development image")
        used.add(name)
    require(used == set(pins).difference(expected_files), "development images do not cover the manifest")
    return visual, text


def check(root, inventory, model_dir):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "GPU assigned to CPU check")
    root = root.resolve(strict=True)
    visual, text = bound_inputs(root, inventory)
    versions = {name: importlib.metadata.version(name) for name in SOURCE_VERSIONS}
    require(versions == SOURCE_VERSIONS, "Swift processor environment changed")
    config = json.loads(read_pinned(model_dir / "config.json", CONFIG_SHA256))
    image_token = config["image_token_id"]

    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.templates import qwen

    require(not torch.cuda.is_available(), "device runtime visible to CPU check")
    read_pinned(Path(qwen.__file__), TEMPLATE_SHA256)
    model, processor = get_model_processor(str(model_dir), model_type="qwen3_vl",
                                           load_model=False, download_model=False, device_map="cpu")
    require(model is None, "processor returned a model")
    template = get_template(processor, template_type="qwen3_vl", max_length=4096,
                            max_pixels=MAX_PIXELS, padding_side="right", padding_free=False,
                            truncation_strategy="raise")
    template.set_mode("transformers")
    counts = {"visual": [], "text_only": []}
    for name, rows in (("visual", visual), ("text_only", text)):
        for row in rows:
            request = {"messages": row["messages"]}
            if name == "visual":
                request["images"] = [str(root / row["images"][0])]
            encoded = template.encode(request)
            ids = encoded["input_ids"]
            require(0 < len(ids) <= 4096 and encoded.get("labels") is None,
                    "unexpected sequence length or inference labels")
            image_count = ids.count(image_token)
            pixels, grid = encoded.get("pixel_values"), encoded.get("image_grid_thw")
            if name == "visual":
                require(image_count > 0 and pixels is not None and grid is not None
                        and pixels.device.type == grid.device.type == "cpu"
                        and pixels.numel() > 0 and tuple(grid.shape) == (1, 3)
                        and bool(torch.isfinite(pixels).all()), "visual preprocessing failed")
            else:
                require(image_count == 0 and pixels is None and grid is None, "text control used vision")
            counts[name].append([len(ids), image_count])
    return {"state": "NATIVE_CPU_DEVELOPMENT_INFERENCE_INPUTS_PASS",
            "dev_pins_SHA256": DEV_PINS_SHA256, "template_SHA256": TEMPLATE_SHA256,
            "versions": versions, "actual_GPU_cards": 0, "actual_generations": 0,
            "arms": {name: {"rows": len(rows), "token_range": [min(r[0] for r in rows), max(r[0] for r in rows)],
                            "image_token_range": [min(r[1] for r in rows), max(r[1] for r in rows)],
                            "counts_SHA256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}
                     for name, rows in counts.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.inventory, args.model_dir), sort_keys=True))
