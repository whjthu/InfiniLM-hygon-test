"""Qualify the native text route for the first ABO orbit clip."""

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import MethodType, SimpleNamespace


CAPTION = "The camera moves around a stationary household product against a neutral studio background."
FRAME_RECEIPT = "0c0749760fcabab3f01be289c1103e1fa47fc167dd736d295cd51f24e7a44b5f"
NATIVE = {
    "hyvideo/models/text_encoders/__init__.py": "3f288a7a36e15a9d69cfd8626401f6bf0e2cca08ade49c4df290c98e81625846",
    "hyvideo/pipelines/hunyuan_video_pipeline.py": "ba4cdec9e195c23638dd27dc5a3026c82aa82444ee97a7fdcb2ced2c85782c3d",
    "hyvideo/commons/__init__.py": "64c8ffb7fcd1f35195f43c8d6bdad2c1eafe1713fd3d5dbe146a5fd672b09a29",
}
TOKENIZER = {
    "config.json": (1374, "77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43"),
    "model.safetensors.index.json": (57619, "73b333b0b16e5286ddba615d2caebcd495cf7e616f52eb217a81781393d79de9"),
    "tokenizer.json": (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (5702, "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56"),
}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, limit, expected, exact_size=None):
    require(not path.is_symlink(), "UNEXPECTED_SYMLINK")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (exact_size is None or len(raw) == exact_size)
            and digest(raw) == expected, "PINNED_ASSET_MISMATCH")
    return raw


def native_glyph_functions(pipeline, filename, torch):
    # Use the deployed method bodies without initializing unrelated video components.
    names = {"_extract_glyph_texts", "_process_single_byt5_prompt"}
    tree = ast.parse(pipeline, filename)
    matches = [
        [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for cls in tree.body if isinstance(cls, ast.ClassDef)
    ]
    groups = [methods for methods in matches if {method.name for method in methods} == names]
    require(len(groups) == 1 and len(groups[0]) == 2, "UNIQUE_NATIVE_GLYPH_IMPLEMENTATION")
    namespace = {"torch": torch, "re": re, "__builtins__": __builtins__}
    module = ast.Module(body=groups[0], type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)
    calls = SimpleNamespace(byt5_max_length=256)
    for name in names:
        setattr(calls, name, MethodType(namespace[name], calls))
    return calls


def qualify(source, qwen, render):
    require(all(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
            "OFFLINE_NATIVE_RUNTIME")
    visible = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    require(all(os.environ.get(name) == "-1" for name in visible)
            and os.environ.get("SLURM_STEP_GPUS", "") in ("", "none", "N/A", "(null)")
            and not re.search("gpu|dcu", os.environ.get("SLURM_TRES_PER_TASK", ""), re.I), "ZERO_GRES_REQUIRED")
    pinned(render, 16384, FRAME_RECEIPT)
    known_sources = {name: pinned(source / name, 1 << 20, pin) for name, pin in NATIVE.items()}
    for name, (size, pin) in TOKENIZER.items():
        pinned(qwen / name, size, pin, exact_size=size)

    sys.path.insert(0, str(source))
    import torch
    from hyvideo.models import text_encoders

    require(torch.cuda.device_count() == 0, "ZERO_VISIBLE_DCU")
    expected = source / "hyvideo/models/text_encoders/__init__.py"
    require(Path(inspect.getfile(text_encoders.TextEncoder)).resolve() == expected.resolve(),
            "EXACT_DEPLOYED_TEXT_ENCODER")
    template = "li-dit-encode-video-json"
    templates = text_encoders.PROMPT_TEMPLATE
    tokenizer, actual_path, processor = text_encoders.load_tokenizer("llm", str(qwen), padding_side="right")
    require(Path(actual_path) == qwen and processor is None and tokenizer.padding_side == "right",
            "EXACT_QWEN_TOKENIZER")
    # Set only the input fields consumed by the native text2tokens method. No encoder is constructed here.
    encoder = text_encoders.TextEncoder.__new__(text_encoders.TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.tokenizer = tokenizer
    encoder.use_template = True
    encoder.use_video_template = True
    encoder.prompt_template_video = deepcopy(templates[template])
    batch = encoder.text2tokens(CAPTION, data_type="video", max_length=300)
    crop = encoder.prompt_template_video["crop_start"]
    ids, mask = batch["input_ids"], batch["attention_mask"]
    require(type(crop) is int and crop > 0 and list(ids.shape) == list(mask.shape) == [1, crop + 300]
            and ids.dtype == mask.dtype == torch.int64, "EXACT_300_SLOT_NATIVE_VIDEO_TOKENIZATION")
    after_crop = mask[:, crop:]
    require(bool(((after_crop == 0) | (after_crop == 1)).all())
            and 0 < int(after_crop.sum()) < 300, "INVALID_NATIVE_MASK")

    pipeline = "hyvideo/pipelines/hunyuan_video_pipeline.py"
    glyph = native_glyph_functions(known_sources[pipeline], str(source / pipeline), torch)
    require(glyph._extract_glyph_texts(CAPTION) == [] and glyph._extract_glyph_texts('"LABEL"') == ["LABEL"],
            "EXACT_UNQUOTED_CAPTION_BRANCH")
    states, glyph_mask = glyph._process_single_byt5_prompt(CAPTION, torch.device("cpu"))
    require(list(states.shape) == [1, 256, 1472] and list(glyph_mask.shape) == [1, 256]
            and states.dtype == torch.float32 and glyph_mask.dtype == torch.int64
            and int(torch.count_nonzero(states)) == int(torch.count_nonzero(glyph_mask)) == 0,
            "NATIVE_EMPTY_GLYPH_CONDITION")
    return dict(actual_tokenized_captions=1, native_empty_glyph_calls=1, frame_receipt_SHA256=FRAME_RECEIPT,
                caption_SHA256=digest(CAPTION.encode()), pad_side="right", crop_start=crop, post_crop_slots=300,
                post_crop_active_slots=int(after_crop.sum()), glyph_shape=list(states.shape),
                glyph_dtype=str(states.dtype), glyph_mask_dtype=str(glyph_mask.dtype), native_glyph_mask_sum=0,
                native_model_forward=0, actual_optimizer_updates=0, training_rows=0, actual_GPU=0, passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("render", type=Path)
    cli = parser.parse_args()
    print(json.dumps(qualify(cli.source, cli.qwen, cli.render), sort_keys=True))
