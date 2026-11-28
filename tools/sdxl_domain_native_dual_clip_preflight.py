"""Read both original SDXL text towers through the pinned training helper on CPU."""

import argparse
import ast
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import stat


PROVIDER_SHA = "420a01868d9748ddb2c165d3f76f0647e504d73506afc87f872c2d00e3c1c613"
TOWERS = (
    ((565, "39b8b2e4b1949e36969caa425b6c81c68bace99198dd9078ce05d16ad401fe7f"),
     (492_265_168, "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc")),
    ((575, "a892d1c3a69a7e9247a24de2bc1d5891e3109a54696e53be20093af671072c34"),
     (2_778_702_264, "3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff")),
)
CAPTIONS = (
    ("tool_chest", "0a2f6b50e1631cb5a5dc5ac47c6c34e694275288f6176f2a01b0bdcea41a4506"),
    ("painted_stool", "c27d943853f15401eafa9efc17bda3b8c32e19da316e45496afd4fb30d30435b"),
)
TOOL_CHEST_TOKENS = (
    "c589ebab6341e39916d867afe024b27b55c93e382f3253ddf4a042383d3fd3b1",
    "757a83130f1a31557f0b50ba82a83e15a8f50a3e47a5db7ff3463ab4eaf82d4e",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest, maximum, exact_size=None, keep_bytes=False):
    before = path.lstat()
    require(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= maximum and (exact_size is None or before.st_size == exact_size),
            "Expected canonical source and size")
    with path.open("rb") as stream:
        if keep_bytes:
            value = stream.read(maximum + 1)
            actual = sha256(value).hexdigest()
        else:
            value = None
            checksum = sha256()
            for block in iter(lambda: stream.read(4 << 20), b""):
                checksum.update(block)
            actual = checksum.hexdigest()
    after = path.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(actual == digest and all(getattr(before, key) == getattr(after, key) for key in identity),
            "Source changed or differs from the pinned file")
    return value


def training_functions(source, torch):
    tree = ast.parse(source)
    names = {"tokenize_prompt", "encode_prompt"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(definitions) == 2 and {node.name for node in definitions} == names
            and all(not node.decorator_list for node in definitions), "Expected exact training helper definitions")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<pinned training functions>", "exec"), namespace)
    return namespace["tokenize_prompt"], namespace["encode_prompt"]


def run(provider, token_root, tower_roots, captions_path):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Zero-GRES CPU environment required")
    source = pinned(provider, PROVIDER_SHA, 256 << 10, keep_bytes=True)
    captions = [pinned(path, expected, 1024, keep_bytes=True).decode("utf-8").strip()
                for path, (_, expected) in zip(captions_path, CAPTIONS, strict=True)]
    require(len(set(captions)) == 2 and all(captions), "Two distinct exact captions required")
    for root, tower in zip(tower_roots, TOWERS, strict=True):
        for name, (size, checksum) in zip(("config.json", "model.safetensors"), tower, strict=True):
            pinned(root / name, checksum, size, exact_size=size)

    import torch
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    require(not torch.cuda.is_initialized(), "Preflight must not initialize a GPU")
    torch.set_num_threads(4)
    tokenize, encode = training_functions(source, torch)
    tokenizers = [AutoTokenizer.from_pretrained(str(token_root / label), local_files_only=True, use_fast=False)
                  for label in ("tokenizer", "tokenizer_2")]
    require(all(type(token).__name__ == "CLIPTokenizer" and token.model_max_length == 77 for token in tokenizers),
            "Training tokenizer provider differs")

    def identity(tensor):
        return sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

    inputs = [tokenize(tokenizer, captions) for tokenizer in tokenizers]
    require(all(value.dtype == torch.int64 and tuple(value.shape) == (2, 77) for value in inputs)
            and tuple(identity(value[0]) for value in inputs) == TOOL_CHEST_TOKENS,
            "Token IDs differ from the first actual native tokenization")
    towers = []
    for model_cls, root in zip((CLIPTextModel, CLIPTextModelWithProjection), tower_roots, strict=True):
        model, loading = model_cls.from_pretrained(str(root), local_files_only=True, use_safetensors=True,
                                                   torch_dtype=torch.float32, output_loading_info=True)
        require(all(not loading.get(field) for field in ("missing_keys", "mismatched_keys", "error_msgs")),
                "Exact original tower failed to load all parameters")
        model.eval().requires_grad_(False)
        require({(value.device.type, value.dtype, value.requires_grad) for value in model.parameters()}
                == {("cpu", torch.float32, False)}, "Frozen FP32 tower CPU identity differs")
        towers.append(model)
    require(tuple(model.config.hidden_size for model in towers) == (768, 1280)
            and towers[1].config.projection_dim == 1280, "Original SDXL text tower shape differs")

    with torch.inference_mode():
        sequence, pooled = encode(towers, None, None, text_input_ids_list=inputs)
    require(tuple(sequence.shape) == (2, 77, 2048) and tuple(pooled.shape) == (2, 1280)
            and all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in (sequence, pooled)),
            "Expected finite native text conditions")
    rows = {}
    for index, (name, caption_sha) in enumerate(CAPTIONS):
        rows[name] = dict(caption_SHA256=caption_sha, token_SHA256=[identity(value[index]) for value in inputs],
                          sequence_SHA256=identity(sequence[index]), pooled_SHA256=identity(pooled[index]),
                          sequence_L2=sequence[index].norm().item(), pooled_L2=pooled[index].norm().item())
    require(all(len({row[key] for row in rows.values()}) == 2 for key in ("sequence_SHA256", "pooled_SHA256"))
            and not torch.cuda.is_initialized(), "Expected distinct conditions and zero GPU initialization")
    return dict(schema="SDXL_DOMAIN_TWO_CC0_NATIVE_DUAL_CLIP_CPU_V1", rows=rows, provider_SHA256=PROVIDER_SHA,
                sequence_shape=list(sequence.shape), pooled_shape=list(pooled.shape), dtype="float32",
                versions={name: metadata.version(name) for name in ("torch", "transformers", "diffusers")},
                actual_towers_loaded=2, actual_captions_conditioned=2, gpu_initialized=False, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", type=Path)
    parser.add_argument("token_root", type=Path)
    parser.add_argument("tower_roots", type=Path, nargs=2)
    parser.add_argument("captions", type=Path, nargs=2)
    args = parser.parse_args()
    print(json.dumps(run(args.provider, args.token_root, args.tower_roots, args.captions), sort_keys=True, allow_nan=False))
