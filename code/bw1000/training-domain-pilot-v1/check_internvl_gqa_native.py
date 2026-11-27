"""Check the real InternVL CPU processor and collator on bound GQA training rows."""
import argparse
import copy
from hashlib import sha256
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path):
    return sha256(path.read_bytes()).hexdigest()


def main(root, expected_rows=4):
    binding = json.loads((root / "BINDING.json").read_bytes())
    data = root / "data"
    model_dir = root / "InternVL3_5-8B-HF"
    train_path = data / "train.jsonl"
    need(pin(train_path) == binding["staged_train_SHA256"], "EXACT_STAGED_TRAINING_RECORDS")
    need(set(p.name for p in model_dir.iterdir()) == set(binding["metadata"]) and len(binding["metadata"]) == 13,
         "ONLY_BOUND_MODEL_METADATA")
    for filename, expected in binding["metadata"].items():
        need(not filename.endswith(".safetensors") and pin(model_dir / filename) == expected["SHA256"],
             "EXACT_FROZEN_PROCESSOR_METADATA")
    rows = [json.loads(line) for line in train_path.read_bytes().splitlines()]
    need(expected_rows in {4, 64} and len(rows) == len(binding["images"]) == expected_rows,
         "FIXED_REAL_TRAINING_ROW_COUNT")
    used_images = set()
    for row in rows:
        media, = row["images"]
        image = Path(media)
        need(image.parent == data and re.fullmatch(r"[0-9]+\.jpg", image.name)
             and pin(image) == binding["images"][image.name], "EXACT_OFFICIAL_STANFORD_JPEG")
        need(image.name not in used_images, "DUPLICATE_TRAINING_IMAGE")
        used_images.add(image.name)
        q, a = row["messages"]
        need(q["role"] == "user" and q["content"].startswith("<image>\n")
             and a["role"] == "assistant" and bool(a["content"].strip()), "ORIGINAL_TRAIN_SUPERVISION_ROLES")

    import torch
    import safetensors
    import safetensors.torch
    from swift.model import get_model_processor
    from swift.template import get_template

    need(not torch.cuda.is_initialized(), "PROCESSOR_STARTED_WITH_CUDA_INITIALIZED")
    expected_versions = {"ms-swift": "4.4.2", "transformers": "5.5.0", "tokenizers": "0.22.2"}
    actual_versions = {key: importlib.metadata.version(key) for key in expected_versions}
    need(actual_versions == expected_versions, "FROZEN_NATIVE_PROVIDER_VERSIONS")
    denied = []

    def forbid(*args, **kwargs):
        denied.append("blocked")
        raise PermissionError("PROCESSOR_ONLY")

    def guard(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            forbid()
        if event == "open" and isinstance(args[0], (str, bytes)) and str(args[0]).endswith(".safetensors"):
            forbid()

    sys.addaudithook(guard)
    torch.cuda._lazy_init = forbid
    torch.load = torch.serialization.load = forbid
    safetensors.safe_open = safetensors.torch.safe_open = safetensors.torch.load_file = forbid
    torch.set_num_threads(2)
    model, processor = get_model_processor(str(model_dir), model_type="internvl", load_model=False,
                                           download_model=False, device_map="cpu",
                                           attn_impl="sdpa", torch_dtype=torch.bfloat16)
    need(model is None, "PROCESSOR_PATH_LOADED_UNEXPECTED_MODEL")
    need(processor.model_meta.template == "internvl_hf", "NATIVE_INTERNVL_PROCESSOR_REQUIRED")
    template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
                            padding_side="right", padding_free=False, truncation_strategy="raise")
    template.set_mode("train")
    eos = processor.tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    need(eos_ids and all(type(value) is int for value in eos_ids), "ORIGINAL_EOS_ID_REQUIRED")
    records = []
    for ordinal, row in enumerate(rows):
        encoded = template.encode(row)
        ids = encoded["input_ids"]
        need(0 < len(ids) <= 4096, "FULL_CONTEXT_LIMIT")
        batch = template.data_collator([copy.deepcopy(encoded)])
        tokens = batch["input_ids"]
        need(tokens.device.type == "cpu" and tuple(tokens.shape) == (1, len(ids))
             and torch.equal(tokens[0], torch.as_tensor(ids, dtype=tokens.dtype)), "COLLATOR_MUST_RETAIN_ALL_IDS")
        labels = batch["labels"]
        need(labels.device.type == "cpu" and labels.shape == tokens.shape, "NATIVE_LABEL_DIMENSIONS")
        mask = labels[0] != -100
        need(bool(mask.any()) and bool((~mask).any()) and int(mask.nonzero()[0]) > 0
             and torch.equal(labels[0][mask], tokens[0][mask]), "REAL_SUPERVISION_AND_MASKED_PREFIX")
        eos_matches = sum(int(((tokens[0] == value) & mask).sum()) for value in eos_ids)
        need(eos_matches > 0, "NATIVE_SUPERVISED_EOS_REQUIRED")
        pixels = batch["pixel_values"]
        need(pixels.device.type == "cpu" and pixels.numel() > 0 and bool(torch.isfinite(pixels).all()),
             "FINITE_NATIVE_IMAGE_TENSOR_REQUIRED")
        records.append(dict(ordinal=ordinal + 1, full_tokens=len(ids), supervised_tokens=int(mask.sum()),
                            supervised_eos=eos_matches, original_prefix_masked=True, pixel_shape=list(pixels.shape)))
    need(len(records) == expected_rows and not denied and not torch.cuda.is_initialized(), "ONLY_CPU_PROCESSOR_ALLOWED")
    providers = {label: pin(Path(inspect.getsourcefile(member))) for label, member in (
        ("get_model_processor", get_model_processor), ("get_template", get_template),
        ("collator", template.data_collator))}
    return dict(passed=True, rows=records, native_rows=len(records), source_train_SHA256=binding["source_train_SHA256"],
                metadata_files=len(binding["metadata"]), versions=actual_versions, provider_SHA256=providers,
                optimization_updates=0, gpu_initialized=False, model_weights_read=0, training_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-rows", type=int, choices=(4, 64), default=4)
    args = parser.parse_args()
    print(json.dumps(main(args.root, args.expected_rows), sort_keys=True), flush=True)
