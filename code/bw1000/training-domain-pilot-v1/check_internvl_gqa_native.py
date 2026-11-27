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
from types import SimpleNamespace


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path):
    return sha256(path.read_bytes()).hexdigest()


def native_fingerprint_provider(framework):
    """Bind the exact projection and Swift mask methods used by the update callback."""
    from swift.trainers.mixin import SwiftMixin
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    native_mixin = Path(inspect.getsourcefile(SwiftMixin.prepare_logits_to_keep))
    fingerprint_source = framework / "vlm/runtime/first_update_audit.py"
    providers = {
        "swift_mixin": pin(native_mixin),
        "swift_prepare_inputs": pin(Path(inspect.getsourcefile(Seq2SeqTrainer._prepare_inputs))),
        "framework_fingerprint": pin(fingerprint_source),
    }
    need(providers == {
        "swift_mixin": "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9",
        "swift_prepare_inputs": "28fe9cb927390ad423add8b98411a90fb2d1c801ead124b354802bdd0779ecf6",
        "framework_fingerprint": "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49",
    }, "EXACT_NATIVE_FINGERPRINT_PROVIDERS")
    sys.path.insert(0, str(framework))
    from vlm.runtime.first_update_audit import fingerprint

    need(Path(inspect.getsourcefile(fingerprint)) == fingerprint_source, "FRAMEWORK_FINGERPRINT_IMPORT")
    return SwiftMixin, fingerprint, providers


def native_logit_policy(swift, torch, processor, template, model_dir):
    from transformers import InternVLConfig, InternVLForConditionalGeneration

    config = InternVLConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        shell = InternVLForConditionalGeneration(config)
    shell.model_meta = processor.model_meta
    shell.model_info = processor.model_info
    need(all(p.device.type == "meta" for p in shell.parameters()), "ONLY_META_CLASS_SIGNATURE")
    holder = SimpleNamespace(args=SimpleNamespace(use_logits_to_keep=None), template=template, model=shell)
    need(template.sequence_parallel_size == 1 and swift.get_use_logits_to_keep(holder, True) is True,
         "ORIGINAL_NATIVE_T03_LOGIT_POLICY")
    return holder


def project_training_input(torch, batch, mask, fingerprints, holder, swift):
    attention = batch["attention_mask"]
    need(attention.shape == batch["input_ids"].shape and bool(torch.all(attention == 1)),
         "COMPLETE_UNPADDED_ATTENTION")
    before_names = ("input_ids", "labels", "pixel_values")
    after_names = ("attention_mask", "labels", "logits_to_keep")
    before = {name: fingerprints(batch[name]) for name in before_names}
    original_labels = batch["labels"].clone()
    loss_scale = batch.get("loss_scale")
    swift.prepare_logits_to_keep(holder, batch)
    logit_mask = batch["logits_to_keep"]
    need(logit_mask.dtype == torch.bool
         and torch.equal(logit_mask, torch.nn.functional.pad(mask[1:], (0, 1), value=True))
         and torch.equal(batch["labels"], torch.nn.functional.pad(original_labels[:, mask], (1, 0), value=-100))
         and int(logit_mask.sum()) == int(mask.sum()) + 1, "REAL_NATIVE_SHIFTED_LABEL_AND_LOGIT_MASK")
    if loss_scale is not None:
        need(torch.equal(batch["loss_scale"], torch.nn.functional.pad(loss_scale[:, mask], (1, 0), value=0)),
             "REAL_NATIVE_SHIFTED_LOSS_SCALE")
    after = {name: fingerprints(batch[name]) for name in after_names}
    need(all(value["finite"] for value in (*before.values(), *after.values())),
         "FINITE_NATIVE_FINGERPRINT_INPUT")
    return {
        "before": {name: value["SHA256"] for name, value in before.items()},
        "after": {name: value["SHA256"] for name, value in after.items()},
        "selected_logit_rows": int(logit_mask.sum()),
    }


def main(root, expected_rows=4, framework=None):
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
    raw_rows = train_path.read_bytes().splitlines()
    rows = [json.loads(line) for line in raw_rows]
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
    logit_providers = None
    if framework is not None:
        need(expected_rows == 64 and framework.is_absolute(), "FIXED_64_ROW_FINGERPRINT_SCOPE")
        swift, fingerprints, logit_providers = native_fingerprint_provider(framework)
        holder = native_logit_policy(swift, torch, processor, template, model_dir)
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
        record = dict(ordinal=ordinal + 1, full_tokens=len(ids), supervised_tokens=int(mask.sum()),
                      supervised_eos=eos_matches, original_prefix_masked=True, pixel_shape=list(pixels.shape))
        if logit_providers:
            record["staged_row_SHA256"] = sha256(raw_rows[ordinal]).hexdigest()
            record["native_logit_fingerprints"] = project_training_input(
                torch, batch, mask, fingerprints, holder, swift)
        records.append(record)
    need(len(records) == expected_rows and not denied and not torch.cuda.is_initialized(), "ONLY_CPU_PROCESSOR_ALLOWED")
    providers = {label: pin(Path(inspect.getsourcefile(member))) for label, member in (
        ("get_model_processor", get_model_processor), ("get_template", get_template),
        ("collator", template.data_collator))}
    result = dict(passed=True, rows=records, native_rows=len(records), source_train_SHA256=binding["source_train_SHA256"],
                  metadata_files=len(binding["metadata"]), versions=actual_versions, provider_SHA256=providers,
                  optimization_updates=0, gpu_initialized=False, model_weights_read=0, training_ready=False)
    if logit_providers:
        keys = {tuple(entry["native_logit_fingerprints"]["before"].values()) for entry in records}
        need(len(keys) == expected_rows, "UNAMBIGUOUS_NATIVE_FORWARD_FINGERPRINTS")
        result.update(staged_train_SHA256=binding["staged_train_SHA256"], fingerprint_providers=logit_providers,
                      native_forward_fingerprint_rows=len(keys), meta_model_only=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-rows", type=int, choices=(4, 64), default=4)
    parser.add_argument("--framework", type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.root, args.expected_rows, args.framework), sort_keys=True), flush=True)
