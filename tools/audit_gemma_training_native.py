"""Audit Gemma's frozen Swift input and collator on CPU before dispatch."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from build_gemma_train_exposure import (
    BINDING_SHA256, SOURCE_ROWS, TRAIN_FILE, TRAIN_SHA256, VIEW_FILE, digest, exposure_plan,
    read_pinned_file,
)


PROCESSOR_MANIFEST_SHA256 = "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b"
FRAMEWORK_SHA256 = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
VIEW_SHA256 = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"
VERSIONS = {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "transformers": "5.12.1",
            "ms-swift": "4.4.2", "tokenizers": "0.22.2", "Pillow": "11.3.0"}


def check_sample(ids, labels, suffix, end_ids):
    if not suffix or not set(suffix).intersection(end_ids) or not 1 < len(ids) == len(labels) <= 512:
        raise ValueError("invalid sequence length or native end-of-turn metadata")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("invalid token identity")
    if labels[0] != -100 or any(type(label) is not int or label not in (-100, token)
                                 for token, label in zip(ids, labels)):
        raise ValueError("invalid causal label mask")
    if ids[-len(suffix):] != suffix or labels[-len(suffix):] != suffix:
        raise ValueError("complete native end-of-turn suffix is not supervised")
    return sum(label != -100 for label in labels[1:])


def audit_rows(lines, full, capped, fingerprint, suffix, end_ids, check_deadline, progress):
    for ordinal, line in enumerate(lines, 1):
        check_deadline()
        original = full.encode(json.loads(line))
        capped_example = capped.encode(json.loads(line))
        fields = fingerprint(capped_example)
        if fingerprint(original)["SHA256"] != fields["SHA256"]:
            raise ValueError("unbounded Gemma input differs at the production limit")
        ids, labels = list(capped_example["input_ids"]), list(capped_example["labels"])
        supervised = check_sample(ids, labels, suffix, end_ids)
        batch = capped.data_collator([capped_example])
        if batch["input_ids"].tolist() != [ids] or batch["labels"].tolist() != [labels]:
            raise ValueError("native one-item collator changed the sample or mask")
        if "attention_mask" in batch and batch["attention_mask"].tolist() != [[1] * len(ids)]:
            raise ValueError("native one-item collator masked a non-padding input")
        progress({"ordinal": ordinal, "source_line_SHA256": digest(line), "tokens": len(ids),
                  "shifted_supervised_tokens": supervised, "encoded": fields, "collated": fingerprint(batch)})
        check_deadline()


def verify_assets(args, check_deadline):
    check_deadline()
    raw = read_pinned_file(args.prepared / TRAIN_FILE, 54246, TRAIN_SHA256)
    lines = raw.splitlines(keepends=True)
    actual_view = read_pinned_file(args.view / VIEW_FILE, 602794, VIEW_SHA256)
    if exposure_plan(lines)[0] != actual_view:
        raise ValueError("target training view differs from the local exposure order")
    binding = json.loads(read_pinned_file(args.prepared / "data-binding-private.json", 10123, BINDING_SHA256))
    paths = {json.loads(line)["images"][0] for line in lines}
    media = {pin["target_path"]: pin for name, pin in binding["media"].items()
             if name.startswith("train/media/")}
    if paths != media.keys() or len(paths) != 12:
        raise ValueError("training view and bound media identities differ")
    for path, pin in media.items():
        check_deadline()
        read_pinned_file(Path(path), pin["bytes"], pin["SHA256"])
    manifest = json.loads(read_pinned_file(args.processor_manifest, 2107, PROCESSOR_MANIFEST_SHA256))["files"]
    if {p.name for p in args.processor_root.iterdir()} != manifest.keys():
        raise ValueError("Gemma processor namespace must contain exactly the 14 pinned metadata files")
    for name, pin in manifest.items():
        check_deadline()
        read_pinned_file(args.processor_root / name, pin["bytes"], pin["SHA256"])
    framework = json.loads(read_pinned_file(args.framework_manifest, 112315, FRAMEWORK_SHA256))
    observer = "vlm/runtime/first_update_audit.py"
    pin = framework["files"][observer]
    read_pinned_file(args.framework_root / observer, pin["bytes"], pin["SHA256"])
    check_deadline()
    return lines


def run(args):
    remaining = args.deadline - time.time()
    if (not all(math.isfinite(v) for v in (args.epoch, args.deadline)) or args.epoch <= 0
            or not 0 < remaining <= args.deadline - args.epoch <= 600):
        raise ValueError("a valid original CPU deadline is required")
    stop = time.monotonic() + remaining

    def check_deadline():
        if time.monotonic() >= stop or time.time() >= args.deadline:
            raise TimeoutError("original CPU deadline reached")

    if any(os.environ.get(k) != "-1" for k in
           ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("native input audit requires hidden devices")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("native input audit requires the installed offline processor")
    lines = verify_assets(args, check_deadline)
    if {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError("target software versions differ from the original Gemma CPU receipt")
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("CUDA is already initialized")

    def forbid_device(*_, **__):
        raise RuntimeError("device initialization forbidden in the CPU input audit")

    torch.cuda._lazy_init = forbid_device
    torch.set_num_threads(2)
    random.seed(42)
    torch.random.default_generator.manual_seed(42)
    sys.path.insert(0, str(args.framework_root.resolve()))
    from vlm.runtime import first_update_audit
    from swift.model import get_model_processor
    from swift.template import get_template

    expected_provider = (args.framework_root / "vlm/runtime/first_update_audit.py").resolve()
    if Path(first_update_audit.__file__).resolve() != expected_provider:
        raise ValueError("Gemma audit imported a different fingerprint provider")
    model, processor = get_model_processor(
        str(args.processor_root), model_type="gemma3_vision", load_model=False, download_model=False,
        device_map="cpu", attn_impl="sdpa", torch_dtype=torch.bfloat16,
    )
    if model is not None:
        raise ValueError("processor audit constructed a model")
    options = dict(template_type="gemma3_vision", max_pixels=262144, padding_side="right",
                   padding_free=False, truncation_strategy="raise")
    full = get_template(processor, max_length=None, **options)
    capped = get_template(processor, max_length=512, **options)
    full.set_mode("train")
    capped.set_mode("train")
    declared = capped.template_meta.suffix
    if not declared or declared != full.template_meta.suffix:
        raise ValueError("native Gemma suffix metadata differs")
    suffix = capped._encode_context_list(declared, [1.] * len(declared))[0]
    if suffix != full._encode_context_list(declared, [1.] * len(declared))[0]:
        raise ValueError("native Gemma suffix tokens differ")
    configured = json.loads((args.processor_root / "generation_config.json").read_bytes())["eos_token_id"]
    eos = {configured} if type(configured) is int else set(configured)
    eos.add(processor.tokenizer.eos_token_id)
    if any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("invalid pinned Gemma EOS metadata")

    args.output.mkdir(mode=0o700)
    private_records = args.output / "native-records-private.jsonl"
    with private_records.open("x", encoding="utf-8", newline="\n") as stream:
        def progress(row):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

        audit_rows(lines, full, capped, first_update_audit.fingerprint, list(suffix), eos, check_deadline, progress)
    if torch.cuda.is_initialized():
        raise ValueError("native input audit initialized CUDA")
    receipt = {"passed": True, "unique_rows": SOURCE_ROWS, "encode_calls": 2 * SOURCE_ROWS,
               "collator_calls": SOURCE_ROWS, "private_capture_SHA256": digest(private_records.read_bytes()),
               "model_loaded": False, "optimizer_updates": 0, "actual_rank_loader_observed": False,
               "training_ready": False, "epoch": args.epoch, "deadline": args.deadline}
    (args.output / "result.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "view", "processor-root", "processor-manifest", "framework-root",
                 "framework-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
