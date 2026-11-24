"""Encode the five staged VizWiz rows with the registered Gemma CPU processor."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time


ROW_SHA256 = "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"
BINDING_SHA256 = "389c4d3d0284b153f25800ae66e78401b26b4c209ce3571f5cba2999c5fdee85"
PROCESSOR_MANIFEST_SHA256 = "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b"
FRAMEWORK_MANIFEST_SHA256 = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
HELPERS = {
    "audit_gemma_training_native.py": "890b4ebcea30972f604ffaa1d4b7f5cca2b76e919722944123fa3368fd29b1ba",
    "build_gemma_train_exposure.py": "c0c380c51ed4af09779fa270b4419ce511571137d95f1676b3dd9a4eaad715a7",
}
OBSERVER = "vlm/runtime/first_update_audit.py"


def pinned(path, expected, maximum):
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("expected bounded regular native input member")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or sha256(raw).hexdigest() != expected:
        raise ValueError("native input identity differs")
    return raw


def pinned_manifest(path, digest):
    return json.loads(pinned(path, digest, 128 * 1024))["files"]


def load_rows(data, deadline):
    binding = json.loads(pinned(data / "data-binding-private.json", BINDING_SHA256, 8192))
    content = pinned(data / "train-swift.jsonl", ROW_SHA256, 8192)
    lines = content.splitlines(keepends=True)
    images = binding["media"]
    if (data.as_posix() != binding["target_root"] or len(lines) != len(images) or len(lines) != 5
            or binding["planned_rows"] != 6 or len(binding["omitted"]) != 1
            or set(p.name for p in (data / "images").iterdir()) != set(images)):
        raise ValueError("five unique prepared images and six planned rows required")
    for ordinal, span in enumerate(lines):
        deadline()
        row = json.loads(span)
        path, = row["images"]
        name = Path(path).name
        media = images[name]
        if (row["split"] != "train" or row["group"] != binding["group_SHA256_order"][ordinal]
                or str(data / "images" / name) != path or media["target_path"] != path):
            raise ValueError("prepared native media scope differs")
        raw = pinned(Path(path), media["SHA256"], 384 * 1024)
        if len(raw) != media["bytes"]:
            raise ValueError("prepared image length differs")
    return lines, binding["omitted"][0]


def run(args):
    now = time.time()
    if not all(math.isfinite(x) for x in (args.epoch, args.deadline)) or not 0 < args.epoch <= now < args.deadline <= args.epoch + 390:
        raise ValueError("one valid original CPU budget required")
    if (os.name != "posix" or os.environ.get("SLURM_JOB_ID") != args.parent
            or not os.environ.get("SLURM_STEP_ID", "").isdecimal()
            or socket.gethostname().split(".")[0] != args.node
            or not 1 <= len(os.sched_getaffinity(0)) <= 4
            or os.environ.get("SLURM_STEP_GPUS") not in (None, "", "(null)")
            or any(os.environ.get(key) != "-1" for key in (
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            or not all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
        raise ValueError("expected the explicitly bound offline CPU step")
    stop = time.monotonic() + (args.deadline - now)

    def deadline():
        if time.monotonic() >= stop or time.time() >= args.deadline:
            raise TimeoutError("original native input budget exhausted")

    lines, omitted = load_rows(args.data, deadline)
    names = pinned_manifest(args.processor_manifest, PROCESSOR_MANIFEST_SHA256)
    if {member.name for member in args.processor.iterdir()} != set(names):
        raise ValueError("expected the exact processor metadata-only directory")
    for name, identity in names.items():
        deadline()
        if Path(name).name != name or len(pinned(args.processor / name, identity["SHA256"], identity["bytes"])) != identity["bytes"]:
            raise ValueError("processor metadata size or member differs")
    files = pinned_manifest(args.framework_manifest, FRAMEWORK_MANIFEST_SHA256)
    expected = args.framework / OBSERVER
    pinned(expected, files[OBSERVER]["SHA256"], files[OBSERVER]["bytes"])
    here = Path(__file__).resolve().parent
    for member, digest in HELPERS.items():
        pinned(here / member, digest, 32768)

    import audit_gemma_training_native as original

    if Path(original.__file__).resolve() != here / "audit_gemma_training_native.py":
        raise ValueError("unexpected original Gemma audit helper provider")
    if {name: importlib.metadata.version(name) for name in original.VERSIONS} != original.VERSIONS:
        raise ValueError("native provider versions differ")
    if "torch" in sys.modules:
        raise ValueError("expected a fresh worker before Torch import")
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("device already initialized")
    device_init_attempts = []

    def forbid_device(*_, **__):
        device_init_attempts.append(True)
        raise RuntimeError("device initialization forbidden")

    torch.cuda._lazy_init = forbid_device
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.framework.resolve()))
    from vlm.runtime import first_update_audit
    from swift.model import get_model_processor
    from swift.template import get_template

    if Path(first_update_audit.__file__).resolve() != expected.resolve():
        raise ValueError("unexpected original fingerprint provider")
    model, processor = get_model_processor(
        str(args.processor), model_type="gemma3_vision", load_model=False, download_model=False,
        device_map="cpu", attn_impl="sdpa", torch_dtype=torch.bfloat16,
    )
    if model is not None:
        raise ValueError("native input audit constructed a model")
    options = dict(template_type="gemma3_vision", max_pixels=262144, padding_side="right",
                   padding_free=False, truncation_strategy="raise")
    full, capped = [get_template(processor, max_length=n, **options) for n in (None, 512)]
    full.set_mode("train")
    capped.set_mode("train")
    suffix = capped.template_meta.suffix
    if not suffix or suffix != full.template_meta.suffix:
        raise ValueError("the native end-of-turn metadata differs")
    trailing = capped._encode_context_list(suffix, [1.] * len(suffix))[0]
    if trailing != full._encode_context_list(suffix, [1.] * len(suffix))[0]:
        raise ValueError("the native end-of-turn tokens differ")
    eos_value = json.loads((args.processor / "generation_config.json").read_bytes())["eos_token_id"]
    eos = {eos_value} if type(eos_value) is int else set(eos_value)
    eos.add(processor.tokenizer.eos_token_id)
    if any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("invalid native EOS metadata")

    args.output.mkdir()
    private = args.output / "native-rows-private.jsonl"
    outcomes = []
    with private.open("x", encoding="utf-8", newline="\n") as stream:
        for ordinal, line in enumerate(lines, 1):
            item = dict(ordinal=ordinal, state="FAILED", source_line_SHA256=sha256(line).hexdigest())

            def progress(record):
                record.update(ordinal=ordinal, state="CHECKED")
                item.update(record)

            try:
                original.audit_rows([line], full, capped, first_update_audit.fingerprint,
                                    list(trailing), eos, deadline, progress)
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                item.update(state="FAILED", error_type=type(error).__name__)
            outcomes.append(item["state"])
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
    counts = dict(planned_source_rows=6, omitted_by_supervision=1, planned_native_rows=5,
                  complete_native_rows=outcomes.count("CHECKED"), failed_native_rows=outcomes.count("FAILED"))
    receipt = dict(
        **counts, omitted_group_SHA256=omitted["group_SHA256"], model_loaded=False,
        optimizer_updates=0, training_READY=False,
        private_rows_SHA256=sha256(private.read_bytes()).hexdigest(),
        torch_lazy_device_initialization_attempts=len(device_init_attempts),
        passed=outcomes == ["CHECKED"] * 5 and not device_init_attempts and not torch.cuda.is_initialized(),
        epoch=args.epoch, deadline=args.deadline,
    )
    with (args.output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "processor", "processor-manifest", "framework", "framework-manifest", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("parent", "node"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
