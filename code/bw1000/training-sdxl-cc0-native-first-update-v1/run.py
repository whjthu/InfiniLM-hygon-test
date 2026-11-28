"""Derive and run a one-update diagnostic through the exact common SDXL backend."""

import argparse
import ast
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from sdxl_cc0_witness import exclusive, require


TRAINER = Path("examples/text_to_image/train_text_to_image_lora_sdxl.py")
BACKEND = Path("gencuda/backends/sdxl_training.py")
TRAINER_SHA = "b3e85543eaeeb42edd67dca0ffbc54391e4d8fa33b44aa1522e24b31ffc6c7bd"
BACKEND_SHA = "1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7"
INPUTS = {
    "painted_stool": ("ebc9641bdc1c3b1db45821abb81585fa92ee65da97bbcef04bf227f47b82cd73",
                      "c27d943853f15401eafa9efc17bda3b8c32e19da316e45496afd4fb30d30435b"),
    "tool_chest": ("df0806d6195e8b78ac45b5833356ae7ca656e6ce3bd2cce85434082d81cbf274",
                   "0a2f6b50e1631cb5a5dc5ac47c6c34e694275288f6176f2a01b0bdcea41a4506"),
}
WEIGHTS = {
    "unet/diffusion_pytorch_model.safetensors": (10270077736, "357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a"),
    "vae/diffusion_pytorch_model.safetensors": (334643268, "1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341"),
    "text_encoder/model.safetensors": (492265168, "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc"),
    "text_encoder_2/model.safetensors": (2778702264, "3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff"),
}
CONFIG_HASHES = {
    "unet/config.json": "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6",
    "vae/config.json": "0b331c8ac22ded5f9997a144a575c1d113d6169aff262c353f39015bd24a6264",
    "text_encoder/config.json": "39b8b2e4b1949e36969caa425b6c81c68bace99198dd9078ce05d16ad401fe7f",
    "text_encoder_2/config.json": "a892d1c3a69a7e9247a24de2bc1d5891e3109a54696e53be20093af671072c34",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "tokenizer/tokenizer_config.json": "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563",
    "tokenizer_2/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer_2/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "tokenizer_2/special_tokens_map.json": "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6",
    "tokenizer_2/tokenizer_config.json": "c9d23941f76a41cbd50eda9290f57be7828f0a7a677939e9ef181f7e12bd1bdf",
}
REQUIRED = set(WEIGHTS) | set(CONFIG_HASHES) | {"model_index.json", "scheduler/scheduler_config.json"}
CONFIG = dict(max_train_steps=25, checkpointing_steps=1, checkpoints_total_limit=3, warmup_steps=0,
              resolution=256, train_batch_size=1, learning_rate=1e-4, lora_rank=4, seed=20260911,
              gradient_checkpointing=False, dataloader_num_workers=2)


def read_pinned(path, expected, size=None, keep=False):
    require(path.is_absolute() and path.resolve(strict=True) == path, "NONCANONICAL_INPUT")
    original = path.lstat()
    identity = ("st_dev", "st_ino", "st_size")
    fields = identity + ("st_mtime_ns", "st_ctime_ns")
    def same(left, right, keys):
        return all(getattr(left, key) == getattr(right, key) for key in keys)

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                         (os.O_NOFOLLOW if sys.platform == "linux" else 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and (size is None or before.st_size == size)
                and (not keep or before.st_size <= 4 << 20)
                and same(original, before, fields if sys.platform == "linux" else identity),
                "INVALID_PINNED_INPUT_SIZE_OR_OPEN_IDENTITY")
        digest = sha256()
        raw = bytearray()
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
            if keep:
                raw.extend(block)
        require(digest.hexdigest() == expected and same(before, os.fstat(stream.fileno()), fields)
                and same(original, path.lstat(), fields), "PIN_OR_IDENTITY_CHANGED")
    return bytes(raw)


def replace_once(source, original, replacement):
    require(source.count(original) == 1, "NATIVE_SOURCE_ANCHOR_NOT_UNIQUE")
    return source.replace(original, replacement, 1)


def trainer_delta(raw):
    require(sha256(raw).hexdigest() == TRAINER_SHA, "ORIGINAL_DIFFUSERS_TRAINER_SHA")
    source = raw.decode()
    substitutions = (
        ("from pathlib import Path\n", "from pathlib import Path\nfrom sdxl_cc0_witness import Witness\n"),
        ("    # Make one log on every process with the configuration for debugging.\n",
         "    witness = Witness(args, accelerator)\n"
         "    # Make one log on every process with the configuration for debugging.\n"),
        ("                model_pred = unet(\n",
         "                witness.before_forward(\n"
         "                    batch, lambda image: train_transforms(train_crop(train_resize(image))),\n"
         "                    pixels=pixel_values, clean_latent=model_input, noise=noise,\n"
         "                    noisy_latent=noisy_model_input, timestep=timesteps, sequence=prompt_embeds,\n"
         "                    pooled=pooled_prompt_embeds, time_ids=add_time_ids,\n"
         "                )\n                model_pred = unet(\n"),
        ("                optimizer.step()\n                lr_scheduler.step()\n",
         "                witness.before_step(unet, optimizer, accelerator, loss, grad_norm, model_pred, target,\n"
         "                                    noise_scheduler.config.prediction_type)\n"
         "                optimizer.step()\n                witness.after_step(optimizer, accelerator)\n"
         "                lr_scheduler.step()\n"),
        ("            if global_step >= args.max_train_steps:\n                break\n\n        if accelerator.is_main_process:",
         "            if global_step == 1:\n                break\n\n        if global_step == 1:\n"
         "            break\n        if accelerator.is_main_process:"),
        ('        save_lora_weights(Path(args.output_dir) / "pytorch_lora_weights.safetensors")\n',
         '        save_lora_weights(Path(args.output_dir) / "pytorch_lora_weights.safetensors")\n'
         '        accelerator.end_training()\n        return\n'),
    )
    for original, replacement in substitutions:
        source = replace_once(source, original, replacement)
    ast.parse(source)
    return source.encode()


def backend_delta(raw):
    require(sha256(raw).hexdigest() == BACKEND_SHA, "ORIGINAL_COMMON_GENCUDA_BACKEND_SHA")
    source = raw.decode()
    source = replace_once(source, '    records = _records(metrics_path)\n',
                          '    records = _records(metrics_path)\n    expected = 1\n')
    source = replace_once(source, 'range(resume_step + 1, max_steps + 1)', 'range(1, 2)')
    source = replace_once(source, '"steps": max_steps,', '"steps": 1,')
    source = replace_once(source, '            "updates_per_second": 1.0 / statistics.median(seconds),\n', '')
    ast.parse(source)
    return source


def assets(root, manifest, manifest_sha):
    records = json.loads(read_pinned(manifest, manifest_sha, keep=True))
    require(type(records) is dict and set(records) == REQUIRED, "EXACT_OFFICIAL_PIPELINE_CLOSURE")
    for relative, pin in sorted(records.items()):
        require(type(pin) is dict and set(pin) == {"bytes", "SHA256"}
                and type(pin["bytes"]) is int and pin["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", pin["SHA256"]), "INVALID_ASSET_PIN")
        require(relative not in WEIGHTS or WEIGHTS[relative] == (pin["bytes"], pin["SHA256"]), "OFFICIAL_WEIGHT_PIN")
        require(relative not in CONFIG_HASHES or CONFIG_HASHES[relative] == pin["SHA256"], "OFFICIAL_CONFIG_PIN")
    actual = {str(path.relative_to(root).as_posix()) for path in root.rglob("*") if not path.is_dir() or path.is_symlink()}
    require(root.resolve(strict=True) == root and actual == REQUIRED, "EXACT_MOUNTED_PIPELINE_CLOSURE")
    for relative, pin in sorted(records.items()):
        require(os.statvfs(root / relative).f_flag & os.ST_RDONLY, "OFFICIAL_SOURCE_MUST_BE_READ_ONLY")
        read_pinned(root / relative, pin["SHA256"], pin["bytes"])


def dataset(raw, target):
    from PIL import Image
    from io import BytesIO

    require({member.name for member in raw.iterdir()} == {
        role + ext for role in INPUTS for ext in (".png", ".txt")}, "EXACT_TWO_INPUTS_ONLY")
    rows = []
    verified = {}
    for role, (image_sha, caption_sha) in sorted(INPUTS.items()):
        png = read_pinned(raw / (role + ".png"), image_sha, keep=True)
        caption = read_pinned(raw / (role + ".txt"), caption_sha, keep=True).decode().strip()
        with Image.open(BytesIO(png)) as image:
            require((image.format, image.mode, image.size) == ("PNG", "RGB", (256, 256)), "EXACT_256_RGB_PNG")
            image.load()
        require(caption and "\n" not in caption, "EXACT_ONE_CAPTION")
        rows.append(dict(file_name=role + ".png", text=caption))
        verified[role] = png
    target.mkdir(mode=0o700)
    for role, png in verified.items():
        (target / (role + ".png")).write_bytes(png)
    with (target / "metadata.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def launch(args):
    require(sys.platform == "linux" and re.fullmatch(r"[0-9a-f]{32}", args.request)
            and args.request == os.environ.get("SDXL_SINGLE_DCU_OWNER_REQUEST")
            and re.fullmatch(r"\d+\.\d+", os.environ.get("SLURM_JOB_ID", "") + "." +
                             os.environ.get("SLURM_STEP_ID", ""))
            and re.fullmatch(r"\d+", os.environ.get("SLURM_STEP_GPUS", "")), "NO_SINGLE_DCU_OWNER_ADMISSION")
    require(Path("/tmp/.sdxl-owner-request").read_text().strip() == args.request
            and Path("/tmp").stat().st_uid == os.geteuid() and stat.S_IMODE(Path("/tmp").stat().st_mode) == 0o700,
            "EXACT_PRIVATE_TMP_BIND_REQUIRED")
    home = Path("/tmp/sdxl-home")
    home.mkdir(mode=0o700)
    db = home / ".config/miopen/user-db"
    cache = home / ".cache/miopen"
    for directory in (db, cache):
        directory.mkdir(mode=0o700, parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                      HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp",
                      MIOPEN_USER_DB_PATH=str(db), MIOPEN_CUSTOM_CACHE_DIR=str(cache))
    require(args.work.is_absolute() and args.work.parent.resolve(strict=True) == args.work.parent,
            "EXACT_PRIVATE_OUTPUT_PARENT")
    args.work.mkdir(mode=0o700)
    exclusive(args.work / "intent.json", dict(request=args.request, planned_updates=1, nominal_steps=25))
    phase = "SOURCE"
    try:
        original = read_pinned(args.original_trainer, TRAINER_SHA, 64125, keep=True)
        backend_raw = read_pinned(args.framework / BACKEND, BACKEND_SHA, keep=True)
        candidate = trainer_delta(original)
        backend_source = backend_delta(backend_raw)
        phase = "ASSETS"
        assets(args.pipeline, args.pipeline_manifest, args.pipeline_manifest_sha)
        phase = "DATA"
        dataset(args.input, args.work / "input")
        overlay = args.work / "trainer" / TRAINER
        overlay.parent.mkdir(parents=True)
        overlay.write_bytes(candidate)
        overlay.with_name("sdxl_cc0_witness.py").write_bytes(Path(__file__).with_name("sdxl_cc0_witness.py").read_bytes())
        sys.path.insert(0, str(args.framework))
        backend = importlib.import_module("gencuda.backends.sdxl_training")
        require(Path(backend.__file__) == args.framework / BACKEND, "ACTIVE_COMMON_MODULE_PATH")
        namespace = dict(vars(backend))
        exec(compile(backend_source, "<reviewed-original-common-backend-delta>", "exec"), namespace)
        phase = "NATIVE_TRAINING"
        out = args.work / "native"
        result = namespace["run_sdxl_training"](args.pipeline, dict(CONFIG, train_data_dir=str(args.work / "input")),
                                               out, args.work / "trainer")
        phase = "NATIVE_SEAL"
        checkpoint = out / "checkpoint-1"
        required = {"pytorch_lora_weights.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"}
        require(required.issubset({member.name for member in checkpoint.iterdir()})
                and {member.name for member in out.glob("checkpoint-*")} == {"checkpoint-1"}, "EXACT_FIRST_NATIVE_CHECKPOINT")
        ckpt = {member.name: file_sha(member) for member in sorted(checkpoint.iterdir()) if member.name in required}
        adapter_identity = namespace["_adapter_tensor_identity"]
        require(adapter_identity(checkpoint / "pytorch_lora_weights.safetensors")
                == adapter_identity(out / "pytorch_lora_weights.safetensors"),
                "NATIVE_CHECKPOINT_ADAPTER_DIFFERS_FROM_FINAL")
        first = json.loads((out / "first-adamw.json").read_bytes())
        require(first["actual_optimizer_updates"] == result["metrics"]["steps"] == 1
                and first["skipped"] is False and first["changed_tensors"] > 0, "EXACT_DURABLE_FIRST_UPDATE")
        exclusive(args.work / "result.json", dict(request=args.request, state="NATIVE_FIRST_UPDATE_PASS",
                  actual_optimizer_updates=1, planned_updates=1, nominal_steps=25, first=first,
                  checkpoint_SHA256=ckpt, forward_SHA256=file_sha(out / "before-forward.json"),
                  original_540_qualification_changed=False, formal_domain_credit=0))
    except Exception as error:
        receipts = {name: (args.work / "native" / name).is_file() for name in
                    ("before-forward.json", "before-adamw.json", "first-adamw.json", "steps.ndjson")}
        witnessed = 1 if receipts["first-adamw.json"] else None if receipts["before-adamw.json"] else 0
        exclusive(args.work / "result.json", dict(request=args.request, state="FAILED", planned_updates=1,
                  nominal_steps=25, phase=phase, actual_optimizer_updates=witnessed,
                  independently_checkpoint_sealed=False, error_type=type(error).__name__,
                  error_SHA256=sha256(str(error).encode()).hexdigest(), evidence_present=receipts,
                  formal_domain_credit=0))
        return 2
    return 0


def file_sha(path):
    require(path.is_file() and not path.is_symlink(), "EXPECTED_REGULAR_OUTPUT")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("verify-source")
    for name in ("trainer", "backend"):
        render.add_argument("--" + name, required=True, type=Path)
    prepare = sub.add_parser("prepare-input")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify-assets")
    for name in ("pipeline", "pipeline-manifest"):
        verify.add_argument("--" + name, required=True, type=Path)
    verify.add_argument("--pipeline-manifest-sha", required=True)
    run = sub.add_parser("run")
    for name in ("original-trainer", "framework", "pipeline", "pipeline-manifest", "input", "work"):
        run.add_argument("--" + name, required=True, type=Path)
    run.add_argument("--pipeline-manifest-sha", required=True)
    run.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.command == "verify-source":
        trainer = trainer_delta(args.trainer.read_bytes())
        backend = backend_delta(args.backend.read_bytes())
        print(json.dumps(dict(trainer_delta_SHA256=sha256(trainer).hexdigest(),
                             backend_delta_SHA256=sha256(backend.encode()).hexdigest(), actual_updates=0)))
        return 0
    if args.command == "prepare-input":
        dataset(args.input, args.output)
        return 0
    if args.command == "verify-assets":
        assets(args.pipeline, args.pipeline_manifest, args.pipeline_manifest_sha)
        print(json.dumps(dict(exact_readonly_pipeline_files=len(REQUIRED), actual_optimizer_updates=0)))
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
