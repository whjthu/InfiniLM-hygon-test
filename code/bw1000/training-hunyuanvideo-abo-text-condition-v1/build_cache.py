"""Build one candidate HunyuanVideo cache with the native text encoder and sealed ABO latent."""

import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time

from native_text import CAPTION, FRAME_RECEIPT, NATIVE, TOKENIZER, digest, native_glyph_functions, pinned, require


LATENT = (610560, "854a0f39980ac592423b9774a3e198dc29eaf57082b777b9c6735f81c64b868e")
SHARDS = (
    (3900233256, 47648, "9b8af80923d85776ed1b7c9c2231dc32a66287705f91bacc6398f57970ad18a0"),
    (3864726320, 11048, "22642924f119f376df06d50cea3db7489f8250ad2cdfcfbae850ca7f08865309"),
    (3864726424, 11152, "ef4612053f44d6e6598fae0a797e12374c13099a4da426eb4d2513124fb7fd94"),
    (3864733680, 11240, "6369f8e1d1a9bf111d8f7071c775c9908c62530df6df43201d9d7e3465a31fd2"),
    (1089994880, 120, "db8d2dd53b59eb138b5232f4c9aa3242303bae47a14486465b5341590cc7d969"),
)
NATIVE_TENSORS = {
    "latents": ((1, 32, 3, 30, 53), "torch.float32"),
    "text_emb": ((1, 300, 3584), "torch.float16"),
    "text_mask": ((1, 300), "torch.int64"),
    "byt5_text_states": ((1, 256, 1472), "torch.bfloat16"),
    "byt5_text_mask": ((1, 256), "torch.int64"),
}
NODE, PARENT = "f13r3n14", "1103606"


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def exact_cpu_step(request):
    env = os.environ
    step = env.get("SLURM_STEP_ID", "")
    require(re.fullmatch(r"[0-9a-f]{32}", request) and re.fullmatch("[0-9]+", step)
            and env.get("SLURM_JOB_ID") == PARENT and socket.gethostname().split(".", 1)[0] == NODE,
            "EXACT_ASSIGNED_NODE_AND_NUMBERED_STEP")
    cgroup = Path("/proc/self/cgroup").read_text("ascii").splitlines()
    require(any({"job_" + PARENT, "step_" + step} <= set(row.split(":", 2)[-1].split("/")) for row in cgroup),
            "EXACT_NUMBERED_CGROUP")
    require(all(env.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"))
            and all(env.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and env.get("SLURM_STEP_GPUS", "") in ("", "none", "N/A", "(null)")
            and not re.search("gpu|dcu", env.get("SLURM_TRES_PER_TASK", ""), re.I), "EXACT_OFFLINE_ZERO_GRES")
    return PARENT + "." + step


def bound_shards(qwen):
    def stamp(st):
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    inputs = {}
    for i, (size, header_size, header_sha) in enumerate(SHARDS, 1):
        name = f"model-{i:05d}-of-00005.safetensors"
        path = qwen / name
        require(not path.is_symlink(), "QWEN_SHARD_SYMLINK")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            prefix = stream.read(8)
            require(before.st_size == size and len(prefix) == 8 and struct.unpack("<Q", prefix)[0] == header_size
                    and digest(stream.read(header_size)) == header_sha, "QWEN_SHARD_LAYOUT_CHANGED")
            inputs[name] = stamp(before)
            require(stamp(path.stat()) == inputs[name], "QWEN_SHARD_CHANGED_DURING_CHECK")
    return inputs


def check_tensors(tensors, torch):
    require(set(tensors) == set(NATIVE_TENSORS), "EXACT_ORIGINAL_CACHE_FIELDS")
    fields = {}
    for name, (shape, dtype) in NATIVE_TENSORS.items():
        value = tensors[name]
        require(tuple(value.shape) == shape and str(value.dtype) == dtype and value.device.type == "cpu"
                and value.is_contiguous() and bool(torch.isfinite(value).all()), "CACHE_TENSOR_SCHEMA_OR_FINITE")
        fields[name] = dict(shape=list(shape), dtype=dtype, SHA256=digest(value.view(torch.uint8).numpy().tobytes()))
    mask = tensors["text_mask"]
    require(bool(((mask == 0) | (mask == 1)).all()) and int(mask.sum()) == 19
            and float(tensors["text_emb"][mask.bool()].std()) > 0,
            "NONCONSTANT_NATIVE_TEXT_AND_19_ACTIVE_SLOTS")
    require(int(torch.count_nonzero(tensors["byt5_text_states"])) == 0
            and int(torch.count_nonzero(tensors["byt5_text_mask"])) == 0, "NATIVE_NO_GLYPH_MASK")
    require(fields["latents"]["SHA256"] == LATENT[1], "EXACT_ORIGINAL_VAE_BYTES")
    return fields


def build(args):
    step = exact_cpu_step(args.request)
    require(args.out.parent.name == args.request and args.out.is_absolute(), "PRIVATE_OUTPUT_NAME")
    args.out.mkdir(mode=0o700)
    begin = time.monotonic()
    pinned(args.render, 16384, FRAME_RECEIPT)
    latent_raw = pinned(args.latent, LATENT[0], LATENT[1], exact_size=LATENT[0])
    pins = {name: pinned(args.source / name, 1 << 20, pin) for name, pin in NATIVE.items()}
    for name, (size, pin) in TOKENIZER.items():
        pinned(args.qwen / name, size, pin, exact_size=size)
    shards = bound_shards(args.qwen)
    write_json(args.out / "start.json", dict(request=args.request, actual_step=step, epoch=time.time(),
                actual_GPU=0, prior_latent_sha256=LATENT[1], actual_optimizer_updates=0))

    sys.path.insert(0, str(args.source))
    import torch
    from hyvideo.models import text_encoders
    require(torch.cuda.device_count() == 0 and sys.byteorder == "little", "EXACT_VISIBLE_CPU_ONLY")
    torch.set_num_threads(12)
    native_file = args.source / "hyvideo/models/text_encoders/__init__.py"
    require(Path(inspect.getfile(text_encoders.TextEncoder)).resolve() == native_file.resolve(),
            "EXACT_NATIVE_ENCODER_PROVIDER")
    templates = text_encoders.PROMPT_TEMPLATE
    load_start = time.monotonic()
    encoder = text_encoders.TextEncoder(text_encoder_type="llm", tokenizer_type="llm", text_encoder_path=str(args.qwen),
        max_length=1000, text_encoder_precision="fp16", prompt_template=deepcopy(templates["li-dit-encode-image-json"]),
        prompt_template_video=deepcopy(templates["li-dit-encode-video-json"]), hidden_state_skip_layer=2,
        apply_final_norm=False, reproduce=False, device=torch.device("cpu"))
    model = encoder.model
    require(not model.training and all(p.device.type == "cpu" and p.dtype == torch.float16 and not p.requires_grad
                                        for p in model.parameters())
            and model.config.hidden_size == 3584 and model.config.num_hidden_layers == 28, "EXACT_FROZEN_QWEN_TEXT_MODEL")
    loaded = time.monotonic()
    write_json(args.out / "loaded.json", dict(request=args.request, actual_step=step,
                model_class=type(model).__name__, model_load_seconds=loaded-load_start,
                text_parameters=sum(p.numel() for p in model.parameters())))

    tokens = encoder.text2tokens(CAPTION, data_type="video", max_length=300)
    require(encoder.prompt_template_video["crop_start"] == 108 and list(tokens["input_ids"].shape) == [1, 408],
            "SAME_EXACT_VIDEO_TOKENS_PATH_AS_4271")
    with torch.inference_mode():
        encoded = encoder.encode(tokens, data_type="video", device=torch.device("cpu"))
        pipeline = "hyvideo/pipelines/hunyuan_video_pipeline.py"
        glyph = native_glyph_functions(pins[pipeline], str(args.source / pipeline), torch)
        require(glyph._extract_glyph_texts(CAPTION) == [], "ONLY_NATIVE_EMPTY_GLYPH_BRANCH")
        glyph_states, glyph_mask = glyph._process_single_byt5_prompt(CAPTION, torch.device("cpu"))
        require(glyph_states.dtype == torch.float32, "PROVEN_NATIVE_ZERO_GLYPH_DTYPE")
        values = dict(latents=torch.frombuffer(bytearray(latent_raw), dtype=torch.float32).clone().view(1, 32, 3, 30, 53),
                      text_emb=encoded.hidden_state.contiguous(), text_mask=encoded.attention_mask.contiguous(),
                      byt5_text_states=glyph_states.to(torch.bfloat16).contiguous(), byt5_text_mask=glyph_mask.contiguous())
        result_fields = check_tensors(values, torch)
    require(bound_shards(args.qwen) == shards, "QWEN_INPUT_FILES_CHANGED_DURING_NATIVE_LOAD")
    completed = time.monotonic()
    path = args.out / "sample.private.pt"
    with path.open("xb") as output:
        torch.save(values, output)
        output.flush()
        os.fsync(output.fileno())
    raw = path.read_bytes()
    require(0 < len(raw) < 4 << 20, "BOUNDED_FIVE_TENSOR_CACHE")
    checked = torch.load(path, map_location="cpu", weights_only=True)
    require(check_tensors(checked, torch) == result_fields, "IMMEDIATE_WEIGHTS_ONLY_READBACK")
    report = dict(request=args.request, actual_step=step, frame_receipt_sha256=FRAME_RECEIPT,
                  caption_sha256=digest(CAPTION.encode()), actual_GPU=0, actual_optimizer_updates=0,
                  actual_native_text_forwards=1, actual_native_empty_glyph_calls=1, native_candidate_cache_rows=1,
                  qualified_training_rows=0, formal_domain_delivery_increment=0, actual_text_device="cpu",
                  native_fields=result_fields, pt_cache=dict(bytes=len(raw), SHA256=digest(raw)),
                  torch=str(torch.__version__), model_class=type(model).__name__,
                  hidden_state_skip_layer=2, text_dtype="float16", model_load_seconds=loaded-load_start,
                  encode_and_validate_seconds=completed-loaded, wall_seconds=time.monotonic()-begin)
    write_json(args.out / "terminal.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "qwen", "render", "latent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request", required=True)
    print(json.dumps(build(parser.parse_args()), sort_keys=True))
