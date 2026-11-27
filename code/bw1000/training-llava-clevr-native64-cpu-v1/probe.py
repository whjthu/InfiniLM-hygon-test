"""Bind CLEVR train-only media and verify real OneVision CPU supervision."""

import argparse
import copy
import hashlib
import inspect
import io
import json
import os
import re
import signal
import stat
import sys
from pathlib import Path


ROWS = 64
TRAIN = "1c88275d408be915584d2b25a5aac4e3e3b7bf97b47cb380cbff70c564932b05"
STAGE = "792698f8beb99faf4d952d323e1227dc1d3c9d07d70dae64da4122d14ec61d3b"
MODEL = "9d3a56a41f66e07a8b45209ae5dba4325b0265e3908a39767112a4dd0ddfd258"
VISION = "75ee6a875b49a53c749cf7091de9c070c3276ef488a1af6ba3d2dab0571d551b"
FRAMEWORK = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
IMAGE = re.compile(r"images/CLEVR_train_[0-9]{6}\.png")
PROMPT = (
    "<image>List the visible objects by color and shape with their counts. "
    "Only include objects that are present."
)


class Gate(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise Gate(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def persist(path, value):
    with path.open("xb") as stream:
        stream.write(wire(value))
        stream.flush()
        os.fsync(stream.fileno())


def read(path, limit, expected=None):
    require(path.resolve(strict=True) == path.absolute(), "CANONICAL_INPUT")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "BOUNDED_REGULAR_INPUT")
        raw = stream.read(limit + 1)
        after = path.stat()
        names = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        path_names = names if sys.platform != "win32" else names[:-1]
        require(all(getattr(os.fstat(stream.fileno()), key) == getattr(before, key) for key in names)
                and all(getattr(after, key) == getattr(before, key) for key in path_names) and
                len(raw) == before.st_size, "CHANGED_INPUT_FILE")
    require(expected is None or sha(raw) == expected, "INPUT_SHA256")
    return raw


def source_rows(data):
    raw = read(data / "train.jsonl", 20635, TRAIN)
    lines = raw.splitlines()
    require(len(raw) == 20635 and len(lines) == ROWS, "EXACT_64_TRAIN_ROWS")
    rows = []
    images = set()
    for line in lines:
        row = json.loads(line)
        require(type(row) is dict and set(row) == {"messages", "images"} and
                type(row["images"]) is list and len(row["images"]) == 1 and
                type(row["images"][0]) is str and IMAGE.fullmatch(row["images"][0]) and
                row["images"][0] not in images, "TRAIN_IMAGE_IDENTITY")
        messages = row["messages"]
        require(type(messages) is list and len(messages) == 2 and
                messages[0] == {"role": "user", "content": PROMPT} and
                type(messages[1]) is dict and set(messages[1]) == {"role", "content"} and
                messages[1]["role"] == "assistant" and type(messages[1]["content"]) is str and
                bool(messages[1]["content"].strip()), "TRAIN_ASSISTANT_SCHEMA")
        images.add(row["images"][0])
        rows.append((row, sha(line)))
    return rows


def image_pin(data, relative):
    from PIL import Image

    raw = read(data / relative, 2 * 1024 * 1024)
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG" and image.size == (480, 320) and
                image.mode in {"RGB", "RGBA"}, "EXACT_CLEVR_PNG")
        image.load()
    return {"path": relative, "bytes": len(raw), "sha256": sha(raw)}


def bind(data, output):
    staged = json.loads(read(data / "RECEIPT.json", 4096, STAGE))
    require(staged["counts"] == {"train": 64, "dev": 16, "calibration": 16} and
            staged["sha256"]["train.jsonl"] == TRAIN, "ORIGINAL_STAGE_IDENTITY")
    pins = [
        dict(image_pin(data, row["images"][0]), ordinal=ordinal, row_sha256=row_sha)
        for ordinal, (row, row_sha) in enumerate(source_rows(data), 1)
    ]
    require(len({pin["sha256"] for pin in pins}) == ROWS, "DUPLICATE_IMAGE_CONTENT")
    persist(output, {"train_sha256": TRAIN, "stage_receipt_sha256": STAGE, "rows": pins})


def check_binding(rows, binding):
    require(set(binding) == {"train_sha256", "stage_receipt_sha256", "rows"} and
            binding["train_sha256"] == TRAIN and binding["stage_receipt_sha256"] == STAGE and
            len(binding["rows"]) == ROWS, "TRAIN_BINDING_HEADER")
    images = set()
    for ordinal, ((row, row_sha), pin) in enumerate(zip(rows, binding["rows"]), 1):
        require(set(pin) == {"ordinal", "row_sha256", "path", "bytes", "sha256"} and
                pin["ordinal"] == ordinal and pin["row_sha256"] == row_sha and
                pin["path"] == row["images"][0] and type(pin["bytes"]) is int and
                24 <= pin["bytes"] <= 2 * 1024 * 1024 and
                re.fullmatch(r"[a-f0-9]{64}", pin["sha256"]) and
                pin["sha256"] not in images, "TRAIN_BINDING_ROW")
        images.add(pin["sha256"])


def supervision(ids, labels, mask, eos_ids, image_id):
    require(1 < len(ids) <= 4096 and len(ids) == len(labels) == len(mask) and
            all(value == 1 for value in mask), "FULL_UNPADDED_INPUT")
    positions = [i for i, label in enumerate(labels) if label != -100]
    require(positions and positions[0] > 0 and
            all(labels[i] == ids[i] and ids[i] != image_id for i in positions),
            "NATIVE_SHIFTED_ASSISTANT_MASK")
    visual = ids.count(image_id)
    values = [ids[i] for i in positions]
    eos = sum(token in eos_ids for token in values)
    require(visual > 0 and eos > 0 and len(values) > eos, "NATIVE_IMAGE_EOS_AND_RESPONSE")
    return values, {"sequence_length": len(ids), "shifted_supervised_tokens": len(values),
                    "supervised_eos": eos, "image_tokens": visual}


def checked_pixels(pixels, torch):
    tensors = pixels if isinstance(pixels, (list, tuple)) else [pixels]
    require(tensors and all(torch.is_tensor(t) and t.device.type == "cpu" and
                            t.is_floating_point() and t.numel() and
                            bool(torch.isfinite(t).all()) for t in tensors), "CPU_FINITE_PIXELS")


def identity(function):
    provider = Path(inspect.getsourcefile(inspect.unwrap(function))).resolve()
    require(provider.is_relative_to("/swift-runtime/site/swift"), "NATIVE_SWIFT_PROVIDER_PATH")
    return {"provider_sha256": sha(read(provider, 2 * 1024 * 1024)),
            "provider_file": provider.relative_to("/swift-runtime/site").as_posix()}


def verified_files(manifest, expected, directory, count, nested=False):
    pins = json.loads(read(manifest, 128 * 1024, expected))
    if nested:
        pins = pins["files"]
    require(len(pins) == count, "ORIGINAL_RUNTIME_FILE_COUNT")
    for name, pin in sorted(pins.items()):
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts, "RUNTIME_MEMBER_PATH")
        raw = read(directory / member, pin["bytes"], pin["SHA256"])
        require(len(raw) == pin["bytes"], "ORIGINAL_RUNTIME_FILE_SIZE")
        yield name, raw


def admission(args):
    require(sys.platform == "linux" and re.fullmatch(r"[0-9]+", args.parent) and
            re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", args.node) and
            os.environ.get("SLURM_JOB_ID") == args.parent and
            os.uname().nodename.split(".")[0] == args.node, "CPU_JOB_AND_NODE")
    step = os.environ.get("SLURM_STEP_ID", "")
    cgroup = Path("/proc/self/cgroup").read_text()
    require(step.isdigit() and re.search(
        rf"/job_{args.parent}(?:\.scope)?/step_{step}(?:\.scope)?(?:/|$)", cgroup,
        re.MULTILINE), "ACTUAL_NUMBERED_STEP_CGROUP")
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)") and
            os.environ.get("SLURM_GPUS_ON_NODE") in (None, "", "0", "(null)") and
            all(os.environ.get(key) == "-1" for key in
                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")) and
            all(os.environ.get(key) == "1" for key in
                ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")),
            "CPU_OFFLINE_ENVIRONMENT")
    return args.parent + "." + step


def no_device_fds():
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        require(not (target == "/dev/kfd" or target.startswith("/dev/dri/")),
                "UNEXPECTED_ACCELERATOR_DESCRIPTOR")


def run(args, result):
    phase = "admission"
    try:
        result["step"] = admission(args)
        no_device_fds()
        phase = "input_binding"
        rows = source_rows(args.data)
        binding = json.loads(read(args.binding, 128 * 1024, args.binding_sha256))
        check_binding(rows, binding)
        allowed_data = {str(args.data / pin["path"]) for pin in binding["rows"]}
        phase = "runtime_metadata"
        view = args.output.parent / "LLaVA-OneVision-1.5-8B-Instruct"
        view.mkdir(mode=0o700)
        for name, raw in verified_files(args.model_pins, MODEL, args.metadata, 14, nested=True):
            require(Path(name).name == name, "PROCESSOR_METADATA_MEMBER")
            with (view / name).open("xb") as stream:
                stream.write(raw)
        for _ in verified_files(args.vision_pins, VISION, Path("/vision"), 7):
            pass
        for _ in verified_files(args.framework_pins, FRAMEWORK, args.framework, 393, nested=True):
            pass

        def audit(event, detail):
            if event in {"socket.connect", "socket.getaddrinfo"}:
                raise Gate("FORBIDDEN_NETWORK")
            if event == "open" and isinstance(detail[0], (str, bytes)):
                path = os.fsdecode(detail[0])
                name = Path(path).name
                require(name not in {"dev.jsonl", "calibration.jsonl", "references.private.jsonl"},
                        "FORBIDDEN_HELDOUT_FILE")
                require(not name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")),
                        "FORBIDDEN_WEIGHT_READ")
                require(not (path == "/dev/kfd" or path.startswith("/dev/dri/")),
                        "FORBIDDEN_ACCELERATOR_OPEN")
                if path.startswith(str(args.data) + "/") or name.startswith("CLEVR_"):
                    require(path in allowed_data, "FORBIDDEN_NON_TRAIN_INPUT")

        sys.addaudithook(audit)
        phase = "native_processor"
        sys.path.insert(0, str(args.framework))
        import torch
        import qwen_vl_utils
        from importlib.metadata import version
        from vlm.plugins import llava_onevision
        from swift.model import get_model_processor
        from swift.template import get_template

        require(torch.cuda.__dict__.get("_initialized") is False and
                Path(qwen_vl_utils.__file__).resolve() == Path("/vision/qwen_vl_utils/__init__.py") and
                version("qwen-vl-utils") == "0.0.14" and Path(llava_onevision.__file__).resolve()
                == args.framework / "vlm/plugins/llava_onevision.py", "ORIGINAL_NATIVE_PROVIDERS")

        def no_gpu(*_, **__):
            raise Gate("FORBIDDEN_GPU_INIT")

        torch.cuda._lazy_init = no_gpu
        torch.set_num_threads(2)
        result["providers"] = {
            "processor": identity(get_model_processor), "template": identity(get_template),
        }
        empty, processor = get_model_processor(str(view), model_type="llava_onevision1_5",
                                               load_model=False, download_model=False, device_map="cpu",
                                               attn_impl="eager", torch_dtype=torch.bfloat16)
        require(empty is None and processor.model_meta.template == "llava_onevision1_5",
                "T04_PROCESSOR_WITHOUT_MODEL")
        template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
                                padding_side="right", padding_free=False)
        template.set_mode("train")
        require(template.template_meta.template_type == "llava_onevision1_5" and
                template.sequence_parallel_size == 1 and not template.padding_free, "T04_NATIVE_TEMPLATE")
        tokenizer = processor.tokenizer
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        model_config = json.loads(read(view / "config.json", 16384))
        image_id = model_config["image_token_id"]
        require(eos_ids and all(type(token) is int for token in eos_ids) and
                type(image_id) is int, "EXACT_EOS_AND_IMAGE_ID")
        result["providers"].update(encode=identity(template.encode),
                                   collator=identity(template.data_collator))
        phase = "native_encode_and_collate"
        for (row, _), pin in zip(rows, binding["rows"]):
            record = {"ordinal": pin["ordinal"], "row_sha256": pin["row_sha256"], "status": "failed"}
            result["rows"].append(record)
            try:
                relative = pin["path"]
                image = {key: pin[key] for key in ("path", "bytes", "sha256")}
                require(image_pin(args.data, relative) == image, "BOUND_TRAIN_IMAGE")
                native = copy.deepcopy(row)
                native["images"] = [str(args.data / relative)]
                encoded = template.encode(native)
                ids_before = list(encoded["input_ids"])
                labels_before = list(encoded["labels"])
                collated = template.data_collator([encoded])
                tensors = [collated[key] for key in ("input_ids", "labels", "attention_mask")]
                require(all(torch.is_tensor(t) and t.device.type == "cpu" and
                            tuple(t.shape) == (1, len(ids_before)) and
                            not t.is_floating_point() and not t.is_complex()
                            for t in tensors), "CPU_COLLATOR_BATCH_SHAPE")
                ids, labels, mask = (tensor[0].tolist() for tensor in tensors)
                require(ids == ids_before and labels == labels_before and
                        "logits_to_keep" not in collated, "EXACT_ENCODED_IDS_LABELS")
                checked_pixels(collated["pixel_values"], torch)
                values, metrics = supervision(ids, labels, mask, eos_ids, image_id)
                observed = tokenizer.decode(values, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                require(observed.strip() == row["messages"][1]["content"], "EXACT_SUPERVISED_CAPTION")
                require(image_pin(args.data, relative) == image, "BOUND_TRAIN_IMAGE_AFTER_NATIVE")
                record.update(metrics, status="passed", image_sha256=pin["sha256"])
            except TimeoutError:
                raise
            except Exception as error:
                record["error"] = failure(error)
        require(torch.cuda.__dict__.get("_initialized") is False, "CPU_NO_GPU_INITIALIZATION")
        no_device_fds()
    except Exception as error:
        result["failure"] = {"phase": phase, **failure(error)}


def failure(error):
    return {"type": type(error).__name__, "errno": getattr(error, "errno", None),
            "code": str(error) if type(error) is Gate else None,
            "message_sha256": sha(str(error).encode("utf-8", errors="replace"))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("bind")
    native = sub.add_parser("native")
    for parser in (build, native):
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
    for flag in ("binding", "metadata", "framework", "model-pins", "vision-pins", "framework-pins"):
        native.add_argument("--" + flag, type=Path, required=True)
    for flag in ("binding-sha256", "parent", "node"):
        native.add_argument("--" + flag, required=True)
    native.add_argument("--seconds", type=int, default=210)
    args = cli.parse_args()
    if args.mode == "bind":
        bind(args.data.absolute(), args.output)
        return 0
    require(re.fullmatch(r"[a-f0-9]{64}", args.binding_sha256) and
            1 <= args.seconds <= 210 and all(path.is_absolute() for path in (
                args.data, args.binding, args.metadata, args.framework, args.output,
                args.model_pins, args.vision_pins, args.framework_pins)), "EXACT_ABSOLUTE_SCOPE")
    require(args.output.parent.resolve(strict=True) == args.output.parent and
            args.output.parent.stat().st_uid == os.geteuid() and
            stat.S_IMODE(args.output.parent.stat().st_mode) == 0o700, "PRIVATE_OUTPUT_ROOT")
    # Reserve this result path before any native work; a surviving empty file is an incomplete attempt.
    with args.output.open("xb") as stream:
        result = {"planned_rows": ROWS, "rows": [], "scope": "CLEVR_TRAIN_ONLY_NATIVE_CPU_SUPERVISION",
                  "binding_sha256": args.binding_sha256, "optimizer_updates": 0,
                  "training_forwards": 0, "gpu_qualified": False}

        def timeout(*_):
            raise TimeoutError("NATIVE_WORK_BUDGET")

        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, args.seconds)
        try:
            run(args, result)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        observed = len(result["rows"])
        passed = sum(row["status"] == "passed" for row in result["rows"])
        result.update(pass_rows=passed, failed_rows=observed - passed, missing_rows=ROWS - observed,
                      passed=passed == ROWS and "failure" not in result)
        stream.write(wire(result))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: result[key] for key in ("planned_rows", "pass_rows", "failed_rows",
                                                  "missing_rows", "passed")}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
