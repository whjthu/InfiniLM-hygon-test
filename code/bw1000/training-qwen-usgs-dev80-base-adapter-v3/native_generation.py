"""Generate the frozen USGS development set with an original Qwen model or checkpoint-2 LoRA."""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import tempfile

from usgs_dev_input import bound_inputs
from torch_provider import write_and_validate
from check_qwen_native_inputs import MAX_PIXELS, SOURCE_VERSIONS, TEMPLATE_SHA256, read_pinned, require


ENGINE_PINS = {
    "infer_engine/transformers_engine.py": "b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb",
    "infer_engine/infer_engine.py": "857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969",
    "infer_engine/utils.py": "5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644",
    "infer_engine/protocol.py": "f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e",
    "template/templates/qwen.py": TEMPLATE_SHA256,
}
MAX_NEW_TOKENS = 64
ADAPTER_NAME = "usgs_step2"


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def run_arm(engine, requests, config, adapter, output, arm, deadline):
    start = time.monotonic()
    counts = {"stop": 0, "length": 0, "empty": 0}
    with (output / (arm + "-raw.jsonl")).open("x", encoding="utf8") as raw, \
            (output / (arm + ".jsonl")).open("x", encoding="utf8") as score:
        for identity, request in requests:
            require(time.time() < deadline, "generation budget expired")
            result, = engine.infer([request], config, use_tqdm=False, adapter_request=adapter)
            choice, = result.choices
            reason = choice.finish_reason
            answer = choice.message.content
            require(reason in ("stop", "length") and (answer is None or isinstance(answer, str)),
                    "unexpected native response")
            counts[reason if answer or reason == "length" else "empty"] += 1
            write_row(raw, {"id": identity, "answer": answer, "finish_reason": reason,
                            "prompt_tokens": result.usage.prompt_tokens,
                            "completion_tokens": result.usage.completion_tokens})
            write_row(score, {"id": identity, "answer": answer if reason == "stop" and answer else None})
            with (output / "progress.jsonl").open("a", encoding="utf8") as progress:
                write_row(progress, dict(event="completed", arm=arm, id=identity, finish_reason=reason))
    return dict(rows=sum(counts.values()), outcome_counts=counts,
                wall_seconds=time.monotonic() - start,
                scoring_file_SHA256=hashlib.sha256((output / (arm + ".jsonl")).read_bytes()).hexdigest(),
                raw_file_SHA256=hashlib.sha256((output / (arm + "-raw.jsonl")).read_bytes()).hexdigest())


def generate(args):
    root = args.inputs.resolve(strict=True)
    requests, _ = bound_inputs(root, args.inventory)
    versions = {name: importlib.metadata.version(name) for name in SOURCE_VERSIONS}
    require(versions == SOURCE_VERSIONS, "Swift inference environment changed")
    for name, sha in ENGINE_PINS.items():
        read_pinned(Path("/swift-runtime/site/swift") / name, sha)

    args.output.mkdir(mode=0o700)
    progress_path = args.output / "progress.jsonl"

    def milestone(event):
        with progress_path.open("a", encoding="utf8") as progress:
            write_row(progress, dict(event=event, arm=args.arm, monotonic_ns=time.monotonic_ns()))

    with progress_path.open("x", encoding="utf8") as progress:
        write_row(progress, dict(event="plan", arm=args.arm, planned_ids=[r["id"] for r in requests], planned_rows=80,
                                 max_new_tokens=64, deadline=args.deadline))
    temp_path = Path(tempfile.gettempdir())
    stats = os.statvfs(temp_path)
    have_private_mount = temp_path.stat().st_dev == args.output.parent.stat().st_dev
    require(have_private_mount and stats.f_blocks * stats.f_frsize > 64 * (1 << 20),
            "Apptainer private scratch mount was not applied")
    with (args.output / "scratch-mount.json").open("x", encoding="utf8") as stream:
        json.dump(dict(same_device_as_run=have_private_mount, total_bytes=stats.f_blocks * stats.f_frsize,
                       available_bytes=stats.f_bavail * stats.f_frsize), stream)
    milestone("private_scratch_mount_bound")
    milestone("before_torch_import")
    import torch
    torch_provider = write_and_validate(torch, args.output/"torch-provider.json")
    require(torch_provider["SHA256"] == "9edf6b378be77655e0641130e18edfc32d4415d9355cf027bef3eba54aeeecc9",
            "Torch bytes differ from three corrected CPU entries")
    milestone("torch_provider_bound")
    from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template

    milestone("swift_native_imports_bound")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected one qualified HIP device")
    milestone("swift_native_apis_and_device_bound")
    def pending():
        return [(row["id"], InferRequest(messages=deepcopy(row["messages"]), images=[str(root / row["images"][0])]))
                for row in requests]
    milestone("before_model_materialization")
    load_start = time.monotonic()
    model, processor = get_model_processor(str(args.model), model_type="qwen3_vl", load_model=True,
                                           download_model=False, device_map={"": 0},
                                           torch_dtype=torch.bfloat16, attn_impl="sdpa")
    require(model.device == torch.device("cuda:0") and model.dtype == torch.bfloat16
            and model.config._attn_implementation == "sdpa" and not hasattr(model, "peft_config"),
            "base model binding differs")
    milestone("model_materialized")
    template = get_template(processor, template_type="qwen3_vl", max_length=4096,
                            max_pixels=MAX_PIXELS, padding_side="right",
                            padding_free=False, truncation_strategy="raise")
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    config = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=0, stream=False, num_beams=1, n=1)
    effective = engine._prepare_generation_config(config)
    require(not effective.do_sample and effective.max_new_tokens == MAX_NEW_TOKENS
            and effective.num_beams == 1 and effective.eos_token_id is not None, "not pinned greedy decoding")
    load_seconds = time.monotonic() - load_start
    require(args.arm in {"base", "domain"}, "unknown frozen arm")
    adapter = None if args.arm == "base" else AdapterRequest(name=ADAPTER_NAME, path=str(args.adapter))
    milestone("before_first_generation")
    generated = run_arm(engine, pending(), config, adapter, args.output, args.arm, args.deadline)
    if args.arm == "base":
        require(engine.model is model and not hasattr(model, "peft_config"), "base arm acquired an adapter")
    else:
        require(engine.model is not model and engine.model.get_base_model() is model
                and set(engine.model.peft_config) == {ADAPTER_NAME}, "fresh arm did not bind its sole LoRA")
    receipt = dict(actual_DCU=1, arm=args.arm, load_seconds=load_seconds, versions=versions,
                   precision="BF16", attention="sdpa", max_new_tokens=MAX_NEW_TOKENS, greedy=True,
                   batch_size=1, fresh_model_process=True, planned_rows_per_arm=80, generated=generated,
                   torch_provider=torch_provider,
                   device_name=torch.cuda.get_device_properties(0).name,
                   device_total_memory=torch.cuda.get_device_properties(0).total_memory)
    with (args.output / "generation-outcome.json").open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "inventory", "model", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--arm", required=True, choices=("base", "domain"))
    parser.add_argument("--deadline", required=True, type=float)
    generate(parser.parse_args())
