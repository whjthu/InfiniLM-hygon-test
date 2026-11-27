"""Check the frozen GQA user-only inference inputs with the real target CPU processor."""

import argparse
from collections import Counter
from importlib import metadata
import inspect
import os
from pathlib import Path
import socket
import sys
import time

import internvl_gqa_val96 as experiment
from internvl_gqa_val96 import need
from internvl_gqa_val96_native import ENGINE_PINS, INPUT_SHA, VERSIONS, native_input_pin, user_input


def check_software_read(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path.endswith((".safetensors", ".pt", ".pth", ".bin", ".ckpt")):
            raise RuntimeError("Weight body access outside the CPU preflight")
        if Path(path).name.startswith("gold."):
            raise RuntimeError("Gold access inside the inference input preflight")


def check(args):
    rows = experiment.inputs(args.inputs, INPUT_SHA)
    need(os.environ.get("SLURM_JOB_ID") == args.parent and socket.gethostname().split(".")[0] == args.node
         and os.environ.get("SLURM_STEP_GPUS") in {None, "", "(null)"}, "CPU-only Slurm placement differs")
    need(time.time() < args.deadline, "Original CPU deadline expired")
    sys.addaudithook(check_software_read)
    need({package: metadata.version(package) for package in VERSIONS} == VERSIONS, "Target Python packages differ")
    for name, pin in ENGINE_PINS.items():
        experiment.read(Path("/swift-runtime/site/swift") / name, 2 << 20, pin)
    import torch
    from swift.infer_engine import TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template
    from vlm.runtime.first_update_audit import fingerprint

    need(torch.version.hip == "6.3.26113" and not torch.cuda.is_initialized(), "Qualified CPU import differs")
    def no_device(*unused, **kwargs):
        raise RuntimeError("GPU initialization in the CPU preflight")

    torch.cuda._lazy_init = no_device
    torch.set_num_threads(2)
    model, processor = get_model_processor(str(args.metadata), model_type="internvl", load_model=False,
                                           download_model=False, device_map="cpu",
                                           torch_dtype=torch.bfloat16, attn_impl="sdpa")
    need(model is None and processor.model_meta.template == "internvl_hf", "Qualified model metadata differs")
    template = get_template(processor, template_type="internvl_hf", max_length=4096, max_pixels=262144,
                            padding_side="right", padding_free=False, truncation_strategy="raise")
    template.set_mode("transformers")
    need(template.template_meta.template_type == "internvl_hf" and template.mode == "transformers",
         "Swift InternVL inference template mode differs")
    receipts, keys = [], set()
    with torch.inference_mode():
        for row in rows:
            need(time.time() < args.deadline, "Original CPU deadline expired")
            encoded = template.encode(user_input(row, args.inputs))
            collated = template.data_collator([encoded])
            keys.update(collated)
            receipts.append({"slot": row["slot"], "kind": row["type"],
                             "processed_SHA256": native_input_pin(collated, fingerprint),
                             "input_tokens": collated["input_ids"].shape[-1]})
    need(not torch.cuda.is_initialized() and len(receipts) == 96, "CPU input scope incomplete")
    result = {"phase": "CPU_NATIVE_USER_ONLY_INPUTS", "input_SHA256": INPUT_SHA, "node": args.node,
              "step": args.parent + "." + os.environ["SLURM_STEP_ID"], "rows": receipts,
              "provider": Path(inspect.getsourcefile(TransformersEngine)).name,
              "input_fields": sorted(keys), "counts": dict(Counter(row["kind"] for row in receipts)),
              "GPU_initialized": False, "model_weights_loaded": False, "gold_read": False}
    experiment.write(args.output, experiment.encode(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs", "metadata", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    check(parser.parse_args())
