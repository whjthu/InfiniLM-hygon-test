"""Exercise the installed Swift Qwen3-VL image path on original COCO64."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import time

BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
OUT = Path("/evidence")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    epoch = float(os.environ["QWEN_CPU_EPOCH"])
    deadline = float(os.environ["QWEN_CPU_DEADLINE"])
    assert deadline == epoch + 230 and epoch <= time.time() < deadline
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    node = socket.gethostname().split(".")[0]
    assert node == "a01r4n16" and os.environ["SLURM_JOB_ID"] == "1087656"
    assert all(os.environ[x] == "-1" for x in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
    import torch
    assert str(torch.__version__) == "2.10.0" and torch.version.hip == "6.3.26113"
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU-only qualification")
    torch.cuda._lazy_init = forbidden
    import qwen_vl_utils
    assert metadata.version("qwen-vl-utils") == "0.0.14"
    assert Path(qwen_vl_utils.__file__).resolve() == Path("/vision/qwen_vl_utils/__init__.py")
    from swift.model import get_model_processor
    from swift.model.models.qwen import Qwen3VLLoader
    from swift.template import get_template
    from swift.template.templates.qwen import Qwen3VLTemplate
    from vlm.runtime.first_update_audit import fingerprint
    assert hash_file(Path(inspect.getsourcefile(Qwen3VLLoader))) == "d500798143d15eae0a2d202312900016402b327e12953715618b5ce9f0842131"
    assert hash_file(Path(inspect.getsourcefile(Qwen3VLTemplate))) == "d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3"
    root = BASE/"training-qwen-original-public-model-v1/model"
    model, processor = get_model_processor(str(root), model_type="qwen3_vl",
        load_model=False, download_model=False, device_map="cpu", attn_impl="sdpa", torch_dtype=torch.bfloat16)
    assert model is None
    template = get_template(processor, template_type="qwen3_vl", max_length=4096, max_pixels=262144,
                            padding_side="right", padding_free=False)
    template.set_mode("train")
    assert isinstance(template,Qwen3VLTemplate)
    # Swift's lazy dummy helper omits the explicitly selected model type.
    # Apply the same meta-only loader with the original recipe's type.
    with torch.device("meta"):
        dummy, _ = get_model_processor(str(root), model_type="qwen3_vl",
            return_dummy_model=True, download_model=False)
    assert dummy is not None and all(p.is_meta for p in dummy.parameters())
    template.dummy_model = dummy
    dataset = BASE/"training-vlm-original-assets-v1/vlm_report_coco64.hip.jsonl"
    raw = dataset.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "1c9aaddad5b681b50a6ddfa79c08e30e4820ee6ffcdc81bd473d0c2e84b35a0f"
    rows = [json.loads(x) for x in raw.splitlines()]
    assert len(rows) == 64 and all(r.get("images") and not r.get("videos") for r in rows)
    lengths = []
    with (OUT/"rows.jsonl").open("x") as stream:
        for ordinal, row in enumerate(rows, 1):
            encoded = template.encode(row)
            collated = template.data_collator([encoded])
            pixels = collated["pixel_values"]
            ids, labels = collated["input_ids"], collated["labels"]
            assert ids.shape == labels.shape and ids.shape[0] == 1 and 0 < ids.shape[1] <= 4096
            assert pixels.numel() > 0 and torch.isfinite(pixels).all() and (labels != -100).any()
            lengths.append(ids.shape[1])
            print(json.dumps(dict(ordinal=ordinal, length=lengths[-1],
                input_ids_SHA256=fingerprint(ids)["SHA256"], labels_SHA256=fingerprint(labels)["SHA256"],
                pixels_SHA256=fingerprint(pixels)["SHA256"])), file=stream, flush=True)
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, step="1087656."+os.environ["SLURM_STEP_ID"], node=node,
        rows_verified=len(lengths), min_length=min(lengths), max_length=max(lengths),
        rows_SHA256=hash_file(OUT/"rows.jsonl"), provider=metadata.version("qwen-vl-utils"),
        actual_model_loaded=False, GPU_cards=0, training_updates=0)
    with (OUT/"compute.json").open("x") as stream:
        json.dump(result,stream,indent=2)


if __name__ == "__main__":
    main()
