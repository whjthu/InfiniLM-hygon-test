"""Run the OneVision generation boundary on one already-exposed CLEVR image."""

import argparse
import importlib.util
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

from generation_boundary import BoundaryTrace, error_evidence


FROZEN = {
    "native.py": "a0af5ad1f92c79f369a710032e78bcfceb47e5f7925e902a7819cb2422f20517",
    "contract.py": "62bd2a9554645ab615320e227b69abb09a93804a5fe390987dea56a3f2947b47",
}
EXPOSED_IMAGE = "CLEVR_train_014344.png"
EXPOSED_BYTES = 193647
EXPOSED_SHA = "e0e9e1c96951283e0ce9db9bf220499301a6d7b147dd1221793157b8b8f0912b"
EXPOSED_BINDING = "d00032b2f0e5eced239d87b2d3f3164c43f369bc3796b0a1180913f3a8950b8b"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def frozen_native(folder):
    from hashlib import sha256

    need(folder.is_absolute() and folder.resolve(strict=True) == folder, "CANONICAL_FROZEN_NATIVE_V3")
    for filename, digest in FROZEN.items():
        path = folder / filename
        with path.open("rb") as stream:
            source = stream.read(64 * 1024 + 1)
        need(path.resolve(strict=True) == path and len(source) <= 64 * 1024
             and sha256(source).hexdigest() == digest, "ACTUAL_FROZEN_NATIVE_V3")
    need("contract" not in sys.modules, "UNSHADOWED_FROZEN_CONTRACT")
    sys.path.insert(0, str(folder))
    try:
        path = folder / "native.py"
        spec = importlib.util.spec_from_file_location("qualified_original_llava_native_v3", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    need(Path(sys.modules["contract"].__file__) == folder / "contract.py", "ACTUAL_FROZEN_CONTRACT_PROVIDER")
    return module


def verify_diagnostic_image(args, native):
    from PIL import Image
    from contract import PROMPT

    need(args.diagnostic_image.name == EXPOSED_IMAGE and args.diagnostic_image_sha == EXPOSED_SHA,
         "EXACT_PREVIOUSLY_EXPOSED_CLEVR_TRAIN_IMAGE")
    data = native.read(args.diagnostic_image, EXPOSED_BYTES, EXPOSED_SHA, EXPOSED_BYTES)
    with Image.open(io.BytesIO(data)) as image:
        need(image.format == "PNG" and image.size == (480, 320) and image.mode in ("RGB", "RGBA"),
             "EXACT_EXPOSED_IMAGE_GEOMETRY")
        image.load()
    return [dict(role="user", content=PROMPT)]


def cpu_invocation(args, native, trace, facts):
    for path in (args.native_source, args.diagnostic_image, args.model, args.model_receipt,
                 native.HELPERS, native.FRAMEWORK, native.SWIFT, Path("/vision")):
        native.readonly(path)
    messages = verify_diagnostic_image(args, native)
    cache = Path("/tmp") / ("llava-zero-gres-modules-" + args.request)
    cache.mkdir(mode=0o700)
    os.environ["HF_MODULES_CACHE"] = str(cache)
    sys.path[:0] = [str(native.HELPERS), str(native.FRAMEWORK), str(native.SWIFT), "/vision"]
    need({name: metadata.version(name) for name in native.VERSIONS} == native.VERSIONS, "ACTUAL_RUNTIME_VERSIONS")
    for name, pin in native.HELPER_PINS.items():
        native.read(native.HELPERS / name, 64 << 10, pin)
    for name, pin in native.ENGINE.items():
        native.read(native.SWIFT / "swift/infer_engine" / name, 2 << 20, pin)
    import probe
    import llava_recovered_native_cpu as recovered
    import llava_target_eager_import as eager
    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template
    from transformers.utils import hub
    from vlm.plugins import llava_onevision

    need(all(Path(mod.__file__) == native.HELPERS / (mod.__name__ + ".py") for mod in (probe, recovered, eager)),
         "ACTUAL_LLAVA_HELPER_PATH")
    for _ in probe.verified_files(native.HELPERS / "FRAMEWORK_FILES.json", probe.FRAMEWORK, native.FRAMEWORK, 393, nested=True):
        pass
    for filename in native.ENGINE:
        suffix = "" if filename == "__init__.py" else "." + filename[:-3]
        module = importlib.import_module("swift.infer_engine" + suffix)
        need(Path(module.__file__) == native.SWIFT / "swift/infer_engine" / filename, "ACTUAL_INFERENCE_PROVIDER")
    need(Path(hub.HF_MODULES_CACHE) == cache and not torch.cuda.is_initialized() and
         Path(llava_onevision.__file__) == native.FRAMEWORK / "vlm/plugins/llava_onevision.py", "CPU_OFFLINE_PROVIDER")
    native.base_files(args.model, args.model_receipt, recovered)
    torch.set_num_threads(6)
    original_lazy_init = torch.cuda._lazy_init

    def forbid_gpu(*_args, **_kwargs):
        raise RuntimeError("ZERO_GRES_CPU_INITIALIZATION_GUARD")

    torch.cuda._lazy_init = forbid_gpu
    try:
        native_class = eager.preimport(args.model, llava_onevision, recovered)
        def cpu_loader(path, **options):
            need(options["device_map"] == {"": 0}, "EXACT_V3_MODEL_LOAD_ARGV")
            return get_model_processor(path, **dict(options, device_map="cpu"))
        current = {}
        model, processor = native.load_original_model(recovered, cpu_loader, args.model, torch, current)
        facts["model_loads"] += 1
        facts["finalizer"] = current["base_diagnostic"]["checkpoint_finalizer"]
        need(type(model) is native_class and all(current["base_diagnostic"]["predicates"].values())
             and not hasattr(model, "peft_config")
             and len(tuple(model.parameters())) == 698
             and {p.device.type for p in model.parameters()} == {"cpu"}
             and {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.bfloat16}
             and processor.model_meta.template == "llava_onevision1_5" and model.config._attn_implementation == "eager"
             and getattr(model, "_vlm_onevision_rope_initialized", False), "EXACT_CPU_BASE_FINALIZER_AND_CONFIGURATION")
        llava_onevision._verify_loaded_native_eager_classes(model, native_class.__module__)
        model.eval().requires_grad_(False)
        template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
                                padding_side="right", padding_free=False, truncation_strategy="raise")
        need(template.template_meta.template_type == "llava_onevision1_5", "EXACT_ONEVISION_TEMPLATE")
        engine = TransformersEngine(model, template=template, max_batch_size=1)
        cfg = RequestConfig(max_tokens=native.CAP, temperature=0, num_beams=1, n=1, stream=False, return_details=True, stop=[])
        policy = engine._prepare_generation_config(cfg)
        need(not policy.do_sample and policy.num_beams == 1 and policy.max_new_tokens == native.CAP,
             "EXACT_ORIGINAL_GREEDY_CONFIG")
        with trace.attach_template(template), trace.attach_model(engine.model):
            need(all(value["file_SHA256"] is not None for value in trace.providers.values()),
                 "ALL_FIVE_REAL_CALLABLE_SOURCE_FILES")
            response, = engine.infer([InferRequest(messages=messages, images=[str(args.diagnostic_image)])],
                                     cfg, use_tqdm=False, adapter_request=None)
            if isinstance(response, BaseException):
                raise response
            facts["unexpected_unstopped_response"] = True
            raise ValueError("CPU_WITNESS_DID_NOT_STOP_BEFORE_RESPONSE")
    finally:
        facts["gpu_initialized"] = torch.cuda.is_initialized()
        torch.cuda._lazy_init = original_lazy_init
        need(facts["gpu_initialized"] is False, "ACTUAL_GPU_INITIALIZATION_FORBIDDEN")


def run(args):
    need(re.fullmatch(r"[a-f0-9]{32}", args.request) and 60 <= args.budget <= 480
         and all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                     "MODELSCOPE_OFFLINE", "PYTHONDONTWRITEBYTECODE"))
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")),
         "EXACT_ZERO_GRES_OFFLINE_AUTHORITY")
    parent = args.output.parent
    need(parent.resolve(strict=True) == parent and parent.stat().st_uid == os.geteuid()
         and stat.S_IMODE(parent.stat().st_mode) == 0o700 and args.output.is_absolute(), "EXACT_PRIVATE_PROBE_OUTPUT")
    native = frozen_native(args.native_source)
    args.output.mkdir(mode=0o700)
    trace = BoundaryTrace(stop_before_forward=True)
    facts = dict(model_loads=0, real_adapter_loads=0, gpu_initialized=None, unexpected_unstopped_response=False)
    result_error = None
    start = time.monotonic()
    def timed_out(*_):
        raise TimeoutError("CPU_BOUNDARY_DEADLINE")
    signal.signal(signal.SIGALRM, timed_out)
    try:
        signal.setitimer(signal.ITIMER_REAL, args.budget)
        cpu_invocation(args, native, trace, facts)
    except Exception as error:
        result_error = error_evidence(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = trace.project()
    same_error = any(row["original_5207_message_match"] for row in (trace.error or result_error or []))
    actual_template = bool(trace.counts.get("template.generate:entry"))
    all_pinned = len(trace.providers) == 5 and all(row["file_SHA256"] is not None for row in trace.providers.values())
    observed = actual_template and trace.real_forward_calls == 0 and all_pinned and (
        trace.pre_forward_reached or same_error)
    passed = observed and facts["model_loads"] == 1 and facts["gpu_initialized"] is False and not facts["unexpected_unstopped_response"]
    report = dict(request=args.request, original_boundary="1103606.5207", actual_forward_calls=trace.real_forward_calls,
                  diagnostic_image_SHA256=EXPOSED_SHA, original_training_binding_SHA256=EXPOSED_BINDING,
                  heldout_inputs_opened=0, facts=facts, boundary=result,
                  outer_error=result_error, passed=passed, original_message_match=same_error,
                  actual_predictions=None if facts["unexpected_unstopped_response"] else 0,
                  seconds=time.monotonic()-start, GPU_equivalence_proven=False)
    from contract import put, wire
    put(args.output / "result.json", wire(report))
    print(json.dumps(dict(request=args.request, passed=passed, forward_calls=trace.real_forward_calls,
                         pre_forward_reached=trace.pre_forward_reached, original_message_match=same_error)), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-source", "diagnostic-image", "model", "model-receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--diagnostic-image-sha", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--budget", type=int, default=420)
    raise SystemExit(run(parser.parse_args()))
