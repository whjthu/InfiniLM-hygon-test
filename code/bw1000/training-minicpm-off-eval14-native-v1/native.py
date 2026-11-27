"""One base model load and two native Swift arms on one allocated HIP device."""
import argparse
import ctypes
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import time

from contract import ARMS, CAP, CP2, CP_ADAPTER, CP_FILES, CP_INVENTORY_SHA, HEX, HEX32, INPUT, MODEL, PROMPT_SHA, encode, generation, inputs, need, put, seal, sha, small, stat_key, verify_file
from internvl_gqa_miopen_home import verify as verify_home

ADAPTER = "minicpm_off_cp2"
ENGINE = {
    "transformers_engine.py": "b1c229267347569f63bc2afd0e263c2d0d919535e596db454e05b8f825493dbb",
    "infer_engine.py": "857f18c2b25a1e6787679a3f96b8fa96f5de7efd040e6933bb62e0a798634969",
    "utils.py": "5aa9dc2376402d4d6a0167c5f1253a6cd085e10b157a0f3f5a7b72c0f8607644",
    "protocol.py": "f7abe5f7fb9ed9cf390d54a3303d28d6c68c28679ab304dc461f645bbd5aac6e",
}
VERSIONS = {"torch": "2.10.0+das.opt1.dtk2604.2606161421.g91a580", "ms-swift": "4.4.2", "transformers": "5.12.1"}


def exact_pci():
    provider = ctypes.CDLL("libamdhip64.so")
    buf = ctypes.create_string_buffer(32)
    child = ctypes.c_int()
    need(provider.hipGetDeviceCount(ctypes.byref(child)) == 0 and child.value == 1
         and provider.hipDeviceGetPCIBusId(buf, len(buf), 0) == 0, "EXACT_ONE_HIP_DEVICE")
    value = buf.value.decode("ascii").lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value):
        value = "0000:" + value
    need(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value), "HIP_PCI_TYPE")
    return value


def verify_launch(stage, ready, request, output):
    parent, node = ready["parent"], ready["node"]
    actual_step = os.environ.get("SLURM_STEP_ID", "")
    need(ready["request"] == stage["request"] == request and os.environ.get("SLURM_JOB_ID") == parent
         and re.fullmatch(r"[0-9]+", actual_step) and node == socket.gethostname().split(".")[0]
         and os.environ.get("SLURM_GPUS_ON_NODE") in (None, "1")
         and os.environ.get("SLURM_CPUS_PER_TASK") == "8" and len(os.sched_getaffinity(0)) >= 8
         and str(output) == ready["output"] and ready["input_SHA256"] == PROMPT_SHA
         and ready["boot"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
         "NATIVE_AUTHORITY_STEP_OR_BOOT")
    return parent + "." + actual_step


def current_files(stage, hash_base):
    need(stage["cp2_manifest_SHA256"] == CP_INVENTORY_SHA and set(stage["adapter"]) == set(CP_FILES)
         and len(stage["base"]) == 10 and "model.safetensors" in stage["base"], "NATIVE_ASSET_SCOPE")
    need(all(os.statvfs(path).f_flag & os.ST_RDONLY for path in (INPUT, MODEL, CP2)), "NATIVE_ASSET_MOUNT_NOT_READONLY")
    for name, original in stage["base"].items():
        path = MODEL / name
        if hash_base:
            pin,identity = verify_file(path,original["bytes"],original["SHA256"])
            need(pin["bytes"] == original["bytes"] and identity == original["identity"], "CURRENT_BASE_FD_BINDING")
        else:
            need(path.resolve(strict=True) == path and stat_key(path.lstat()) == original["identity"], "CURRENT_BASE_FILE_BINDING")
    for name, original in stage["adapter"].items():
        need({k:original[k] for k in ("bytes","SHA256")} == CP_FILES[name], "OFF_CP2_MEMBER_PIN")
        pin, stamp = verify_file(CP2 / name, 128*1024**2, original["SHA256"])
        need(pin["bytes"] == original["bytes"] and stamp == original["identity"], "CURRENT_CP2_FILE_BINDING")


def run(args):
    need(HEX32.fullmatch(args.request) and HEX.fullmatch(args.stage_sha) and HEX.fullmatch(args.ready_sha)
         and args.output.is_absolute(), "NATIVE_ARG_SCOPE")
    stage = json.loads(small(args.stage, 256*1024, args.stage_sha))
    ready = json.loads(small(args.ready, 16384, args.ready_sha))
    need(ready["stage_SHA256"] == args.stage_sha and stage["prompt_SHA256"] == PROMPT_SHA
         and type(ready["epoch"]) in (float,int) and math.isfinite(ready["epoch"])
         and type(ready["deadline"]) in (float,int) and 0 < ready["deadline"] - ready["epoch"] <= 960, "NATIVE_EPOCH_SCOPE")
    actual_step = verify_launch(stage, ready, args.request, args.output)
    args.output.mkdir(mode=0o700)
    rows = {arm: {} for arm in ARMS}
    witnesses = {arm: {} for arm in ARMS}
    facts = dict(actual_step=actual_step, model_loads=0, actual_native_calls=0, actual_live_adapter=None)
    failure = None
    current = {}
    phase = "inputs"

    def deadline(*_):
        if not ready["epoch"] <= time.time() < ready["deadline"]:
            raise TimeoutError("NATIVE_SHARED_EPOCH")

    signal.signal(signal.SIGALRM, deadline)
    try:
        deadline()
        signal.setitimer(signal.ITIMER_REAL, max(.001, ready["deadline"] - time.time()))
        prompts = inputs(INPUT)
        current_files(stage,hash_base=True)
        facts["current_base_members_SHA_count"] = len(stage["base"])
        phase = "providers"
        need({k: metadata.version(k) for k in VERSIONS} == VERSIONS, "EXACT_MINICPM_RUNTIME_VERSIONS")
        for name, pin in ENGINE.items():
            small(Path("/swift-runtime/site/swift/infer_engine") / name, 2*1024**2, pin)
        for name,pin in stage["framework_helpers"].items():
            need(name in ("vlm/runtime/first_update_audit.py","vlm/runtime/first_update_capture.py")
                 and len(small(Path("/mini-framework")/name,1024**2,pin["SHA256"]))==pin["bytes"], "DIGEST_PROVIDER_PIN")
        put(args.output/"miopen-home.json", encode(verify_home(args.home)))
        from providers import verify_wheels
        facts["official_wheels"] = verify_wheels()
        import numpy as np
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
        from swift.model import get_model_processor
        from swift.template import get_template
        from vlm.runtime.first_update_audit import fingerprint
        from vlm.runtime.first_update_capture import cpu

        for name in ENGINE:
            module = importlib.import_module("swift.infer_engine." + name[:-3])
            need(Path(module.__file__) == Path("/swift-runtime/site/swift/infer_engine")/name, "ACTUAL_SWIFT_ENGINE_PROVIDER")
        need(Path(importlib.import_module(fingerprint.__module__).__file__) == Path("/mini-framework/vlm/runtime/first_update_audit.py")
             and Path(importlib.import_module(cpu.__module__).__file__) == Path("/mini-framework/vlm/runtime/first_update_capture.py"),
             "ACTUAL_NAMED_DIGEST_PROVIDER")
        need(torch.version.hip == "6.3.26113" and not torch.version.cuda and torch.cuda.device_count() == 1, "EXACT_HIP_RUNTIME")
        torch.cuda.set_device(0)
        pci = exact_pci()
        need(pci in ready["pci"] and len(set(ready["pci"])) == 8, "PCI_OUTSIDE_ORIGINAL_EIGHT")
        facts["PCI"] = pci
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        phase = "CP2_state"
        state = load_file(str(CP2/"adapter_model.safetensors"), device="cpu")
        need(fingerprint(state) == CP_ADAPTER and all(re.search(r"\.lora_[AB]\.weight$", key) for key in state),
             "CP2_DIFFERS_FROM_INDEPENDENT_CPU_NAMED_VALUES")
        phase = "model_load"
        model, processor = get_model_processor(str(MODEL), model_type="minicpmv4_6", load_model=True,
                                             download_model=False, device_map={"": 0}, torch_dtype=torch.bfloat16, attn_impl="sdpa")
        facts["model_loads"] += 1
        need(model is not None and not hasattr(model, "peft_config") and {str(p.device) for p in model.parameters()} == {"cuda:0"}
             and processor.model_meta.template == "minicpmv4_6", "UNADAPTED_ORIGINAL_BASE")
        model.eval()
        model.requires_grad_(False)
        template = get_template(processor, template_type="minicpmv4_6", max_length=4096, max_pixels=262144,
                                padding_side="right", padding_free=False, truncation_strategy="raise")
        need(template.template_meta.template_type == "minicpmv4_6", "MINICPM_TEMPLATE")
        engine = TransformersEngine(model, template=template, max_batch_size=1)
        config = RequestConfig(max_tokens=CAP, temperature=0, num_beams=1, n=1, stream=False, return_details=True, stop=[])
        effective = engine._prepare_generation_config(config)
        need(not effective.do_sample and effective.num_beams == 1 and effective.max_new_tokens == CAP, "FIXED_GREEDY_POLICY")
        collate, generate, get_ids = template.data_collator, template.generate, template.get_generate_ids

        def observe_collate(*params, **options):
            value = collate(*params, **options)
            ids = value["input_ids"]
            pixels = value.get("pixel_values")
            need(current["input"] is None and value.get("labels") is None and ids.ndim == 2 and ids.shape[0] == 1
                 and 0 < ids.shape[-1] <= 4096 and pixels is not None
                 and (not isinstance(pixels,(list,tuple)) or bool(pixels)), "NATIVE_USER_ONLY_IMAGE_COLLATION")
            observed = fingerprint(cpu(value))
            current["input"] = {"SHA256": observed["SHA256"], "tokens": int(ids.shape[-1])}
            old = witnesses["base"].get(current["slot"])
            if current["arm"] == "adapter":
                need(old == current["input"], "PAIRED_IMAGE_PROMPT_DIFFERENCE")
            return value

        def observe_generate(*params, **options):
            dc = options["generation_config"]
            need(current["eos"] is None and current["input"] is not None and not dc.do_sample and dc.num_beams == 1
                 and dc.max_new_tokens == CAP and not engine.model.training and options.get("past_key_values") is None,
                 "ACTUAL_NATIVE_GREEDY_PATH")
            if current["arm"] == "base":
                need(engine.model is model and not hasattr(model, "peft_config") and not options.get("adapter_names"), "BASE_ADAPTER_CONTAMINATION")
            else:
                need(options.get("adapter_names") == [ADAPTER] and engine.model.get_base_model() is model
                     and ADAPTER in engine.model.peft_config, "CP2_ACTIVE_ADAPTER_OR_BASE_IDENTITY")
                live = get_peft_model_state_dict(engine.model, adapter_name=ADAPTER)
                if facts["actual_live_adapter"] is None:
                    match = fingerprint(cpu(live))
                    need(match == CP_ADAPTER and set(live) == set(state)
                         and all(torch.equal(live[key].detach().cpu(), state[key]) for key in state), "LIVE_CP2_DIFFERS_FROM_CPU_NAMED_VALUES")
                    facts["actual_live_adapter"] = match
            raw_eos = dc.eos_token_id
            eos = raw_eos if isinstance(raw_eos, list) else [raw_eos]
            need(eos and all(type(x) is int and x >= 0 for x in eos), "NATIVE_EOS")
            current["eos"] = eos
            facts["actual_native_calls"] += 1
            return generate(*params, **options)

        def observe_ids(*params, **options):
            ids = get_ids(*params, **options)
            need(current["generated"] is None and ids.ndim == 2 and ids.shape[0] == 1, "ONE_NATIVE_GENERATED_SEQUENCE")
            current["generated"] = ids[0].detach().cpu().tolist()
            return ids

        template.data_collator, template.generate, template.get_generate_ids = observe_collate, observe_generate, observe_ids
        phase = "paired_native"
        for arm in ARMS:
            adapter = None if arm == "base" else AdapterRequest(name=ADAPTER, path=str(CP2))
            for prompt in prompts:
                deadline()
                slot = prompt["slot"]
                current = dict(arm=arm, slot=slot, input=None, eos=None, generated=None)
                try:
                    response, = engine.infer([InferRequest(messages=[{"role":"user", "content":"<image>\n"+prompt["question"]}],
                                                         images=[str(INPUT/prompt["image"])])],
                                             config, use_tqdm=False, adapter_request=adapter)
                    if isinstance(response, BaseException):
                        raise response
                    choice, = response.choices
                    pad = engine.tokenizer.pad_token_id
                    finish, _ = generation(current["generated"], current["eos"], pad)
                    need(choice.token_ids == [x for x in current["generated"] if x != pad]
                         and type(choice.message.content) is str and len(choice.message.content) <= 4096, "ACTUAL_NATIVE_DECODE")
                    row = dict(slot=slot, answer=choice.message.content, finish_reason=finish)
                    witnesses[arm][slot] = current["input"]
                    rows[arm][slot] = row
                    put(args.output/f"{arm}-{slot}.private.json", encode(row))
                except Exception:
                    rows[arm][slot] = dict(slot=slot, answer=None, finish_reason="error")
                    put(args.output/f"{arm}-{slot}.error.json", encode(rows[arm][slot]))
                    raise
        phase = "post_asset_continuity"
        current_files(stage,hash_base=False)
    except BaseException as error:
        failure = dict(phase=phase, arm=current.get("arm"), slot=current.get("slot"), error_type=type(error).__name__,
                       detail_SHA256=sha(str(error).encode()), generation_entered=current.get("eos") is not None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    summary = seal(args.output, args.request, rows, witnesses, failure, facts)
    print(json.dumps(dict(request=args.request, planned=28, passed=summary["passed"], counts=summary["counts"]),
                     sort_keys=True), flush=True)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("stage", "ready", "output", "home"):
        parser.add_argument("--"+k, type=Path, required=True)
    for k in ("request", "stage-sha", "ready-sha"):
        parser.add_argument("--"+k, required=True)
    raise SystemExit(run(parser.parse_args()))
