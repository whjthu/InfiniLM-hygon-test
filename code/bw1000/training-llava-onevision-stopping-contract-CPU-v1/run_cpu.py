"""Run original OneVision stopping-policy construction on the known exposed image."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from types import ModuleType

from stopping import hash_json, id_info, need, observe

V4 = {
    "native.py": "4fdaa114b90109b361d46bae8c61a139f8c51b69cd907407b27b68532b3faecc",
    "contract.py": "62bd2a9554645ab615320e227b69abb09a93804a5fe390987dea56a3f2947b47",
    "repair.py": "22e3f95eb52192146834dcb4285080d9bc928a597bceab41d54c87e2ce119abe",
    "two_forward.py": "3f594f945ca414173fb83e7361ba10ae6d548b4a8c04d80f34f66e47a9204e0e",
}
BOUNDARY = {
    "cpu_probe.py": "342717bdbd24946c67b1856fd32b7c9719f6c6fdd9997d6324b1692d5eb03327",
    "generation_boundary.py": "1b651a99f60738a6c3439a4ccaf7d2cc26f6171656c21fb95085ee7ff6e1720b",
}
ORIGINAL = "fed9c042b5f9f5418ada35ece4853705cbad2dc011323e8c2f312efbec1b87c5"
PROVIDER_PROOF = "cb12278fd957a7898f7bca0915d055ac2aab32bec39c99048a008cb52d823abc"
MODEL_FILES = {
    "config.json": (2505, "6f28a8d792b02e1d6756398229bdf46e4c976d589439552af414227aa3cb4243"),
    "generation_config.json": (225, "f58d4aaa30f8b5c1453e287454ee10385af19770510dd59cbb7afbab5c2d7781"),
    "tokenizer_config.json": (4730, "e04be4dd17e9f661bc487ba848a37f66a8f1adf2eef412f5d4ecb40cab4d39e7"),
}
SWIFT_TEMPLATE = {
    "swift/template/base.py": (107643, "cc1629c8a84b764b530895692e4e04322aa6732281be229efce9e98752e113cc"),
    "swift/template/templates/llava.py": (17769, "65ae2986ba1ffb173c1c19a9f532a5cb1b6f9f85931782e151c6e2206d458f0c"),
    "swift/template/templates/qwen.py": (76766, "d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3"),
}
OLD_PREFIXES = ("7344da28bdbd", "354e5ebd0e01", "0f17a330cc2b")
ANCHOR = """        with trace.attach_template(template), trace.attach_model(engine.model):
            need(all(value["file_SHA256"] is not None for value in trace.providers.values()),
                 "ALL_FIVE_REAL_CALLABLE_SOURCE_FILES")
            response, = engine.infer([InferRequest(messages=messages, images=[str(args.diagnostic_image)])],
                                     cfg, use_tqdm=False, adapter_request=None)
            if isinstance(response, BaseException):
                raise response
            facts["unexpected_unstopped_response"] = True
            raise ValueError("CPU_WITNESS_DID_NOT_STOP_BEFORE_RESPONSE")
"""


def derive(raw):
    need(sha256(raw).hexdigest() == BOUNDARY["cpu_probe.py"], "EXACT_PINNED_EXPOSED_CPU_PROBE")
    text = raw.decode()
    need(text.count(ANCHOR) == 1, "EXACT_ORIGINAL_EXPOSED_MODEL_INVOCATION")
    replacement = """        _observe_stopping(args, native, native_class, model, template, engine, cfg,
                          InferRequest, messages, torch, facts)
"""
    patched = text.replace(ANCHOR, replacement, 1)
    ast.parse(patched)
    return patched.encode()


def readonly(path):
    need(path.is_absolute() and path.resolve(strict=True) == path
         and os.statvfs(path).f_flag & os.ST_RDONLY, "EXACT_SCOPE_READONLY")


def read(path, digest):
    readonly(path)
    need(stat.S_ISREG(path.lstat().st_mode), "EXACT_SCOPE_REGULAR_FILE")
    with path.open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    need(0 < len(raw) <= 64 << 10 and sha256(raw).hexdigest() == digest, "EXACT_ORIGINAL_DIAGNOSTIC_SOURCE")
    return raw


def verify_original(raw):
    need(sha256(raw).hexdigest() == ORIGINAL, "EXACT_5232_ORIGINAL_TERMINAL")
    report = json.loads(raw)
    result = report["independent_numbered_terminal"]
    need(report["gpu"] == "1103606.5232" and result["accounting"]["gpu"][2:4] == ["COMPLETED", "0:0"]
         and result["cleanup_verified"] is True and result["planned"] == 32
         and all(result["counts"][arm] == dict(cap=16, eos=0, error=0, missing=0, stop=0)
                 for arm in ("base", "cp25")), "ORIGINAL_TWICE_16_ALL_CAP")
    return result


def bind(boundary_folder, frozen):
    for path in (boundary_folder, frozen):
        readonly(path)
    raw = {key: read(boundary_folder / key, pin) for key, pin in BOUNDARY.items()}
    for key, pin in V4.items():
        read(frozen / key, pin)
    need(not {"contract", "generation_boundary", "repair", "two_forward"}.intersection(sys.modules),
         "EXACT_UNSHADOWED_NATIVE_PROVIDERS")
    boundary = ModuleType("generation_boundary")
    boundary.__file__ = str(boundary_folder / "generation_boundary.py")
    sys.modules[boundary.__name__] = boundary
    exec(compile(raw["generation_boundary.py"], boundary.__file__, "exec"), boundary.__dict__)
    cpu = ModuleType("llava_exposed_cpu_stopping")
    cpu.__file__ = str(boundary_folder / "cpu_probe.py")
    cpu._observe_stopping = invoke
    exec(compile(derive(raw["cpu_probe.py"]), cpu.__file__, "exec"), cpu.__dict__)
    cpu.FROZEN = {key: V4[key] for key in ("native.py", "contract.py")}
    return cpu, boundary


def declarations(args, native):
    values = {}
    for name, (size, sha) in MODEL_FILES.items():
        path = args.model / name
        native.readonly(path)
        data = json.loads(native.read(path, size, sha, size))
        named = data.get("eos_token")
        entry = dict(SHA256=sha, eos_token_id=id_info(data.get("eos_token_id")),
                     pad_token_id=id_info(data.get("pad_token_id")),
                     eos_name_present=named is not None,
                     eos_name_SHA256=hash_json(named) if named is not None else None)
        if name == "config.json":
            entry["text_config_EOS"] = id_info((data.get("text_config") or {}).get("eos_token_id"))
        values[name] = entry
    return values


def invoke(args, native, native_class, model, template, engine, cfg, infer_request, messages, torch, facts):
    need(engine.model is model and not hasattr(model, "peft_config"), "EXACT_UNADAPTED_EXPOSED_MODEL")
    facts["read_only_config_declarations"] = declarations(args, native)
    provider_paths = {}
    for name, (size, sha) in SWIFT_TEMPLATE.items():
        path = native.SWIFT / name
        native.readonly(path)
        native.read(path, size, sha, size)
        provider_paths[path] = sha
    sys.path.insert(0, str(args.native_source))
    try:
        install, cache, mixin = native.cache_patch()
    finally:
        sys.path.pop(0)

    def inference():
        response, = engine.infer([infer_request(messages=messages, images=[str(args.diagnostic_image)])], cfg,
                                 use_tqdm=False, adapter_request=None)
        if isinstance(response, BaseException):
            raise response
        raise ValueError("UNEXPECTED_RESPONSE_WITHOUT_MODEL_FORWARD")

    with install(model, native_class, torch, cache, mixin):
        observe(model, template, engine, cfg, inference, torch, facts, provider_paths)


def run(args):
    need(re.fullmatch(r"[a-f0-9]{32}", args.request) and not args.request.startswith(OLD_PREFIXES)
         and 60 <= args.budget <= 600 and all(os.environ.get(key) == "-1" for key in
                     ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and all(os.environ.get(key) == "1" for key in
                     ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE", "PYTHONDONTWRITEBYTECODE"))
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")),
         "EXACT_NEW_ZERO_GRES_EXPOSED_STOPPING_SCOPE")
    verify_original(read(args.original, ORIGINAL))
    parent = args.output.parent
    need(args.output.is_absolute() and parent.resolve(strict=True) == parent and parent.stat().st_uid == os.geteuid()
         and stat.S_IMODE(parent.stat().st_mode) == 0o700, "EXACT_PRIVATE_STOPPING_RESULT")
    probe, boundary = bind(args.boundary_source, args.native_source)
    native = probe.frozen_native(args.native_source)
    args.output.mkdir(mode=0o700)
    facts = dict(model_loads=0, first_forward_guard_hits=0, pre_forward_stop=False, gpu_initialized=None)
    failure = None
    start = time.monotonic()

    def timeout(*_):
        raise TimeoutError("STOPPING_EXPOSED_CPU_DEADLINE")

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, args.budget)
        probe.cpu_invocation(args, native, None, facts)
    except Exception as error:
        failure = boundary.error_evidence(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    passed = (failure is None and facts["model_loads"] == facts["first_forward_guard_hits"] == 1
              and facts["pre_forward_stop"] is True and facts["gpu_initialized"] is False
              and set(facts["controlled_suffix"]) == {"eos", "ordinary", "cap"})
    report = dict(request=args.request, actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  original_5232_SHA256=ORIGINAL, exposed_image_SHA256=probe.EXPOSED_SHA, heldout_inputs_opened=0,
                  original_provider_readback_SHA256=PROVIDER_PROOF,
                  historical_token_ids_available=False, generated_outputs=0, original_model_forward_calls=0,
                  passed=bool(passed), seconds=time.monotonic() - start, facts=facts, failure=failure)
    from contract import put, wire
    put(args.output / "result.json", wire(report))
    print(json.dumps({key: report[key] for key in ("request", "actual_step", "passed", "original_model_forward_calls")}),
          flush=True)
    return 0 if passed else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("boundary-source", "native-source", "original", "diagnostic-image", "model", "model-receipt", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--diagnostic-image-sha", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--budget", type=int, default=520)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
