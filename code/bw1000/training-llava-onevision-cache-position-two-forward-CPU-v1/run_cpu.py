"""Qualify scoped cache repair through real prefill and first decode using the exposed image."""

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

from repair import install
from two_forward import TwoForwardComplete, TwoForwards, need


SOURCES = {
    "cpu_probe.py": "342717bdbd24946c67b1856fd32b7c9719f6c6fdd9997d6324b1692d5eb03327",
    "generation_boundary.py": "1b651a99f60738a6c3439a4ccaf7d2cc26f6171656c21fb95085ee7ff6e1720b",
}
SOURCE_AUDIT = "5de9d05b399c906e5a587537096fb90a8c26897112e4d784854d317ce3065c74"
BEFORE = '''        with trace.attach_template(template), trace.attach_model(engine.model):
            need(all(value["file_SHA256"] is not None for value in trace.providers.values()),
                 "ALL_FIVE_REAL_CALLABLE_SOURCE_FILES")
            response, = engine.infer([InferRequest(messages=messages, images=[str(args.diagnostic_image)])],
                                     cfg, use_tqdm=False, adapter_request=None)
            if isinstance(response, BaseException):
                raise response
            facts["unexpected_unstopped_response"] = True
            raise ValueError("CPU_WITNESS_DID_NOT_STOP_BEFORE_RESPONSE")
'''
AFTER = '''        _two_forwards(args, native_class, model, template, engine, cfg, InferRequest, messages, torch, facts)
'''


def derive(raw):
    need(sha256(raw).hexdigest() == SOURCES["cpu_probe.py"], "EXACT_ORIGINAL_5225_CPU_PROBE")
    text = raw.decode()
    need(text.count(BEFORE) == 1, "EXACT_ORIGINAL_NATIVE_INVOCATION_ANCHOR")
    patched = text.replace(BEFORE, AFTER, 1)
    ast.parse(patched)
    return patched.encode()


def bind(folder):
    need(folder.is_absolute() and folder.resolve(strict=True) == folder and
         os.statvfs(folder).f_flag & os.ST_RDONLY, "READONLY_FROZEN_5225_CPU_PROBE")
    sources = {}
    for name, expected in SOURCES.items():
        path = folder / name
        need(path.resolve(strict=True) == path and path.is_file() and path.stat().st_size < 64 << 10,
             "EXACT_BOUNDED_PROBE_SOURCE_FILE")
        raw = path.read_bytes()
        need(sha256(raw).hexdigest() == expected, "EXACT_FROZEN_5225_SOURCE")
        sources[name] = raw
    need("generation_boundary" not in sys.modules, "UNSHADOWED_BOUNDARY_SOURCE")
    boundary = ModuleType("generation_boundary")
    boundary.__file__ = str(folder / "generation_boundary.py")
    sys.modules[boundary.__name__] = boundary
    exec(compile(sources["generation_boundary.py"], boundary.__file__, "exec"), boundary.__dict__)
    probe = ModuleType("qualified_5225_two_forward_probe")
    probe.__file__ = str(folder / "cpu_probe.py")
    probe._two_forwards = invoke
    exec(compile(derive(sources["cpu_probe.py"]), probe.__file__, "exec"), probe.__dict__)
    return probe, boundary


def invoke(args, native_class, model, template, engine, cfg, infer_request, messages, torch, facts):
    from transformers.cache_utils import DynamicCache
    from transformers.generation.utils import GenerationMixin

    need(engine.model is model and not hasattr(model, "peft_config"), "ACTUAL_UNADAPTED_CPU_MODEL")
    witness = TwoForwards(torch, DynamicCache)
    with install(model, native_class, torch, DynamicCache, GenerationMixin) as decision, witness.install(template, model):
        try:
            response, = engine.infer([infer_request(messages=messages, images=[str(args.diagnostic_image)])],
                                     cfg, use_tqdm=False, adapter_request=None)
            if isinstance(response, BaseException):
                raise response
            choice, = response.choices
            # Check only the first token and the finish reason in the bounded early-EOS branch.
            need(witness.early_eos(choice), "UNEXPECTED_COMPLETE_RESPONSE_WITHOUT_TWO_FORWARD_STOP")
            facts["early_eos"] = True
        except TwoForwardComplete:
            need(witness.complete and len(witness.records) == 2, "EXACT_TWO_FORWARD_STOP")
            need(len(decision.events) == 2 and all(
                record["visual_present"] is not visual["drop_visual"]
                and record["forwarded_cache_position_is_none"] ==
                (visual["position"] == "none_native_forward_fallback")
                and (visual["position"] != "none_native_forward_fallback" or
                     (visual["cache_length"], visual["query_tokens"]) == (record["cache_before"], record["query_tokens"]))
                for record, visual in zip(witness.records, decision.events, strict=True)), "ACTUAL_VISUAL_AND_POSITION_BRANCH")
            facts["two_forward_stop"] = True
        finally:
            facts.update(forwards_entered=witness.entered, returned_forwards=witness.records,
                         visual_decisions=decision.events)


def run(args):
    need(re.fullmatch(r"[a-f0-9]{32}", args.request) and not args.request.startswith("0f17a330cc2b")
         and 60 <= args.budget <= 600 and all(os.environ.get(key) == "-1" for key in
                     ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
         and all(os.environ.get(key) == "1" for key in
                     ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE", "PYTHONDONTWRITEBYTECODE"))
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")),
         "EXACT_NEW_ZERO_GRES_OFFLINE_SCOPE")
    parent = args.output.parent
    need(args.output.is_absolute() and parent.resolve(strict=True) == parent and parent.stat().st_uid == os.geteuid()
         and stat.S_IMODE(parent.stat().st_mode) == 0o700, "EXACT_PRIVATE_PROBE_OUTPUT")
    probe, boundary = bind(args.boundary_source)
    native = probe.frozen_native(args.native_source)
    args.output.mkdir(mode=0o700)
    facts = dict(model_loads=0, real_adapter_loads=0, gpu_initialized=None, forwards_entered=0, returned_forwards=[],
                 visual_decisions=[], early_eos=False, two_forward_stop=False)
    failure = None
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError("CPU_TWO_FORWARD_DEADLINE")
    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, args.budget)
        probe.cpu_invocation(args, native, None, facts)
    except Exception as error:
        failure = boundary.error_evidence(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    passed = failure is None and facts["two_forward_stop"] and facts["model_loads"] == 1 and (
        facts["forwards_entered"] == len(facts["returned_forwards"]) == len(facts["visual_decisions"]) == 2
        and facts["gpu_initialized"] is False)
    stage = "EXACT_TWO_CPU_FORWARD_RETURNS" if passed else ("EARLY_EOS_DECODE_UNCOVERED" if facts["early_eos"] and
                                                                   failure is None else "FAILURE_OR_UNCOVERED")
    result = dict(request=args.request, phase=stage, passed=bool(passed), actual_step=
                  os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"], original_step="1103606.5225",
                  exact_source_audit_SHA256=SOURCE_AUDIT, diagnostic_image_SHA256=probe.EXPOSED_SHA,
                  actual_generated_outputs_persisted=0, heldout_inputs_opened=0, GPU_equivalence_proven=False,
                  facts=facts, failure=failure, seconds=time.monotonic()-start)
    from contract import put, wire
    put(args.output / "result.json", wire(result))
    print(json.dumps(dict(request=args.request, phase=stage, actual_returned_forwards=len(facts["returned_forwards"]))),
          flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boundary-source", "native-source", "diagnostic-image", "model", "model-receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--diagnostic-image-sha", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--budget", type=int, default=520)
    raise SystemExit(run(parser.parse_args()))
