"""Audit all released train rows through the registered native Gemma tokenizer and collator."""

from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import audit_gemma_training_native as original
import stage_receiver as stage


STAGE = Path("/audit-source")
DATA = stage.TARGET / "data"
PROCESSOR = Path("/gemma-processor")
FRAMEWORK = Path("/input-framework")
OUT = Path("/audit-out/worker")
OBSERVER = "vlm/runtime/first_update_audit.py"


def exact(path, pin):
    if path.is_symlink() or not path.is_file():
        raise ValueError("native processor asset type differs")
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != pin["bytes"]:
            raise ValueError("native processor asset size differs")
        return stage.checked_member(stream, pin)


def verified_json(path, pin):
    return json.loads(exact(path, pin))


def audited_scope(claim, check_deadline):
    offer = claim["data_stage"]
    stage.check_claim(offer)
    observed = verified_json(DATA.parent / "stage-outcome.json", claim["data_stage_outcome"])
    if observed != dict(request=offer["request"], state="DATA_STAGED", role="train", completed=offer["train_rows"] + 2,
                        train_rows=offer["train_rows"], payload_bytes=offer["payload_bytes"], slurm=0, gpu=0,
                        optimizer_updates=0, training_READY=False):
        raise ValueError("original receive receipt differs")
    binding = exact(DATA / "data-binding-private.json", offer["binding"])
    rows = exact(DATA / "train-swift.jsonl", offer["rows"])
    media = stage.validate_scope(offer, binding, rows)
    if DATA.is_symlink() or (DATA / "images").is_symlink() or set(x.name for x in DATA.iterdir()) != {
            "data-binding-private.json", "train-swift.jsonl", "images"}:
        raise ValueError("staged data includes an unexpected role")
    if set(x.name for x in (DATA / "images").iterdir()) != media.keys():
        raise ValueError("staged images differ from the exact release")
    for name, pin in media.items():
        check_deadline()
        exact(DATA / "images" / name, pin)
    return rows.splitlines(keepends=True)


def load_template(claim, torch, check_deadline):
    files = verified_json(STAGE / "MODEL_PROCESSOR_FILES.json", claim["processor_manifest"])["files"]
    if set(p.name for p in PROCESSOR.iterdir()) != files.keys() or len(files) != 14:
        raise ValueError("offline processor namespace changed")
    for name, pin in files.items():
        check_deadline()
        exact(PROCESSOR / name, pin)
    f = verified_json(STAGE / "FRAMEWORK_FILES.json", claim["framework_manifest"])["files"][OBSERVER]
    expected = FRAMEWORK / OBSERVER
    exact(expected, {k: f[k] for k in ("bytes", "SHA256")})
    sys.path.insert(0, str(FRAMEWORK))
    from vlm.runtime import first_update_audit
    from swift.model import get_model_processor
    from swift.template import get_template

    if Path(first_update_audit.__file__).resolve() != expected.resolve():
        raise ValueError("unbound native input fingerprint provider")
    model, processor = get_model_processor(
        str(PROCESSOR), model_type="gemma3_vision", load_model=False, download_model=False,
        device_map="cpu", attn_impl="sdpa", torch_dtype=torch.bfloat16,
    )
    if model is not None:
        raise ValueError("native preprocessing constructed a model")
    kwargs = dict(template_type="gemma3_vision", max_pixels=262144, padding_side="right",
                  padding_free=False, truncation_strategy="raise")
    full, capped = [get_template(processor, max_length=n, **kwargs) for n in (None, 512)]
    full.set_mode("train")
    capped.set_mode("train")
    suffix = capped.template_meta.suffix
    if not suffix or suffix != full.template_meta.suffix:
        raise ValueError("original supervised turn delimiter differs")
    trailing = capped._encode_context_list(suffix, [1.] * len(suffix))[0]
    if trailing != full._encode_context_list(suffix, [1.] * len(suffix))[0]:
        raise ValueError("original turn delimiter tokens differ")
    eos_value = json.loads((PROCESSOR / "generation_config.json").read_bytes())["eos_token_id"]
    eos = {eos_value} if type(eos_value) is int else set(eos_value)
    eos.add(processor.tokenizer.eos_token_id)
    if any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("invalid original EOS metadata")
    return full, capped, first_update_audit.fingerprint, list(trailing), eos


def exercise(lines, template, check_deadline, stream):
    summary = {"checked": 0, "failed": 0, "not_started": 0, "max_tokens": None, "supervised_tokens": 0}
    reached_deadline = False
    for ordinal, line in enumerate(lines, 1):
        record = dict(ordinal=ordinal, source_line_SHA256=sha256(line).hexdigest(), state="NOT_STARTED")
        if not reached_deadline:
            def checked(fields):
                record.update(fields, ordinal=ordinal, state="CHECKED")

            try:
                original.audit_rows([line], *template, check_deadline, checked)
            except TimeoutError:
                if record["state"] != "CHECKED":
                    record["state"] = "FAILED"
                    record["error_type"] = "TimeoutError"
                reached_deadline = True
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                record.update(state="FAILED", error_type=type(error).__name__)

        summary[record["state"].lower()] += 1
        if record["state"] == "CHECKED":
            summary["max_tokens"] = max(summary["max_tokens"] or 0, record["tokens"])
            summary["supervised_tokens"] += record["shifted_supervised_tokens"]
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return summary


def main():
    claim = json.loads((STAGE / "CLAIM.json").read_bytes())
    epoch, deadline = claim["epoch"], claim["worker_deadline"]
    now = time.time()
    job, step = (os.environ.get(key, "") for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    if (not all(math.isfinite(n) for n in (epoch, deadline)) or not 0 < epoch <= now < deadline <= epoch + 390
            or job != claim["parent"] or not step.isdecimal() or socket.gethostname().split(".")[0] != claim["node"]
            or not 1 <= len(os.sched_getaffinity(0)) <= 4
            or os.environ.get("SLURM_STEP_GPUS") not in (None, "", "(null)")
            or any(os.environ.get(k) != "-1" for k in (
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
        raise ValueError("expected one original offline CPU step")
    stop = time.monotonic() + deadline - now

    def check_deadline():
        if time.monotonic() >= stop or time.time() >= deadline:
            raise TimeoutError("original audit deadline passed")

    lines = audited_scope(claim, check_deadline)
    if Path(original.__file__).resolve() != (STAGE / "audit_gemma_training_native.py").resolve():
        raise ValueError("unexpected native Gemma audit helper")
    if {k: importlib.metadata.version(k) for k in original.VERSIONS} != original.VERSIONS:
        raise ValueError("original native runtime versions differ")
    if "torch" in sys.modules:
        raise ValueError("expected native Torch import in this fresh worker")
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("unexpected device initialization")
    attempted = []

    def forbid_device(*_args, **_kwargs):
        attempted.append(True)
        raise RuntimeError("device initialization forbidden in this CPU step")

    torch.cuda._lazy_init = forbid_device
    torch.set_num_threads(2)
    template = load_template(claim, torch, check_deadline)
    OUT.mkdir()
    capture = OUT / "native-train-rows-private.jsonl"
    with capture.open("x", encoding="utf-8", newline="\n") as stream:
        totals = exercise(lines, template, check_deadline, stream)
    passed = totals["checked"] == len(lines) and not attempted and not torch.cuda.is_initialized()
    result = dict(total_train_rows=len(lines), actual_step=job + "." + step, model_loaded=False,
                  gpu=0, optimizer_updates=0, training_READY=False, **totals, passed=passed, epoch=epoch, deadline=deadline,
                  rows_SHA256=sha256(capture.read_bytes()).hexdigest(), device_init_attempts=len(attempted))
    stage.save(OUT / "result.json", stage.encode(result))
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(not passed)


if __name__ == "__main__":
    sys.exit(main())
