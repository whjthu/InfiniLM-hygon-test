"""Run continuous SDXL 25 and exact full-state CP10 or CP20 recovery in a second native process."""

import argparse
from hashlib import sha256
import importlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

from derive import BACKEND, HELPER, SCHEMA, TRAINER, backend_source, helper_source, need, trainer_source


PIPE_MANIFEST = "b0a2515ad9a6422d83991788f6af7749bc50812ad0b4e028e52b86350c519175"
FIRST_UPDATE_SOURCE = "15145c880bd9d3ce08c09607b8ed43a93fae7ddb9f3c73459a81e7c42bcf4c63"
FIRST_UPDATE_WITNESS = "6305e21a7a85a11b6706b5fe17e7d9a6a852a68a49b7a5f4c8cf804706856568"
ORIGINAL_FIRST_UPDATE = "24624803a2f89d04f89d832c0cef459e2fda57eb3cf85092351b56c87781d2b7"
ORIGINAL_STATIC_RNG = "adee3624410f35a60be1166a275f6deda481f23c7fa69203c4c01000b75ea9a2"
TRAINER_FILE = "train_text_to_image_lora_sdxl.py"
CONFIG = dict(max_train_steps=25, checkpointing_steps=10, extra_checkpoint_steps="25",
              checkpoints_total_limit=3, warmup_steps=5, resolution=256, train_batch_size=1,
              learning_rate=1e-4, lora_rank=4, seed=20260911, gradient_checkpointing=False,
              dataloader_num_workers=0, sdxl_hip_full_checkpoint=True)


def report_file(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def source_bytes(inputs):
    return {
        TRAINER_FILE: trainer_source(inputs[0]),
        "sdxl_hip_common.py": backend_source(inputs[1]),
        "sdxl_hip_checkpoint.py": helper_source(inputs[2]),
    }


def checkpoint(helper, path, step, pins):
    doc = helper.preflight(path)
    need(doc["schema"] == SCHEMA and doc["step"] == step and doc["next_step"] == step + 1,
         "EXACT_NATIVE_HIP_CHECKPOINT_BOUNDARY")
    contract = doc["contract"]
    need(contract["hip"] is True and len(contract["data"]["records"]) == 2
         and contract["trainer_sha256"] == pins[TRAINER_FILE]
         and contract["helper_sha256"] == pins["sdxl_hip_checkpoint.py"]
         and contract["backend_sha256"] == pins["sdxl_hip_common.py"], "ACTIVE_HIP_STACK_CONTRACT")
    return doc


def updates(work, start, stop):
    raw = (work / "steps.ndjson").read_bytes()
    need(len(raw) <= 2 << 20, "BOUNDED_UPDATE_LEDGER")
    rows = [json.loads(line) for line in raw.splitlines()]
    need([row["global_step"] for row in rows] == list(range(start, stop + 1)), "COMPLETE_ACTUAL_UPDATE_LEDGER")
    for row in rows:
        witness = row["sample_witness"]
        step = row["global_step"]
        need(type(step) is int and math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
             and row["grad_norm"] > 0 and (witness["epoch"], witness["batch_index"]) == divmod(step - 1, 2)
             and len(witness["sample_ids"]) == len(witness["samples"]) == 1
             and witness["sample_ids"][0] in (0, 1)
             and set(witness["tensors"]) == {"pixel_values", "input_ids_one", "input_ids_two"},
             "NATIVE_UPDATE_OR_CURSOR_NOT_PROVEN")
    return rows


def compare(continuous, resumed, source_doc, final_a, final_b, witness_a, witness_b, resume_step):
    need(len(continuous) == 25 and len(resumed) == 25 - resume_step, "PLANNED_PAIR_DENOMINATOR")
    original = continuous[resume_step:]
    need([row["global_step"] for row in original] == [row["global_step"] for row in resumed],
         "EXACT_RESUMED_UPDATE_ORDINALS")
    need(witness_a["completed_step"] == 0 and witness_b["completed_step"] == resume_step
         and witness_b["next_step"] == resume_step + 1 and witness_b["restored_full_state_exact"] is True
         and witness_b["state_sha256"] == witness_b["restored"]["state_sha256"] == source_doc["state_sha256"]
         and all(type(row[key]) is int and row[key] > 0 for row in (witness_a, witness_b)
                 for key in ("process_pid", "process_startticks"))
         and (witness_a["process_pid"], witness_a["process_startticks"]) !=
         (witness_b["process_pid"], witness_b["process_startticks"]), "NEW_PROCESS_EXACT_RESTORE_BEFORE_UPDATE")
    different = {}
    for left, right in zip(original, resumed, strict=True):
        fields = [name for name in ("sample_witness", "loss", "grad_norm") if left[name] != right[name]]
        if fields:
            different[left["global_step"]] = fields
    final_equal = final_a["state_sha256"] == final_b["state_sha256"]
    return dict(passed=not different and final_equal, first_different_update=min(different, default=None),
                differing_updates=different, final_full_state_exact=final_equal, resumed_from=resume_step,
                restored_full_state_exact=True, continuous_updates=len(continuous), resumed_updates=len(resumed),
                actual_updates=len(continuous) + len(resumed), final_step=25,
                continuous_final_state_SHA256=final_a["state_sha256"], resume_final_state_SHA256=final_b["state_sha256"],
                source_checkpoint_state_SHA256=source_doc["state_sha256"])


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def firstupdate_tools(root):
    for name, digest in (("run.py", FIRST_UPDATE_SOURCE), ("sdxl_cc0_witness.py", FIRST_UPDATE_WITNESS)):
        need(sha256((root / name).read_bytes()).hexdigest() == digest, "EXACT_ORIGINAL_INPUT_AND_ASSET_VERIFIER")
    sys.path.insert(0, str(root))
    try:
        return load_module(root / "run.py", "sdxl_original_firstupdate_tools")
    finally:
        sys.path.pop(0)


def run(args):
    need(sys.platform == "linux" and re.fullmatch(r"[a-f0-9]{32}", args.request) and
         args.request != "e2f599910efc3e11758a4afb5ef1c7bf" and
         os.environ.get("SDXL_SINGLE_DCU_OWNER_REQUEST") == args.request
         and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_STEP_GPUS", ""))
         and all(os.environ.get(key, "").isdecimal() for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
         and (Path("/tmp/.sdxl-owner-request").read_text().strip() == args.request)
         and stat.S_IMODE(Path("/tmp").stat().st_mode) == 0o700 and Path("/tmp").stat().st_uid == os.geteuid()
         and os.environ.get("PYTHONPATH") == "/training-runtime/site"
         and re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_id), "FRESH_HIP_RUNTIME_AND_OWNER")
    for root in (args.firstupdate_source, args.original_trainer, args.framework, args.pipeline, args.input):
        need(root.resolve(strict=True) == root and os.statvfs(root).f_flag & os.ST_RDONLY, "READONLY_ORIGINAL_MOUNTS")
    tools = firstupdate_tools(args.firstupdate_source)
    home = Path("/tmp/sdxl-home")
    home.mkdir(mode=0o700)
    db, cache = home / ".config/miopen/user-db", home / ".cache/miopen"
    for directory in (db, cache):
        directory.mkdir(mode=0o700, parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                      HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp",
                      MIOPEN_USER_DB_PATH=str(db), MIOPEN_CUSTOM_CACHE_DIR=str(cache))
    need(args.work.is_absolute() and args.work.parent.resolve(strict=True) == args.work.parent, "EXACT_DURABLE_OUTPUT")
    args.work.mkdir(mode=0o700)
    counters = dict(continuous_updates=None, resumed_updates=None, scientific_pass=False)
    planned = 25 + (25 - args.resume_step)
    report_file(args.work / "intent.json", dict(request=args.request, planned_updates=planned, resume_from=args.resume_step,
                original_first_update_SHA256=ORIGINAL_FIRST_UPDATE, original_rng_projection_SHA256=ORIGINAL_STATIC_RNG))
    phase = "assets"
    try:
        tools.assets(args.pipeline, args.pipeline_manifest, PIPE_MANIFEST)
        tools.dataset(args.input, args.work / "inputs")
        phase = "provider"
        old_backend = args.framework / "gencuda/backends/sdxl_training.py"
        old_helper = args.framework / "genmlu/sdxl_training_resume_checkpoint.py"
        sources = source_bytes(tuple(tools.read_pinned(path, pin, keep=True) for path, pin in (
            (args.original_trainer, TRAINER), (old_backend, BACKEND), (old_helper, HELPER))))
        overlay = args.work / "source"
        child_source = overlay / "examples/text_to_image"
        child_source.mkdir(parents=True)
        pins = {}
        for name, raw in sources.items():
            with (child_source / name).open("xb") as stream:
                stream.write(raw)
            pins[name] = sha256(raw).hexdigest()
        helper = load_module(child_source / "sdxl_hip_checkpoint.py", "sdxl_hip_checkpoint_readonly")
        sys.path.insert(0, str(args.framework))
        stock = importlib.import_module("gencuda.backends.sdxl_training")
        need(Path(stock.__file__) == old_backend, "EXACT_BOUND_GENCUDA")
        namespace = dict(vars(stock))
        exec(compile(sources["sdxl_hip_common.py"], "sdxl_hip_common.py", "exec"), namespace)
        base = dict(CONFIG, runtime_image_id=args.image_id, train_data_dir=str(args.work / "inputs"))
        phase = "continuous"
        continuous, resume = args.work / "continuous", args.work / "resume"
        namespace["run_sdxl_training"](args.pipeline, base, continuous, overlay)
        rows = updates(continuous, 1, 25)
        counters["continuous_updates"] = len(rows)
        continuous_ckpts = {step: checkpoint(helper, continuous / f"checkpoint-{step}", step, pins)
                            for step in (10, 20, 25)}
        source = continuous / f"checkpoint-{args.resume_step}"
        source_doc = continuous_ckpts[args.resume_step]
        phase = "new_process_resume"
        namespace["run_sdxl_training"](args.pipeline, dict(base, resume_from_checkpoint=str(source)), resume, overlay)
        resumed = updates(resume, args.resume_step + 1, 25)
        counters["resumed_updates"] = len(resumed)
        final = checkpoint(helper, resume / "checkpoint-25", 25, pins)
        need(all(doc["contract"] == final["contract"] for doc in continuous_ckpts.values())
             and checkpoint(helper, source, args.resume_step, pins) == source_doc, "UNMODIFIED_IDENTICAL_RECIPE_AND_SOURCE")
        phase = "compare"
        witnesses = [json.loads((path / "preupdate-witness.json").read_bytes()) for path in (continuous, resume)]
        need(witnesses[1]["restored"]["manifest_sha256"] == helper.file_sha(source / helper.MANIFEST)
             and witnesses[1]["restored"]["path"] == str(source), "EXACT_RESTORED_SOURCE_MANIFEST")
        result = compare(rows, resumed, source_doc, continuous_ckpts[25], final, *witnesses, args.resume_step)
        report_file(args.work / "result.json", dict(request=args.request, planned_updates=planned, **result,
                    source_SHA256=pins, original_first_update_is_resume_source=False, formal_domain_credit=0))
        return 0 if result["passed"] else 2
    except BaseException as error:
        report_file(args.work / "result.json", dict(request=args.request, planned_updates=planned, phase=phase,
                    failure_class=type(error).__name__, failure_SHA256=sha256(str(error).encode()).hexdigest(),
                    **counters, formal_domain_credit=0))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    check = subparsers.add_parser("verify-source")
    for name in ("trainer", "backend", "helper"):
        check.add_argument("--" + name, type=Path, required=True)
    launch = subparsers.add_parser("run")
    for name in ("firstupdate-source", "original-trainer", "framework", "pipeline", "pipeline-manifest", "input", "work"):
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--resume-step", type=int, choices=(10, 20), required=True)
    launch.add_argument("--request", required=True)
    launch.add_argument("--image-id", required=True)
    args = parser.parse_args()
    if args.mode == "verify-source":
        print(json.dumps({name: sha256(raw).hexdigest() for name, raw in source_bytes(
            tuple(path.read_bytes() for path in (args.trainer, args.backend, args.helper))).items()}, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
