"""Test real frozen source derivation and scopes while keeping devices and models untouched."""

import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from types import SimpleNamespace

from derive import SCHEMA
from run import CONFIG, ORIGINAL_STATIC_RNG, TRAINER_FILE, compare, source_bytes


def reject(call, code):
    try:
        call()
    except ValueError as error:
        assert str(error) == code, str(error)
    else:
        raise AssertionError(code)


def native_argv(source, config, checkpoint):
    body = next(node.body for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                and node.name == "run_sdxl_training")
    assignments = {target.id: i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    root = PurePosixPath("/fixture")
    ns = dict(config=config, sys=SimpleNamespace(executable="/native-python"), script=root / TRAINER_FILE,
              weights=root / "weights", dataset=root / "inputs", output_dir=root / "out",
              metrics_path=root / "out/steps.ndjson", initial_adapter=root / "out/initial.safetensors",
              max_steps=25, expected=25 if checkpoint is None else 25 - int(checkpoint.name.split("-")[-1]),
              checkpointing_steps=10, checkpoints_total_limit=3, resume_checkpoint=checkpoint)
    nodes = [body[assignments["warmup_steps"]], *body[assignments["command"]:assignments["environment"]]]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "exact_backend_command_and_warmup", "exec"), ns)
    return ns["command"]


def exercise_sources(repo):
    parent = repo / "local-only/training-sdxl-style18-public-candidate-cli-v1-original"
    trainer = (parent / "candidate-source-root/examples/text_to_image" / TRAINER_FILE).read_bytes()
    backend = (parent / "sdxl_training_196501.py").read_bytes()
    with tarfile.open(repo / "code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz") as archive:
        helper = archive.extractfile("framework/generation/src/genmlu/sdxl_training_resume_checkpoint.py").read()
    sources = source_bytes((trainer, backend, helper))
    reject(lambda: source_bytes((trainer, backend, helper + b" ")), "EXACT_ORIGINAL_PROVIDER")
    sys.path.insert(0, str(repo / "code/bw1000/training-sdxl-style18-public-candidate-cli-v1"))
    from check_interface_once import parser_only
    config = dict(CONFIG, runtime_image_id="sha256:" + "a" * 64)
    for original in (None, PurePosixPath("/frozen/continuous/checkpoint-10"), PurePosixPath("/frozen/continuous/checkpoint-20")):
        argv = native_argv(sources["sdxl_hip_common.py"].decode(), config, original)
        args = parser_only(sources[TRAINER_FILE])(argv[2:])
        assert args.genmlu_full_checkpoint is True and args.dataloader_num_workers == 0
        assert (args.max_train_steps, args.checkpointing_steps, args.genmlu_checkpoint_steps) == (25, 10, "25")
        assert (args.train_batch_size, args.gradient_accumulation_steps, args.mixed_precision) == (1, 1, "bf16")
        assert (args.learning_rate, args.lr_warmup_steps, args.lr_scheduler, args.rank) == (1e-4, 0, "constant", 4)
        assert args.genmlu_external_resume_from_checkpoint == (None if original is None else str(original))
    namespace = {"__file__": "fake-no-real-import", "__name__": "hip_checkpoint_fixture"}
    exec(compile(sources["sdxl_hip_checkpoint.py"], "exact-derived-hip-helper", "exec"), namespace)
    assert namespace["SCHEMA"] == SCHEMA and b"torch_mlu_manual_seed" not in sources["sdxl_hip_checkpoint.py"]
    api = SimpleNamespace(device_count=lambda: 1, current_device=lambda: 0)
    fake = SimpleNamespace(cuda=api, version=SimpleNamespace(hip="6.3.26113", cuda=None),
                           device=lambda kind, ordinal: (kind, ordinal))
    accel = SimpleNamespace(device=SimpleNamespace(type="cuda", index=None))
    assert namespace["native_training_device"](accel, fake) == ("cuda", 0)
    fake.version.hip = None
    reject(lambda: namespace["native_training_device"](accel, fake), "native full checkpoint requires one visible HIP")
    rng = repo / "local-only/sdxl-first1-v3-original-RNG-static-contract-e0c72d94d6004939a2929973bbaf6969.json"
    raw = rng.read_bytes()
    assert sha256(raw).hexdigest() == ORIGINAL_STATIC_RNG
    actual = json.loads(raw)["response"]
    assert set(actual["allowed_key_symbols"]) == {"numpy_random_seed", "random_state", "step",
        "torch_cuda_manual_seed", "torch_manual_seed"} and actual["tensor_storage_extents"]["1"] == 16
    assert actual["pickle_executed"] is False
    return {name: sha256(raw).hexdigest() for name, raw in sources.items()}


def exercise_comparison():
    rows = [dict(global_step=step, sample_witness={"marker": step}, loss=0.1, grad_norm=1.0) for step in range(1, 26)]
    source, final = {"state_sha256": "a" * 64}, {"state_sha256": "b" * 64}
    a = dict(completed_step=0, process_pid=100, process_startticks=10000)
    for start in (10, 20):
        b = dict(completed_step=start, next_step=start+1, restored_full_state_exact=True,
                 state_sha256=source["state_sha256"], restored=dict(source), process_pid=200, process_startticks=11000)
        assert compare(rows, rows[start:], source, final, final, a, b, start)["actual_updates"] == 50 - start
        drift = [dict(row) for row in rows[start:]]
        drift[0]["loss"] = 0.2
        bad = compare(rows, drift, source, final, final, a, b, start)
        assert bad["first_different_update"] == start + 1 and bad["passed"] is False
        dup = dict(b, process_pid=100, process_startticks=10000)
        reject(lambda: compare(rows, rows[start:], source, final, final, a, dup, start),
               "NEW_PROCESS_EXACT_RESTORE_BEFORE_UPDATE")


if __name__ == "__main__":
    source = exercise_sources(Path(__file__).resolve().parents[3])
    exercise_comparison()
    print(json.dumps(dict(exact_source_derived=source, actual_trainer_cli=True, static_2670_RNG_evidence=True,
                          HIP_gate_fixture_only=True, real_Torch_device_calls=0, target_calls=0), sort_keys=True))
