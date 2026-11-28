"""Probe exact continued target layout and Trainer chronology without imports or writes."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import owner
import resume_layout as layout


TRAINER = (Path("/usr/local/lib/python3.10/dist-packages/transformers/trainer.py"),
           "17e212935057f58b52efe79c3f952bf044bbada8afd4f13c8e905142b1d2ef0a")
MIXIN = (Path("/swift-runtime/site/swift/trainers/mixin.py"),
         "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9")
SHARD = (Path("/swift-runtime/site/swift/dataloader/shard.py"),
         "ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c")


def call_sites(node, name):
    return sorted(item.lineno for item in ast.walk(node) if isinstance(item, ast.Call)
                  and isinstance(item.func, ast.Attribute) and item.func.attr == name)


def chronology(tree):
    objects = [item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Trainer"]
    trainer, = objects
    methods = {item.name: item for item in trainer.body if isinstance(item, ast.FunctionDef)}
    loop, epoch = methods["_inner_training_loop"], methods["_run_epoch"]
    initial = methods["_init_training_state"]
    assignments = [ast.unparse(item) for item in ast.walk(initial) if isinstance(item, ast.Assign)]
    owner.need(any("self.state.global_step // num_update_steps_per_epoch" in line for line in assignments)
               and "steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch" in assignments,
               "ACTUAL_NATIVE_CP2_INTEGER_CURSOR")
    begin, = call_sites(loop, "on_train_begin")
    run, = call_sites(loop, "_run_epoch")
    prepared, = call_sites(loop, "_prepare_for_training")
    direct_optimizer = call_sites(loop, "_load_optimizer_and_scheduler")
    optimizer = direct_optimizer or call_sites(methods["_prepare_for_training"], "_load_optimizer_and_scheduler")
    owner.need(begin < run and prepared < begin and len(optimizer) == 1
               and (not direct_optimizer or direct_optimizer[0] < begin), "NATIVE_RESTORE_BEFORE_TRAIN_BEGIN")
    adapter = [(name, line) for name in ("train", "_inner_training_loop", "_prepare_for_training")
               for line in call_sites(methods[name], "_load_from_checkpoint")]
    owner.need(adapter and any(name == "train" or line < begin for name, line in adapter), "NATIVE_ADAPTER_RESUME_CALL")
    begin_rng = [child.lineno for item in ast.walk(epoch) if isinstance(item, ast.If)
                 and ast.unparse(item.test) == "steps_trained_in_current_epoch == 0"
                 for body in item.body for child in ast.walk(body) if isinstance(child, ast.Call)
                 and isinstance(child.func, ast.Attribute) and child.func.attr == "_load_rng_state"]
    first_draw, = call_sites(epoch, "get_batch_samples")
    owner.need(len(begin_rng) == 1 and begin_rng[0] < first_draw, "NATIVE_ZERO_SKIP_RNG_BEFORE_FIRST_FETCH")
    return dict(adapter_load_sites=adapter, optimizer_scheduler_load_line=optimizer[0],
                train_begin=begin, first_epoch_call=run, rng_at_epoch_boundary=begin_rng[0], first_fetch=first_draw)


def ro(path):
    owner.need(path.resolve(strict=True) == path and os.statvfs(path).f_flag & os.ST_RDONLY,
               "EXACT_GUEST_READONLY_MOUNT")


def guest(request, step):
    owner.need(os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"] == step
               and all(os.environ.get(key) == "-1" for key in
                       ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
               "ZERO_DEVICE_ASSEMBLY_ENVIRONMENT")
    path = Path("/llava-source")
    for member in (path, Path("/llava-helpers"), Path("/llava-train-only"), Path("/llava-framework"),
                   Path("/swift-runtime"), Path("/vision"), Path("/t04-model"), layout.CP2):
        ro(member)
    for name in ("4983.json", "CP2_NATIVE_RESUME_PLAN.private.json"):
        ro(Path("/llava-resume-authority") / name)
    for name, expected in (("TENSORS.json", layout.TENSORS), ("INPUTS.json", layout.INPUTS)):
        member = Path("/llava-helpers") / name
        ro(member)
        owner.pin(member, expected)
    owner.here_manifest(path, layout.SCIENCE)
    manifest_path = Path("/llava-resume-authority/CP2_NATIVE_RESUME_PLAN.private.json")
    raw, _ = owner.pin(manifest_path, layout.PLAN_SHA)
    plan = json.loads(raw)
    owner.need(len(raw) == layout.PLAN_BYTES and len(plan["updates"]) == 23
               and sum(len(row["rows"]) for row in plan["updates"]) == 736
               and plan["start"]["skip_microbatches"] == 0, "ORIGINAL_4993_736_AND_ZERO_SKIP")
    archive, _ = owner.document(Path("/llava-resume-authority/4983.json"), layout.ARCHIVE_SHA)
    files = archive["training"]["checkpoint_manifest"]
    owner.need({p.name for p in layout.CP2.iterdir()} == set(files) and len(files) == 17, "ORIGINAL_CP2_MEMBER_COUNT")
    for name, member in files.items():
        current = layout.CP2 / name
        ro(current)
        owner.need(current.stat().st_size == member["bytes"], "ORIGINAL_CP2_MOUNTED_MEMBER_EXTENT")
    state, _ = owner.document(layout.CP2 / "trainer_state.json", files["trainer_state.json"]["SHA256"])
    owner.need((state["global_step"], state["epoch"], state["max_steps"]) == (2, 1.0, 25), "EXACT_REAL_CP2_CURSOR")
    facts = {}
    for name, (source, digest) in dict(trainer=TRAINER, swift_mixin=MIXIN, swift_shard=SHARD).items():
        data, _ = owner.pin(source, digest, limit=512 << 10)
        facts[name] = dict(bytes=len(data), SHA256=digest)
        if name == "trainer":
            facts["chronology"] = chronology(ast.parse(data))
    owner.need(not any(name in sys.modules for name in ("torch", "swift", "transformers")), "NO_MODEL_PROVIDER_IMPORTS")
    return dict(passed=True, request=request, step=step, actual_GPU=0, model_loaded=False,
                original_cp2_members=17, plan_SHA256=layout.PLAN_SHA, science_SHA256=layout.SCIENCE,
                source=facts)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("host", "guest"))
    cli.add_argument("--request", required=True)
    cli.add_argument("--step")
    args = cli.parse_args()
    owner.need(re.fullmatch(r"[0-9a-f]{32}", args.request), "EXACT_NEW_OWNER_REQUEST")
    if args.mode == "guest":
        print(owner.encode(guest(args.request, args.step)).decode(), end="", flush=True)
        return 0
    label = "training-llava-clevr-resume25-" + args.request[:12]
    step = owner.local_step("assembly", {"label": label})
    owner.need(socket.gethostname().split(".")[0] == owner.NODE, "EXACT_LIVE_ASSEMBLY_NODE")
    layout.archive_requirements(owner.pin, owner.need)
    owner.pin(owner.STAGED / "staged-ready.json", owner.STAGED_SHA)
    owner.here_manifest(layout.RESUME_STAGE / "source", layout.SCIENCE)
    bindings = [*layout.mount_arguments(Path("/"), resume_authority=False),
                (owner.HERE, "/llava-resume-owner", "ro"), (owner.STAGED / "helpers", "/llava-helpers", "ro"),
                (owner.INPUT, "/llava-train-only", "ro"), (owner.FRAMEWORK, "/llava-framework", "ro"),
                (owner.RUNTIME, "/swift-runtime", "ro"), (owner.VISION, "/vision", "ro"),
                (owner.MODEL, "/t04-model", "ro")]
    argv = [owner.APPTAINER, "exec", "--contain", "--no-mount", "home,cwd", "--pwd", "/"]
    for source, target, mode in bindings:
        argv.extend(("--bind", f"{source.as_posix()}:{target}:{mode}"))
    argv.extend((owner.IMAGE.as_posix(), "/usr/bin/python3", "-B", "/llava-resume-owner/assembly.py",
                 "guest", "--request", args.request, "--step", step))
    os.chdir("/")
    env = dict(os.environ)
    for device in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env["APPTAINERENV_" + device] = "-1"
    reply = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=35)
    owner.need(reply.returncode == 0 and 0 < len(reply.stdout) <= 16384, "BOUNDED_CONTAINER_ASSEMBLY_RECEIPT")
    proof = json.loads(reply.stdout)
    owner.need(proof["request"] == args.request and proof["step"] == step and proof["passed"] is True,
               "ACTUAL_CONTAINER_ASSEMBLY_RECEIPT")
    print(owner.encode(proof).decode(), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
