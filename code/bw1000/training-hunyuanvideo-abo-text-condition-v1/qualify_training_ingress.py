"""Qualify the exact HunyuanVideo domain-cache binding in the target CPU runtime."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import sys


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
FRAMEWORK = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
BACKEND = FRAMEWORK / "generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"
BACKEND_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"
EXPECTED = {"training_cache", "training_cache_sha"}


def main(root, request):
    numbered = os.environ.get("SLURM_STEP_ID", "")
    assert socket.gethostname().split(".", 1)[0] == "f13r3n14" and os.environ.get("SLURM_JOB_ID") == "1103606"
    assert re.fullmatch("[0-9]+", numbered) and re.fullmatch("[0-9a-f]{32}", request)
    assert os.environ.get("SLURM_STEP_GPUS", "") in ("", "(null)", "none", "N/A")
    assert all(os.environ.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    cgroup = Path("/proc/self/cgroup").read_text("ascii")
    assert any({ "job_1103606", "step_" + numbered } <= set(line.split(":", 2)[-1].split("/"))
               for line in cgroup.splitlines())
    sys.path.insert(0, str(root / "source"))
    from contract import CACHE, CACHE_BYTES, CACHE_SHA, CLI_SHA, MODEL, NATIVE_ALIAS, LAUNCH_SHA, read
    from domain_ingress import bind, load_common_cli

    cache, cache_pin = read(CACHE, CACHE_BYTES)
    assert len(cache) == CACHE_BYTES and cache_pin["SHA256"] == CACHE_SHA
    backend, backend_pin = read(BACKEND, 1 << 20)
    assert backend_pin["SHA256"] == BACKEND_SHA and len(backend) > 0
    cli = FRAMEWORK / "tools/hygon_training_recipe.py"
    module, pin = load_common_cli(cli)
    assert pin["SHA256"] == CLI_SHA
    builder = module["build_generation_plan"]
    launch = (root / "source/ORIGINAL_LAUNCH.json").read_bytes()
    assert sha256(launch).hexdigest() == LAUNCH_SHA
    original = json.loads(launch)
    mapping = {original["config"]["training_cache"]: str(CACHE), original["weights"]: str(MODEL),
               original["source"]: str(NATIVE_ALIAS)}
    rebound = bind(builder, "T09.sft", launch, mapping)
    config = rebound["config"]
    changed = {k for k in config.keys() | original["config"].keys() if config.get(k) != original["config"].get(k)}
    assert changed == EXPECTED and config["training_cache"] == str(CACHE) and config["training_cache_sha"] == CACHE_SHA
    assert callable(module["run_generation"]) and module["run_generation"].__globals__["build_generation_plan"] is builder

    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    assert torch.cuda.device_count() == 0 and not torch.cuda.is_initialized()
    hook = register_optimizer_step_post_hook(lambda *_: None)
    hook.remove()
    assert not torch.cuda.is_initialized()
    result = dict(request=request, actual_step="1103606." + numbered, actual_GPU=0, optimizer_updates=0,
                  model_loads=0, changed_config_keys=sorted(changed), optimizer_hook_register_remove=True,
                  deployed_cli_SHA256=CLI_SHA, original_launch_SHA256=LAUNCH_SHA, actual_cache_SHA256=CACHE_SHA,
                  common_cli_sibling_provider=True,
                  actual_cache_bytes=CACHE_BYTES, original_optimizer_config_preserved=True,
                  native_train_method_present=True, actual_train_method_invoked=False, passed=True)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with (root / "worker.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
