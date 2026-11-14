"""Build the frozen eight-device HIP Qwen invocation with ordered native callbacks."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from tools.hygon_vlm_recipe import build_launch

CASE = "T01.sft"
BINDING_SHA256 = "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
CALLBACKS = ("native_initial_trainables", "native_first_updates")
BASE = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
DATASET = BASE / "training-vlm-original-assets-v1/vlm_report_coco64.hip.jsonl"
PLUGINS = BASE / "training-vlm-original-assets-v1/plugins"
MODEL_PARENT = BASE / "training-qwen-original-public-model-v1/models"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(original, binding_path, swift, framework, output, port, *, epoch, deadline, max_seconds):
    """Return the invocation. The caller owns the Slurm lease, runtime and process."""
    require(all(math.isfinite(v) for v in (epoch, deadline, max_seconds))
            and 0 < epoch < deadline <= epoch + max_seconds, "Invalid native epoch budget")
    raw = Path(binding_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BINDING_SHA256, "Target input binding changed")
    binding = json.loads(raw)
    require(binding["spec"]["case"] == CASE and binding["spec"]["training"]["max_steps"] == 25
            and binding["spec"]["training"]["gradient_accumulation_steps"] == 4
            and binding["spec"]["training"]["per_device_train_batch_size"] == 1
            and binding["spec"]["training"]["bf16"] is True, "Target training scope changed")
    root = PurePosixPath(framework)
    output = PurePosixPath(output)
    sources = [root / "vlm/plugins" / (name + ".py") for name in CALLBACKS]
    launch = build_launch(CASE, original, PurePosixPath(swift), MODEL_PARENT, DATASET, PLUGINS,
                          output / "train", port, diagnostic_plugin=sources,
                          diagnostic_callback=list(CALLBACKS), diagnostic_callback_position="before")
    argv = launch["command"]
    at = argv.index("--callbacks") + 1
    require(argv[at:at + 3] == [*CALLBACKS, "nvidia_baseline"], "Invalid callback order")
    env = launch["environment_overrides"]
    require(env["NPROC_PER_NODE"] == "8" and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7",
            "Eight-device topology changed")
    env.update(FIRST_UPDATE_PLATFORM="hip", FIRST_UPDATE_BINDING=str(binding_path),
               FIRST_UPDATE_BINDING_SHA256=BINDING_SHA256, FIRST_UPDATE_CAPTURE=str(output / "capture"),
               FIRST_UPDATE_EPOCH=str(epoch), FIRST_UPDATE_DEADLINE=str(deadline),
               FIRST_UPDATE_MAX_SECONDS=str(max_seconds))
    return launch
