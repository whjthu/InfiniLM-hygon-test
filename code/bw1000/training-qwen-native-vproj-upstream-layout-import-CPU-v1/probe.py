"""Zero-card verification of the optional callback against the actual Swift runtime."""

from hashlib import sha256
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path, PurePosixPath
import sys

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from tools.hygon_vlm_recipe import build_launch


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    require(not torch.cuda.is_initialized(), "CUDA initialized before provider check")
    sources = Path("/layout-swift")
    frozen = Path("/layout-upstream")
    sys.path.insert(0, str(sources))
    from swift_args import add_observer

    plugin = sources / "swift_plugin.py"
    engine = importlib.util.spec_from_file_location("qwen_native_layout_swift_probe", plugin)
    adapter = importlib.util.module_from_spec(engine)
    engine.loader.exec_module(adapter)
    require(callbacks_map[adapter.KEY] is adapter.QwenLayoutCallback
            and issubclass(adapter.QwenLayoutCallback, TrainerCallback), "Actual Swift callback registry differs")
    upstream = adapter.load_layout(frozen)
    require(callable(upstream.evaluate), "Exact loaded diagnostic callable is unavailable")
    raw = Path("/original-launch.json").read_bytes()
    require(sha256(raw).hexdigest() == "28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400",
            "Frozen Qwen recipe differs")
    launch = build_launch("T01.sft", raw, PurePosixPath("/qualified-framework/vlm/runtime/swift_entry.py"),
                          PurePosixPath("/qwen-model"), PurePosixPath("/real-data.jsonl"),
                          PurePosixPath("/original-plugin"), PurePosixPath("/unused-training-output"), 29968)
    original = launch["command"]
    command = add_observer(original, str(plugin))
    require(command.count(adapter.KEY) == command.count(str(plugin)) == 1
            and [arg for arg in command if arg not in (adapter.KEY, str(plugin))] == original,
            "Original launch changed outside observer addition")
    signature = inspect.signature(Seq2SeqTrainer.training_step)
    require(list(signature.parameters)[:3] == ["self", "model", "inputs"], "Native Swift training_step interface differs")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "ms-swift", "peft")}
    require(versions["ms-swift"] == "4.4.2" and versions["transformers"] == "5.12.1"
            and versions["peft"] == "0.19.1" and not torch.cuda.is_initialized(), "Target provider or zero-device scope differs")
    print(json.dumps(dict(state="ACTUAL_SWIFT_OBSERVER_IMPORT_AND_RECIPE_PASS", versions=versions,
                          actual_callback_registered=True, upstream_frozen_source_loaded=True,
                          only_two_argv_tokens_added=True, training_step_parameter_names=list(signature.parameters),
                          actual_training_steps_invoked=0, actual_CUDA_initialized=False, actual_model_loaded=False)), flush=True)


if __name__=="__main__":
    main()
