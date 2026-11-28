"""Read-only zero-GRES qualification of the staged OneVision Swift entry."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sys


def main():
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert os.environ[key] == "-1"
    assert not os.environ.get("SLURM_STEP_GPUS") and not os.environ.get("SLURM_GPUS_ON_NODE")
    sys.path[:0] = ["/llava-helpers", "/llava-framework", "/swift-runtime/site", "/vision", "/llava-source"]
    import train
    import probe
    import llava_recovered_native_cpu as recovered
    import llava_target_eager_import as target_eager

    assert sha256(Path(train.__file__).read_bytes()).hexdigest() == "757d8be1b62bf4bc7f21ba526c2ff284c88f46d6b6de48560b52bba0881c6ee8"
    for name, digest in train.HELPER_PINS.items():
        probe.read(train.HELPERS / name, 64 * 1024, digest)
    original = json.loads(probe.read(train.HELPERS / "original-launch.json", 64 * 1024, train.ORIGINAL))
    model = Path("/t04-model")
    entry = train.RUNTIME / "swift/cli/sft.py"
    probe.read(entry, 583, train.CLI_SHA)
    from vlm.plugins import llava_onevision

    train.EXPECTED_CLASS = target_eager.preimport(model, llava_onevision, recovered)
    arguments = train.cli(original, model, Path("/tmp/train-bound.jsonl"), Path("/tmp/t04-out"),
                          train.HERE / "callback.py")
    import swift.pipelines as pipelines
    from transformers import HfArgumentParser

    real_main = pipelines.sft_main
    swift_sft = real_main.__globals__["SwiftSft"]
    sft_args = swift_sft.args_class
    for symbol in (real_main, swift_sft, sft_args):
        module = sys.modules[symbol.__module__]
        assert Path(module.__file__).resolve().is_relative_to(train.RUNTIME)
    passed = {}

    def check_entry_arguments(*args, **kwargs):
        assert not args and not kwargs and sys.argv == [str(entry), *arguments]
        parser = HfArgumentParser(sft_args)
        parsed, unknown = parser.parse_known_args(arguments)
        assert not unknown
        assert (parsed.attn_impl, parsed.gradient_checkpointing, parsed.max_steps,
                parsed.max_length, parsed.gradient_accumulation_steps,
                parsed.per_device_train_batch_size, parsed.sequence_parallel_size) == ("eager", False, 25, 4096, 4, 1, 1)
        import callback
        from swift.callbacks import callbacks_map

        assert callbacks_map[train.CALLBACK] is callback.LlavaClevrFirstTwo
        passed.update(cli_executed=True, option_tokens=len(arguments), recognized_all_options=True,
                      swift_sft_class=swift_sft.__name__, swift_argument_class=sft_args.__name__,
                      native_callback=True, native_eager_class=train.EXPECTED_CLASS.__name__)

    sys.argv = [str(entry), *arguments]
    pipelines.sft_main = check_entry_arguments
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        pipelines.sft_main = real_main
    assert passed and pipelines.sft_main is real_main
    gpu_fds = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target == "/dev/kfd" or re.fullmatch(r"/dev/dri/(?:renderD|card)\d+", target):
            gpu_fds.append(fd.name)
    assert not gpu_fds
    print(json.dumps(dict(passed, zero_device_fds=True, modeled_forward_calls=0, optimizer_updates=0,
                          stopped_at_swift_sft_entry=True), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
