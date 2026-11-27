"""Check the real Swift dispatcher and CPU TorchElastic log placement without loading a model."""

from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from elastic_argv import Dispatch, SFT_PATH, SFT_SHA, SWIFT_CLI_SHA, file_pin, log_root, pin_dispatch


def main(root, request):
    assert socket.gethostname().split(".", 1)[0] == "f13r1n12" and os.environ["SLURM_JOB_ID"] == "1103606"
    step = os.environ["SLURM_STEP_ID"]
    assert step.isdecimal() and os.environ.get("SLURM_STEP_GPUS", "") in ("", "(null)", "none")
    assert all(os.environ.get(k) == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    short = root / "short-tmp"
    short.mkdir(mode=0o700)
    (short / ".scope").write_text(request + "\n", encoding="ascii")
    os.environ.update(TMPDIR=str(short), TEMP=str(short), TMP=str(short),
                      NPROC_PER_NODE="8", MASTER_PORT="32143", MASTER_ADDR="127.0.0.1")

    import torch
    from torch.distributed.run import get_args_parser
    from swift.cli.main import cli_main

    assert torch.cuda.device_count() == 0
    body = file_pin(Path(cli_main.__globals__["__file__"]), 3264, SWIFT_CLI_SHA)
    pin_dispatch(body, (root / "source/SWIFT_DISPATCH_WITNESS.json").read_bytes())
    file_pin(SFT_PATH, 583, SFT_SHA)
    private = log_root(root, short)
    seen = []

    def capture(argv):
        parser = get_args_parser()
        options = parser.parse_args(argv[3:])
        assert options.log_dir == str(private) and str(options.nproc_per_node) == "8"
        assert options.training_script == SFT_PATH and options.training_script_args == ["--help"]
        seen.append(sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest())
        return SimpleNamespace(returncode=0)

    namespace = cli_main.__globals__
    previous = namespace["subprocess"], sys.argv
    namespace["subprocess"], sys.argv = SimpleNamespace(run=capture), ["swift_entry.py", "sft", "--help"]
    try:
        Dispatch(cli_main, private, request, 32143).invoke()
    finally:
        namespace["subprocess"], sys.argv = previous
    assert len(seen) == 1
    intent = json.loads((root / "native-torchelastic-intent.json").read_bytes())
    assert seen[0] == intent["argv_SHA256"] and intent["log_dir_is_sole_change"] and intent["rank_count"] == 8

    child_source = root / "source/elastic_child.py"
    native = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
              "--nproc-per-node", "1", "--log-dir", str(private), "--redirects", "3", str(child_source), str(root)]
    run = subprocess.run(native, stdin=subprocess.DEVNULL, capture_output=True, timeout=55)
    assert run.returncode == 0 and (root / "elastic-child.json").is_file()
    children = tuple(private.iterdir())
    assert children and any(p.name == "stdout.log" for p in private.rglob("stdout.log"))
    assert {p.name for p in short.iterdir()} == {".scope"}
    result = dict(request=request, actual_step="1103606." + step, actual_GPU=0, actual_training_updates=0,
                  model_loaded=False, actual_swift_dispatch_parser_calls=len(seen), production_rank_argv_validated=8,
                  actual_native_fixture_ranks=1, actual_native_fixture_log_parents=len(children),
                  short_tmp_marker_only=True, escaped_logs_inside_short_root=0,
                  exact_swift_dispatcher_SHA256=SWIFT_CLI_SHA, actual_canonical_train_ranks_launched=0, passed=True)
    with (root / "result.json").open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
