"""Qualify eight OneVision dynamic imports with the production rank-local cache."""

from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def qualify_rank(rank):
    sys.path[:0] = ["/tmp/patch", "/llava-helpers", "/llava-framework", "/swift-runtime/site", "/vision"]
    import train

    cache = train.bind_dynamic_module_cache(Path("/tmp"), rank)
    from transformers import AutoConfig
    from transformers.utils import hub
    from vlm.plugins import llava_onevision
    import llava_recovered_native_cpu as recovered
    import llava_target_eager_import as target_eager

    assert Path(hub.HF_MODULES_CACHE) == cache and cache.stat().st_mode & 0o777 == 0o700
    model = Path("/t04-model")
    native = target_eager.preimport(model, llava_onevision, recovered)
    config = type(AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True))
    assert config.__name__ == "Llavaonevision1_5Config"
    files = {}
    for cls in (config, native):
        path = Path(inspect.getfile(cls)).resolve()
        assert path.is_relative_to(cache)
        original = model / path.name
        raw = path.read_bytes()
        assert raw == original.read_bytes()
        files[cls.__name__] = sha256(raw).hexdigest()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        assert target != "/dev/kfd" and not re.fullmatch(r"/dev/dri/(?:renderD|card)\d+", target)
    print(json.dumps(dict(rank=rank, cache=cache.name, source_SHA256=files, no_GPU_descriptors=True),
                     sort_keys=True), flush=True)


def main():
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert os.environ[key] == "-1"
    if len(sys.argv) == 2:
        qualify_rank(int(sys.argv[1]))
        return
    assert len(sys.argv) == 1 and not os.environ.get("SLURM_STEP_GPUS") and not os.environ.get("SLURM_GPUS_ON_NODE")
    workers = []
    deadline = time.monotonic() + 145
    for rank in range(8):
        out = Path(f"/tmp/dynamic-import-{rank:02d}.out")
        err = Path(f"/tmp/dynamic-import-{rank:02d}.err")
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.Popen([sys.executable, "-B", __file__, str(rank)], stdout=stdout, stderr=stderr)
        workers.append((process, out, err))
    receipts = []
    try:
        for rank, (process, out, err) in enumerate(workers):
            code = process.wait(timeout=max(1, deadline - time.monotonic()))
            if code:
                raise RuntimeError(f"DYNAMIC_IMPORT_CPU_RANK_{rank:02d}_EXIT_{code}")
            records = out.read_bytes().splitlines()
            receipt = json.loads(records[-1])
            assert receipt["rank"] == rank and receipt["no_GPU_descriptors"]
            receipts.append(receipt)
    finally:
        for process, _, _ in workers:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
    assert len({record["cache"] for record in receipts}) == 8
    assert all(record["source_SHA256"] == receipts[0]["source_SHA256"] for record in receipts)
    print(json.dumps(dict(case="EIGHT_CONCURRENT_DYNAMIC_IMPORTS", ranks=len(receipts),
                          distinct_private_caches=8, source_SHA256=receipts[0]["source_SHA256"],
                          model_tensor_payloads_read=0, training_updates=0, actual_GPU_file_descriptors=0),
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
