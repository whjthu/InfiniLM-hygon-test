"""Enter native Swift OneVision training using only the original CLEVR64 training rows."""

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import stat
import sys
import time


HERE = Path(__file__).resolve().parent
RUNTIME = Path("/swift-runtime/site")
HELPERS = Path("/llava-helpers")
ASSETS = Path("/llava-train-only")
FRAMEWORK = Path("/llava-framework")
ORIGINAL = "849e72c9dde3d3e4a7bd40a4c1ebc700854af944fa6fbf86317f8447f55f7c9b"
NATIVE = "0e8bc1decb2f98037cd99499432146beebf6a5ac75c1466835e7a089b8603586"
IMAGE_BIND = "d00032b2f0e5eced239d87b2d3f3164c43f369bc3796b0a1180913f3a8950b8b"
CPU_CASES = {
    "4860.json": "b38a6efd40ae58d4dba233bc92bc5c5700e9330c7901cc6d6953826c72889ae2",
    "4719.json": "6457ef3edfd2df18aef8667354a947cdc9d6b1f36f261a30415f6748c31a2caa",
}
HELPER_PINS = {
    "probe.py": "446d4d0b27d9eff6488a6ebe07bce959fdf6e81b4f708e805c7c6cd237183aaf",
    "llava_recovered_native_cpu.py": "d94ddf5a68f9eeb2f3fe19396989fd87a2bd5df3bbfe0ab16ceb525f84e782ad",
    "cache_alias.py": "f843bf60d441c82813680f323796aaf9177df8256c8f25fada42fe5164ec9c94",
    "llava_target_eager_import.py": "fd998f6c192cf6cf8021ba37d34397894852d4fb505759b6e9b46441d4d12ba1",
}
CLI_SHA = "97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0"
CALLBACK = "llava_clevr_first2_native_v1"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def cli(original, model, dataset, output, callback):
    command = original["command"]
    need(command[1] == "sft", "T04_SWIFT_SFT")
    args = command[2:]
    for flag, value in {
        "--model": str(model), "--dataset": str(dataset), "--output_dir": str(output),
        "--callbacks": CALLBACK,
    }.items():
        need(args.count(flag) == 1, "EXACT_T04_CLI_ANCHOR")
        args[args.index(flag) + 1] = value
    i = args.index("--external_plugins")
    need(args[i + 3] == "--callbacks", "EXACT_T04_PLUGIN_ANCHOR")
    # OneVision has already been imported through its qualified package name.
    args[i + 1:i + 3] = [str(callback)]
    return args


def roster(document, source, binding, expected_train):
    need(set(document) == {"binding_sha256", "train_sha256", "rows"}
         and document["binding_sha256"] == IMAGE_BIND and document["train_sha256"] == expected_train
         and len(document["rows"]) == len(source) == len(binding["rows"]) == 64, "EXACT_64_CPU_TENSOR_BINDING")
    rows = {}
    for i, (observed, (_, content_sha), image) in enumerate(zip(document["rows"], source, binding["rows"]), 1):
        need(set(observed) == {"ordinal", "row_sha256", "image_sha256", "input_sha256"}
             and (observed["ordinal"], observed["row_sha256"], observed["image_sha256"])
             == (i, content_sha, image["sha256"]) and re.fullmatch(r"[0-9a-f]{64}", observed["input_sha256"])
             and observed["input_sha256"] not in rows, "EXACT_NATIVE_TENSOR_AND_SOURCE_ROW")
        rows[observed["input_sha256"]] = observed
    return rows


def inputs(probe, fingerprint_pin):
    for name, digest in CPU_CASES.items():
        report = json.loads(probe.read(HELPERS / name, 256 * 1024, digest))
        if name == "4860.json":
            need(report["step"] == "1103606.4860" and report["lora_manifest_exact"] is True
                 and report["target_membership_exact"] is True, "ORIGINAL_4860")
        else:
            check = report["ledgers"]["independent"]["response"]
            need(check["rows"] == check["unique_png"] == check["verified_source_occurrences"] == 64
                 and check["private_binding_SHA256"] == IMAGE_BIND, "ORIGINAL_4719")
    binding = json.loads(probe.read(HELPERS / "INPUTS.json", 128 * 1024, IMAGE_BIND))
    source = probe.source_rows(ASSETS)
    probe.check_binding(source, binding)
    need({p.name for p in ASSETS.iterdir()} == {"train.jsonl", "images"}
         and {p.name for p in (ASSETS / "images").iterdir()} == {Path(row["path"]).name for row in binding["rows"]},
         "TRAIN_ONLY_READONLY_VIEW")
    for path in (ASSETS, ASSETS / "train.jsonl", ASSETS / "images", HELPERS, FRAMEWORK, RUNTIME):
        need(path.resolve(strict=True) == path and os.statvfs(path).f_flag & os.ST_RDONLY, "READONLY_T04_ASSETS")
    for row in binding["rows"]:
        path = ASSETS / row["path"]
        need(os.statvfs(path).f_flag & os.ST_RDONLY
             and len(probe.read(path, row["bytes"], row["sha256"])) == row["bytes"], "EXACT_ORIGINAL_TRAIN_PNG")
    document = json.loads(probe.read(HELPERS / "TENSORS.json", 32 * 1024, fingerprint_pin))
    result = roster(document, source, binding, probe.TRAIN)
    transformed = []
    for row, _ in source:
        transformed.append(dict(messages=row["messages"], images=[str(ASSETS / row["images"][0])]))
    return result, b"".join(probe.wire(row) for row in transformed)


def run(root, model, request, fingerprint_pin):
    global EXPECTED_CLASS, CHECKPOINT_ASSIGNMENT
    sys.path[:0] = [str(HELPERS), str(FRAMEWORK), str(RUNTIME), "/vision", str(HERE)]
    # Hash the source before either helper can be imported.
    from hashlib import sha256
    for name, digest in HELPER_PINS.items():
        raw = (HELPERS / name).read_bytes()
        need(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == digest, "EXACT_READONLY_HELPER")
    import probe
    import llava_recovered_native_cpu as recovered
    import llava_target_eager_import as target_eager

    rank = int(os.environ["RANK"])
    owner = json.loads(probe.read(root / "OWNER.json", 8192, os.environ["LLAVA_OWNER_SHA256"]))
    epoch, deadline = owner["epoch"], owner["deadline"]
    job, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    need(job.isdecimal() and step.isdecimal() and owner["request"] == request
         and owner["step"] == job + "." + step and owner["GPU_GO"] is True
         and owner["fingerprint_sha256"] == fingerprint_pin and owner["gpu_count"] == 8
         and str(owner["port"]) == os.environ["MASTER_PORT"] and os.environ["MASTER_ADDR"] == "127.0.0.1"
         and owner["node"] == os.uname().nodename.split(".")[0] and 0 < deadline - epoch <= 1500
         and epoch <= time.time() < deadline and os.environ["WORLD_SIZE"] == "8"
         and int(os.environ["LOCAL_RANK"]) == rank and 0 <= rank < 8
         and re.search(rf"/job_{job}(?:\.scope)?/step_{step}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE), "AUTHORIZED_NUMBERED_RANK_SCOPE")
    need(root.resolve(strict=True) == root and root.stat().st_uid == os.geteuid()
         and stat.S_IMODE(root.stat().st_mode) == 0o700, "PRIVATE_TRAIN_OUTPUT")
    need(all(os.environ.get(key) == "1" for key in
             ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")), "OFFLINE_RUNTIME")
    _, raw = inputs(probe, fingerprint_pin)
    dataset = root / "train-bound.jsonl"
    if rank == 0:
        # Publish only a completely written file for the other seven ranks.
        temporary = root / "train-bound.part"
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, dataset)
        temporary.unlink()
    while not dataset.is_file():
        need(time.time() < min(deadline, epoch + 120), "TRAIN_DATASET_PUBLICATION_DEADLINE")
        time.sleep(.1)
    need(probe.read(dataset, len(raw), probe.sha(raw)) == raw, "SHARED_EXACT_TRAIN_ONLY_VIEW")
    model_receipt = recovered.pinned_document(HELPERS / "MODEL_RECEIPT.json", recovered.MODEL_RECEIPT_SHA)
    need(model.resolve(strict=True) == model and os.statvfs(model).f_flag & os.ST_RDONLY
         and {p.name for p in model.iterdir()} == set(model_receipt["files"])
         and model_receipt["verified_complete_shards"] == 4, "RECOVERED_MODEL18_READONLY")
    for name, pin in model_receipt["files"].items():
        path = model / name
        status = path.stat()
        need(path.resolve(strict=True) == path and os.statvfs(path).f_flag & os.ST_RDONLY
             and (status.st_size, status.st_ino, str(status.st_mtime_ns))
             == (pin["bytes"], pin["inode"], pin["mtime_ns"]), "RECOVERED_ORIGINAL_MODEL_FILE_IDENTITY")
    original = json.loads(probe.read(HELPERS / "original-launch.json", 64 * 1024, ORIGINAL))
    entry = RUNTIME / "swift/cli/sft.py"
    probe.read(entry, 583, CLI_SHA)
    for _ in probe.verified_files(HELPERS / "FRAMEWORK_FILES.json", probe.FRAMEWORK, FRAMEWORK, 393, nested=True):
        pass
    from vlm.plugins import llava_onevision
    plugin = FRAMEWORK / "vlm/plugins/llava_onevision.py"
    need(Path(llava_onevision.__file__) == plugin, "ORIGINAL_ONEVISION_PLUGIN")
    EXPECTED_CLASS = target_eager.preimport(model, llava_onevision, recovered)
    os.environ.update(LLAVA_RUN=str(root), LLAVA_REQUEST=request, LLAVA_EPOCH=str(epoch),
                      LLAVA_DEADLINE=str(deadline), LLAVA_TENSORS_SHA256=fingerprint_pin)
    sys.argv = [str(entry), *cli(original, model, dataset, root / "train", HERE / "callback.py")]
    sys.modules["train"] = sys.modules[__name__]
    with recovered.observe_checkpoint_assignment() as assignment:
        CHECKPOINT_ASSIGNMENT = assignment
        runpy.run_path(str(entry), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--fingerprint-sha256", required=True)
    args = parser.parse_args()
    need(args.run.is_absolute() and args.model.is_absolute() and re.fullmatch(r"[0-9a-f]{32}", args.request)
         and re.fullmatch(r"[0-9a-f]{64}", args.fingerprint_sha256), "EXACT_LAUNCH_ARGUMENTS")
    run(args.run, args.model, args.request, args.fingerprint_sha256)


if __name__ == "__main__":
    main()
