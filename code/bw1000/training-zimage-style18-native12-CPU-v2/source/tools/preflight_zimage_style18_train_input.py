"""Check the frozen training split through the installed DiffSynth image dataset."""

import argparse
import csv
from hashlib import sha256
import importlib
import io
import json
import os
from pathlib import Path
import stat


REPO = Path(__file__).resolve().parents[1]
BINDING = REPO / "coordination/training/zimage-style18-target-train-inputs-v1.json"
BINDING_SHA = "a06c29c8af3d56ed8192c39f63063db82bf67a14b75ba1fb2913202155d1ea09"
MODULE = "diffsynth.core.data"
DATASET_SOURCES = {
    "unified_dataset": "b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f",
    "operators": "c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4",
}
TRAIN_COUNT = 12


def digest(raw):
    return sha256(raw).hexdigest()


def read_regular(path, ceiling):
    if path.resolve(strict=True) != path:
        raise ValueError("input uses a noncanonical or linked path")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= ceiling:
            raise ValueError("input exceeds its expected size or is irregular")
        raw = stream.read(ceiling + 1)
    after = path.stat()
    if len(raw) != before.st_size or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during verification")
    return raw


def read_pinned(path, pin):
    raw = read_regular(path, pin["bytes"])
    if len(raw) != pin["bytes"] or digest(raw) != pin["SHA256"]:
        raise ValueError("frozen input bytes differ")
    return raw


def load_binding():
    raw = read_regular(BINDING, 16_384)
    if digest(raw) != BINDING_SHA:
        raise ValueError("training input binding differs")
    binding = json.loads(raw)
    if binding["schema"] != "ZIMAGE_STYLE18_TRAIN_ONLY_CPU_INPUT_V1" or len(binding["rows"]) != TRAIN_COUNT:
        raise ValueError("training input scope differs")
    return binding


def inspect_inputs(train_dir, binding):
    if train_dir.name != "train" or train_dir.resolve(strict=True) != train_dir:
        raise ValueError("expected canonical train split directory")
    csv_path = train_dir / "metadata.csv"
    raw = read_pinned(csv_path, binding["CSV"])
    with io.StringIO(raw.decode("utf-8"), newline="") as handle:
        csv_reader = csv.DictReader(handle)
        if csv_reader.fieldnames != ["image", "prompt"]:
            raise ValueError("metadata columns differ")
        metadata = list(csv_reader)
    if len(metadata) != TRAIN_COUNT:
        raise ValueError("metadata row count differs")
    for ordinal, (row, expected) in enumerate(zip(metadata, binding["rows"], strict=True)):
        relative = f"images/{ordinal:03d}.png"
        if (
            expected["ordinal"] != ordinal or expected["path"] != relative or row["image"] != relative
            or not isinstance(row["prompt"], str) or digest(row["prompt"].encode("utf-8")) != expected["prompt_SHA256"]
        ):
            raise ValueError("frozen CSV row order, path or prompt differs")
        read_pinned(train_dir / relative, expected["image"])
    return csv_path


def check_native_entry(providers):
    package = providers / "diffsynth/core/data"
    for module, pin in DATASET_SOURCES.items():
        if digest(read_regular(package / (module + ".py"), 65_536)) != pin:
            raise ValueError("DiffSynth dataset provider source differs")
    importlib.invalidate_caches()
    dataset = importlib.import_module(MODULE + ".unified_dataset")
    operators = importlib.import_module(MODULE + ".operators")
    for module in (dataset, operators):
        expected = package / (module.__name__.rsplit(".", 1)[-1] + ".py")
        if Path(module.__file__).resolve(strict=True) != expected:
            raise ValueError("imported DiffSynth provider differs from the pinned deployment")
    if dataset.ImageCropAndResize is not operators.ImageCropAndResize:
        raise ValueError("dataset did not bind its original image operator")
    return dataset.UnifiedDataset, operators


def compare_samples(dataset, binding, image_type):
    if len(dataset) != TRAIN_COUNT:
        raise ValueError("native dataset count differs")
    rows = []
    for expected in binding["rows"]:
        ordinal = expected["ordinal"]
        result = dict(ordinal=ordinal, image_matches=False, prompt_matches=False)
        try:
            sample = dataset[ordinal]
            prompt = sample["prompt"]
            result["prompt_matches"] = isinstance(prompt, str) and digest(
                prompt.encode("utf-8")
            ) == expected["prompt_SHA256"]
            image = sample["image"]
            if isinstance(image, image_type) and image.mode == "RGB" and image.size == (256, 256):
                result["observed_RGB256_SHA256"] = digest(image.tobytes())
                result["image_matches"] = result["observed_RGB256_SHA256"] == expected["RGB256_SHA256"]
        except (KeyError, TypeError, OSError, ValueError, RuntimeError) as error:
            result["error_type"] = type(error).__name__
        rows.append(result)
    return rows


def inspect_native(train_dir, csv_path, binding, providers):
    masks = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    if any(os.environ.get(mask) != "-1" for mask in masks) or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("native check requires the zero-GRES, accelerator-hidden environment")
    entry, ops = check_native_entry(providers)
    if ops.torch.cuda.is_initialized():
        raise ValueError("unexpected initialized accelerator runtime")
    tvision = ops.torchvision
    functional = tvision.transforms.functional
    pil_backend = functional.F_pil
    versions = {
        "torch": str(ops.torch.__version__),
        "torchvision": str(tvision.__version__),
        "Pillow": str(ops.Image.__version__),
    }
    provider_files = {
        "torchvision.functional": digest(read_regular(Path(functional.__file__).resolve(strict=True), 128 * 1024)),
        "torchvision._functional_pil": digest(read_regular(Path(pil_backend.__file__).resolve(strict=True), 65_536)),
    }
    dataset = entry(
        base_path=str(train_dir), metadata_path=str(csv_path), data_file_keys=("image",),
        main_data_operator=entry.default_image_operator(
            base_path=str(train_dir), height=256, width=256, height_division_factor=16, width_division_factor=16,
        ),
    )
    rows = compare_samples(dataset, binding, ops.Image.Image)
    inspect_inputs(train_dir, binding)
    if ops.torch.cuda.is_initialized():
        raise ValueError("accelerator runtime initialized during the input check")
    return versions, provider_files, rows


def run(train_dir, providers=None):
    report = dict(input_binding_SHA256=BINDING_SHA, planned_train_rows=TRAIN_COUNT,
                  read_development_or_blind_rows=0, model_loads=0, optimizer_updates=0, training_ready=False)
    phase = "INPUT_BYTES"
    try:
        binding = load_binding()
        csv_path = inspect_inputs(train_dir, binding)
        report["verified_source_train_rows"] = TRAIN_COUNT
        if providers is None:
            return dict(report, state="TRAIN_INPUT_BYTES_VERIFIED", native_dataset_rows=0)
        phase = "NATIVE_DATASET"
        versions, files, rows = inspect_native(train_dir, csv_path, binding, providers)
        passed = sum(row["image_matches"] and row["prompt_matches"] for row in rows)
        return dict(report, state="NATIVE_TRAIN_INPUT_PASS" if passed == TRAIN_COUNT else "NATIVE_TRAIN_INPUT_DIFFERS",
                    native_dataset_rows=len(rows), passed=passed, failed=TRAIN_COUNT-passed,
                    versions=versions, native_provider_SHA256=files, rows=rows)
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, ImportError, AttributeError) as error:
        return dict(report, state="TRAIN_INPUT_CHECK_FAILED", phase=phase, error_type=type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs-only", action="store_true")
    mode.add_argument("--diffsynth-source", type=Path)
    args = parser.parse_args()
    result = run(args.train_dir, args.diffsynth_source)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(result["state"] not in ("TRAIN_INPUT_BYTES_VERIFIED", "NATIVE_TRAIN_INPUT_PASS"))


if __name__ == "__main__":
    raise SystemExit(main())
