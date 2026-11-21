"""Bind relocated SDXL training data to a frozen common backend."""
import hashlib
import json
from pathlib import Path
import re
import stat


BINDING_SHA256 = "bc268682bdfad89dbd3a05a93214fdeffa7415ca3b3d9dc3c8804c9ec9f96c58"
BACKEND_SHA256 = {
    "1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7",
    "65814732b3d2eb4bec29198c2594bbfbd0e84ceb364592ac542a23c133c11ffb",
}
TRAINER_RELATIVE = "examples/text_to_image/train_text_to_image_lora_sdxl.py"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local_stat(path, kind):
    info = path.lstat()
    if not kind(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError("training input requires a regular file or directory")
    return info


def verify_file(path, pin):
    expected_size = pin["bytes"]
    if local_stat(path, stat.S_ISREG).st_size != expected_size:
        raise ValueError("training input size differs")
    with path.open("rb") as stream:
        raw = stream.read(expected_size + 1)
    if len(raw) != expected_size or digest(raw) != pin["SHA256"]:
        raise ValueError("training input bytes differ")


def bind_config(train_data_dir):
    """Verify a staged train-only directory before passing it to ImageFolder."""
    path = Path(train_data_dir)
    if not path.is_absolute():
        raise ValueError("training directory must be absolute")
    local_stat(path, stat.S_ISDIR)
    path = path.resolve(strict=True)
    raw = Path(__file__).with_name("DATA_BINDING.json").read_bytes()
    if digest(raw) != BINDING_SHA256:
        raise ValueError("training data binding differs")
    binding = json.loads(raw)
    observed = {file.name: file for file in path.iterdir()}
    expected = binding["train_images_by_filename_sha256"]
    metadata = observed.pop("metadata.jsonl", None)
    images = {digest(name.encode("utf-8")): file for name, file in observed.items()}
    if metadata is None or images.keys() != expected.keys():
        raise ValueError("training directory entries differ")
    verify_file(metadata, binding["train_metadata"])
    for name_sha, file in images.items():
        verify_file(file, expected[name_sha])
    return {"train_data_dir": str(path), **binding["path_independent_config"]}


def run_via_common_backend(
    backend, *, weights, output_dir, source_root, train_data_dir, trainer_sha256
):
    config = bind_config(train_data_dir)
    provider = Path(backend.run_sdxl_training.__code__.co_filename)
    if digest(provider.read_bytes()) not in BACKEND_SHA256:
        raise ValueError("common SDXL backend provider differs")
    if not isinstance(trainer_sha256, str) or not re.fullmatch("[0-9a-f]{64}", trainer_sha256):
        raise ValueError("a pinned deployed SDXL trainer SHA256 is required")
    source_root = Path(source_root)
    trainer = source_root / TRAINER_RELATIVE
    trainer_stat = local_stat(trainer, stat.S_ISREG)
    if not 0 < trainer_stat.st_size <= 1024 * 1024:
        raise ValueError("deployed SDXL trainer size is out of bounds")
    verify_file(trainer, {"bytes": trainer_stat.st_size, "SHA256": trainer_sha256})
    return backend.run_sdxl_training(Path(weights), config, Path(output_dir), source_root)
