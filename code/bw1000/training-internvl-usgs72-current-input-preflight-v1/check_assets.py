"""Read the original USGS72 inputs and rank-zero trainables inside the current T03 image."""

from hashlib import sha256
import json
import os
import socket
import stat
import time

import native
from scope import DATASET, INITIAL, INITIAL_BYTES, INITIAL_SHA, NAMED_SHA, need, pin


MODEL_DATASET_SHA = "92e3bd193f395f6cd8108f62f86f2e6368619e6210cc3d14997953bd7ff0d538"


def file_stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def initial_payload(torch, manifest, fingerprint):
    need(INITIAL.is_absolute() and INITIAL.resolve(strict=True) == INITIAL and not INITIAL.is_symlink(),
         "canonical original archive required")
    fd = os.open(INITIAL, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size == INITIAL_BYTES, "initial archive extent differs")
        digest = sha256()
        for piece in iter(lambda: source.read(1 << 20), b""):
            digest.update(piece)
        need(digest.hexdigest() == INITIAL_SHA, "initial archive SHA256 differs")
        source.seek(0)
        archive = torch.load(source, map_location="cpu", weights_only=True)
        need(file_stamp(os.fstat(source.fileno())) == file_stamp(before) == file_stamp(INITIAL.lstat()),
             "initial archive changed while reading")
    need(type(archive) is dict and set(archive) == {"parameters", "RNG", "manifest", "global_step"}
         and type(archive["global_step"]) is int and archive["global_step"] == 0
         and archive["manifest"] == manifest and type(archive["parameters"]) is dict,
         "initial archive schema or ordered manifest differs")
    parameters = archive["parameters"]
    need(len(manifest) == len(parameters) == 504 and {row["name"] for row in manifest} == set(parameters),
         "initial tensor inventory differs")
    named = []
    for row in manifest:
        name = row["name"]
        value = parameters[name]
        need(type(value) is torch.Tensor and value.device.type == "cpu" and value.layout == torch.strided
             and value.is_contiguous() and value.is_floating_point() and bool(torch.isfinite(value).all())
             and list(value.shape) == row["shape"] and value.numel() == row["numel"] and str(value.dtype) == row["dtype"],
             "initial tensor layout, shape, dtype or finite check differs")
        named.append((name, value))
    need(fingerprint(named) == NAMED_SHA, "initial named parameters differ")
    return sum(value.numel() for _, value in named)


def main():
    node, parent = "f13r1n14", "1103606"
    step = os.environ["SLURM_STEP_ID"]
    need(socket.gethostname().split(".")[0] == node and os.environ["SLURM_JOB_ID"] == parent
         and step.isdecimal() and os.environ.get("SLURM_STEP_GPUS") in (None, "", "none", "N/A", "(null)")
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "expected zero-GRES current T03 node")
    started = time.monotonic()
    data = pin(DATASET, expected=MODEL_DATASET_SHA, extent=53958)
    rows = data.splitlines()
    need(len(rows) == 72 and all(type(json.loads(row)) is dict for row in rows), "72 complete canonical data rows required")
    binding = native.binding()
    ordinals = sorted(map(int, binding["domain_source_rows"]))
    need(ordinals == list(range(1, 73)) and binding["initial_trainables_SHA256"] == NAMED_SHA,
         "domain membership or initialization scope differs")

    import torch
    from vlm.runtime.initial_trainables import named_parameter_sha256

    torch.set_num_threads(1)
    runtime = native.runtime_identity()
    need(runtime["torch_device_initialized"] is False, "runtime initialized accelerator")
    initial_elements = initial_payload(torch, binding["trainable_manifest"], named_parameter_sha256)
    need(initial_elements == 43646976 and not torch.cuda.is_initialized(), "unexpected initialization extent or device")
    result = {"node": node, "parent": parent, "step": f"{parent}.{step}", "target_dataset_SHA256": MODEL_DATASET_SHA,
              "training_rows_verified": 72, "original_native_signature_members": len(ordinals),
              "domain_contract_members_verified": 14, "initial_archive_SHA256": INITIAL_SHA,
              "initial_named_SHA256": NAMED_SHA, "initial_archive_bytes": INITIAL_BYTES,
              "initial_tensor_count": 504, "initial_element_count": initial_elements,
              "runtime": runtime, "observed_epoch": time.time(), "work_seconds": time.monotonic() - started,
              "torch_accelerator_initialized": False, "base_weight_body_bytes": 0,
              "actual_GPU_steps": 0, "actual_optimizer_updates": 0, "RNG_restored": False}
    print("RESULT_JSON " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
