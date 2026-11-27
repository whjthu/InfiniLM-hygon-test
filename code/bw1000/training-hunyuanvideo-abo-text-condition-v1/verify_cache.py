"""Independently read the candidate cache on the cluster without importing PyTorch."""

import ast
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import struct
from typing import Any
from zipfile import ZipFile


EXPECTED_LAYOUT = {
    "latents": ("FloatStorage", (1, 32, 3, 30, 53), 4, "torch.float32"),
    "text_emb": ("HalfStorage", (1, 300, 3584), 2, "torch.float16"),
    "text_mask": ("LongStorage", (1, 300), 8, "torch.int64"),
    "byt5_text_states": ("BFloat16Storage", (1, 256, 1472), 2, "torch.bfloat16"),
    "byt5_text_mask": ("LongStorage", (1, 256), 8, "torch.int64"),
}
EXPECTED_METADATA = {"frames": int, "height": int, "width": int, "prompt": str, "seed": int, "source_image": str}
VAE_SHA = "854a0f39980ac592423b9774a3e198dc29eaf57082b777b9c6735f81c64b868e"
CAPTION_SHA = "9cf1b347c692c3b667a10d92d00eb10bcd0f669182364c2482fb7cf910b078ee"
BACKEND_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def hex_sha(data):
    return sha256(data).hexdigest()


def read_bounded(path, maximum):
    require(not path.is_symlink(), "NO_OUTPUT_SYMLINK")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, "BOUNDED_OUTPUT_READ")
    return raw


def contiguous(shape):
    return tuple(math.prod(shape[i+1:]) for i in range(len(shape)))


def framework_validator(path):
    raw = read_bounded(path, 1 << 20)
    require(hex_sha(raw) == BACKEND_SHA, "EXACT_DEPLOYED_COMMON_FRAMEWORK_BACKEND")
    tree = ast.parse(raw, str(path))
    validators = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_hunyuanvideo_cache"]
    constants = [n.value for n in tree.body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQUIRED_CACHE_SHAPES" for t in n.targets))
                 or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "REQUIRED_CACHE_SHAPES")]
    require(len(validators) == len(constants) == 1, "ONE_CANONICAL_CACHE_VALIDATOR_AND_SHAPE_TABLE")
    namespace = {"REQUIRED_CACHE_SHAPES": ast.literal_eval(constants[0]), "Any": Any}
    method = ast.Module(body=validators, type_ignores=[])
    exec(compile(method, str(path), "exec"), namespace)
    return namespace["validate_hunyuanvideo_cache"]


def verify_original_views(path, shape_reader, expected):
    original = read_bounded(path, 8 << 20)
    require(len(original) == 4296921
            and hex_sha(original) == "ae18b701a130899f66da7c736b913d547cea14e1c0b4d6087b780ff4ac570c2b",
            "EXACT_PRESERVED_ORIGINAL_ARCHIVE")
    with ZipFile(io.BytesIO(original)) as archive:
        manifests = [item for item in archive.infolist() if item.filename.endswith("/data.pkl")]
        require(len(manifests) == 1 and manifests[0].file_size <= 32768, "ORIGINAL_BOUNDED_DESCRIPTOR")
        descriptors = shape_reader(io.BytesIO(archive.read(manifests[0]))).load()
        prefix = manifests[0].filename.rsplit("/", 1)[0]
        for name, (storage, shape, width, _) in EXPECTED_LAYOUT.items():
            t = descriptors[name]
            require(t.shape == shape and t.stride[1:] == contiguous(shape)[1:] and shape[0] == 1
                    and t.storage.kind == storage, "ORIGINAL_VIEW_LAYOUT")
            member = archive.getinfo(prefix + "/data/" + t.storage.key)
            require(member.file_size == width * t.storage.elements, "ORIGINAL_STORAGE_EXTENT")
            start = t.offset * width
            end = start + width * math.prod(shape)
            payload = archive.read(member)[start:end]
            require(len(payload) == end - start and hex_sha(payload) == expected[name]["SHA256"],
                    "BITWISE_EQUAL_ORIGINAL_SELECTED_VIEW")


def verify(root, accounting, shape_reader, deployed_backend, original_archive=None):
    require(len(accounting) == 9 and accounting[2:4] == ["COMPLETED", "0:0"]
            and accounting[5] == "f13r3n14" and accounting[6] in
            ("cpu=12,mem=112G,node=1", "cpu=12,mem=114688M,node=1",
             "cpu=2,mem=4G,node=1", "cpu=2,mem=4096M,node=1"),
            "EXACT_TERMINAL_ZERO_GRES_ACCOUNTING")
    out = root / "cache-output"
    terminal_raw = read_bounded(out / "terminal.json", 16_384)
    report = json.loads(terminal_raw)
    require(report["actual_step"] == accounting[0] and report["request"] == root.name
            and report["native_candidate_cache_rows"] == 1
            and report["actual_GPU"] == report["actual_optimizer_updates"] == report["qualified_training_rows"] == 0,
            "ORIGINAL_WORKER_ATTESTATION")
    repacked = report.get("recovery_method") == "storage_clone_of_verified_original_views"
    if repacked:
        require(report["actual_native_text_forwards"] == 0 and report["upstream_native_text_forwards"] == 1
                and report["recovered_from_original_step"] == "1103606.4282"
                and report["original_serialized_cache"] == {
                    "bytes": 4296921, "SHA256": "ae18b701a130899f66da7c736b913d547cea14e1c0b4d6087b780ff4ac570c2b"},
                "EXACT_RECOVERED_TEXT_FORWARD_PROVENANCE")
        require(original_archive is not None, "EXACT_ORIGINAL_VIEW_COMPARISON_REQUIRED")
    else:
        require(report["actual_native_text_forwards"] == 1, "EXACT_DIRECT_TEXT_FORWARD_PROVENANCE")
    cache = read_bounded(out / "sample.private.pt", 4 << 20)
    require(report["pt_cache"] == {"bytes": len(cache), "SHA256": hex_sha(cache)}, "EXACT_SERIALIZED_CACHE")
    with ZipFile(io.BytesIO(cache)) as z:
        names = z.namelist()
        descriptors = [item for item in z.infolist() if item.filename.endswith("/data.pkl")]
        require(len(names) == len(set(names)) and len(descriptors) == 1 and descriptors[0].file_size <= 32768,
                "ONE_UNIQUE_BOUNDED_TORCH_DIRECTORY")
        descriptor = z.read(descriptors[0])
        prefix = descriptors[0].filename.rsplit("/", 1)[0]
        order = z.getinfo(prefix + "/byteorder")
        require(order.file_size == 6 and z.read(order) == b"little", "LITTLE_ENDIAN_TORCH_STORAGE")
        fields = shape_reader(io.BytesIO(descriptor)).load()
        require(set(fields) == set(EXPECTED_LAYOUT) | set(EXPECTED_METADATA), "EXACT_11_NATIVE_FIELDS")
        require(all(type(fields[name]) is kind for name, kind in EXPECTED_METADATA.items())
                and [fields[name] for name in ("frames", "height", "width", "seed")] == [9, 480, 848, 20261127]
                and hex_sha(fields["prompt"].encode()) == CAPTION_SHA == report["caption_sha256"]
                and Path(fields["source_image"]).name == "frame-00.private.png", "FROZEN_METADATA_AND_CAPTION")
        framework_validator(deployed_backend)(fields)
        members = {}
        digests = {}
        keys = set()
        for name, (storage, shape, width, dtype) in EXPECTED_LAYOUT.items():
            tensor = fields[name]
            require(tensor.shape == shape and tensor.stride == contiguous(shape) and tensor.offset == 0
                    and tensor.storage.kind == storage and tensor.storage.elements == math.prod(shape)
                    and tensor.storage.key not in keys, "INDEPENDENT_CACHE_LAYOUT_STRIDE_OR_STORAGE")
            keys.add(tensor.storage.key)
            info = z.getinfo(prefix + "/data/" + tensor.storage.key)
            require(info.file_size == width * tensor.storage.elements, "INDEPENDENT_CACHE_EXACT_PAYLOAD_SIZE")
            data = z.read(info)
            members[name] = data
            digests[name] = dict(shape=list(shape), dtype=dtype, SHA256=hex_sha(data))
        require(digests == report["native_fields"] and digests["latents"]["SHA256"] == VAE_SHA,
                "INDEPENDENT_TENSOR_DIGESTS_AND_SEALED_VAE")
    if repacked:
        verify_original_views(original_archive, shape_reader, digests)
    mask = [number for number, in struct.iter_unpack("<q", members["text_mask"])]
    require(set(mask) == {0, 1} and sum(mask) == 19 and not any(members["byt5_text_states"]) and
            not any(members["byt5_text_mask"]), "SAME_19_ACTIVE_SLOTS_AND_NO_GLYPH")
    summary = {}
    for name, fmt in (("latents", "<f"), ("text_emb", "<e")):
        lower, upper = math.inf, -math.inf
        for value, in struct.iter_unpack(fmt, members[name]):
            require(math.isfinite(value), "CACHE_NONFINITE")
            lower, upper = min(lower, value), max(upper, value)
        require(lower < upper, "CACHE_EMPTY_OR_CONSTANT")
        summary[name] = dict(finite=True, nonconstant=True)
    return dict(request=root.name, accounting=accounting, original_terminal_SHA256=hex_sha(terminal_raw),
                verified_cache=report["pt_cache"], tensor_digests=digests, verified_shapes=5, metadata_fields=6,
                descriptor_SHA256=hex_sha(descriptor), text_active_slots=sum(mask), numeric=summary,
                common_framework_backend_SHA256=BACKEND_SHA, common_framework_cache_metadata_validated=True,
                candidate_training_cache_rows=1, training_updates=0, qualified_training_rows=0,
                formal_delivery_increment=0, actual_GPU=0, recovered_without_new_forward=repacked,
                original_view_values_bitwise_equal=repacked,
                original_worker_model_load_seconds=report.get("upstream_model_load_seconds", report.get("model_load_seconds")),
                original_worker_encode_seconds=report.get("encode_and_validate_seconds"),
                original_worker_wall_seconds=report["wall_seconds"], original_worker_model_class=report["model_class"])
