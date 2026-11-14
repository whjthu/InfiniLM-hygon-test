"""Recheck the original shard and hand bytes to the typed owner on this host."""
import ctypes as ct
import hashlib
import json
import os
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(stream):
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_verified(model, binding, rank):
    if type(rank) is not int or rank not in (0, 8):
        raise ValueError("selected-value measurement requires rank 0 or 8")
    if sha((model / "config.json").read_bytes()) != binding["config_SHA256"]:
        raise ValueError("checkpoint config changed")
    index_bytes = (model / "model.safetensors.index.json").read_bytes()
    if sha(index_bytes) != binding["index_SHA256"]:
        raise ValueError("checkpoint index changed")
    index = json.loads(index_bytes)["weight_map"]
    shard = binding["shard"]
    prefix = "model.layers.0.self_attn.kv_b_proj."
    expected = ((prefix + "weight", rank, "F8_E4M3", [1792, 512], 917504),
                (prefix + "weight_scale_inv", None, "F32", [224, 4], 3584))
    if any(index[key] != shard for key, *_ in expected):
        raise ValueError("checkpoint shard binding changed")
    payloads = []
    with (model / shard).open("rb") as stream:
        before = identity(stream)
        if before[2] != binding["shard_bytes"]:
            raise ValueError("checkpoint shard size changed")
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size != binding["header_bytes"] or not 0 < header_size <= (16 << 20):
            raise ValueError("checkpoint header size changed")
        header = stream.read(header_size)
        if len(header) != header_size or sha(header) != binding["header_SHA256"]:
            raise ValueError("checkpoint header hash changed")
        tensors = json.loads(header)
        for key, owner, dtype, shape, size in expected:
            pins = [r for r in binding["ranges"] if r["name"] == key and r["rank"] == owner]
            if len(pins) != 1:
                raise ValueError("missing or ambiguous checkpoint range")
            pin = pins[0]
            tensor = tensors[key]
            start, end = tensor["data_offsets"]
            offset = 8 + header_size + start + (owner or 0)*size
            global_shape = [28672, 512] if owner is not None else shape
            global_size = size*16 if owner is not None else size
            if (pin["absolute_offset"] != offset or pin["bytes"] != size
                    or pin["shape"] != shape or pin["dtype"] != dtype
                    or tensor["dtype"] != dtype or tensor["shape"] != global_shape
                    or end-start != global_size or not 0 <= start < end <= before[2]-8-header_size):
                raise ValueError("checkpoint range geometry changed")
            stream.seek(offset)
            raw = stream.read(size)
            if len(raw) != size or sha(raw) != pin["SHA256"]:
                raise ValueError("checkpoint range payload hash changed")
            payloads.append(raw)
        if identity(stream) != before:
            raise ValueError("checkpoint changed while reading pinned ranges")
    return tuple(payloads)


def load_owner(lib, context, model, binding, rank):
    weight, scale = read_verified(model, binding, rank)
    lib.glm_selected_device_load.argtypes = [
        ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t
    ]
    lib.glm_selected_device_load.restype = ct.c_int
    result = lib.glm_selected_device_load(context, weight, len(weight), scale, len(scale))
    if result != 0:
        raise RuntimeError(f"selected-value owner load failed with {result}")
