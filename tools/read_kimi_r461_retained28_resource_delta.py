"""Compare passive compiler resource metadata for the two original Kimi score objects."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import runpy
import struct

from read_training_slurm_membership import allocated_dcu, completed_numbered_step


REPO = Path(__file__).resolve().parents[1]
ELF_AUDIT = REPO / "code/bw1000/GLM-C1-layer-observer-r474/routed-i128-cpu-artifact-audit-v1/audit_once.py"
ELF_AUDIT_SHA = "5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
ARMS = {
    "full-tree": ("e5ce43bacc53bc64c75a74005e5e249c", "Kimi461-output-residual-wave-score-gfx936-CPU-v1",
                  "f12r2n11", "d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6", 2),
    "retained28": ("7f4239bb5f8fd6dc21da6a248b5c7469", "Kimi461-output-residual-retained28-score-gfx936-CPU-v2",
                   "f12r2n10", "9d3efa3a83f6733a479b23f95bbf4dcadb94a88c7b1a01216f2b889b2d6caf04", 1),
}
RESOURCE_KEYS = ("vgpr_count", "sgpr_count", "private_segment_fixed_size", "group_segment_fixed_size",
                 "wavefront_size", "max_flat_workgroup_size")
MAX_RECORD = 2**20
MAX_OBJECT = 2**24


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, limit):
    with path.open("rb") as fd:
        raw = fd.read(limit + 1)
    need(0 < len(raw) <= limit, "missing or oversized input: " + path.name)
    return raw


def elf_tools():
    need(digest(read(ELF_AUDIT, MAX_RECORD)) == ELF_AUDIT_SHA, "frozen ELF reader differs")
    return runpy.run_path(str(ELF_AUDIT))


def decode_metadata(raw):
    """Decode the bounded MessagePack values used by kernel metadata."""
    cursor = 0

    def take(count):
        nonlocal cursor
        need(0 <= count <= len(raw) - cursor, "truncated AMDGPU metadata")
        start = cursor
        cursor += count
        return raw[start:cursor]

    def integer(count, signed=False):
        return int.from_bytes(take(count), "big", signed=signed)

    def value(depth):
        need(depth < 20, "AMDGPU metadata nesting exceeded")
        tag = integer(1)
        if tag < 0x80:
            return tag
        if tag >= 0xe0:
            return tag - 256
        if 0xa0 <= tag <= 0xbf:
            return take(tag - 0xa0).decode("utf-8")
        if tag in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[tag]
        if 0xcc <= tag <= 0xd3:
            signed = tag >= 0xd0
            return integer(1 << (tag - (0xd0 if signed else 0xcc)), signed)
        if 0xd9 <= tag <= 0xdb:
            return take(integer(1 << (tag - 0xd9))).decode("utf-8")
        if 0x90 <= tag <= 0x9f or tag in (0xdc, 0xdd):
            count = tag - 0x90 if tag < 0xa0 else integer(2 if tag == 0xdc else 4)
            need(count <= 4096, "AMDGPU metadata array exceeds bound")
            return [value(depth + 1) for _ in range(count)]
        if 0x80 <= tag <= 0x8f or tag in (0xde, 0xdf):
            count = tag - 0x80 if tag < 0x90 else integer(2 if tag == 0xde else 4)
            need(count <= 4096, "AMDGPU metadata map exceeds bound")
            data = {}
            for _ in range(count):
                key = value(depth + 1)
                need(isinstance(key, str) and key not in data, "invalid or repeated AMDGPU metadata key")
                data[key] = value(depth + 1)
            return data
        raise ValueError(f"unsupported AMDGPU metadata marker 0x{tag:02x}")

    result = value(0)
    need(cursor == len(raw) and isinstance(result, dict), "invalid AMDGPU metadata payload")
    return result


def score_kernel(image):
    table = struct.unpack_from("<Q", image, 40)[0]
    _, count = struct.unpack_from("<HH", image, 58)
    notes = []
    for index in range(count):
        section = struct.unpack_from("<IIQQQQIIQQ", image, table + index * 64)
        if section[1] != 7:  # SHT_NOTE
            continue
        offset, amount = section[4:6]
        end = offset + amount
        need(end <= len(image), "AMDGPU note section escaped the image")
        while offset < end:
            need(end - offset >= 12, "truncated AMDGPU note header")
            name_size, desc_size, kind = struct.unpack_from("<III", image, offset)
            name_at = offset + 12
            desc_at = name_at + ((name_size + 3) & ~3)
            offset = desc_at + ((desc_size + 3) & ~3)
            need(offset <= end, "AMDGPU note fields escaped the section")
            if kind == 32 and image[name_at:name_at + name_size] == b"AMDGPU\0":
                need(0 < desc_size <= MAX_RECORD, "AMDGPU note exceeds evidence bound")
                notes.append(decode_metadata(image[desc_at:desc_at + desc_size]))
    need(len(notes) == 1, "expected one uncompressed AMDGPU metadata note")
    note = notes[0]
    target = note.get("amdhsa.target")
    if target is not None:
        need(isinstance(target, str) and re.fullmatch(r"amdgcn-amd-amdhsa--gfx936(?::[^ ]+)?", target),
             "AMDGPU metadata target differs from gfx936")
    kernels = note.get("amdhsa.kernels")
    need(isinstance(kernels, list) and all(isinstance(kernel, dict) for kernel in kernels),
         "missing AMDGPU kernel descriptors")
    matches = [kernel for kernel in kernels if "scores_kernel" in kernel.get(".name", "")]
    need(len(matches) == 1, "ambiguous original score device kernel")
    kernel = matches[0]
    name = kernel[".name"]
    need(kernel[".symbol"] == name + ".kd", "metadata does not bind the score kernel descriptor")
    fields = {key: kernel["." + key] for key in RESOURCE_KEYS}
    need(all(type(v) is int and 0 <= v < 2**32 for v in fields.values())
         and fields["vgpr_count"] > 0 and fields["sgpr_count"] > 0 and fields["wavefront_size"] in (32, 64)
         and fields["max_flat_workgroup_size"] >= 256, "score kernel resource values differ")
    return dict(symbol=name, descriptor=kernel[".symbol"], target=target, **fields)


def compiler_object(arm, reconcile_raw, receipt_raw, object_raw):
    request, label, node, manifest, total = ARMS[arm]
    terminal, receipt = map(json.loads, (reconcile_raw, receipt_raw))
    row = terminal["row"]
    step = receipt["original_step"]
    verified = completed_numbered_step(terminal["columns"], row, step, scope=arm)
    tres = dict(field.split("=", 1) for field in verified["AllocTRES"].split(","))
    need(terminal["request"] == request and terminal["state"] == "COMPILE_VERIFIED"
         and terminal["independent_observation"] is True and terminal["columns"] == COLUMNS
         and terminal["observations"] == [row] and terminal["score_receipt_SHA256"] == digest(receipt_raw)
         and terminal["terminal_row_SHA256"] == digest(json.dumps(row, separators=(",", ":")).encode())
         and re.fullmatch(r"1103606\.[0-9]+", step) and verified["JobName"] == label and verified["NodeList"] == node
         and verified["ElapsedRaw"].isdecimal() and int(verified["ElapsedRaw"]) <= 300
         and tres.get("cpu") == "4" and tres.get("node") == "1" and tres.get("mem") in ("8G", "8192M")
         and allocated_dcu(verified["AllocTRES"]) == 0
         and all(value == "0" for key, value in tres.items() if key.startswith("gres/")),
         "original terminal compile scope differs: " + arm)
    need(receipt["manifest_SHA256"] == manifest and receipt["node"] == node
         and type(receipt["epoch"]) is float and math.isfinite(receipt["epoch"]) and receipt["epoch"] > 0
         and receipt["deadline"] == receipt["epoch"] + 300
         and receipt["planned_device_TUs"] == receipt["actual_device_TUs"] == total
         and receipt["actual_GPU_steps"] == receipt["runtime_calls"] == 0
         and all(type(receipt[key]) is int for key in
                 ("planned_device_TUs", "actual_device_TUs", "actual_GPU_steps", "runtime_calls"))
         and receipt["passed"] is receipt["completed"] is receipt["all_fresh_M_before_compile"] is True
         and receipt["device_qualified"] is False, "original worker receipt differs: " + arm)
    original, check = receipt["arms"][arm], terminal["arms"][arm]
    obj_pin = digest(object_raw)
    need(obj_pin == original["object_SHA256"] == check["object_SHA256"]
         and original["object"] == f"/refactor/candidates/{label}/{arm}/fused_scores.o"
         and original["exports"] == ["kimi_output_residual_scores_v1_launch"]
         and all(original[key] == check[key] for key in ("fresh_M_SHA256", "actual_MD_SHA256")),
         "original score object or dependency pins differ: " + arm)
    return dict(step=step, object_SHA256=obj_pin, score_receipt_SHA256=digest(receipt_raw),
                reconcile_SHA256=digest(reconcile_raw))


def compare(original, retained):
    helpers = elf_tools()
    result = {}
    for arm, raw in (("full-tree", original), ("retained28", retained)):
        identity = compiler_object(arm, *raw)
        object_raw = raw[2]
        helpers["host_elf"](object_raw)
        images = helpers["embedded_amdgpu"](object_raw)
        identity["resource_metadata"] = None
        try:
            need(len(images) == 1, "expected one bounded uncompressed AMDGPU image")
            [info] = images
            start = info["offset"]
            identity["device_image"] = info
            identity["resource_metadata"] = score_kernel(object_raw[start:start + info["bytes"]])
        except (ValueError, KeyError, TypeError, struct.error) as error:
            identity["metadata_unresolved"] = str(error)[:160]
        result[arm] = identity
    old, new = (result[arm]["resource_metadata"] for arm in ARMS)
    comparable = bool(old and new and old["target"] and old["target"] == new["target"]
                      and all(old[k] == new[k] for k in ("symbol", "descriptor", "wavefront_size")))
    delta = {key: new[key] - old[key] for key in RESOURCE_KEYS} if comparable else None
    identical_image = (result["full-tree"]["device_image"]["SHA256"] == result["retained28"]["device_image"]["SHA256"]
                       if comparable else None)
    return dict(arms=result, resource_delta=delta, actual_target_identity_verified=comparable,
                no_added_fixed_scratch=(delta["private_segment_fixed_size"] <= 0 if delta is not None else None),
                identical_device_images=identical_image,
                observed_input_reload_eliminated=False if identical_image else None,
                spill_instruction_delta=None, warmed_performance_passed=None,
                GPU_GO=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("baseline", "candidate"):
        for item in ("reconcile", "receipt", "object"):
            parser.add_argument(f"--{role}-{item}", type=Path, required=True)
    args = parser.parse_args()
    inputs = [[read(getattr(args, role + "_" + item), limit)
               for item, limit in (("reconcile", MAX_RECORD), ("receipt", MAX_RECORD), ("object", MAX_OBJECT))]
              for role in ("baseline", "candidate")]
    print(json.dumps(compare(*inputs), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
