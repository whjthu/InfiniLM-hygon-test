"""Passively inspect the gfx936 images of the two original RoPE width objects."""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess


READELF = Path("/opt/dtk/llvm/bin/llvm-readelf")
OBJDUMP = Path("/opt/dtk/aillvm/bin/llvm-objdump")
ELF64LE = b"\x7fELF\x02\x01"
MEMORY = re.compile(r"(?:global|flat|buffer)_(?:load|store)_[a-z0-9_]+\Z")
RESOURCE_FIELDS = (
    "sgpr_count", "vgpr_count", "agpr_count", "wavefront_size", "group_segment_fixed_size",
    "private_segment_fixed_size", "kernarg_segment_size", "sgpr_spill_count", "vgpr_spill_count",
)


def need(valid, message):
    if not valid:
        raise ValueError(message)


def pin(data):
    return {"bytes": len(data), "SHA256": sha256(data).hexdigest()}


def read(path, cap, expected=None):
    need(path.is_file() and path.resolve(strict=True) == path, "Expected a canonical regular file")
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    need(0 < len(data) <= cap, "Unexpected input byte count")
    need(expected is None or pin(data) == expected, "Frozen object identity differs")
    return data


def extract_image(data):
    need(data.startswith(ELF64LE) and struct.unpack_from("<H", data, 18)[0] == 62,
         "Expected the original 64-bit x86 host wrapper")
    images = []
    for marker in re.finditer(re.escape(ELF64LE), data):
        offset = marker.start()
        start = data[offset:]
        if len(start) < 64 or struct.unpack_from("<H", start, 18)[0] != 224:
            continue
        table_start = struct.unpack_from("<Q", start, 40)[0]
        entry_size, count = struct.unpack_from("<HH", start, 58)
        table_end = table_start + entry_size * count
        need(entry_size == 64 and 0 < count < 256 and 64 <= table_start < table_end <= len(start),
             "Invalid AMDGPU section table")
        end = table_end
        for index in range(count):
            _, kind, _, _, location, size, _, _, _, _ = struct.unpack_from(
                "<IIQQQQIIQQ", start, table_start + index * entry_size
            )
            if kind != 8:  # SHT_NOBITS has no file payload.
                need(location + size <= len(start), "AMDGPU section extends past the wrapper")
                end = max(end, location + size)
        images.append((offset, start[:end]))
    need(len(images) == 1, "Exactly one embedded AMDGPU image is required")
    return images[0]


def run_tool(program, args, work, label):
    result = subprocess.run([program, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=12)
    need(len(result.stdout) <= 1024 * 1024 and len(result.stderr) <= 16384, "Tool output exceeds its bound")
    with (work / (label + ".stdout")).open("xb") as out:
        out.write(result.stdout)
    with (work / (label + ".stderr")).open("xb") as out:
        out.write(result.stderr)
    need(result.returncode == 0, f"DTK tool failed: {label}")
    return result.stdout.decode("utf-8"), {"stdout": pin(result.stdout), "stderr": pin(result.stderr)}


def kernel_resources(notes):
    _, sep, body = notes.partition("amdhsa.kernels:")
    need(sep, "Missing AMDHSA kernel metadata")
    body = re.split(r"(?m)^amdhsa\.", body, maxsplit=1)[0]
    kernels = {}
    for block in re.split(r"(?m)^  - (?=\.)", body)[1:]:
        name = re.search(r"(?m)^(?:    )?\.name:\s+(\S+)", block)
        if not name or "inverse_output" not in name[1]:
            continue
        resources = {}
        for field in RESOURCE_FIELDS:
            value = re.search(r"(?m)^(?:    )?\." + field + r":\s+(\d+)\s*$", block)
            if value:
                resources[field] = int(value[1])
        kernels[name[1]] = resources
    need(kernels and all({"sgpr_count", "vgpr_count", "wavefront_size"} <= row.keys()
                         for row in kernels.values()), "Missing kernel register metadata")
    return kernels


def disassembly_counts(assembly, symbols):
    counts = {name: Counter() for name in symbols}
    active = None
    for line in assembly.splitlines():
        header = re.match(r"^\s*[0-9a-f]+ <([^<>]+)>:\s*$", line)
        if header and header[1] in counts:
            active = header[1]
        elif header and re.search(r"inverse_output|\.kd$", header[1]):
            active = None
        elif active:
            opcode = line.strip().split(maxsplit=1)
            if opcode and MEMORY.fullmatch(opcode[0]):
                counts[active][opcode[0]] += 1
    need(all(counts.values()), "Missing device memory instructions for an original kernel")
    return {name: dict(sorted(row.items())) for name, row in counts.items()}


def inspect(obj, expected, output):
    offset, image = extract_image(read(obj, 128 * 1024, expected))
    device_image = output / "gfx936.elf"
    with device_image.open("xb") as out:
        out.write(image)
    notes, note_pins = run_tool(READELF, ["--file-header", "--notes", device_image], output, "notes")
    dis, isa_pins = run_tool(OBJDUMP, ["-d", "--mcpu=gfx936", device_image], output, "isa")
    need(re.search(r"gfx936\b", notes, flags=re.I), "Original ELF header or notes do not identify gfx936")
    resources = kernel_resources(notes)
    counts = disassembly_counts(dis, resources)
    return {"object": expected, "elf_file_offset": offset, "elf_size_and_SHA256": pin(image),
            "ELF_flags": struct.unpack_from("<I", image, 48)[0],
            "kernels": {name: {"resources": row, "memory_instruction_counts": counts[name]}
                        for name, row in resources.items()},
            "notes": note_pins, "disassembly": isa_pins}


def main():
    scope = Path(__file__).resolve().parent
    claim = json.loads((scope / "CLAIM.json").read_bytes())
    node = socket.gethostname().split(".")[0]
    parent, step = (os.environ[name] for name in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    need(parent == claim["parent"] and node == claim["node"] and step.isdecimal(), "Unexpected CPU step")
    need(not os.environ.get("SLURM_STEP_GPUS") and all(os.environ.get(name) == "-1" for name in
         ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")), "Accelerator mask differs")
    output = Path("/audit-out")
    report = {"node": node, "step": parent + "." + step, "request": claim["request"],
              "actual_GPU_steps": 0, "runtime_calls": 0, "arms": {}, "passed": False}
    try:
        for width, row in claim["objects"].items():
            arm = output / ("pack" + width)
            arm.mkdir(mode=0o700)
            try:
                report["arms"][width] = inspect(Path("/refactor") / row["path"], row["pin"], arm)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                report["arms"][width] = {"error": type(error).__name__, "message": str(error)[:180]}
        report["passed"] = all("kernels" in arm for arm in report["arms"].values())
    finally:
        with (output / "original-result.json").open("x", encoding="utf-8") as out:
            json.dump(report, out, sort_keys=True, allow_nan=False)
            out.write("\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
