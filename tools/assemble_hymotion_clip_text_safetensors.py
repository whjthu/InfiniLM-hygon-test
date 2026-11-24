"""Assemble the exact official CLIP text-only safetensors file from three authenticated ranges."""

from hashlib import sha256
import json
import time

from inspect_hymotion_clip_text_weight_ranges import decode, summarize


CHUNK = 512 * 1024


def projection(header, full_size, expected_header, retained_end, expected_intervals, expected_count):
    if len(header) < 8 or sha256(header).hexdigest() != expected_header:
        raise ValueError("official checkpoint header is unbound")
    original_header_size = int.from_bytes(header[:8], "little")
    if original_header_size != len(header) - 8:
        raise ValueError("official checkpoint header extent differs")
    _, entries = decode(header[8:], full_size - len(header), len(header))
    summary = summarize(entries, retained_end)
    if summary["official_file_intervals"] != expected_intervals or len(entries) != expected_count:
        raise ValueError("actual CLIP text graph intervals or count differ")
    text_tensors, cursor = {}, 0
    for name, row, begin, end in entries:
        length = end - begin
        text_tensors[name] = dict(dtype=row["dtype"], shape=row["shape"], data_offsets=[cursor, cursor + length])
        cursor += length
    header = json.dumps(dict(__metadata__={"format": "pt"}, **text_tensors),
                        sort_keys=True, separators=(",", ":")).encode()
    padding = (-len(header)) % 8
    header += b" " * padding
    return len(header).to_bytes(8, "little") + header, cursor


def copy_checked(output, source, source_offset, count, expected, deadline):
    if source_offset is not None:
        source.seek(source_offset)
    remaining = count
    digest = sha256()
    while remaining:
        if time.monotonic() >= deadline:
            raise TimeoutError("original CLIP range assembly budget expired")
        block = source.read(min(CHUNK, remaining))
        if not block or len(block) > remaining:
            raise EOFError("authenticated CLIP range was truncated")
        if output.write(block) != len(block):
            raise OSError("CLIP derivative write was truncated")
        digest.update(block)
        remaining -= len(block)
    if digest.hexdigest() != expected:
        raise ValueError("authenticated CLIP tensor range digest differs")


def body_copy(output, retained, tail, record, deadline):
    for index, entry in enumerate(record):
        (begin, end), length = entry["interval"], entry["bytes"]
        if end - begin != length or not 0 <= begin < end:
            raise ValueError("invalid source or destination CLIP range")
        owner, offset = (tail, None) if index == len(record) - 1 else (retained, begin)
        copy_checked(output, owner, offset, length, entry["SHA256"], deadline)
    if tail.read(1):
        raise ValueError("CLIP tail input exceeded its authenticated extent")


def assemble(output, retained, tail, header, claim, deadline):
    compacted, bytes_expected = projection(header, claim["official_bytes"], claim["official_header_SHA256"],
                                           claim["retained_end"], claim["official_intervals"], claim["text_tensor_count"])
    ranges = claim["source_ranges"]
    expected = [[claim["official_intervals"][0][0], claim["official_intervals"][0][1]],
                [claim["official_intervals"][1][0], claim["retained_end"]],
                [claim["retained_end"], claim["official_intervals"][1][1]]]
    if [entry["interval"] for entry in ranges] != expected or sum(entry["bytes"] for entry in ranges) != bytes_expected:
        raise ValueError("expected the precise retained prefix and missing tail inside the official text-only layout")
    if output.write(compacted) != len(compacted):
        raise OSError("CLIP derivative header write was truncated")
    body_copy(output, retained, tail, ranges, deadline)
    return dict(text_tensors=claim["text_tensor_count"], tensor_bytes=bytes_expected, file_bytes=len(compacted) + bytes_expected,
                all_source_range_SHA256_matched=True, vision_weights_included=False, whole_official_checkpoint_verified=False)
