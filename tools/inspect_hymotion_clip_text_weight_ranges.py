"""Measure the official CLIP-L text-weight ranges without opening tensor values."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
from urllib.request import Request, build_opener

from probe_hymotion_official_clip_range import BASE, CHECKPOINT_SIZE, OfficialRedirects, REVISION


HEADER_BYTES = 71848
PREFIX_SHA256 = "27b2cbb669999f71cedcb139f695fb67300d71c7952f685de534d6ef9c5d1402"
RETAINED_BYTES = 472383488
WIDTH = {"BOOL": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "U16": 2, "I16": 2,
         "F32": 4, "U32": 4, "I32": 4, "F64": 8, "U64": 8, "I64": 8}


def distinct(keys):
    value = {}
    for key, member in keys:
        if key in value:
            raise ValueError("duplicate safetensors field")
        value[key] = member
    return value


def decode(raw, remaining, payload_start):
    table = json.loads(raw, object_pairs_hook=distinct)
    metadata = table.pop("__metadata__", {})
    if not 1 <= len(table) < 1000 or not isinstance(metadata, dict):
        raise ValueError("unexpected official tensor table")
    full = []
    texts = []
    for name, row in table.items():
        if set(row) != {"dtype", "shape", "data_offsets"} or row["dtype"] not in WIDTH:
            raise ValueError("unexpected official tensor representation")
        begin, end = row["data_offsets"]
        shape = row["shape"]
        if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= remaining
                or not isinstance(shape, list) or len(shape) > 12
                or any(type(x) is not int or x <= 0 for x in shape)
                or math.prod(shape) * WIDTH[row["dtype"]] != end - begin):
            raise ValueError("tensor byte interval differs from its exact shape and dtype")
        full.append((begin, end))
        if name.startswith("text_model."):
            texts.append((name, row, begin + payload_start, end + payload_start))
    ordered = sorted(full)
    if (ordered[0][0] != 0 or ordered[-1][1] != remaining
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))):
        raise ValueError("official tensor ranges overlap or leave an unaccounted gap")
    if not texts:
        raise ValueError("no CLIP text-model tensors in the official checkpoint")
    return table, sorted(texts, key=lambda x: x[2])


def summarize(text, retained):
    intervals = []
    for _, _, begin, end in text:
        if intervals and intervals[-1][1] == begin:
            intervals[-1][1] = end
        else:
            intervals.append([begin, end])
    size = sum(end - begin for begin, end in intervals)
    retained_text_bytes = sum(max(0, min(end, retained) - begin) for begin, end in intervals)
    return dict(text_tensors=len(text), official_file_intervals=intervals,
                official_text_tensor_bytes=size, range_intersection_with_recorded_partial_bytes=retained_text_bytes,
                missing_text_tensor_bytes=size-retained_text_bytes, partial_content_sha_verified=False)


def request_header(opener):
    end = 7 + HEADER_BYTES
    request = Request(BASE + "model.safetensors", headers={
        "Range": f"bytes=0-{end}", "Accept-Encoding": "identity"})
    with opener.open(request, timeout=12) as response:
        exact = f"bytes 0-{end}/{CHECKPOINT_SIZE}"
        if (response.status != 206 or response.headers.get("Content-Range") != exact
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("publisher response did not honor the exact header-only request")
        raw = response.read(end + 2)
    if (len(raw) != end + 1 or sha256(raw[:8]).hexdigest() != PREFIX_SHA256
            or int.from_bytes(raw[:8], "little") != HEADER_BYTES):
        raise ValueError("header length does not match the original publisher prefix")
    return raw


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    header = request_header(build_opener(OfficialRedirects()))
    payload = 8 + HEADER_BYTES
    table, text = decode(header[8:], CHECKPOINT_SIZE - payload, payload)
    evidence = dict(revision=REVISION, official_checkpoint_bytes=CHECKPOINT_SIZE,
                    header_read_bytes=len(header), header_SHA256=sha256(header).hexdigest(),
                    official_tensor_count=len(table), checkpoint_tensor_values_read=0,
                    retained_model_file_bytes=RETAINED_BYTES,
                    text_tensor_metadata_SHA256=sha256(json.dumps([(n, p) for n, p, _, _ in text],
                                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    true_installation_updated=False,gpu=0,encoder_rows=0,training_READY=False,
                    **summarize(text, RETAINED_BYTES))
    with args.output.open("xb") as handle:
        handle.write((json.dumps(evidence, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
