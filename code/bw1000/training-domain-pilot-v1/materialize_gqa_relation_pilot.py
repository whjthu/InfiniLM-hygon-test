"""Materialize an identity-bound GQA training pilot from official JPEG members."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

from PIL import Image
from build_gqa_relation_train import build, deflate_parts, load_pinned, question_rows


URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
SIZE = 21_817_965_542
ETAG = '"5c55f4cb-51473bbe6"'
OLD = "gqa-official-eight-media-source-v1"
HOLDOUT = "gqa-original-four-train-smoke-v1/holdout-identities.json"
PINS = {
    OLD + "/receipt.json": "a023f6dd5e794cdd63bfcc3348a54e5c97d7db48f8d3597bee245a0856cb42ff",
    HOLDOUT: "22fe71b7205e3b19ebe087d4ff9a9f5829834e551971a5d1a49bef1e6de1db57",
    "receipt.json": "a9374b426f870e49018d4b8904f14e5c720fdd220b0595f1923037ac816dad65",
    "coco-vg-candidates-original-v1.json": "fca123196d7ae669b959cae6deceeba8123998f5290d61c6d33796a987df9f1e",
}
TRAIN_ROWS = 64
FETCH_LIMIT = 84
MAX_MEDIA_RANGES = 32 * 1024**2
MAX_JPEG_BYTES = 2 * 1024**2


def need(value, error):
    if not value:
        raise ValueError(error)


def fetch_once(offset, count):
    need(0 <= offset < offset + count <= SIZE, "range outside publisher archive")
    end = offset + count - 1
    request = Request(URL, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
                                    "If-Match": ETAG})
    with urlopen(request, timeout=17) as response:
        need(response.status == 206 and response.geturl() == URL
             and response.headers.get("Content-Range") == f"bytes {offset}-{end}/{SIZE}"
             and response.headers.get("ETag") == ETAG
             and response.headers.get("Content-Encoding", "identity") == "identity", "publisher archive identity differs")
        data = response.read(count + 1)
    need(len(data) == count, "publisher returned an incomplete range")
    return data


def fetch(offset, count):
    for attempt in range(3):
        try:
            return fetch_once(offset, count)
        except HTTPError as error:
            if attempt == 2 or error.code not in {429, 500, 502, 503, 504}:
                raise
            interval = error.headers.get("Retry-After", "")
            if interval.isdigit() and int(interval) > 30:
                raise
            time.sleep(max(int(interval) if interval.isdigit() else 0, 1 << attempt))


class ZipIndex:
    def __init__(self):
        self.position = 0
        self.bytes_read = 0

    def tell(self):
        return self.position

    def seek(self, offset, origin=0):
        next_position = {0: 0, 1: self.position, 2: SIZE}[origin] + offset
        need(0 <= next_position <= SIZE, "ZIP index seek outside archive")
        self.position = next_position
        return self.position

    def read(self, count=-1):
        remaining = SIZE - self.position
        count = remaining if count < 0 else min(count, remaining)
        if not count:
            return b""
        need(self.bytes_read + count <= 16 * 1024**2, "ZIP index exceeded its read bound")
        result = fetch(self.position, count)
        self.bytes_read += count
        self.position += count
        return result


def select_manifest(index, identities):
    members = {}
    for info in index.infolist():
        identity = info.filename.removeprefix("images/").removesuffix(".jpg")
        if identity not in identities or info.filename != f"images/{identity}.jpg":
            continue
        need(identity not in members and info.flag_bits & 1 == 0
             and info.compress_type in {ZIP_STORED, ZIP_DEFLATED}
             and 0 < info.compress_size <= MAX_JPEG_BYTES and 0 < info.file_size <= MAX_JPEG_BYTES,
             "ambiguous or oversize JPEG member")
        members[identity] = info
    need(identities == set(members), "publisher image member missing")
    return members


def read_jpeg(info):
    header = fetch(info.header_offset, 30)
    magic, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack("<4s5H3I2H", header)
    name = info.filename.encode("ascii")
    need(magic == b"PK\x03\x04" and flags == info.flag_bits and method == info.compress_type
         and name_size == len(name) and extra_size <= 8192, "local ZIP header differs")
    extent = name_size + extra_size + info.compress_size
    block = fetch(info.header_offset + len(header), extent)
    need(block[:name_size] == name, "local ZIP filename differs")
    data = block[name_size + extra_size:]
    if method == ZIP_DEFLATED:
        dec = zlib.decompressobj(-zlib.MAX_WBITS)
        data = dec.decompress(data, info.file_size + 1)
        need(dec.eof and not dec.unconsumed_tail and not dec.unused_data, "invalid bounded image deflate")
    need(len(data) == info.file_size and zlib.crc32(data) == info.CRC, "JPEG byte count or CRC differs")
    return data, len(header) + len(block)


def image_identity(data, dimensions):
    with Image.open(BytesIO(data)) as image:
        need(image.format == "JPEG" and list(image.size) == dimensions, "JPEG dimensions or format differ")
        image.load()
        pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    dhash = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    return sha256(data).hexdigest(), dhash, dimensions[0] / dimensions[1]


def near_duplicate(first, second):
    return first[0] == second[0] or (
        abs(first[2] / second[2] - 1) <= 0.02 and (first[1] ^ second[1]).bit_count() <= 4
    )


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def document(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def main(source, output, previous=None, previous_pin=None):
    inputs = {name: load_pinned(source / name, expected) for name, expected in PINS.items()}
    old_images = {str(row["gqa_image_id"]): row for row in inputs[OLD + "/receipt.json"]["images"]}
    holdout = inputs[HOLDOUT]
    candidates = inputs["coco-vg-candidates-original-v1.json"]["candidates"]
    flattened = {str(row["gqa_image_id"]): row for row in candidates}
    need(len(candidates) == len(flattened) == 4331 and len(old_images) == 8
         and len(holdout["image_ids"]) == 4 and set(holdout["image_ids"]) <= set(old_images),
         "candidate or prior split denominator differs")
    need(len({row["source_group"] for row in candidates}) == len(candidates), "ambiguous source group")
    picked = sorted(
        (key for key in flattened if key not in old_images
         and flattened[key]["source_group"] not in holdout["source_groups"]),
        key=lambda key: sha256(f"gqa64-license-pilot-v1:{key}".encode()).digest(),
    )[:FETCH_LIMIT]
    cursor = ZipIndex()
    with ZipFile(cursor) as archive:
        need(len(archive.infolist()) == 148855, "publisher ZIP member count differs")
        members = select_manifest(archive, set(picked) | old_images.keys())
    need(sum(members[key].compress_size + 30 + 8192 + 64 for key in picked) <= MAX_MEDIA_RANGES,
         "prospective media exceeds total range budget")
    protected = []
    original = []
    for key, row in old_images.items():
        info = members[key]
        need(info.filename == row["official_member"] and f"{info.CRC:08x}" == row["CRC32"],
             "archive differs from prior member anchors")
        data = (source / OLD / f"{key}.jpg").read_bytes()
        identity = image_identity(data, row["JPEG_dimensions"])
        need(identity[0] == row["SHA256"], "local original JPEG changed")
        if key in holdout["image_ids"]:
            need(identity[0] in holdout["image_SHA256"] and row["source_group"] in holdout["source_groups"],
                 "prior holdout binding differs")
            protected.append(identity)
        else:
            original.append((key, data, identity))
    prior = [*protected]
    selected = []
    for key, data, identity in original:
        if not any(near_duplicate(identity, earlier) for earlier in prior):
            prior.append(identity)
            selected.append((key, data, identity))
    original_count = len(selected)
    if previous:
        old_receipt = load_pinned(previous / "media-receipt.json", previous_pin)
        need(old_receipt["summary"]["source_input_SHA256"] == PINS
             and old_receipt["summary"]["publisher_ETag"] == ETAG,
             "prior media receipt belongs to another scope")
        for key, row in old_receipt["admitted"].items():
            need(key in members and (key in picked or key in old_images)
                 and row["path"] == f"{key}.jpg", "unexpected earlier JPEG identity")
            if key in old_images:
                continue
            data = (previous / "media" / row["path"]).read_bytes()
            identity = image_identity(data, flattened[key]["visual_genome_dimensions"])
            need(identity[0] == row["SHA256"] and zlib.crc32(data) == members[key].CRC,
                 "earlier JPEG fails digest and publisher CRC")
            if not any(near_duplicate(identity, earlier) for earlier in prior):
                prior.append(identity)
                selected.append((key, data, identity))
    resumed_count = len(selected) - original_count
    transferred = 0
    failures = []
    selected_ids = {value[0] for value in selected}
    pending = [key for key in picked if key not in selected_ids]
    for chunk in range(0, len(pending), 4):
        batch = pending[chunk:chunk + 4]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {key: pool.submit(read_jpeg, members[key]) for key in batch}
            for key in batch:
                try:
                    data, ranged = jobs[key].result()
                    transferred += ranged
                    identity = image_identity(data, flattened[key]["visual_genome_dimensions"])
                except HTTPError as error:
                    failures.append({"image_id": key, "failure_type": f"HTTP_{error.code}"})
                    continue
                except (OSError, ValueError, zlib.error) as error:
                    failures.append({"image_id": key, "failure_type": type(error).__name__})
                    continue
                if any(near_duplicate(identity, item) for item in prior):
                    failures.append({"image_id": key, "failure_type": "near_duplicate"})
                    continue
                prior.append(identity)
                selected.append((key, data, identity))
        if len(selected) >= TRAIN_ROWS:
            break
    output.mkdir()
    media = output / "media"
    media.mkdir()
    admitted = {}
    for key, data, identity in selected:
        candidate = flattened[key]
        relative = f"{key}.jpg"
        write(media / relative, data)
        admitted[key] = {"path": relative, "source_group": candidate["source_group"],
                         "license": candidate["coco_license_url"], "source_url": candidate["source_url"],
                         "SHA256": identity[0]}
    receipt = inputs["receipt.json"]
    parts = deflate_parts(source / "train_balanced_questions.json.deflate",
                          receipt["uncompressed_SHA256"], receipt["compressed_SHA256"],
                          receipt["uncompressed_bytes"])
    projection = {"publisher_ETag": ETAG, "ZIP_entries": 148855, "index_range_bytes": cursor.bytes_read,
                  "successful_transfer_range_bytes": transferred, "media_budget_bytes": MAX_MEDIA_RANGES,
                  "candidate_pool": len(candidates), "selected_original_train_reused": original_count,
                  "existing_media_reused_after_recheck": resumed_count, "previous_receipt_SHA256": previous_pin,
                  "admitted_media": len(admitted), "protected_prior_holdout": len(protected),
                  "failures": len(failures), "source_input_SHA256": PINS,"GPU": 0,"optimizer_updates": 0}
    write(output / "media-receipt.json", document({"summary": projection, "admitted": admitted, "failures": failures}))
    rows, lineage = build(question_rows(parts), admitted, holdout, media, TRAIN_ROWS, "internvl-gqa-relation-train-v1")
    train = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    write(output / "train.jsonl", train)
    write(output / "training-lineage.json", document(lineage))
    summary = {**projection, "source_questions": lineage["source_questions"], "train_rows": len(rows),
               "train_SHA256": sha256(train).hexdigest(), "independent_blind_release_qualified": False,
               "target_native_processor_rows": 0, "training_ready": False}
    write(output / "summary.json", document(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--previous-receipt-sha")
    args = parser.parse_args()
    need(bool(args.previous) == bool(args.previous_receipt_sha), "prior media needs an exact receipt")
    need(not args.output.exists() and args.output.parent.is_dir(), "fresh workspace output required")
    main(args.source.resolve(strict=True), args.output.resolve(), args.previous, args.previous_receipt_sha)
