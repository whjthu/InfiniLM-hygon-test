"""Bind the sealed COCO development roster to its exact official captions."""

import argparse
from hashlib import sha256
from io import RawIOBase
import json
from pathlib import Path
import struct
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED
import zlib


URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ARCHIVE_BYTES = 252907541
MEMBER = "annotations/captions_val2017.json"
MEMBER_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
MEMBER_HEADER_SHA256 = "80e5e7f1592d23905a4d51ed6e179177e6120c303d6747840b368f9b2fede6fd"
MEMBER_AFTER_HEADER_SHA256 = "ba2348b66158aea7083493f3152b8a6c8f8fc5ee1e46df0a48741526941ce7f3"
MEMBER_COMPRESSED_SHA256 = "7f74a81ef4359c4594180b8be025a3bb11e7ff6c8409d60c6c9d33d074b378c0"
ROSTER_SHA256 = "9e1603e5626fdca032cf8b59b63a69c4e26ee410064f507b9f521df8c8234a63"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


class Ranges(RawIOBase):
    def __init__(self):
        super().__init__()
        self.offset = 0
        self.received_bytes = 0
        self.requests = 0

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        new = (0, self.offset, ARCHIVE_BYTES)[whence] + offset
        require(0 <= new <= ARCHIVE_BYTES, "ZIP seek outside publisher archive")
        self.offset = new
        return self.offset

    def read(self, n=-1):
        available = ARCHIVE_BYTES - self.offset
        count = available if n < 0 else min(n, available)
        if not count:
            return b""
        require(count < 2 << 20 and self.received_bytes + count <= 4 << 20
                and self.requests < 16, "range transfer budget")
        first, last = self.offset, self.offset + count - 1
        request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=10) as stream:
            require(stream.status == 206
                    and stream.headers.get("Content-Range") == f"bytes {first}-{last}/{ARCHIVE_BYTES}",
                    "publisher did not return the exact byte range")
            raw = stream.read(count + 1)
        require(len(raw) == count, "short or oversized archive range")
        self.requests += 1
        self.received_bytes += count
        self.offset += count
        return raw


def read_annotations(remote):
    with ZipFile(remote) as catalog:
        [entry] = [candidate for candidate in catalog.infolist() if candidate.filename == MEMBER]
    require(not entry.flag_bits & 1 and entry.compress_type == ZIP_DEFLATED
            and 0 < entry.file_size < 4 << 20, "invalid compressed annotation member")
    remote.seek(entry.header_offset)
    header = remote.read(30)
    require(sha256(header).hexdigest() == MEMBER_HEADER_SHA256, "frozen fixed annotation header differs")
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack("<4s5H3I2H", header)
    require(signature == b"PK\x03\x04" and not flags & 1 and method == ZIP_DEFLATED
            and name_size == len(MEMBER) and extra_size < 4096, "local annotation header differs")
    name = remote.read(name_size)
    require(name == MEMBER.encode("ascii"), "local annotation name differs")
    extra = remote.read(extra_size)
    compressed = remote.read(entry.compress_size)
    require(sha256(name + extra + compressed).hexdigest() == MEMBER_AFTER_HEADER_SHA256
            and sha256(compressed).hexdigest() == MEMBER_COMPRESSED_SHA256, "frozen compressed annotations differ")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    raw = inflater.decompress(compressed, entry.file_size + 1)
    require(inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail
            and len(raw) == entry.file_size and zlib.crc32(raw) == entry.CRC
            and sha256(raw).hexdigest() == MEMBER_SHA256, "frozen decoded annotations differ")
    return json.loads(raw)


def bind(annotations, roster):
    image_index = {entry["id"]: entry for entry in annotations["images"]}
    annotation_index = {}
    for entry in annotations["annotations"]:
        annotation_id = entry["id"]
        require(annotation_id not in annotation_index, "reused publisher caption ID")
        annotation_index[annotation_id] = entry
    require(len(image_index) == 5000 and len(annotation_index) == 25014, "invalid COCO image/caption coverage")
    rows = []
    dev = roster["subsets"]["development"]
    require(len(dev) == len({row["image_id"] for row in dev}) == 256, "development cohort differs")
    for index, row in enumerate(dev):
        source = image_index[row["image_id"]]
        original = annotation_index[row["caption_id"]]
        prompt = original["caption"]
        require(original["image_id"] == row["image_id"] and source["file_name"] == row["filename"]
                and source["license"] == row["license_id"] and isinstance(prompt, str)
                and sha256(prompt.encode("utf-8")).hexdigest() == row["prompt_SHA256"],
                "caption, image, or license identity differs")
        rows.append({"ordinal": index, "image_id": row["image_id"], "caption_id": row["caption_id"],
                     "prompt": prompt, "prompt_SHA256": row["prompt_SHA256"], "seed": row["seed"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    roster = args.roster.read_bytes()
    require(sha256(roster).hexdigest() == ROSTER_SHA256, "frozen roster differs")
    with Ranges() as remote:
        annotations = read_annotations(remote)
        row_list = bind(annotations, json.loads(roster))
        receipt = {"ordinal_count": len(row_list), "cohort": "development", "roster_SHA256": ROSTER_SHA256,
                   "annotation_member_SHA256": MEMBER_SHA256, "annotation_member_compressed_SHA256": MEMBER_COMPRESSED_SHA256,
                   "fixed_member_header_SHA256": MEMBER_HEADER_SHA256, "member_after_header_SHA256": MEMBER_AFTER_HEADER_SHA256,
                   "publisher_range_bytes": remote.received_bytes,
                   "publisher_range_requests": remote.requests, "measured_images": 0}
    private = dict(schema="training.zimage.coco.development_prompts.v1", **receipt, rows=row_list)
    raw = (json.dumps(private, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(receipt, private_manifest_bytes=len(raw), private_manifest_SHA256=sha256(raw).hexdigest()),
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
