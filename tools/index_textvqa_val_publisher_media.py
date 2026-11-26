"""Read only the publisher ZIP index for the exact TextVQA validation images."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from project_textvqa_val_identities import ANNOTATION_SHA256, IMAGE_ID


ARCHIVE_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
ARCHIVE_ETAG = '"79ddfdcdb3f40b3466b9c0dc2f0cda25-68"'
ARCHIVE_SIZE = 7_072_297_970
INDEX_READ_LIMIT = 8 * 1024**2


class PublisherIndexRanges:
    def __init__(self):
        self.position = 0
        self.transferred = 0
        self.requests = 0

    def tell(self):
        return self.position

    def seek(self, offset, origin=0):
        position = {0: 0, 1: self.position, 2: ARCHIVE_SIZE}[origin] + offset
        if not 0 <= position <= ARCHIVE_SIZE:
            raise ValueError("Seek outside the pinned publisher archive")
        self.position = position
        return position

    def read(self, size=-1):
        remainder = ARCHIVE_SIZE - self.position
        amount = remainder if size < 0 else min(size, remainder)
        if amount == 0:
            return b""
        if self.transferred + amount > INDEX_READ_LIMIT or self.requests >= 16:
            raise ValueError("Requested bytes exceed the bounded ZIP index scope")
        end = self.position + amount - 1
        headers = {"Range": f"bytes={self.position}-{end}",
                   "Accept-Encoding": "identity", "If-Match": ARCHIVE_ETAG}
        with urlopen(Request(ARCHIVE_URL, headers=headers), timeout=15) as response:
            if (response.status != 206 or response.geturl() != ARCHIVE_URL
                    or response.headers.get("Content-Range") != f"bytes {self.position}-{end}/{ARCHIVE_SIZE}"
                    or response.headers.get("ETag") != ARCHIVE_ETAG
                    or response.headers.get("Content-Encoding", "identity") != "identity"):
                raise ValueError("Publisher archive identity or exact byte range differs")
            raw = response.read(amount + 1)
        if len(raw) != amount:
            raise ValueError("Incomplete publisher range")
        self.position += amount
        self.transferred += amount
        self.requests += 1
        return raw


def validate_identities(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(300_001)
    if not 0 < len(raw) <= 300_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen answerless identity projection differs")
    data = json.loads(raw)
    if ((data["schema"], data["annotation_SHA256"], data["split"], data["version"])
            != ("TEXTVQA_V051_VAL_IDENTITY_ONLY_V1", ANNOTATION_SHA256, "val", "0.5.1")):
        raise ValueError("Wrong TextVQA source identity")
    rows = data["rows"]
    if any(set(row) != {"image_id", "question_id"} or not IMAGE_ID.fullmatch(row["image_id"])
           or type(row["question_id"]) is not int for row in rows):
        raise ValueError("Expected identity-only rows")
    images = {row["image_id"] for row in rows}
    if len(rows) != 5000 or len({row["question_id"] for row in rows}) != 5000 or len(images) != 3166:
        raise ValueError("Wrong TextVQA denominator")
    return images


def inspect(archive, expected_images):
    result = {}
    with ZipFile(archive) as index:
        for item in index.infolist():
            path = PurePosixPath(item.filename)
            match = re.fullmatch(r"([0-9a-f]{16})\.(?:jpg|jpeg)", path.name)
            if not match or match[1] not in expected_images:
                continue
            image = match[1]
            if (image in result or path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or item.flag_bits & 1 or item.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                    or not 0 < item.file_size <= 32 * 1024**2
                    or not 0 < item.compress_size <= 32 * 1024**2):
                raise ValueError("Ambiguous or unsupported publisher JPEG in the frozen validation set")
            result[image] = dict(zip_name=item.filename, zip_CRC32=f"{item.CRC:08x}",
                                 member_offset=item.header_offset, compressed_bytes=item.compress_size,
                                 member_bytes=item.file_size)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identities", type=Path)
    parser.add_argument("identities_sha256")
    parser.add_argument("index_output", type=Path)
    args = parser.parse_args()
    if args.index_output.exists():
        raise FileExistsError("Frozen index output already exists")
    expected = validate_identities(args.identities, args.identities_sha256)
    source = PublisherIndexRanges()
    index = inspect(source, expected)
    summary = dict(expected_images=len(expected), indexed_images=len(index),
                   zip_index_bytes_read=source.transferred, index_range_requests=source.requests,
                   compressed_image_bytes=sum(row["compressed_bytes"] for row in index.values()),
                   unpacked_image_bytes=sum(row["member_bytes"] for row in index.values()),
                   actual_image_bytes_read=0)
    raw = (json.dumps(dict(identities_SHA256=args.identities_sha256, archive=ARCHIVE_URL, ETag=ARCHIVE_ETAG,
                           archive_bytes=ARCHIVE_SIZE, missing_ids=sorted(expected-index.keys()), rows=index,
                           **summary), sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.index_output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({**summary, "SHA256": hashlib.sha256(raw).hexdigest()}))
    return 0 if len(index) == len(expected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
