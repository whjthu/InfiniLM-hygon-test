"""Check the original 20 Z-Image training JPEGs against public COCO ranges."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import RawIOBase
import json
import struct
import time
import urllib.request
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib


URL = "http://images.cocodataset.org/zips/val2017.zip"
LENGTH = 815585330
TRAINING_SHA256 = "070013087a083bdec7defafaef92e781579e4ea491665536c48abe2336a21f55"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_range(start, size):
    require(0 <= start < LENGTH and 0 < size <= 2 << 20 and start + size <= LENGTH, "unbounded archive range")
    last = start + size - 1
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{last}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=12) as source:
        require(source.status == 206 and source.headers.get("Content-Range") == f"bytes {start}-{last}/{LENGTH}",
                "publisher returned another range")
        raw = source.read(size + 1)
        require(len(raw) == size, "publisher returned a short range")
    return raw


class RemoteDirectory(RawIOBase):
    def __init__(self):
        super().__init__()
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        origin = (0, self.position, LENGTH)[whence]
        target = origin + offset
        require(0 <= target <= LENGTH, "invalid ZIP directory offset")
        self.position = target
        return target

    def tell(self):
        return self.position

    def read(self, size=-1):
        length = LENGTH - self.position if size < 0 else min(size, LENGTH - self.position)
        if not length:
            return b""
        data = read_range(self.position, length)
        self.position += len(data)
        return data


def encoded_jpeg(item, pin, fetch=read_range):
    header = fetch(item.header_offset, 30)
    signature, _, flags, method, _, _, _, _, _, filename_size, extra_size = struct.unpack("<4s5H3I2H", header)
    require(signature == b"PK\x03\x04" and not flags & 1 and method == item.compress_type
            and filename_size == len(item.filename.encode()) and extra_size < 4096,
            "local ZIP member disagrees with the directory")
    offset = item.header_offset + 30
    blob = fetch(offset, filename_size + extra_size + item.compress_size)
    require(blob[:filename_size].decode("ascii") == item.filename, "local ZIP name differs")
    payload = blob[filename_size + extra_size:]
    if method == ZIP_DEFLATED:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        image = decoder.decompress(payload, pin["bytes"] + 1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "JPEG deflate extent differs")
    elif method == ZIP_STORED:
        image = payload
    else:
        raise ValueError("unexpected JPEG compression method")
    require(len(image) == pin["bytes"] == item.file_size and image.startswith(b"\xff\xd8")
            and zlib.crc32(image) == item.CRC and sha256(image).hexdigest() == pin["SHA256"],
            "public archive image disagrees with original training image")
    return len(header) + len(blob), len(image)


def verify(training_raw):
    require(sha256(training_raw).hexdigest() == TRAINING_SHA256, "original training bytes changed")
    started = time.monotonic()
    manifest = json.loads(training_raw)
    paths = manifest["row_order"]
    require(len(paths) == len(set(paths)) == 20 and set(paths) == manifest["images"].keys(), "wrong training count")
    with ZipFile(RemoteDirectory()) as archive:
        entries = archive.infolist()
        require(len(entries) in (5000, 5001) and len({i.filename for i in entries}) == len(entries), "wrong ZIP member roster")
        checks = [(archive.getinfo("val2017/" + path.removeprefix("images/")), manifest["images"][path]) for path in paths]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: encoded_jpeg(*item), checks))
    return dict(training_manifest_SHA256=TRAINING_SHA256, archive_size=LENGTH, actual_training_jpeg_count=len(results),
                actual_training_jpeg_bytes=sum(row[1] for row in results),
                range_body_bytes_for_members=sum(row[0] for row in results), elapsed_s=time.monotonic()-started,
                training_JPEGs_match_SHA256=True, archive_SHA256_verified=False,
                checked_heldout_images=0, near_duplicate_reviewed=False)
