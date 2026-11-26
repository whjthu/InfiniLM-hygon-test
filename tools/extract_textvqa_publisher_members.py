"""Read a bounded batch of pinned validation JPEGs from the official TextVQA ZIP."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import zlib

from index_textvqa_val_publisher_media import ARCHIVE_ETAG, ARCHIVE_SIZE, ARCHIVE_URL, PublisherIndexRanges


INDEX_SHA256 = '5bfcd3ddc2be183c4d693dadbc3d8a62d7974b683b9e0bc0457fccfee4f73552'
MEMBER_LIMIT = 2 * 1024**2
BATCH_LIMIT = 6 * 1024**2
IMAGE_ID = re.compile(r'[0-9a-f]{16}')


def read_index(path):
    with path.open('rb') as stream:
        raw = stream.read(600_001)
    if len(raw) > 600_000 or sha256(raw).hexdigest() != INDEX_SHA256:
        raise ValueError('Frozen publisher member index differs')
    index = json.loads(raw)
    if (index['archive'], index['archive_bytes'], index['ETag'], index['indexed_images']) != (
            ARCHIVE_URL, ARCHIVE_SIZE, ARCHIVE_ETAG, 3166) or len(index['rows']) != 3166:
        raise ValueError('Publisher archive identity differs')
    return index


def image_bytes(reader, img, record):
    size = record['member_bytes']
    packed_size = record['compressed_bytes']
    if not (IMAGE_ID.fullmatch(img) and 0 < size <= MEMBER_LIMIT and 0 < packed_size <= MEMBER_LIMIT):
        raise ValueError('Member outside batch policy')
    reader.seek(record['member_offset'])
    header = reader.read(30)
    magic, _version, flags, method, _mtime, _mdate, _crc, _packed, _plain, nlen, xlen = struct.unpack(
        '<4s5H3I2H', header)
    if magic != b'PK\x03\x04' or flags & 1 or method not in (0, 8) or not 1 <= nlen <= 512 or xlen > 4096:
        raise ValueError('Invalid canonical ZIP member header')
    name = reader.read(nlen).decode('utf8')
    if name != record['zip_name'] or name.rsplit('/', 1)[-1] not in (img + '.jpg', img + '.jpeg'):
        raise ValueError('ZIP name differs from the independent central directory')
    reader.seek(xlen, 1)
    packed = reader.read(packed_size)
    if method == 8:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(packed, size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid compressed media extent')
    else:
        data = packed
    if (len(data) != size or f'{zlib.crc32(data):08x}' != record['zip_CRC32']
            or not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9')):
        raise ValueError('Publisher byte length, CRC32 or JPEG framing differs')
    return data


def extract(index, requested, output):
    if (not 1 <= len(requested) <= 4 or len(set(requested)) != len(requested)
            or any(not IMAGE_ID.fullmatch(img) or img not in index['rows'] for img in requested)):
        raise ValueError('Use one to four unique indexed validation images')
    if any(sum(index['rows'][img][key] for img in requested) > BATCH_LIMIT
           for key in ('compressed_bytes', 'member_bytes')):
        raise ValueError('Batch exceeds total media byte budget')
    output.mkdir()
    reader = PublisherIndexRanges()
    records = []
    for img in requested:
        data = image_bytes(reader, img, index['rows'][img])
        name = img + '.jpg'
        with (output / name).open('xb') as stream:
            stream.write(data)
        records.append(dict(image_id=img, relative_path=name, bytes=len(data), SHA256=sha256(data).hexdigest()))
    result = dict(index_SHA256=INDEX_SHA256, rows=records, range_requests=reader.requests,
                  transferred_bytes=reader.transferred, jpeg_decoder_used=False)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    with (output / 'batch.json').open('xb') as stream:
        stream.write(raw)
    return dict(images=len(records), range_requests=reader.requests, transferred_bytes=reader.transferred,
                stored_bytes=sum(row['bytes'] for row in records), batch_SHA256=sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('image_ids', nargs='+')
    args = parser.parse_args()
    print(json.dumps(extract(read_index(args.index), args.image_ids, args.output)))


if __name__ == '__main__':
    main()
