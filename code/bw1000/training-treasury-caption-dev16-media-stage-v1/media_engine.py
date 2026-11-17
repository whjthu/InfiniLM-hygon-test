"""One fixed public image GET per frozen row; no retry, resize, scoring or pixels decoding."""
import hashlib
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.request

MAX_BYTES = 2*1024**2
HOST = 's3.us-east-1.amazonaws.com'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def image_url(row):
    identity = row['coco_image_id']
    require(type(identity) is int and identity > 0 and row['file_name'] == f'{identity:012d}.jpg', 'Unsafe frozen image identity')
    return 'https://'+HOST+'/images.cocodataset.org/train2017/'+row['file_name']


def remaining(deadline, monotonic_end):
    value = min(deadline-time.time(), monotonic_end-time.monotonic())-7
    require(value > 0, 'Original media epoch exhausted')
    return value


def validate_headers(response, expected_url):
    require(response.status == 200 and response.geturl() == expected_url, 'Fixed image TLS URL must return200 before body')
    headers = response.headers
    lengths = headers.get_all('Content-Length')
    require(lengths is not None and len(lengths) == 1 and re.fullmatch(r'[1-9][0-9]*', lengths[0]), 'Single image Content-Length required before body')
    length = int(lengths[0]); require(0 < length <= MAX_BYTES, 'Image exceeds2MiB before body')
    require(headers.get_all('Content-Encoding') in (None, ['identity']) and headers.get_all('Transfer-Encoding') is None
            and headers.get_all('Content-Range') is None, 'Image encoding/range rejected before body')
    content_types = headers.get_all('Content-Type')
    require(content_types == ['image/jpeg'], 'Publisher image/jpeg required before body')
    etags = headers.get_all('ETag')
    require(etags is not None and len(etags) == 1 and 1 <= len(etags[0]) <= 256, 'Single actual image ETag required before body')
    return dict(Content_Length=length, Content_Type='image/jpeg', ETag=etags[0])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, 'Media redirects prohibited', headers, None)


def jpeg_header(path, width, height):
    """Pillow passive open/size only. Full JPEG entropy/pixel decode is explicitly not qualified."""
    from PIL import Image
    with path.open('rb') as stream:
        require(stream.read(2) == b'\xff\xd8', 'Actual image lacks JPEG SOI')
        stream.seek(-2, 2); require(stream.read(2) == b'\xff\xd9', 'Actual image lacks JPEG EOI')
    with Image.open(path) as image:
        require(image.format == 'JPEG' and image.size == (width, height), 'JPEG header dimensions differ from frozen publisher metadata')
        require(image.width > 0 and image.height > 0 and image.width*image.height <= 640*640, 'Frozen image dimensions exceed limit')
        return dict(format=image.format, width=image.width, height=image.height,
                    passive_header_only=True, full_pixel_decode=False, resized=False)


def content_dedup(records, old_hashes):
    """Reject all members of an intra-roster collision, including its first occurrence."""
    groups = {}; old_overlap = 0; duplicate_members = 0
    for index, row in enumerate(records):
        digest = row.get('SHA256')
        if digest in old_hashes:
            row.update(actual_image_qualified=False, state='ORIGINAL64_CONTENT_SHA_OVERLAP'); old_overlap += 1
        if digest is not None:
            groups.setdefault(digest, []).append(index)
    for indices in groups.values():
        if len(indices) > 1:
            duplicate_members += len(indices)
            for index in indices:
                records[index].update(actual_image_qualified=False, state='CANDIDATE16_CONTENT_SHA_DUPLICATE')
    return old_overlap, duplicate_members


def acquire(row, directory, deadline, monotonic_end, progress):
    url = image_url(row); partial = directory/(row['file_name']+'.partial')
    result = dict(state='NOT_STARTED', HTTP_open_attempts=0, bytes_read=0, bytes_saved=0, URL=url,
                  planned_images=1, actual_image_qualified=False, replay_allowed=False, retries=False)
    response = None
    try:
        require(remaining(deadline, monotonic_end) >= 15, 'Insufficient original epoch to begin image GET')
        context = ssl.create_default_context()
        require(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED, 'Normal TLS required')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(), urllib.request.HTTPSHandler(context=context))
        request = urllib.request.Request(url, headers={'Accept-Encoding':'identity', 'User-Agent':'InfiniLM-caption-dev16-fixed-media-v1'}, method='GET')
        result.update(state='HTTP_OPEN', HTTP_open_attempts=1, normal_TLS=True, redirects=False, proxy_environment=False)
        progress(result)
        response = opener.open(request, timeout=min(15, remaining(deadline, monotonic_end)))
        result['response_headers'] = validate_headers(response, url); result['headers_qualified_before_body'] = True
        progress(result); digest = hashlib.sha256(); length = result['response_headers']['Content_Length']
        with partial.open('xb') as stream:
            while result['bytes_read'] < length:
                remaining(deadline, monotonic_end)
                chunk = response.read(min(64*1024, length-result['bytes_read']))
                require(chunk and len(chunk) <= length-result['bytes_read'], 'Truncated/overrun image')
                result['bytes_read'] += len(chunk); digest.update(chunk); stream.write(chunk)
                result['bytes_saved'] += len(chunk); progress(result)
            stream.flush()
        require(partial.stat().st_size == result['bytes_saved'] == length, 'Image actual byte count differs')
        result['SHA256'] = digest.hexdigest(); result['JPEG'] = jpeg_header(partial, row['width'], row['height'])
        final = directory/row['file_name']; require(not final.exists(), 'Media target already exists')
        partial.rename(final)
        result.update(state='ACTUAL_MEDIA_HEADER_AND_SHA_RECORDED', file_name=final.name, actual_image_qualified=True)
    except BaseException as error:
        result.update(state='IMAGE_FAILED_NO_BACKFILL', error_type=type(error).__name__, actual_image_qualified=False)
        if isinstance(error, urllib.error.HTTPError):
            error.close()
    finally:
        if response is not None:
            response.close()
        progress(result)
    return result
