"""Final missing range data binding, separate from exact reused parser/Job/HTTP engines."""
import hashlib
import json
from pathlib import Path
import re
import stat

HERE = Path(__file__).resolve().parent
BINDING = json.loads((HERE/'BINDING.json').read_bytes())
G = Path(BINDING['root'])
ORIGINAL_REQUEST = BINDING['prior_request']
ORIGINAL_PUBLICATION = '71ee6ebbc8116571968f037f0cb43094652b3fce'
FAILURE_PUBLICATION = BINDING['prior_failure_publication']
FAILURE_RESULT = BINDING['prior_failure_public_path']
FAILURE_SHA = BINDING['prior_failure_public_SHA256']
CONSUMER = G/BINDING['prior_independent_result']
CONSUMER_SHA = BINDING['prior_independent_result_SHA256']
PREFIX = G/BINDING['prefix_path']
PREFIX_BYTES, PREFIX_SHA = BINDING['prefix_bytes'], BINDING['prefix_SHA256']
TAIL_START, TAIL_END, TAIL_BYTES = BINDING['range_start'], BINDING['range_end'], BINDING['new_HTTP_body_bytes']
FULL_BYTES, ETAG = BINDING['complete_member_bytes'], BINDING['ETag']


def require(ok, why):
    if not ok:
        raise ValueError(why)


def ready():
    require(BINDING['state'] == 'FROZEN_SECOND_FAILURE_AUDIT_AND_PUBLICATION_BOUND'
            and BINDING['HTTP_GO'] is True, 'Actual prefix audit/publication not yet bound')
    require(re.fullmatch('[0-9a-f]{40}', FAILURE_PUBLICATION or '')
            and all(re.fullmatch('[0-9a-f]{64}', value or '') for value in (FAILURE_SHA, CONSUMER_SHA, PREFIX_SHA)),
            'Missing actual prerequisite SHA/publication')
    require((PREFIX_BYTES, TAIL_START, TAIL_END, TAIL_BYTES, FULL_BYTES) ==
            (17039360,176326406,178103193,1776788,18816148)
            and TAIL_START == 159287046+PREFIX_BYTES and TAIL_END-TAIL_START+1 == TAIL_BYTES,
            'Only the exact missing final range is permitted')
    require((BINDING['whole_epoch_seconds'], BINDING['CPU_cores'], BINDING['memory_limit_bytes'],
             BINDING['HTTP_open_attempts'], BINDING['HTTP_prefix_bytes_requested']) == (600,2,2147483648,1,0),
            'Final tail resource/request scope differs')


def consumer_record():
    ready()
    require(CONSUMER.is_file() and not CONSUMER.is_symlink() and CONSUMER.stat().st_size <= 65536,
            'Unsafe second failure independent record')
    raw = CONSUMER.read_bytes(); require(hashlib.sha256(raw).hexdigest() == CONSUMER_SHA, 'Second failure audit SHA differs')
    value = json.loads(raw)
    require(value['passed'] is True and value['acquisition_passed'] is False
            and value['state'] == 'KNOWN_TERMINAL_SUFFIX_FAILURE_RETAINED' and value['request'] == ORIGINAL_REQUEST
            and value['actual_guard']['returncode'] == 124 and value['actual_guard']['timed_out'] is True
            and value['actual_suffix_bytes_saved'] == value['actual_suffix_bytes_read'] == 8388608
            and value['actual_metadata_selected'] == 0 and value['missing_slots'] == value['planned_images'] == 16,
            'Second request must be an independently verified actual terminal failure')
    partial = value['failed_assembled']
    require(partial['bytes'] == PREFIX_BYTES and partial['SHA256'] == PREFIX_SHA and partial['stable_stat'] is True
            and partial['original_prefix_bytes'] == BINDING['first_verified_prefix_bytes']
            and partial['original_prefix_SHA256'] == BINDING['first_verified_prefix_SHA256']
            and partial['exact_missing_suffix_range_start'] == TAIL_START
            and partial['exact_missing_suffix_range_end'] == TAIL_END and partial['exact_missing_suffix_bytes'] == TAIL_BYTES,
            'Actual cumulative prefix or final range differs')
    return value


def copy_prefix(destination, original, remaining):
    """Execution only: hash once while copying the second closed request's cumulative prefix."""
    expected = original['failed_assembled']; before = PREFIX.lstat()
    require(stat.S_ISREG(before.st_mode) and not PREFIX.is_symlink() and before.st_size == PREFIX_BYTES
            and all(getattr(before, k) == expected[k] for k in ('st_dev','st_ino','st_mtime_ns','st_ctime_ns')),
            'Independently verified cumulative prefix identity changed')
    digest = hashlib.sha256(); count = 0
    with PREFIX.open('rb') as source, destination.open('xb') as out:
        while count < PREFIX_BYTES:
            remaining(); chunk = source.read(min(256*1024, PREFIX_BYTES-count)); require(chunk, 'Cumulative prefix truncated')
            count += len(chunk); digest.update(chunk); out.write(chunk)
        out.flush()
    after = PREFIX.lstat()
    require(all(getattr(before,k) == getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
            and count == PREFIX_BYTES and digest.hexdigest() == PREFIX_SHA, 'Cumulative prefix changed or SHA differs')
    return digest, dict(bytes=count, SHA256=digest.hexdigest(), original_not_modified=True, HTTP_prefix_bytes_requested=0)
