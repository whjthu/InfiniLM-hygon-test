"""Reuse the exact terminal relay prefix and fetch remaining COCO ranges at the experiment host."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time
import urllib.error
from zipfile import ZipFile

from seal_zimage_coco_media import (ARCHIVE_BYTES, ARCHIVE_SHA256, BASE, LABEL, ROSTER_SHA256,
                                    TRAINING_SHA256, audit_zip, encode, expire, need, write_file)
from stream_zimage_coco_media import copy_spans, fetch


PUBLISHER = "http://images.cocodataset.org/zips/val2017.zip"
ORIGINAL_REQUEST = "3dd89d53b2dd47858deafb5444471a45"
ORIGINAL_PUBLICATION = "4eca093289d7dd66c17cc1302326109307fd0a39"
CONTINUATION_LABEL = LABEL + "-continuation"
SECONDS = 1800


def publisher_chunk(span):
    for attempt in range(3):
        try:
            return fetch(span, PUBLISHER)
        except urllib.error.HTTPError as error:
            if attempt == 2 or error.code < 500:
                raise
        except OSError:
            if attempt == 2:
                raise
        time.sleep(1)


def sealed_prefix():
    source = BASE / LABEL
    intent = json.loads((source / "intent.json").read_bytes())
    ended = json.loads((source / "failure.json").read_bytes())
    partial = source / "val2017.zip.part"
    need(intent["request"] == ended["request"] == ORIGINAL_REQUEST and intent["publication"] == ORIGINAL_PUBLICATION
         and intent["archive_SHA256"] == ARCHIVE_SHA256 and not (source / "receipt.json").exists()
         and ended["error_type"] in {"TimeoutError", "ValueError"}, "original relay has no matching failed terminal")
    size = ended["partial_bytes"]
    need(type(size) is int and 0 < size < ARCHIVE_BYTES and not partial.is_symlink()
         and partial.stat().st_size == size, "original archive prefix changed")
    return partial, size


def continue_archive(request, publication, roster_raw, training_raw, expected_bytes, expected_sha):
    need(re.fullmatch(r"[a-f0-9]{32}", request) and re.fullmatch(r"[a-f0-9]{40}", publication), "wrong request")
    need(sha256(roster_raw).hexdigest() == ROSTER_SHA256 and sha256(training_raw).hexdigest() == TRAINING_SHA256,
         "frozen holdout or training identity changed")
    roster, training = json.loads(roster_raw), json.loads(training_raw)
    original, size = sealed_prefix()
    need(size == expected_bytes and re.fullmatch(r"[a-f0-9]{64}", expected_sha), "recorded prefix extent differs")
    need(shutil.disk_usage(BASE).free >= 2 << 30, "insufficient experimental storage")
    scope = BASE / CONTINUATION_LABEL
    os.umask(0o077)
    scope.mkdir(mode=0o700)
    epoch = time.time()
    write_file(scope / "intent.json", encode(dict(request=request, publication=publication, epoch=epoch,
               deadline_epoch=epoch+SECONDS, original_request=ORIGINAL_REQUEST, reused_prefix_bytes=size,
               reused_prefix_SHA256=expected_sha, publisher=PUBLISHER, retry_original=False)))
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(SECONDS)
    partial = scope / "val2017.zip.part"
    try:
        state = sha256()
        with original.open("rb") as prefix, partial.open("xb") as destination:
            while block := prefix.read(1 << 20):
                destination.write(block)
                state.update(block)
            need(state.hexdigest() == expected_sha and destination.tell() == size and original.stat().st_size == size,
                 "recorded prefix digest differs")
            write_file(scope / "prefix-reused.json", encode(dict(request=request, bytes=size, SHA256=expected_sha)))
            archive_sha = copy_spans(destination, begin=size, span_size=1 << 20, prefetch=6,
                                     state=state, read_span=publisher_chunk)
            need(destination.tell() == ARCHIVE_BYTES and archive_sha == ARCHIVE_SHA256,
                 "assembled archive differs from the publisher digest")
            destination.flush()
            os.fsync(destination.fileno())
        with ZipFile(partial) as archive:
            media, overlaps, repeated = audit_zip(archive, training["images"], roster["subsets"])
        private = encode(dict(schema="zimage-coco-val2017-media-v1", source_roster_SHA256=ROSTER_SHA256,
                              archive_SHA256=ARCHIVE_SHA256, images=media,
                              training_byte_overlaps=overlaps, repeated_holdout_bytes=repeated))
        write_file(scope / "private-media.json", private)
        partial.rename(scope / "val2017.zip")
        directory = os.open(scope, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        result = dict(request=request, publication=publication, original_request=ORIGINAL_REQUEST,
                      reused_prefix_bytes=size, reused_prefix_SHA256=expected_sha, fetched_suffix_bytes=ARCHIVE_BYTES-size,
                      archive_bytes=ARCHIVE_BYTES, archive_SHA256=ARCHIVE_SHA256, exact_original_training_images=20,
                      holdout_counts={k: len(v) for k, v in media.items()}, holdout_byte_overlaps=len(overlaps),
                      duplicate_holdout_bytes=len(repeated), image_byte_identity_verified=True,
                      eligible_for_content_review=not overlaps and not repeated, private_media_SHA256=sha256(private).hexdigest(),
                      private_media_bytes=len(private), near_duplicates_checked=False, scored_images=0,
                      published_archive="val2017.zip", elapsed_seconds=time.time()-epoch)
        write_file(scope / "receipt.json", encode(result))
        print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as error:
        write_file(scope / "failure.json", encode(dict(request=request, error_type=type(error).__name__,
                   partial_bytes=partial.stat().st_size if partial.is_file() else None)))
        raise
    finally:
        signal.alarm(0)
