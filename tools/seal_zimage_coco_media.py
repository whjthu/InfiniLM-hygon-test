"""Receive the immutable COCO validation archive and seal per-image byte identities."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from zipfile import ZipFile


ARCHIVE_BYTES = 815585330
ARCHIVE_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ROSTER_SHA256 = "9e1603e5626fdca032cf8b59b63a69c4e26ee410064f507b9f521df8c8234a63"
TRAINING_SHA256 = "070013087a083bdec7defafaef92e781579e4ea491665536c48abe2336a21f55"
LABEL = "training-zimage-t07-coco-val2017-publisher-media-v1"
BASE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
JPEG = re.compile(r"val2017/[0-9]{12}\.jpg")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_file(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def audit_zip(archive, training, subsets):
    accepted = {"val2017/" + key.removeprefix("images/"): value for key, value in training.items()}
    expected = {"val2017/" + row["filename"]: (name, row["image_id"]) for name, rows in subsets.items() for row in rows}
    need(len(expected) == sum(map(len, subsets.values())) and not accepted.keys() & expected.keys(),
         "training and holdout membership overlap or duplicate")
    expected_names = accepted.keys() | expected.keys()
    images = {}
    directory_count = 0
    for item in archive.infolist():
        if item.filename == "val2017/" and item.is_dir():
            directory_count += 1
            continue
        need(JPEG.fullmatch(item.filename) and item.filename not in images and not item.flag_bits & 1
             and 0 < item.file_size <= 16 << 20, "unexpected or repeated archive member")
        images[item.filename] = item
    need(directory_count <= 1 and images.keys() == expected_names, "archive image roster differs")

    originals = set()
    for name, pin in accepted.items():
        data = archive.read(images[name])
        actual = sha256(data).hexdigest()
        need(len(data) == pin["bytes"] and actual == pin["SHA256"] and data[:2] == b"\xff\xd8",
             "original training image differs from the archive")
        originals.add(actual)

    records = {name: {} for name in subsets}
    seen = {}
    train_overlaps = []
    repeated_holdout = []
    for name, (subset, image_id) in expected.items():
        need(name == f"val2017/{image_id:012d}.jpg", "image name and ID disagree")
        data = archive.read(images[name])
        identity = sha256(data).hexdigest()
        need(data[:2] == b"\xff\xd8", "holdout member is not a JPEG")
        if identity in originals:
            train_overlaps.append(image_id)
        if identity in seen:
            repeated_holdout.append([seen[identity], image_id])
        else:
            seen[identity] = image_id
        records[subset][str(image_id)] = dict(archive_member=name, bytes=len(data), SHA256=identity)
    return records, train_overlaps, repeated_holdout


def copy_archive(wire, path, size, expected_sha):
    state = sha256()
    received = 0
    with path.open("xb") as stream:
        while received < size:
            block = wire.read(min(1 << 20, size - received))
            need(block, "short archive stream")
            stream.write(block)
            state.update(block)
            received += len(block)
        need(not wire.read(1), "archive stream exceeded its frozen size")
        stream.flush()
        os.fsync(stream.fileno())
    need(state.hexdigest() == expected_sha, "whole COCO archive SHA256 differs")


def expire(_signal, _frame):
    raise TimeoutError("receiver deadline")


def receive(request, publication, wire):
    need(re.fullmatch(r"[a-f0-9]{32}", request) and re.fullmatch(r"[a-f0-9]{40}", publication), "wrong receipt identity")
    scope = BASE / LABEL
    need(shutil.disk_usage(BASE).free >= 2 << 30, "insufficient experimental storage")
    os.umask(0o077)
    header_size = int.from_bytes(wire.read(4), "big")
    need(1_000_000 <= header_size <= 1_200_000, "unexpected roster envelope size")
    envelope = wire.read(header_size)
    need(len(envelope) == header_size, "short roster envelope")
    roster_raw, training_raw = envelope.split(b"\x00", 1)
    need(sha256(roster_raw).hexdigest() == ROSTER_SHA256 and sha256(training_raw).hexdigest() == TRAINING_SHA256,
         "frozen holdout or training identity differs")
    roster, training = json.loads(roster_raw), json.loads(training_raw)
    subsets = roster["subsets"]
    need(roster["case"] == "T07.sft" and {k: len(v) for k, v in subsets.items()} == {"development": 256, "blind": 4724}
         and len(training["images"]) == 20, "unexpected evaluation cohort")
    scope.mkdir(mode=0o700)
    epoch = time.time()
    write_file(scope / "intent.json", encode(dict(request=request, publication=publication, epoch=epoch,
               deadline_epoch=epoch+720, archive_bytes=ARCHIVE_BYTES, archive_SHA256=ARCHIVE_SHA256, replay=False)))
    print(json.dumps(dict(ready=True, request=request, archive_bytes=ARCHIVE_BYTES)), flush=True)
    partial = scope / "val2017.zip.part"
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(720)
    try:
        copy_archive(wire, partial, ARCHIVE_BYTES, ARCHIVE_SHA256)
        with ZipFile(partial) as archive:
            media, overlaps, repeated = audit_zip(archive, training["images"], subsets)
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
        result = dict(request=request, publication=publication, archive_bytes=ARCHIVE_BYTES, archive_SHA256=ARCHIVE_SHA256,
                      exact_original_training_images=20, holdout_counts={k: len(v) for k, v in media.items()},
                      holdout_byte_overlaps=len(overlaps), duplicate_holdout_bytes=len(repeated),
                      image_byte_identity_verified=True, eligible_for_content_review=not overlaps and not repeated,
                      private_media_SHA256=sha256(private).hexdigest(),
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


if __name__ == "__main__":
    receive(sys.argv[1], sys.argv[2], sys.stdin.buffer)
