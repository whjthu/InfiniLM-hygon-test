"""Stream one official TUM archive and retain only frozen development RGB frames."""

import argparse
from bisect import bisect_left
from contextlib import closing
import hashlib
from http.client import IncompleteRead, RemoteDisconnected
import io
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.error
import urllib.request

from bind_vggt_tum_dev_rgb import ANCHOR_SHA, MAX_IMAGE_BYTES, MAX_TIME_ERROR, bind_pairs, parse_rgb_index, read_bounded


BASE = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset"
ARCHIVES = {
    "freiburg1_plant": (695620135, "d6361193aaf5236d4c55011bbe47acd7ce8d85f200ad846023c76d752edebf5b"),
    "freiburg2_dishes": (1518207248, "b6730ff1b53d3e110fcc1fdb7427d2168aec0e11ed26cb91d29e8b2a7e6d2d38"),
}
PREFIX = 16 * 1024**2
MAX_INDEX = 3 * 1024**2
RECOVERY_SEQUENCE = "freiburg2_dishes"
RECOVERY_SECONDS = 6000
ORIGINAL_TIMEOUT_SHA = "946c4435f9b2ce155584420c9f348eaf5970a3072e7d3fa38e8c4324f8412be3"
SOCKET_TIMEOUT_SHA = "ef2c72fe4b75b1af8f63158b24b82274cfc56ba01ab70fd8ab1c1fbe0d1b9c86"
DISHES_ETAG = '"5a7e0110-4ae2a216577c0"'
RANGE_BYTES = 64 * 1024**2
RANGE_RETRIES = 8


class RangedArchive:
    """Keep the gzip consumer alive while the HTTP connection moves across exact ranges."""

    def __init__(self, url, size, etag, deadline, *, opener=urllib.request.urlopen,
                 clock=time.monotonic, segment_bytes=RANGE_BYTES, retry_limit=RANGE_RETRIES):
        self.url = url
        self.size = size
        self.etag = etag
        self.deadline = deadline
        self.opener = opener
        self.clock = clock
        self.segment_bytes = segment_bytes
        self.retry_limit = retry_limit
        self.offset = self.stop = self.retries = self.requests = 0
        self.response = None

    def close(self):
        if self.response is not None:
            self.response.close()
            self.response = None

    def connect(self, timeout):
        self.stop = min(self.offset + self.segment_bytes, self.size)
        requested = f"bytes={self.offset}-{self.stop-1}"
        expected = f"bytes {self.offset}-{self.stop-1}/{self.size}"
        req = urllib.request.Request(self.url, headers={"Range": requested, "If-Match": self.etag,
                                                         "Accept-Encoding": "identity"})
        self.requests += 1
        response = self.opener(req, timeout=timeout)
        if (response.status != 206 or response.geturl() != self.url or response.headers.get("ETag") != self.etag
                or response.headers.get("Content-Range") != expected
                or response.headers.get("Content-Length") != str(self.stop-self.offset)
                or response.headers.get("Content-Encoding") not in (None, "identity")):
            response.close()
            raise ValueError("official archive range or immutable object identity differs")
        self.response = response

    def recover(self, error):
        self.close()
        self.retries += 1
        if self.retries > self.retry_limit:
            raise TimeoutError("official archive exhausted the bounded range reconnects") from error

    def read(self, amount):
        target = min(self.offset + amount, self.size)
        parts = []
        while self.offset < target:
            remaining = self.deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("official archive range budget exhausted")
            try:
                if self.response is None:
                    self.connect(min(35.0, remaining))
                wanted = min(target, self.stop) - self.offset
                raw = self.response.read(wanted)
            except urllib.error.HTTPError as error:
                error.close()
                if error.code not in (408, 429, 500, 502, 503, 504):
                    raise
                self.recover(error)
                continue
            except (TimeoutError, ConnectionError, IncompleteRead, RemoteDisconnected, urllib.error.URLError) as error:
                self.recover(error)
                continue
            if not raw:
                self.recover(EOFError("official range ended before the expected boundary"))
                continue
            if len(raw) > wanted:
                raise ValueError("official range returned bytes beyond the request")
            self.offset += len(raw)
            parts.append(raw)
            if self.offset == self.stop:
                self.close()
        return b"".join(parts)


class ArchiveInput:
    def __init__(self, source, size, prefix_sha, deadline, notify):
        self.source = source
        self.size = size
        self.prefix_sha = prefix_sha
        self.deadline = deadline
        self.notify = notify
        self.count = 0
        self.hasher = hashlib.sha256()
        self.next_progress = 64 * 1024**2

    def read(self, amount):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("original public RGB archive deadline exhausted")
        raw = self.source.read(amount)
        end = self.count + len(raw)
        if end > self.size:
            raise ValueError("archive exceeds the official byte count")
        if self.count < PREFIX:
            self.hasher.update(raw[:PREFIX-self.count])
            if end >= PREFIX and self.hasher.hexdigest() != self.prefix_sha:
                raise ValueError("official archive prefix differs from the prior original")
        self.count = end
        if self.count >= self.next_progress:
            self.notify(self.count)
            self.next_progress += 64 * 1024**2
        return raw


def frame_bytes(tar, member, maximum):
    if not member.isfile() or not 0 < member.size <= maximum:
        raise ValueError("unexpected mounted archive member")
    data = tar.extractfile(member).read(member.size + 1)
    if len(data) != member.size:
        raise ValueError("truncated archive member")
    return data


def scan_archive(stream, sequence, anchors, output):
    from PIL import Image

    folder = f"rgbd_dataset_{sequence}"
    pattern = re.compile(rf"{re.escape(folder)}/(rgb/([0-9]+\.[0-9]+)\.png)")
    targets = sorted(time for anchor in anchors if anchor["state"] == "POSE_PAIR_CANDIDATE"
                     for time in anchor["original_pose_times"])
    if not targets:
        raise ValueError("no frozen viable anchors")
    retained = {}
    rgb_count = 0
    index, wanted = None, None
    with tarfile.open(fileobj=stream, mode="r|gz", bufsize=128*1024) as archive:
        for member in archive:
            if member.name == f"{folder}/rgb.txt":
                if index is not None:
                    raise ValueError("duplicate official RGB index")
                index = frame_bytes(archive, member, MAX_INDEX)
                match = bind_pairs(anchors, parse_rgb_index(index))
                wanted = {path for row in match if row["state"] == "RGB_INDEX_MATCHED" for path in row["frames"]}
                if wanted <= retained.keys():
                    break
                continue
            hit = pattern.fullmatch(member.name)
            if hit is None:
                continue
            name, timestamp = hit[1], float(hit[2])
            rgb_count += 1
            i = bisect_left(targets, timestamp)
            neighbors = targets[max(0, i-1):i+1]
            if not (any(abs(t-timestamp) <= MAX_TIME_ERROR for t in neighbors) if wanted is None else name in wanted):
                continue
            if name in retained or len(retained) >= 256:
                raise ValueError("duplicate or oversized set of original RGB candidates")
            raw = frame_bytes(archive, member, MAX_IMAGE_BYTES)
            with Image.open(io.BytesIO(raw)) as picture:
                if picture.format != "PNG" or picture.size != (640, 480) or picture.mode != "RGB":
                    raise ValueError("original RGB archive frame shape differs")
                picture.verify()
            with (output / name).open("xb") as dest:
                dest.write(raw)
            retained[name] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
            if wanted is not None and wanted <= retained.keys():
                break
    if index is None or wanted is None or not wanted <= retained.keys():
        raise ValueError("complete original RGB index or indexed RGB frames unavailable")
    with (output / "rgb.txt").open("xb") as dest:
        dest.write(index)
    return dict(rgb_headers_read=rgb_count, rgb_index_sha256=hashlib.sha256(index).hexdigest(),
                official_rgb_index_rows=len(parse_rgb_index(index)), indexed_required_frames=len(wanted),
                original_png_writes=len(retained), selected_frames={p: retained[p] for p in sorted(wanted)})


def recovery_budget(sequence, seconds, original_timeout):
    if original_timeout is None:
        if not 60 <= seconds <= 3600:
            raise ValueError("ordinary archive budget must be 60–3600 seconds")
        return None
    if sequence != RECOVERY_SEQUENCE or seconds != RECOVERY_SECONDS:
        raise ValueError("recovery is pinned to the Freiburg 2 timeout and 6000-second budget")
    raw = read_bounded(original_timeout, 65536)
    receipt_sha = hashlib.sha256(raw).hexdigest()
    if receipt_sha != ORIGINAL_TIMEOUT_SHA:
        raise ValueError("original timeout receipt SHA differs")
    d = json.loads(raw)
    if (d["sequence"] != sequence or d["passed"] is not False or d["failure_type"] != "TimeoutError"
            or d["actual_compressed_bytes_read"] != 1198260224 or d["frozen_anchor_SHA256"] != ANCHOR_SHA
            or d["archive_written_to_disk"] is not False):
        raise ValueError("expected the original terminal timeout with no retained compressed prefix")
    return receipt_sha


def socket_recovery(sequence, socket_timeout, original_timeout):
    if socket_timeout is None:
        return None
    if original_timeout is None or sequence != RECOVERY_SEQUENCE:
        raise ValueError("range recovery requires both prior terminal receipts")
    raw = read_bounded(socket_timeout, 65536)
    identity = hashlib.sha256(raw).hexdigest()
    if identity != SOCKET_TIMEOUT_SHA:
        raise ValueError("original socket timeout receipt SHA differs")
    prior = json.loads(raw)
    if (prior["sequence"] != sequence or prior["passed"] is not False
            or prior["failure_type"] != "TimeoutError" or prior["failure"] != "The read operation timed out"
            or prior["prior_timeout_receipt_SHA256"] != ORIGINAL_TIMEOUT_SHA):
        raise ValueError("expected the original terminated socket timeout")
    return identity


def extract_source(sequence, source, output, seconds, original_timeout=None, socket_timeout=None):
    recovery = recovery_budget(sequence, seconds, original_timeout)
    socket_recovery_sha = socket_recovery(sequence, socket_timeout, original_timeout)
    for previous in (original_timeout, socket_timeout):
        if previous is not None and output.resolve() == previous.parent.resolve():
            raise ValueError("use a distinct output directory for the explicit recovery")
    anchor_raw = read_bounded(source / "dev-anchor-plan-private.json", 8192)
    if hashlib.sha256(anchor_raw).hexdigest() != ANCHOR_SHA:
        raise ValueError("frozen anchor file differs")
    anchors = json.loads(anchor_raw)[sequence]
    size, prefix = ARCHIVES[sequence]
    family = sequence.split("_", 1)[0]
    url = f"{BASE}/{family}/rgbd_dataset_{sequence}.tgz"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size-1}", "Accept-Encoding": "identity"})
    output.mkdir(mode=0o700)
    (output / "rgb").mkdir(mode=0o700)
    started = time.monotonic()
    reader = None
    summary = {"sequence": sequence, "archive": url, "verified_official_length": size,
               "frozen_anchor_SHA256": ANCHOR_SHA, "archive_written_to_disk": False, "passed": False}
    if recovery is not None:
        summary.update(prior_timeout_receipt_SHA256=recovery, budget_seconds=seconds, compressed_resume=False)
    ranged = None
    if socket_recovery_sha is not None:
        ranged = RangedArchive(url, size, DISHES_ETAG, started + seconds)
        summary.update(prior_socket_timeout_receipt_SHA256=socket_recovery_sha,
                       HTTP_range_ETag=DISHES_ETAG, HTTP_range_segment_bytes=RANGE_BYTES,
                       HTTP_range_reconnect_budget=RANGE_RETRIES)
    try:
        with (closing(ranged) if ranged else urllib.request.urlopen(request, timeout=35)) as response:
            if ranged is None and (response.geturl() != url or response.status not in (200, 206)
                    or response.headers.get("Content-Length") != str(size)
                    or (response.status == 206 and response.headers.get("Content-Range") != f"bytes 0-{size-1}/{size}")):
                raise ValueError("original archive identity or range differs")
            reader = ArchiveInput(response, size, prefix, started + seconds,
                                  lambda count: print(json.dumps(dict(sequence=sequence, read_bytes=count)), flush=True))
            summary.update(scan_archive(reader, sequence, anchors, output))
            if reader.count < PREFIX:
                raise ValueError("official archive prefix was not independently verified")
            summary["passed"] = True
    except (OSError, ValueError, tarfile.TarError) as error:
        summary["failure_type"] = type(error).__name__
        summary["failure"] = str(error)[:180]
    summary.update(actual_compressed_bytes_read=reader.count if reader else 0,
                   wall_seconds=time.monotonic() - started)
    if ranged is not None:
        summary.update(HTTP_range_requests=ranged.requests, HTTP_range_reconnects=ranged.retries,
                       HTTP_confirmed_offset=ranged.offset)
    with (output / "official-stream-receipt-private.json").open("x", encoding="utf-8") as dest:
        json.dump(summary, dest, sort_keys=True)
    print(json.dumps({k: v for k,v in summary.items() if k != "selected_frames"}, sort_keys=True), flush=True)
    return summary["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", choices=ARCHIVES)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--recover-original-timeout", type=Path)
    parser.add_argument("--recover-socket-timeout", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if extract_source(args.sequence, args.source, args.output, args.seconds,
                                         args.recover_original_timeout, args.recover_socket_timeout) else 1)


if __name__ == "__main__":
    main()
