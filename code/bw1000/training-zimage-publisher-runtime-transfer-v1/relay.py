"""Fetch exact official Z-Image asset ranges in memory and relay to an explicit receiver."""

from hashlib import sha256
import math
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from receiver import assets, storage_spec
import stage_internvl_weight_segment as stage


MAX_METADATA_BODY = 256 * 1024


def official_endpoint(url):
    endpoint = urlsplit(url)
    return (endpoint.scheme == "https" and endpoint.username is None and endpoint.password is None
            and endpoint.port in (None, 443) and (endpoint.hostname == "huggingface.co"
            or (endpoint.hostname or "").endswith((".cdn.hf.co", ".hf.co"))))


class OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, new_url):
        stage.need(official_endpoint(new_url), "publisher redirect leaves approved HTTPS endpoints")
        return super().redirect_request(req, fp, code, msg, headers, new_url)


def fetch(row, offset, deadline, *, open_url=None):
    open_url = build_opener(OfficialRedirect()).open if open_url is None else open_url
    count = stage.segment_length(storage_spec(row), offset)
    end = offset + count - 1
    raw = bytearray()
    stage.need(official_endpoint(row["publisher_URL"]), "publisher HTTPS source differs")
    for attempt in range(4):
        first = offset + len(raw)
        remaining = deadline - time.monotonic()
        stage.need(remaining > 0, "publisher range budget exhausted")
        request = Request(row["publisher_URL"], headers={"Range": f"bytes={first}-{end}",
                                                        "Accept-Encoding": "identity"})
        try:
            with open_url(request, timeout=min(20, remaining)) as response:
                stage.need(official_endpoint(response.geturl()), "unexpected publisher response endpoint")
                permitted_small_body = (response.status == 200 and first == 0 and count == row["bytes"]
                                        and count <= MAX_METADATA_BODY and not row["tensor_payload"])
                exact_range = (response.status == 206 and
                               response.headers.get("Content-Range") == f"bytes {first}-{end}/{row['bytes']}")
                stage.need((permitted_small_body or exact_range)
                           and response.headers.get("Content-Length") == str(end - first + 1)
                           and response.headers.get("Content-Encoding", "identity").lower() == "identity",
                           "publisher did not return the exact immutable range")
                while len(raw) < count:
                    stage.need(time.monotonic() < deadline, "publisher range budget exhausted")
                    block = response.read1(min(1024 * 1024, count - len(raw)))
                    stage.need(bool(block), "short publisher range")
                    raw.extend(block)
                stage.need(response.read(1) == b"" and time.monotonic() < deadline, "unexpected publisher body tail")
                if count == row["bytes"]:
                    stage.need(sha256(raw).hexdigest() == row["SHA256"], "publisher body differs from the manifest")
                return raw
        except (TimeoutError, ConnectionError, URLError) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            if (not isinstance(reason, (TimeoutError, ConnectionError)) or len(raw) == count or attempt == 3
                    or time.monotonic() >= deadline):
                raise


def advance(asset_id, transport, *, max_segments, deadline, record):
    """Run a bounded window; a transport error stops the window with no replay."""
    choices = [row for row in assets() if row["asset_id"] == asset_id]
    stage.need(len(choices) == 1 and 1 <= max_segments <= 16, "asset window differs")
    row = choices[0]

    def call(operation, payload=None, **args):
        remain = math.floor(deadline - time.monotonic())
        if remain < 5:
            raise TimeoutError(f"receiver window exhausted before {operation}")
        command = [operation, "--asset", str(asset_id), "--budget", str(min(180, remain - 2))]
        for name, value in args.items():
            command.extend(["--" + name, str(value)])
        result = transport(command, payload, remain)
        stage.need(result["file_id"] == asset_id, "receiver replied for a different asset")
        record(operation, result)
        return result

    next_state = call("inspect")
    if next_state.get("whole_file_qualified_at_seal"):
        return next_state

    offset = next_state["confirmed_bytes"]
    for _ in range(max_segments):
        if offset == row["bytes"]:
            break
        raw = fetch(row, offset, deadline)
        digest = sha256(raw).hexdigest()
        next_state = call("append", raw, offset=offset, digest=digest)
        stage.need(next_state["confirmed_bytes"] == offset + len(raw) and next_state["segment_SHA256"] == digest,
                   "target durable prefix or digest differs")
        offset = next_state["confirmed_bytes"]
        del raw
    if offset == row["bytes"]:
        next_state = call("seal")
        stage.need(next_state["whole_file_qualified"] and next_state["SHA256"] == row["SHA256"],
                   "target whole-file qualification differs")
    return next_state
