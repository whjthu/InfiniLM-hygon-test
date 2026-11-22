"""Stream the official pinned Z-Image files into the original durable receiver."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import ssl
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPSHandler, HTTPRedirectHandler, ProxyHandler, Request, build_opener


REVISION = "7ec975f8151f31281afad66b11443f0d08c125bb"
ENDPOINT = "https://www.modelscope.cn/api/v1/models/Tongyi-MAI/Z-Image-Turbo/repo"
CA_FILE = "/etc/pki/tls/certs/ca-bundle.crt"
HOSTS = {"www.modelscope.cn", "modelscope.cn", "cdn-lfs-cn-1.modelscope.cn"}
PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RECEIVER = PROJECT / "incoming/training-zimage-publisher-runtime-transfer-v1"
WIRE = PROJECT / "incoming/training-zimage-runtime-wire-pieces-v2/wire.py"
MEMORY_CAP = 32 << 20
SELECTED = (5, 6, 16, 14, 15)


def need(value, detail):
    if not value:
        raise ValueError(detail)


def allowed(url):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.hostname in HOSTS and parsed.port in (None, 443)
            and not parsed.username and not parsed.password)


class PublisherRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        need(allowed(newurl), "Refused redirect outside the verified publisher hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(opener, row, offset, count, absolute_deadline):
    end = offset + count - 1
    need(0 <= offset <= end < row["bytes"] and count <= MEMORY_CAP, "Out-of-bounds source range")
    url = ENDPOINT + "?" + urlencode({"Revision": REVISION, "FilePath": row["relative_path"]})
    request = Request(url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
    started = time.monotonic()
    with opener.open(request, timeout=12) as response:
        need(allowed(response.geturl()) and response.status == 206
             and response.headers.get("Content-Range") == f"bytes {offset}-{end}/{row['bytes']}"
             and response.headers.get("Content-Length") == str(count)
             and response.headers.get("Content-Encoding", "identity").lower() == "identity",
             "Publisher response is not the exact requested range")
        data = bytearray()
        while len(data) < count:
            if time.time() >= absolute_deadline:
                raise TimeoutError("Direct publisher request exceeded its phase budget")
            chunk = response.read1(min(1 << 20, count - len(data)))
            need(chunk, "Publisher range ended early")
            data.extend(chunk)
        need(response.read(1) == b"", "Publisher response exceeds its declared length")
    return data, time.monotonic() - started


def save(path, result):
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def load_receiver(claim):
    data = WIRE.read_bytes()
    need(WIRE.resolve(strict=True) == WIRE and len(data) == claim["wire"]["bytes"]
         and sha256(data).hexdigest() == claim["wire"]["SHA256"], "Installed wire helper differs")
    scope = {"__name__": "frozen_wire"}
    exec(compile(data, str(WIRE), "exec"), scope)
    receiver, _, stage = scope["load_legacy"](RECEIVER)
    return receiver, stage, scope["Pieces"](receiver.TARGET, receiver, stage)


def transfer(claim, until, report, record):
    receiver, stage, pieces = load_receiver(claim)
    rows = receiver.assets()
    by_id = {row["asset_id"]: row for row in rows}
    selected = tuple(claim["asset_order"])
    need(selected == SELECTED, "Direct asset order differs")
    for key in selected:
        row = by_id[key]
        pin = claim["publisher_members"][str(key)]
        need((pin["path"], pin["bytes"], pin["SHA256"]) == (row["relative_path"], row["bytes"], row["SHA256"]),
             "Publisher metadata disagrees with the frozen receiver")
    remaining = 0
    for key in selected:
        row = by_id[key]
        state = pieces.state(row)
        need(state["wire_bytes"] == 0, "Original wire pieces require their existing owner")
        remaining += row["bytes"] - state["confirmed_bytes"]
        report["initial"][str(key)] = state["confirmed_bytes"]
        report["confirmed"][str(key)] = state["confirmed_bytes"]
    need(shutil.disk_usage(receiver.TARGET).free >= remaining + (1 << 30), "Target capacity is insufficient")
    opener = build_opener(ProxyHandler({}), PublisherRedirect(),
                          HTTPSHandler(context=ssl.create_default_context(cafile=CA_FILE)))
    for key in selected:
        row = by_id[key]
        while True:
            status = pieces.state(row)
            offset = status["confirmed_bytes"]
            need(status["wire_bytes"] == 0 and offset == report["confirmed"][str(key)],
                 "Original durable offset was concurrently advanced")
            if offset == row["bytes"]:
                if not status["whole_file_qualified_at_seal"]:
                    if until - time.time() < 85:
                        return
                    started = time.monotonic()
                    proof = receiver.seal(receiver.TARGET, row)
                    need(proof["whole_file_qualified"] and proof["SHA256"] == row["SHA256"]
                         and proof["bytes"] == row["bytes"], "Whole original file did not qualify")
                    record(dict(event="seal", asset=key, bytes=row["bytes"], SHA256=row["SHA256"],
                                wall_s=time.monotonic() - started))
                report["sealed"].append(key)
                break
            if until - time.time() < 32:
                return
            count = stage.segment_length(receiver.storage_spec(row), offset)
            data, source_time = fetch(opener, row, offset, count, until-12)
            digest = sha256(data).hexdigest()
            need(pieces.state(row, offset)["wire_bytes"] == 0, "Original wire scope became active")
            started = time.monotonic()
            received = receiver.receive(receiver.TARGET, row, offset, digest, BytesIO(data))
            need(received["confirmed_bytes"] == offset + count and received["segment_SHA256"] == digest,
                 "Receiver did not attest to this range")
            report["confirmed"][str(key)] = offset + count
            record(dict(event="commit", asset=key, bytes=count, offset=offset, confirmed=offset+count,
                        SHA256=digest, download_s=source_time, receiver_s=time.monotonic()-started))
    if until - time.time() >= 20:
        report["runtime_publication"] = receiver.publish_runtime(receiver.TARGET, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", required=True, type=float)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    claim = json.loads((root / "CLAIM.json").read_bytes())
    need(claim["request"] == args.request and claim["label"] == root.name, "Original scope differs")
    need(os.environ["SLURM_JOB_ID"] == claim["parent"] and os.environ["SLURM_STEP_ID"].isdigit()
         and socket.gethostname().split(".")[0] == claim["node"] and not os.environ.get("SLURM_STEP_GPUS"),
         "Expected the original zero-GRES compute step")
    until = args.epoch + claim["worker_seconds"]
    need(args.epoch <= time.time() < until-60, "Direct work epoch already exhausted")
    report = dict(request=args.request, step=claim["parent"] + "." + os.environ["SLURM_STEP_ID"],
                  epoch=args.epoch, until=until, initial={}, confirmed={}, sealed=[], runtime_publication=None,
                  GPU_calls=0, training_ready=False, status="BOUNDED_PROGRESS")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Direct transfer work deadline")))
    signal.setitimer(signal.ITIMER_REAL, until-time.time())
    started = time.monotonic()
    try:
        save(root/"worker-started.json", {key: report[key] for key in ("request", "epoch", "step", "until")})
        with (root / "worker-progress.jsonl").open("x", encoding="utf8") as f:
            def record(value):
                value["elapsed_s"] = time.monotonic()-started
                f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            transfer(claim, until, report, record)
            if report["runtime_publication"] is not None:
                report["status"] = "PUBLISHED_ALL_ORIGINAL_RUNTIME_FILES"
    except Exception as error:
        report.update(status="STOP_AT_ORIGINAL_DURABLE_OFFSET", error_type=type(error).__name__,
                      error_code=getattr(error, "code", None),
                      brief=str(error)[:120] if type(error) in (ValueError, TimeoutError) else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    report["elapsed_s"] = time.monotonic()-started
    save(root/"worker-result.json", report)
    return 1 if report["status"] == "STOP_AT_ORIGINAL_DURABLE_OFFSET" else 0


if __name__ == "__main__":
    raise SystemExit(main())
