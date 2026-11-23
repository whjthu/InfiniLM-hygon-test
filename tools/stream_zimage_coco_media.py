"""Stream the pinned public COCO archive over an already authorized SSH command."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import subprocess
from threading import Thread, Timer
import urllib.request

from seal_zimage_coco_media import ARCHIVE_BYTES, ARCHIVE_SHA256, ROSTER_SHA256, TRAINING_SHA256, need


URL = ("https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/"
       "fbbee70f52f540987cca6936c9f1d7b1ff969ad1/val2017.zip")
SPAN = 16 << 20
PREFETCH = 3


def fetch(span):
    first, end = span
    request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{end}", "Accept-Encoding": "identity"})
    length = end - first + 1
    with urllib.request.urlopen(request, timeout=30) as response:
        need(response.status == 206 and response.headers.get("Content-Range") == f"bytes {first}-{end}/{ARCHIVE_BYTES}",
             "mirror response range differs")
        data = response.read(length + 1)
        need(len(data) == length, "mirror response has the wrong byte count")
        return data


def copy_mirror(destination):
    spans = iter((start, min(start+SPAN, ARCHIVE_BYTES)-1) for start in range(0, ARCHIVE_BYTES, SPAN))
    futures = deque()
    state = sha256()
    with ThreadPoolExecutor(max_workers=PREFETCH) as executor:
        for span in (next(spans, None) for _ in range(PREFETCH)):
            if span is not None:
                futures.append(executor.submit(fetch, span))
        while futures:
            data = futures.popleft().result()
            need(destination.write(data) == len(data), "short SSH write")
            state.update(data)
            span = next(spans, None)
            if span is not None:
                futures.append(executor.submit(fetch, span))
    return state.hexdigest()


def transfer(ssh_argv, request, roster, training):
    need(sha256(roster).hexdigest() == ROSTER_SHA256 and sha256(training).hexdigest() == TRAINING_SHA256,
         "source manifest or roster differs")
    payload = roster + b"\x00" + training
    diagnostics = sha256()
    with subprocess.Popen(ssh_argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as session:
        def drain_stderr():
            while data := session.stderr.read(64 << 10):
                diagnostics.update(data)

        def terminate_local_transport():
            if session.poll() is None:
                session.kill()

        drainer = Thread(target=drain_stderr, daemon=True)
        drainer.start()
        deadline = Timer(745, terminate_local_transport)
        deadline.daemon = True
        deadline.start()
        try:
            session.stdin.write(len(payload).to_bytes(4, "big") + payload)
            session.stdin.flush()
            ready_bytes = session.stdout.readline(1025)
            need(0 < len(ready_bytes) <= 1024, "missing bounded remote ready receipt")
            ready = json.loads(ready_bytes)
            need(ready == dict(ready=True, request=request, archive_bytes=ARCHIVE_BYTES), "wrong remote receiver receipt")
            digest = copy_mirror(session.stdin)
            session.stdin.close()
            need(digest == ARCHIVE_SHA256, "mirror bytes differ from the original archive")
            raw = session.stdout.readline(8193)
            need(0 < len(raw) <= 8192, "missing bounded remote terminal receipt")
            receipt = json.loads(raw)
            need(receipt["request"] == request and receipt["archive_SHA256"] == digest
                 and receipt["archive_bytes"] == ARCHIVE_BYTES, "wrong remote byte-seal receipt")
            need(session.wait(timeout=8) == 0, "receiver returned failure")
            drainer.join(timeout=4)
            return dict(receipt, sender_verified_SHA256=digest, stderr_SHA256=diagnostics.hexdigest())
        except Exception:
            terminate_local_transport()
            raise
        finally:
            deadline.cancel()
