"""Stage only the five pinned public Microsoft pairs on the cluster."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


STAGE = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/trellis-native-generation-microsoft-v1")
ASSET_SHA = "cf83b1c4d52ca7c3978778bfa454a2ca6b5ed8608b3bf728032d646479e53dd7"
CHUNK = 32 * 1024**2


def record(path, data):
    temp = path.with_suffix(".tmp")
    with temp.open("x", encoding="utf8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
    temp.replace(path)


def budget(deadline):
    remain = deadline - time.monotonic()
    if remain <= 0:
        raise TimeoutError("original transfer deadline")
    return min(24, remain)


def range_bytes(context, url, headers, start, count, total, deadline):
    last = start + count - 1
    request = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{last}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=budget(deadline), context=context) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{last}/{total}":
                    raise ValueError("upstream byte range differs")
                raw = response.read(count + 1)
            if len(raw) != count:
                raise OSError("short range")
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(min(2, budget(deadline)))
    raise AssertionError("range retry loop exhausted")


def header_info(path, size):
    with path.open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < n <= 8 * 1024**2:
            raise ValueError("invalid safetensors header length")
        raw = stream.read(n)
    if len(raw) != n:
        raise ValueError("short safetensors header")
    entries = json.loads(raw)
    entries.pop("__metadata__", None)
    cursor = 0
    for left, right in sorted(entry["data_offsets"] for entry in entries.values()):
        if left != cursor or right < left:
            raise ValueError("safetensors offsets are not contiguous")
        cursor = right
    if not entries or size != 8 + n + cursor:
        raise ValueError("safetensors data extent differs")
    return {"header_bytes": n, "header_SHA256": sha256(raw).hexdigest(),
            "tensor_count": len(entries), "dtypes": sorted({entry["dtype"] for entry in entries.values()})}


def download(context, component, name, pin, deadline):
    url = "https://modelscope.cn/api/v1/models/" + component["repo"] + "/repo?" + urllib.parse.urlencode(
        {"Revision": "master", "FilePath": name}
    )
    dest = STAGE / component["target_subdir"] / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    expected = pin["bytes"]
    headers = {"Accept-Encoding": "identity"}
    received = 0
    digest = sha256()
    with part.open("xb") as stream:
        if pin["kind"] == "config":
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=budget(deadline), context=context) as reply:
                if reply.status != 200:
                    raise ValueError("unexpected config response")
                raw = reply.read(expected + 1)
            if len(raw) != expected or json.loads(raw)["name"] != component["native_class"]:
                raise ValueError("config size or class differs")
            stream.write(raw)
            digest.update(raw)
            received = len(raw)
        else:
            while received < expected:
                raw = range_bytes(context, url, headers, received, min(CHUNK, expected - received), expected, deadline)
                stream.write(raw)
                digest.update(raw)
                received += len(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if received != expected or digest.hexdigest() != pin["SHA256"]:
        raise ValueError("downloaded body differs from the pinned official file")
    result = {"bytes": received, "SHA256": digest.hexdigest(), "kind": pin["kind"]}
    if pin["kind"] == "weights":
        result.update(header_info(part, received))
    # Link atomically fails if an unexpected final filename appeared.
    os.link(part, dest)
    part.unlink()
    return result


def one_component(context, spec, deadline):
    rows = {}
    state = "VERIFIED"
    error = None
    try:
        for name, pin in spec["files"].items():
            rows[name] = download(context, spec, name, pin, deadline)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        state = "FAILED"
        reason = getattr(exc, "reason", exc)
        error = {"type": type(exc).__name__, "reason_type": type(reason).__name__,
                 "http_status": getattr(exc, "code", None), "errno": getattr(reason, "errno", None),
                 "ssl_verify_code": getattr(reason, "verify_code", None)}
    result = {"component": spec["component"], "revision": spec["revision"], "repo": spec["repo"],
              "target_subdir": spec["target_subdir"], "files": rows, "state": state, "error": error}
    record(STAGE / (spec["component"] + ".json"), result)
    return result


def main():
    start = time.monotonic()
    if len(sys.argv) != 2 or len(sys.argv[1]) != 40 or set(sys.argv[1]) - set("0123456789abcdef"):
        raise ValueError("publication commit identity required")
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536 or sha256(raw).hexdigest() != ASSET_SHA:
        raise ValueError("asset manifest differs")
    asset = json.loads(raw)
    ca = next(p for p in ("/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt")
              if Path(p).is_file())
    context = ssl.create_default_context(cafile=ca)
    if shutil.disk_usage(STAGE.parent).free < asset["expected_total_bytes"] + 256 * 1024**2:
        raise OSError("insufficient project storage")
    STAGE.mkdir(mode=0o700)
    epoch = time.time()
    proc = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
    record(STAGE / "intent.json", {"publication": sys.argv[1], "asset_SHA256": ASSET_SHA,
                                   "epoch": epoch, "deadline_epoch": epoch + 880, "PID": os.getpid(),
                                   "proc_startticks": int(proc[19]), "expected_bytes": asset["expected_total_bytes"]})
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [pool.submit(one_component, context, spec, start + 850) for spec in asset["components"]]
        for run in as_completed(runs):
            item = run.result()
            rows[item["component"]] = item
    total = sum(pin["bytes"] for row in rows.values() for pin in row["files"].values())
    result = {"state": "VERIFIED" if len(rows) == 5 and all(x["state"] == "VERIFIED" for x in rows.values())
              and total == asset["expected_total_bytes"] else "INCOMPLETE",
              "components": rows, "verified_bytes": total, "expected_bytes": asset["expected_total_bytes"],
              "publication": sys.argv[1], "asset_SHA256": ASSET_SHA, "elapsed_seconds": time.monotonic() - start,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "new_Slurm_steps": 0}
    record(STAGE / "outcome.json", result)
    print(json.dumps(result))
    return int(result["state"] != "VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
