"""Stage the exact TRELLIS Shape Flow training assets in private cluster storage."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import time
import urllib.parse
import urllib.request

root = Path(__file__).resolve().parent
base_url = "https://modelscope.cn/api/v1/models/microsoft/TRELLIS.2-4B/repo?"
prefix = "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
weights_bytes = 2584574424
weights_sha = "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f"
metadata = {
    "pipeline.json": "222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6",
    prefix + ".json": "ddbd7a1d34ce9a8e7af2f6ef9ae686d9ab49f69e62dacb98d8409b511cdbd5e2",
}


def uri(name):
    return base_url + urllib.parse.urlencode({"Revision": "master", "FilePath": name})


def publish(state, received, **extra):
    result = {"state": state, "received_bytes": received, "expected_bytes": weights_bytes, **extra}
    temp = root / "status.tmp"
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(root / "status.json")


def main():
    ca = next((p for p in ("/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt")
               if Path(p).is_file()), None)
    context = ssl.create_default_context(cafile=ca)
    model = root / "model"
    (model / "ckpts").mkdir(parents=True)
    if shutil.disk_usage(root).free < weights_bytes + 256 * 1024**2:
        raise OSError("insufficient storage")
    end = time.monotonic() + 850
    received = 0
    publish("FETCHING_CONFIG", received)
    try:
        for name, expected in metadata.items():
            with urllib.request.urlopen(uri(name), timeout=20, context=context) as reply:
                data = reply.read(32769)
            if len(data) > 32768 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("component configuration differs")
            (model / name).write_bytes(data)
        name = prefix + ".safetensors"
        part = model / (name + ".part")
        digest = hashlib.sha256()
        with part.open("xb") as stream:
            while received < weights_bytes:
                last = min(weights_bytes, received + 64 * 1024**2) - 1
                for attempt in range(3):
                    if time.monotonic() >= end:
                        raise TimeoutError("overall checkpoint transfer deadline")
                    request = urllib.request.Request(uri(name), headers={"Range": f"bytes={received}-{last}"})
                    try:
                        with urllib.request.urlopen(request, timeout=25, context=context) as reply:
                            expected_range = f"bytes {received}-{last}/{weights_bytes}"
                            if reply.status != 206 or reply.headers.get("Content-Range") != expected_range:
                                raise OSError("unexpected response range")
                            data = reply.read(last - received + 2)
                        if len(data) != last - received + 1:
                            raise OSError("short response range")
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
                stream.write(data)
                stream.flush()
                digest.update(data)
                received += len(data)
                publish("FETCHING_WEIGHTS", received)
            os.fsync(stream.fileno())
        if digest.hexdigest() != weights_sha:
            raise ValueError("full checkpoint SHA-256 differs")
        part.rename(model / name)
        publish("VERIFIED_A100_SHAPE_FLOW", received, weights_SHA256=weights_sha, metadata_SHA256=metadata)
    except (OSError, ValueError) as error:
        publish("FAILED", received, error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
