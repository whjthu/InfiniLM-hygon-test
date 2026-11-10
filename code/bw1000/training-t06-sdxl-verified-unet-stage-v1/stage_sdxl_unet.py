"""Stage the byte-identical SDXL UNet pair required by the A100 first-step replay."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.request

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
STAGE = ROOT / "results/training-t06-sdxl-unet-stage-v1"
BASE = ("https://modelscope.cn/api/v1/models/AI-ModelScope/stable-diffusion-xl-base-1.0/"
        "repo?Revision=master&FilePath=unet%2F")
CONFIG_SHA = "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6"
WEIGHT_SHA = "357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a"
WEIGHT_SIZE = 10270077736
BLOCK = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(os.environ["SLURM_JOB_ID"] == "1087656" and socket.gethostname().split(".")[0] == "f11r4n12",
            "unexpected T06 step identity")
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    deadline = time.monotonic() + 6000
    ca = next((path for path in ("/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt")
               if Path(path).is_file()), None)
    context = ssl.create_default_context(cafile=ca)
    out = STAGE / "model" / "unet"
    out.mkdir(parents=True, exist_ok=False)
    status = {"step": step, "node": "f11r4n12", "expected_weight_bytes": WEIGHT_SIZE}

    def publish(state, **extra):
        status.update(state=state, **extra)
        temporary = STAGE / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(STAGE / "status.json")

    publish("FETCHING_CONFIG", weight_bytes=0)
    require(shutil.disk_usage(out).free >= WEIGHT_SIZE + 256 * 1024**2,
            "insufficient staging disk space")
    try:
        with urllib.request.urlopen(BASE + "config.json", timeout=25, context=context) as response:
            config = response.read(4097)
        require(len(config) == 1680 and hashlib.sha256(config).hexdigest() == CONFIG_SHA,
                "configuration differs from A100 bytes")
        (out / "config.json").write_bytes(config)
        name = "diffusion_pytorch_model.safetensors"
        partial = out / (name + ".part")
        digest = hashlib.sha256()
        transferred = 0
        with partial.open("xb") as stream:
            while transferred < WEIGHT_SIZE:
                last = min(transferred + BLOCK, WEIGHT_SIZE) - 1
                for attempt in range(3):
                    require(time.monotonic() < deadline, "staging deadline reached")
                    request = urllib.request.Request(BASE + name, headers={"Range": f"bytes={transferred}-{last}"})
                    try:
                        with urllib.request.urlopen(request, timeout=30, context=context) as response:
                            expected_range = f"bytes {transferred}-{last}/{WEIGHT_SIZE}"
                            require(response.status == 206 and response.headers.get("Content-Range") == expected_range,
                                    "unexpected HTTP range or source size")
                            raw = response.read(last - transferred + 2)
                            require(len(raw) == last - transferred + 1, "incomplete weight range")
                        break
                    except (OSError, RuntimeError):
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
                stream.write(raw)
                digest.update(raw)
                transferred += len(raw)
                stream.flush()
                publish("FETCHING_WEIGHTS", weight_bytes=transferred, config_SHA256=CONFIG_SHA)
            os.fsync(stream.fileno())
        require(transferred == WEIGHT_SIZE and digest.hexdigest() == WEIGHT_SHA,
                "complete weight SHA256 differs from A100 bytes")
        partial.replace(out / name)
        publish("VERIFIED_A100_UNET", weight_SHA256=digest.hexdigest(), GPU_requested=0, training_updates=0)
    except BaseException as error:
        publish("FAILED", error_type=type(error).__name__, error=str(error)[:180])
        raise


if __name__ == "__main__":
    main()
