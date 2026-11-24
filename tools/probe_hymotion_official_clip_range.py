"""Check the pinned official CLIP-L configuration and bounded checkpoint range."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
BASE = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/" + REVISION + "/"
CHECKPOINT_SIZE = 1710540580


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        parsed = urlsplit(new_url)
        host = parsed.hostname or ""
        official_hosts = {"huggingface.co", "us.aws.cdn.hf.co"}
        if parsed.scheme != "https" or parsed.username or parsed.password or not (
                host in official_hosts or host.endswith(".huggingface.co") or host.endswith(".xethub.hf.co")):
            raise ValueError("unexpected official asset redirect")
        return super().redirect_request(request, response, code, message, headers, new_url)


def config(opener):
    with opener.open(BASE + "config.json", timeout=8) as response:
        raw = response.read(65537)
    if not 0 < len(raw) <= 65536:
        raise ValueError("configuration response exceeds the bound")
    value = json.loads(raw)
    if value["model_type"] != "clip" or value["text_config"]["hidden_size"] != 768:
        raise ValueError("CLIP-L text configuration differs")
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest(), "text_hidden_size": 768}


def checkpoint_prefix(opener):
    request = Request(BASE + "model.safetensors", headers={"Range": "bytes=0-7"})
    with opener.open(request, timeout=8) as response:
        received_range = response.headers.get("Content-Range")
        if response.status != 206 or received_range != f"bytes 0-7/{CHECKPOINT_SIZE}":
            return {"status": response.status, "exact_requested_range": False, "body_bytes": 0}
        raw = response.read(9)
    if len(raw) != 8 or not 1 <= int.from_bytes(raw, "little") <= 1024 * 1024:
        raise ValueError("unexpected safetensors header length")
    return {"status": 206, "exact_requested_range": True, "body_bytes": len(raw),
            "safetensors_header_bytes": int.from_bytes(raw, "little"), "prefix_SHA256": sha256(raw).hexdigest()}


def attempt(opener, operation):
    try:
        return operation(opener)
    except HTTPError as error:
        return {"error": type(error).__name__, "status": error.code}
    except (OSError, URLError, ValueError, KeyError, TypeError) as error:
        return {"error": type(error).__name__, "underlying_error": type(getattr(error, "reason", None)).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    opener = build_opener(OfficialRedirects())
    value = {"revision": REVISION, "official_configuration": attempt(opener, config),
             "checkpoint_access": attempt(opener, checkpoint_prefix), "local_checkpoint_files_written": 0}
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"SHA256": sha256(raw).hexdigest(), "observation": value}, sort_keys=True))


if __name__ == "__main__":
    main()
