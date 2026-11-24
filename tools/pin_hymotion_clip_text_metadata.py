"""Pin official CLIP-L text metadata without retaining downloaded files."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from probe_hymotion_official_clip_range import BASE, REVISION, OfficialRedirects, build_opener


LIMITS = {"config.json": 8192, "merges.txt": 600000, "special_tokens_map.json": 4096,
          "tokenizer.json": 2400000, "tokenizer_config.json": 8192, "vocab.json": 1100000}
CONFIG_SHA256 = "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"


def fetch_member(member):
    name, maximum = member
    with build_opener(OfficialRedirects()).open(BASE + name, timeout=10) as response:
        raw = response.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError(f"unexpected official bytes for {name}")
    digest = sha256(raw).hexdigest()
    if name == "config.json":
        model = json.loads(raw)
        if digest != CONFIG_SHA256 or model.get("text_config", {}).get("hidden_size") != 768:
            raise ValueError("CLIP-L configuration differs from the pinned target")
    return name, {"bytes": len(raw), "SHA256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as threads:
        result = dict(threads.map(fetch_member, LIMITS.items()))
    document = dict(revision=REVISION, asset_identity=result, model_weight_body_bytes=0,
                    local_payload_files_written=0, target_installation=False)
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(SHA256=sha256(raw).hexdigest(), bytes=sum(row["bytes"] for row in result.values()),
                          members=len(result))))


if __name__ == "__main__":
    main()
