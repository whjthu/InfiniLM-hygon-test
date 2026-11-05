"""Derive friend-only class headers for GLM952 from exact archived source bytes."""

import argparse
from hashlib import sha256
from pathlib import Path

HEADERS = {
    "glm_moe_dsa_projected_text_model_r22.hpp": "99552503026068ea9dbdc834947a91b0a522f3c1b84ffa425dcc2fa33f438447",
    "glm_moe_dsa_projected_decoder_r22.hpp": "6868fddec9fca2a2f6e195667ef572c560b4b23ed16dbb1d5ad0ecf6e87b66b9",
    "glm_moe_dsa_projected_cache_r22.hpp": "8641388d870c8efc05743608b1b609d7966962e5c460287f85b9f2c418915745",
    "glm_moe_dsa_indexer_selection.hpp": "dc55e5b38e697aba3cb31b39ea93836f668e1281edbc2064619bcb75faf5c22d",
}


def add_access(data):
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    scope = b"namespace infinilm::models::glm_moe_dsa {" + newline
    private = b"private:" + newline
    if data.count(scope) != 1 or data.count(private) != 1:
        raise ValueError("unexpected Native class declaration")
    return data.replace(scope, scope + b"namespace block_verify { struct NativeLengthsAccess; }" + newline).replace(
        private, private + b" friend struct block_verify::NativeLengthsAccess;" + newline)


def derive(parent, output):
    if output.exists():
        raise ValueError("fresh output required")
    sources = {}
    for name, expected in HEADERS.items():
        raw = (parent / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"unexpected Native parent header: {name}")
        sources[name] = add_access(raw)
    output.mkdir()
    for name, data in sources.items():
        with (output / name).open("xb") as file:
            file.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    derive(args.parent, args.output)
