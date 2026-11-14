"""Check runtime providers and network channels for the GLM TP16 transport comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re

RCCL_SHA256 = "32aa13da8bbd4d9a4ec37be08014f0783f5f576c7439dc10a3e1c661aa35427a"
PLUGIN_SHA256 = "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
INFINICCL_SHA256 = "76eb65adacc8a6832970991ca276b32562543b7fb7e8dd09cda9f4c269f24809"
NETWORK = {"socket": "Socket", "shca": "IBext_v8"}
BASENAMES = {
    "infiniccl": re.compile(r"libinfiniccl\.so(?:\.\d+)*"),
    "rccl": re.compile(r"librccl\.so(?:\.\d+)*"),
    "plugin": re.compile(r"librccl-net-shca\.so(?:\.\d+)*"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_providers(maps, mode, *, hash_file=sha256):
    """Run in the collective process after its first successful collective."""
    require(mode in NETWORK, "unknown transport mode")
    selected = {name: set() for name in BASENAMES}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[-1].startswith("/"):
            continue
        mapped = fields[-1]
        path = Path(mapped.removesuffix(" (deleted)"))
        for name, pattern in BASENAMES.items():
            if pattern.fullmatch(path.name):
                require(mapped == path.as_posix(), "mapped provider was deleted")
                selected[name].add(path)
    expected = {"infiniccl": INFINICCL_SHA256, "rccl": RCCL_SHA256}
    if mode == "shca":
        expected["plugin"] = PLUGIN_SHA256
    else:
        require(not selected["plugin"], "SHCA plugin unexpectedly mapped in Socket arm")
    identities = {}
    for name, digest in expected.items():
        require(len(selected[name]) == 1, f"expected a unique mapped {name} image")
        path, = selected[name]
        require(hash_file(path) == digest, f"unexpected mapped {name} SHA256")
        identities[name] = {"path": path.as_posix(), "SHA256": digest}
    return identities


def network_channels(log, mode):
    """Require measured cross-node channels using the requested transport."""
    require(mode in NETWORK, "unknown transport mode")
    selected = re.findall(r"\bUsing network\s+([\w.-]+)", log)
    channels = re.findall(r"\bvia NET/([\w.-]+)", log)
    expected = NETWORK[mode]
    permitted = {expected, "IBext"} if mode == "shca" else {expected}
    require(bool(selected) and set(selected) == {expected}, "RCCL selected a different network")
    require(bool(channels) and set(channels) <= permitted, "missing channels or transport fallback")
    return {"network": expected, "channel_count": len(channels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=NETWORK)
    parser.add_argument("log", type=Path, help="Original RCCL diagnostic log")
    args = parser.parse_args()
    print(json.dumps(network_channels(args.log.read_text(encoding="utf-8"), args.mode)))


if __name__ == "__main__":
    main()
