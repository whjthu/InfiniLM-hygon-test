"""Read metadata for exact HYTextModel checkpoint candidates on the target."""

from hashlib import sha256
import json
from pathlib import Path
import re


SHARED = Path("/public/home/aclsylqidf/shared")
MODELS = {
    "qwen3": {
        "exact_roots": ("Qwen3-8B", "Qwen/Qwen3-8B"),
        "hidden_size": 4096,
        "model_type": "qwen3",
        "tokenizer_members": ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
    },
    "clipl": {
        "exact_roots": ("clip-vit-large-patch14", "openai/clip-vit-large-patch14"),
        "hidden_size": 768,
        "model_type": "clip",
        "tokenizer_members": ("tokenizer_config.json", "vocab.json", "merges.txt"),
    },
}


def metadata(path, maximum):
    if not path.is_file():
        return None, None
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("bounded text asset metadata exceeded")
    value = json.loads(raw)
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}, value


def sized_file(path):
    return path.stat().st_size if path.is_file() else None


def probe(path, spec):
    config_pin, config = metadata(path / "config.json", 256 * 1024)
    result = {"path": str(path), "config": config_pin}
    if config_pin is None:
        return result
    text = config.get("text_config", config)
    model_type = config.get("model_type")
    result.update(model_type=model_type, hidden_size=text.get("hidden_size"),
                  configuration_matches=model_type == spec["model_type"] and text.get("hidden_size") == spec["hidden_size"])
    result["tokenizer_members_bytes"] = {name: sized_file(path / name) for name in spec["tokenizer_members"]}
    index_pin, index = metadata(path / "model.safetensors.index.json", 2 * 1024 * 1024)
    if index_pin:
        shard_names = sorted(set(index["weight_map"].values()))
        if not 1 <= len(shard_names) <= 64 or not all(
                re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name) for name in shard_names):
            raise ValueError("unexpected weight shard index names")
        result["weight_index"] = index_pin
        result["weight_shards_bytes"] = {name: sized_file(path / name) for name in shard_names}
    else:
        result["weight_shards_bytes"] = {name: sized_file(path / name) for name in ("model.safetensors", "pytorch_model.bin")}
    return result


def main():
    result = {name: [probe(SHARED / root, spec) for root in spec["exact_roots"]]
              for name, spec in MODELS.items()}
    print(json.dumps({"exact_candidate_observations": result, "weight_bodies_read": 0,
                      "native_encoder_calls": 0, "GPU_steps": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
