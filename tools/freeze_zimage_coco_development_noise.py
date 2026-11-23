"""Freeze and replay byte-identical BF16 starting noise for the T07 development cohort."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


ROSTER_SHA256 = "9e1603e5626fdca032cf8b59b63a69c4e26ee410064f507b9f521df8c8234a63"
SHAPE = (1, 16, 32, 32)
COUNT = 16 * 32 * 32
BYTES_PER_IMAGE = 2 * COUNT
DEVELOPMENT_COUNT = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def data_sha256(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def bf16_rne(values, np):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def freeze(roster_path, output):
    import numpy as np

    raw = roster_path.read_bytes()
    require(sha256(raw).hexdigest() == ROSTER_SHA256, "original COCO roster changed")
    plans = json.loads(raw)["subsets"]["development"]
    seeds = [row["seed"] for row in plans]
    ids = [row["image_id"] for row in plans]
    require(len(plans) == len(set(seeds)) == len(set(ids)) == DEVELOPMENT_COUNT
            and all(type(seed) is int and 0 <= seed < 2**64 for seed in seeds), "invalid development cohort or seeds")
    output.mkdir(mode=0o700)
    bank = output / "development-bf16-le.private.bin"
    combined = sha256()
    rows = {}
    with bank.open("xb") as stream:
        for index, row in enumerate(plans):
            seed = row["seed"]
            rng = np.random.Generator(np.random.PCG64(seed))
            values = rng.standard_normal(COUNT, dtype=np.float32)
            require(values.shape == (COUNT,) and np.isfinite(values).all(), "non-finite generated noise")
            exact = bf16_rne(values, np)
            require(((exact & 0x7F80) != 0x7F80).all(), "non-finite frozen BF16 noise")
            data = exact.tobytes()
            stream.write(data)
            combined.update(data)
            rows[str(row["image_id"])] = {
                "ordinal": index, "offset": index * BYTES_PER_IMAGE, "seed": seed,
                "prompt_SHA256": row["prompt_SHA256"], "bytes": len(data), "SHA256": sha256(data).hexdigest(),
            }

    document = {
        "schema": "training.zimage.coco.development_noise_bank.v1",
        "roster_SHA256": ROSTER_SHA256, "shape": SHAPE, "dtype": "BF16 little endian", "cohort": "development",
        "generator": "NumPy Generator(PCG64), float32 standard_normal, round-to-nearest-even to BF16",
        "numpy_version": np.__version__, "decoder_independent": True,
        "bank": {"filename": bank.name, "bytes": DEVELOPMENT_COUNT * BYTES_PER_IMAGE, "SHA256": combined.hexdigest()},
        "rows": rows, "blind_noise_frozen": False, "measured_images": 0,
    }
    require(data_sha256(bank) == document["bank"]["SHA256"], "written bank identity differs")
    manifest = output / "development-noise.private.json"
    manifest_raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with manifest.open("xb") as stream:
        stream.write(manifest_raw)
    print(json.dumps({"cohort": "development", "images": len(rows), "bank": document["bank"],
                      "manifest": {"filename": manifest.name, "bytes": len(manifest_raw),
                                   "SHA256": sha256(manifest_raw).hexdigest()},
                      "generator": document["generator"], "numpy_version": np.__version__,
                      "measured_images": 0}, sort_keys=True), flush=True)


class Replay:
    def __init__(self, manifest_path, expected_manifest_sha, roster_path):
        raw = manifest_path.read_bytes()
        require(sha256(raw).hexdigest() == expected_manifest_sha, "development noise manifest differs")
        roster = roster_path.read_bytes()
        require(sha256(roster).hexdigest() == ROSTER_SHA256, "original COCO roster changed")
        doc = json.loads(raw)
        require(doc["schema"] == "training.zimage.coco.development_noise_bank.v1" and doc["roster_SHA256"] == ROSTER_SHA256
                and tuple(doc["shape"]) == SHAPE and doc["dtype"] == "BF16 little endian", "noise manifest layout differs")
        require(doc["bank"]["filename"] == "development-bf16-le.private.bin"
                and doc["bank"]["bytes"] == DEVELOPMENT_COUNT * BYTES_PER_IMAGE, "noise bank layout differs")
        self._bank = manifest_path.parent / doc["bank"]["filename"]
        self._rows = doc["rows"]
        plans = json.loads(roster)["subsets"]["development"]
        require(len(self._rows) == len(plans) == DEVELOPMENT_COUNT, "noise manifest cohort differs")
        for index, row in enumerate(plans):
            fields = self._rows[str(row["image_id"])]
            require(fields["ordinal"] == index and fields["offset"] == index * BYTES_PER_IMAGE
                    and fields["bytes"] == BYTES_PER_IMAGE and fields["seed"] == row["seed"]
                    and fields["prompt_SHA256"] == row["prompt_SHA256"], "noise row does not match the frozen roster")
        require(self._bank.stat().st_size == doc["bank"]["bytes"]
                and data_sha256(self._bank) == doc["bank"]["SHA256"], "sealed noise bank differs")

    def sample(self, image_id, prompt_sha, seed):
        row = self._rows[str(image_id)]
        require(row["prompt_SHA256"] == prompt_sha and row["seed"] == seed, "noise is bound to another request")
        with self._bank.open("rb") as stream:
            stream.seek(row["offset"])
            data = stream.read(BYTES_PER_IMAGE)
        require(len(data) == row["bytes"] and sha256(data).hexdigest() == row["SHA256"], "noise payload changed")
        return data, row["SHA256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.roster, args.output)


if __name__ == "__main__":
    main()
