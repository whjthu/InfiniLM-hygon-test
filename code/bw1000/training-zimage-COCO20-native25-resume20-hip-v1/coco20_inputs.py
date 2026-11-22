"""Bind every original 20-row COCO occurrence to the actual target preprocessing."""

from hashlib import sha256
import json

from coco20_step import MEDIA, PROOF, exact, prepare, require


def load_dataset(root, claim, deadline):
    from diffsynth.core import UnifiedDataset

    _, first = prepare(root, claim, deadline)
    raw = exact(PROOF, claim["preprocess_stdout"])
    [pre] = [json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
    by_step = pre["input_values"]
    require(len(by_step) == 25 and [
        (row["step"], row["index"]) for row in by_step
    ] == [(s, (s - 1) % 20) for s in range(1, 26)], "Wrong original COCO20 occurrences")
    by_index = by_step[:20]
    for row in by_step[20:]:
        first_occurrence = by_index[row["index"]]
        require(all(row[k] == first_occurrence[k] for k in ("RGB256_SHA256", "prompt_SHA256")),
                "Repeated original COCO20 input bytes differ")
    require(by_index[0]["RGB256_SHA256"] == first["RGB256_SHA256"], "Step 1 target preprocessing drift")
    dataset = UnifiedDataset(base_path=str(MEDIA), metadata_path=str(MEDIA / "metadata-zimage-20.csv"),
        repeat=1, data_file_keys=("image",), main_data_operator=UnifiedDataset.default_image_operator(
            base_path=str(MEDIA), height=256, width=256, height_division_factor=16, width_division_factor=16))
    require(len(dataset) == len(by_index) == 20, "The complete original training dataset is required")
    return dataset, by_index, first["original_manifest_present"]


def sample(dataset, rows, index):
    expected = rows[index]
    data = dataset[index]
    image = data["image"]
    identity = {"RGB256_SHA256": sha256(image.tobytes()).hexdigest(),
                "prompt_SHA256": sha256(data["prompt"].encode()).hexdigest()}
    require(image.mode == "RGB" and image.size == (256, 256) and expected["index"] == index
            and all(identity[k] == expected[k] for k in identity), "Live COCO20 input differs from the frozen roster")
    return data, identity
