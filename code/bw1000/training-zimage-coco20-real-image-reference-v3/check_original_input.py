"""Compare the original COCO20 native PIL image with the saved replay operand."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


OLD_RUN = Path("/original-run")
FRAMEWORK = Path("/common-framework/generation/src")
IMAGE_REFERENCE_PROGRESS = "a1694d2ffc3491d5c89d896cd73260f9ded215bd83e6ea5eba78e5b14403d2d0"


def main():
    import torch

    sys.path[:0] = [str(OLD_RUN), str(FRAMEWORK)]
    from coco20_step import prepare, require
    from gencuda.zimage_first_update import safe_input
    from gencuda.zimage_fixed_reference import load_fixed_reference

    require(Path(importlib.util.find_spec("gencuda.zimage_fixed_reference").origin).is_relative_to(FRAMEWORK),
            "Unexpected fixed replay source")
    require(not torch.cuda.is_available(), "Zero-GRES input audit can see an accelerator")
    claim = json.loads((OLD_RUN / "CLAIM.json").read_bytes())
    ends = time.monotonic() + 60

    def deadline():
        require(time.monotonic() < ends, "Native image validation deadline elapsed")

    data, identity = prepare(OLD_RUN, claim, deadline)
    image = data["image"]
    replay = load_fixed_reference(torch, Path("/original-first"), IMAGE_REFERENCE_PROGRESS)
    marker = replay.reference["inputs"]["input_image"]
    require(replay._place(image, marker, "input_image") is image,
            "Native image object did not survive the replay boundary")
    require(safe_input(image) == marker, "Native image did not serialize to the original marker")
    require(hashlib.sha256(image.tobytes()).hexdigest() == identity["RGB256_SHA256"],
            "Original decoded image differs")

    mutated = image.copy()
    pixel = mutated.getpixel((0, 0))
    mutated.putpixel((0, 0), (pixel[0] ^ 1, *pixel[1:]))
    try:
        replay._place(mutated, marker, "input_image")
    except ValueError:
        pass
    else:
        raise ValueError("Changed native image passed fixed replay")
    deadline()
    print(json.dumps({"passed": True, "mode": image.mode, "size": list(image.size),
                      "decoded_pixels_SHA256": identity["RGB256_SHA256"],
                      "validated_dataset_rows": identity["rows"],
                      "original_preprocess_step": identity["preprocessing_step"],
                      "original_reference_progress_SHA256": IMAGE_REFERENCE_PROGRESS,
                      "image_identity_preserved": True, "changed_pixel_rejected": True,
                      "actual_updates": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
