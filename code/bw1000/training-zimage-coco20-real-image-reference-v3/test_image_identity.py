"""Regression fixture for the real PIL input observed in step 1103606.1696."""

import hashlib
from pathlib import Path
import runpy
from types import SimpleNamespace

from PIL import Image


path = Path(__file__).resolve().parent / "framework-delta/generation/src/gencuda/zimage_fixed_reference.py"
FixedReference = runpy.run_path(str(path))["FixedReference"]
boundary = object.__new__(FixedReference)
boundary.torch = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))


def test():
    image = Image.new("RGB", (256, 256), (29, 100, 220))
    identity = dict(kind="PIL.Image", mode=image.mode, size=list(image.size),
                    decoded_pixels_SHA256=hashlib.sha256(image.tobytes()).hexdigest())
    prepared = boundary._place({"input_image": image}, {"input_image": identity}, "inputs")
    assert prepared["input_image"] is image
    assert hashlib.sha256(image.tobytes()).hexdigest() == identity["decoded_pixels_SHA256"]

    altered = image.copy()
    altered.putpixel((255, 1), (29, 100, 221))
    bad_images = [altered, image.resize((257, 256)), image.convert("L"), dict(identity)]
    invalid_captures = [(native, identity) for native in bad_images]
    invalid_captures.append((image, {**identity, "extra": 0}))
    rejected = 0
    for native, marker in invalid_captures:
        try:
            boundary._place(native, marker, "input_image")
        except ValueError as error:
            assert str(error) == "Realized Z-Image image pixels differ at input_image"
            rejected += 1
        else:
            raise AssertionError("Invalid image identity reached the native forward")
    print(f"live image identity preserved; {rejected} invalid captures refused")


if __name__ == "__main__":
    test()
