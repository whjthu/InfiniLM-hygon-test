"""Check CPU input boundaries with generated RGB-D fixtures."""

import copy
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from tum_vggt_pair import digest, prepare_pair


def write_frame(path, values):
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(values).save(path)


def validate(root):
    def write(name, text):
        (root / name).write_bytes(text.encode())

    write("rgb.txt", "100.100000 rgb/100.100000.png\n100.300000 rgb/100.300000.png\n")
    write("depth.txt", "100.090000 depth/100.090000.png\n100.295000 depth/100.295000.png\n100.310000 depth/100.310000.png\n")
    write("groundtruth.txt", "100.090000 3 4 5 0 0 0 1\n"
          "100.290000 3.10 4 5.20 0 0 0.70710678 0.70710678\n100.310000 3.10 4 5.20 0 0 0.70710678 0.70710678\n")
    first_depth = np.full((480, 640), 10000, dtype=np.uint16)
    second_depth = np.full((480, 640), 9000, dtype=np.uint16)
    write_frame(root / "depth/100.090000.png", first_depth)
    write_frame(root / "depth/100.295000.png", second_depth)
    first_image = np.zeros((480, 640, 3), dtype=np.uint8)
    first_image[:, :, 0] = 200
    second_image = np.zeros_like(first_image)
    second_image[:, :, 2] = 200
    write_frame(root / "rgb/100.100000.png", first_image)
    write_frame(root / "rgb/100.300000.png", second_image)

    names = ("rgb.txt", "depth.txt", "groundtruth.txt", "depth/100.090000.png", "depth/100.295000.png",
             "rgb/100.100000.png", "rgb/100.300000.png")
    receipt = dict(sequence="freiburg3_fixture", rgb_indices=[0, 1],
                   files_SHA256={name: digest((root / name).read_bytes()) for name in names})
    batch, evidence = prepare_pair(root, receipt)
    assert np.asarray(batch["images"]).shape == (2, 224, 224, 3)
    assert np.asarray(batch["depths"]).shape == (2, 224, 224)
    assert np.asarray(batch["point_masks"]).shape == (2, 224, 224)
    assert np.array_equal(batch["images"][0][0, 0], [200, 0, 0])
    assert np.array_equal(batch["images"][1][0, 0], [0, 0, 200])
    assert np.allclose(batch["extrinsics"][0], np.eye(4)[:3])
    assert np.allclose(batch["extrinsics"][1], ((0, 1, 0, 0), (-1, 0, 0, 0.1), (0, 0, 1, -0.2)))
    assert np.allclose(batch["intrinsics"][0][0], (535.4 * 224 / 480, 0, (320.1 - 80 + 0.5) * 224 / 480 - 0.5))
    assert evidence["matches"][1]["pose"] == "100.290000"
    assert evidence["depth_reprojection"][0]["median_abs_depth_error_m"] < 1e-6
    assert evidence["depth_reprojection"][1]["median_abs_depth_error_m"] < 1e-6

    def rejects(payload, message):
        try:
            prepare_pair(root, payload)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"expected rejection of {message}")

    def replace(name, content):
        original = (root / name).read_bytes()
        (root / name).write_bytes(content)
        changed = copy.deepcopy(receipt)
        changed["files_SHA256"][name] = digest(content)
        return changed, original

    rejects(dict(receipt, sequence="freiburg1_fixture"), "calibration")
    rejects(dict(receipt, rgb_indices=[1, 0]), "ordered")
    changed, original = replace("rgb.txt", b"100.100000 rgb/100.100000.png\n100.300000 rgb/100.300004.png\n")
    rejects(receipt, "changed input")
    rejects(changed, "timestamp differ")
    (root / "rgb.txt").write_bytes(original)
    changed, original = replace("groundtruth.txt", b"100.090000 3 4 5 0 0 0 1\n100.999000 3 4 5 0 0 0 1\n")
    rejects(changed, "20 ms")
    (root / "groundtruth.txt").write_bytes(original)
    depth_name = "depth/100.295000.png"
    original = (root / depth_name).read_bytes()
    write_frame(root / depth_name, np.full_like(first_depth, 19000))
    changed = copy.deepcopy(receipt)
    changed["files_SHA256"][depth_name] = digest((root / depth_name).read_bytes())
    rejects(changed, "metric depth agreement")
    (root / depth_name).write_bytes(original)
    print("PASS 2-frame RGB-D CPU fixture, both reprojections, 6 rejection cases")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        validate(Path(directory))
