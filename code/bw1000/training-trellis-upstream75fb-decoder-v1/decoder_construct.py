"""Load the original decoder checkpoints and exercise pinned dual-grid geometry."""

import gc
import json
import os
from pathlib import Path
import resource
from time import perf_counter

import torch
from gencuda import trellis_cpu_decode as decode


SOURCE = Path("/checkout")
WEIGHTS = Path("/trellis-weights")
PINS = {
    "shape": ("e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
              "5f1b856dffce79466fb18f5f0eefc08f08c673b24b3c05b20359c0b6d318a209", 292),
    "material": ("97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
                 "22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1", 284),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.set_num_threads(2)
    require(not torch.cuda.is_available() and os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0",
            "decoder construction is CPU-only")
    source = decode._verify_source(SOURCE)
    cls, _, path = decode._load_official_decoder(SOURCE)
    rows = {}
    for role, (weights_sha, config_sha, keys) in PINS.items():
        begin = perf_counter()
        prefix = WEIGHTS / "ckpts" / decode.DECODER_FILENAMES[role]
        require(decode._sha256(prefix.with_suffix(".json")) == config_sha, f"{role} official config differs")
        model, config, receipt = decode._load_decoder(prefix, role, 512, cls, weights_sha)
        require(receipt["strict_key_count"] == keys and receipt["config_sha256"] == config_sha,
                f"{role} decoder receipt differs")
        rows[role] = {**receipt, "model_class": config["name"], "cpu_wall_seconds": perf_counter() - begin}
        del model
        gc.collect()

    coords = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 2, 1]], dtype=torch.int32)
    intersected = torch.zeros((4, 3), dtype=torch.bool)
    intersected[0, 0] = True
    dual_vertices = torch.full((4, 3), 0.5)
    vertices, faces = decode._geometry(SOURCE, coords, dual_vertices, intersected, None,
                                       resolution=512, max_points=4)
    require(tuple(vertices.shape) == (4, 3) and tuple(faces.shape) == (2, 3),
            "real dual-grid converter did not produce one quad")
    triangles = vertices[faces.to(torch.int64)]
    normal = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    surface_area = float(torch.linalg.vector_norm(normal, dim=1).sum() * 0.5)
    require(surface_area == 512**-2, "geometry fixture area differs")
    require(decode._verify_source(SOURCE) == source, "checkout changed during decoder construction")
    evidence = {"step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                "official_source_sha256": source, "official_decoder_path": str(path),
                "loaded_decoders": rows, "decoder_model_forwards": 0,
                "geometry_fixture": {"vertices": len(vertices), "faces": len(faces), "surface_area": surface_area,
                                     "scope": "analytical CPU fixture through the original O-Voxel function"},
                "torch": str(torch.__version__), "cpu_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print("TRELLIS_PINNED_DECODER_RECEIPT=" + json.dumps(evidence, allow_nan=False))


if __name__ == "__main__":
    main()
