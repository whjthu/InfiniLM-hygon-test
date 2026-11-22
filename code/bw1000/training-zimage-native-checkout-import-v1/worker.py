"""Verify and import the original Z-Image SFT source in a zero-DCU container."""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


ROOT = Path("/native-source")
ENTRY = "examples/z_image/model_training/train.py"
PROVIDERS = {
    ENTRY: "309170a93b0033930cac1e45c5baa7af66fd792240d494efa05f96dc0ab5cbc1",
    "diffsynth/pipelines/z_image.py": "fc9c5c47937ce8a29531015cebec7814e598f015f4dcf256536c908f1a8d7b75",
    "diffsynth/models/z_image_dit.py": "ee266edab9df1dc3476c8e90b2c1478466a0071a2ef2b4b13491648582a38ca7",
    "diffsynth/diffusion/base_pipeline.py": "13d916193a7838de41ca60af36dbddd5813075dff78846f5ef8e8ac1f2d1f43b",
    "diffsynth/diffusion/flow_match.py": "4000f836cd019673bbf5e0c34c34a3a8a15325c5fce615481cc81dc066656a87",
}
TREE_SHA256 = "22f7084b30a5d84588e8496ba972f97a9405088d19ce2a85f899d4f02a6fed14"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_tree():
    tree = hashlib.sha256()
    files = sorted((path for path in ROOT.rglob("*") if not path.is_dir()),
                   key=lambda path: path.relative_to(ROOT).as_posix())
    need(len(files) == 651, "Original source file count differs")
    total = 0
    for path in files:
        need(path.is_file() and path.resolve(strict=True) == path and path.stat().st_size <= 16 << 20,
             "Unsafe source file")
        relative = path.relative_to(ROOT).as_posix()
        name = relative.encode()
        raw = path.read_bytes()
        for part in (name, raw):
            tree.update(len(part).to_bytes(8, "big"))
            tree.update(part)
        total += len(raw)
        if relative in PROVIDERS:
            need(hashlib.sha256(raw).hexdigest() == PROVIDERS[relative], "Pinned provider differs: " + relative)
    need(total == 3000854 and tree.hexdigest() == TREE_SHA256, "Original source tree digest differs")
    return len(files), total


def no_model_or_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network attempted during native import")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if Path(path).suffix.lower() in {".safetensors", ".pt", ".pth", ".ckpt"}:
            raise RuntimeError("Checkpoint open attempted during native import")


def main():
    started = time.monotonic()
    need(all(os.environ.get(key) == "" for key in
             ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
         "Accelerator masks are not empty")
    count, total = verify_tree()
    sys.addaudithook(no_model_or_network)
    import torch

    need(not torch.cuda.is_initialized(), "Accelerator initialized before import")
    spec = importlib.util.spec_from_file_location("zimage_frozen_sft", ROOT / ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    pipeline = importlib.import_module("diffsynth.pipelines.z_image")
    dit = importlib.import_module("diffsynth.models.z_image_dit")
    diffusion = importlib.import_module("diffsynth.diffusion")
    need(entry.ZImagePipeline is pipeline.ZImagePipeline and entry.ModelConfig is pipeline.ModelConfig
         and pipeline.ZImageDiT is dit.ZImageDiT
         and issubclass(entry.ZImageTrainingModule, diffusion.DiffusionTrainingModule),
         "Native training symbols do not bind to the original provider")
    imports = sorted({Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                      if (name == "diffsynth" or name.startswith("diffsynth."))
                      and getattr(module, "__file__", None)})
    need(imports and all(path.is_relative_to(ROOT) for path in imports), "DiffSynth module escaped the pinned tree")
    for module in (pipeline, dit, sys.modules["diffsynth.diffusion.flow_match"],
                   sys.modules["diffsynth.diffusion.base_pipeline"]):
        need(Path(module.__file__).resolve().relative_to(ROOT).as_posix() in PROVIDERS,
             "Original provider path differs")
    need(not torch.cuda.is_initialized(), "Accelerator initialized during import")
    print(json.dumps(dict(passed=True, node=socket.gethostname().split(".")[0],
                          step_id=os.environ["SLURM_STEP_ID"], job_id=os.environ["SLURM_JOB_ID"],
                          request=os.environ["ZIMAGE_IMPORT_REQUEST"], source_files=count, source_bytes=total,
                          source_tree_SHA256=TREE_SHA256, providers=PROVIDERS,
                          eager_diffsynth_source_modules=len(imports), torch=str(torch.__version__),
                          cuda_initialized=False, model_constructed=False, optimizer_updates=0,
                          elapsed_seconds=time.monotonic()-started), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
