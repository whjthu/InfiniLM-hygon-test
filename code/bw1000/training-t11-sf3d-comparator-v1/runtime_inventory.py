import importlib.metadata
import importlib.util
import json
import shutil
import torch

modules=("omegaconf","open_clip","jaxtyping","einops","transformers","trimesh","rembg",
         "pynanoinstantmeshes","gpytoolbox","texture_baker","uv_unwrapper","safetensors","torchvision")
versions={}
for package in ("omegaconf","open-clip-torch","jaxtyping","einops","transformers","trimesh","safetensors","torchvision"):
    try:
        versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package]=None
print(json.dumps({"scope":"DTK package discovery; no upstream imports or execution","torch":torch.__version__,
    "hip":torch.version.hip,"visible_cards":torch.cuda.device_count(),"versions":versions,
    "discoverable_modules":{name:importlib.util.find_spec(name) is not None for name in modules},
    "tools":{tool:shutil.which(tool) is not None for tool in ("g++","hipcc","cmake","ninja")}}))
