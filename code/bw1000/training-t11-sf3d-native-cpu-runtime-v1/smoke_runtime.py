import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import os

import torch
import texture_baker._C as baker_ext
import uv_unwrapper._C as unwrap_ext
from texture_baker import TextureBaker
from uv_unwrapper import Unwrapper
from sf3d.system import SF3D
from sf3d.models.image_estimator.clip_based_estimator import ClipBasedHeadEstimator

assert torch.version.hip and not torch.cuda.is_initialized() and torch.cuda.device_count()==0
assert not Path(torch.__file__).resolve().is_relative_to(Path("/runtime"))
assert Path(__import__("sf3d.system",fromlist=["SF3D"]).__file__).resolve().is_relative_to(Path("/sf3d-source"))
assert isinstance(Unwrapper(),torch.nn.Module)
baker=TextureBaker()
uv=torch.tensor([[0.125,0.125],[0.875,0.125],[0.125,0.875]],dtype=torch.float32)
faces=torch.tensor([[0,1,2]],dtype=torch.int32)
raster=baker.rasterize(uv,faces,16)
mask=baker.get_mask(raster)
out=baker.interpolate(torch.ones(3,3),raster,faces)
assert raster.shape==(16,16,4) and out.shape==(16,16,3)
assert 0<int(mask.sum())<256 and torch.isfinite(out).all()
assert torch.allclose(out[mask],torch.ones_like(out[mask]),rtol=0,atol=1e-6)
operators=("texture_baker_cpp::rasterize","texture_baker_cpp::interpolate","UVUnwrapper::assign_faces_uv_to_atlas_index")
assert all(torch._C._dispatch_has_kernel_for_dispatch_key(op,"CPU") for op in operators)
assert not any(torch._C._dispatch_has_kernel_for_dispatch_key(op,"CUDA") for op in operators)
libraries={}
env=dict(os.environ)
env["LD_LIBRARY_PATH"]=str(Path(torch.__file__).parent/"lib")+":"+env.get("LD_LIBRARY_PATH","")
for module in (baker_ext,unwrap_ext):
    path=Path(module.__file__)
    assert path.resolve().is_relative_to(Path("/runtime/site"))
    text=subprocess.check_output(["ldd",str(path)],env=env,text=True,timeout=10)
    assert "not found" not in text
    libraries[path.name]={"bytes":path.stat().st_size,"SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
                          "ldd_sha256":hashlib.sha256(text.encode()).hexdigest()}
versions={p:importlib.metadata.version(p) for p in ("torch","torchvision","numpy","transformers","huggingface-hub",
          "open-clip-torch","trimesh","omegaconf","jaxtyping","texture-baker","uv-unwrapper")}
assert versions["numpy"]=="1.26.4" and versions["transformers"]=="4.42.3"
print(json.dumps({"scope":"CPU native imports and registered operators","imports":[SF3D.__name__,ClipBasedHeadEstimator.__name__],
                  "torch_hip":torch.version.hip,"gpu_initialized":torch.cuda.is_initialized(),"versions":versions,
                  "CPU_operators":operators,"sample_occupied_pixels":int(mask.sum()),"libraries":libraries}))
