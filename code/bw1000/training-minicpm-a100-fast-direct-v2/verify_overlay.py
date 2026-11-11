"""Inspect the isolated original fast-runtime overlay with devices masked."""
import hashlib,importlib,importlib.metadata as m,json,os,sys
from pathlib import Path
ROOT=Path(os.environ["FAST_OVERLAY_ROOT"])
assert all(os.environ[n]=="-1" for n in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
plan=json.loads((ROOT/"PLAN.json").read_bytes())
versions={}
for name,expected in plan["packages"].items():
    version=m.version(name);assert version==expected
    versions[name]=version
assert m.version("torch")=="2.8.0a0+5228986c39.nv25.6"
import torch
assert str(torch.__version__)=="2.8.0a0+5228986c39.nv25.06" and torch.version.cuda=="12.9" and not torch.cuda.is_initialized()
checks={}
for name in ("safetensors","regex","causal_conv1d_cuda","causal_conv1d","fla","fla.ops.gated_delta_rule"):
    try:
        module=importlib.import_module(name)
        p=Path(module.__file__).resolve()
        assert p.is_relative_to(ROOT/"site"),name
        checks[name]={"imported":True,"module_path":str(p)}
    except Exception as error:
        checks[name]={"imported":False,"error_type":type(error).__name__,"message":str(error)[:250]}
assert not torch.cuda.is_initialized()
result={"state":"ORIGINAL_FAST_OVERLAY_CPU_IMPORTS_QUALIFIED" if all(x["imported"] for x in checks.values()) else "OVERLAY_FILES_INSTALLED_IMPORT_FAILURE",
    "versions":versions,"imports":checks,"torch":str(torch.__version__),"CUDA":torch.version.cuda,
    "torch_file":torch.__file__,"device_initialized":False,"explicit_GPU_calls":False,"model_loaded":False}
with (ROOT/"direct-v2/verification.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
