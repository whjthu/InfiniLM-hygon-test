"""Qualify only the new private Transformers overlay and protected runtime."""
import hashlib,importlib.metadata as m,json,os
from pathlib import Path
from packaging.requirements import Requirement
assert os.environ["CUDA_VISIBLE_DEVICES"]=="-1"
assert m.version("transformers")=="5.12.1"
assert m.version("torch")=="2.10.0+das.opt1.dtk2604.2606161421.g91a580"
assert m.version("torchvision")=="0.25.0+das.opt1.dtk2604.torch2100.2605071719.g7ffc50"
checks=[]
for text in m.requires("transformers"):
    r=Requirement(text)
    if r.marker and not r.marker.evaluate({"extra":""}):continue
    version=m.version(r.name)
    assert version in r.specifier,(r.name,version,str(r.specifier))
    checks.append(dict(requirement=text,actual=version))
import torch,transformers
from transformers import AutoConfig
assert not torch.cuda.is_initialized()
config=AutoConfig.from_pretrained("/processor-model",local_files_only=True)
assert config.model_type=="minicpmv4_6"
assert str(Path(transformers.__file__)).startswith("/replay/runtime-site/")
result={"state":"PRIVATE_TRANSFORMERS_512_OVERLAY_QUALIFIED","requirements":checks,
        "transformers":transformers.__version__,"transformers_file":transformers.__file__,
        "config_class":type(config).__name__,"torch":m.version("torch"),"torchvision":m.version("torchvision"),
        "device_initialized":torch.cuda.is_initialized(),"model_loaded":False}
with Path("/replay/overlay.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
