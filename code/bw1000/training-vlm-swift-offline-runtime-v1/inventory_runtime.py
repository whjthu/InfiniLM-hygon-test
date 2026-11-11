"""Inspect the selected DTK Python environment for the frozen Swift SFT entry."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import signal
import sys
import time

signal.alarm(100)
started=time.monotonic()
root=Path("/replay")
assert all(os.environ[name]=="-1" for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
contract=json.loads((root/"SWIFT_REQUIREMENTS.json").read_bytes())
distributions={}
for requirement in contract["requirements"]:
    try:
        from packaging.requirements import Requirement
        parsed=Requirement(requirement)
        needed=parsed.marker is None or parsed.marker.evaluate({"extra":""})
        installed=metadata.version(parsed.name) if needed else None
        satisfied=installed in parsed.specifier if installed is not None else None
        distributions[requirement]={"needed":needed,"installed":installed,"satisfied":satisfied}
    except metadata.PackageNotFoundError:
        distributions[requirement]={"needed":True,"installed":None,"satisfied":False}
imports={}
for name in ("torch","torchvision","transformers","peft","accelerate","datasets","swift","swift.pipelines"):
    stdout,stderr=io.StringIO(),io.StringIO()
    try:
        with redirect_stdout(stdout),redirect_stderr(stderr):
            module=importlib.import_module(name)
            if name=="swift.pipelines":
                assert callable(getattr(module,"sft_main"))
        row={"loaded":True,"module_path":getattr(module,"__file__",None)}
    except Exception as error:
        row={"loaded":False,"error_type":type(error).__name__,"reason":str(error)[:300]}
    row["stdout_bytes"]=len(stdout.getvalue().encode())
    row["stderr_bytes"]=len(stderr.getvalue().encode())
    imports[name]=row
runtime={}
if imports["torch"]["loaded"]:
    import torch
    runtime={"torch":str(torch.__version__),"hip":torch.version.hip,"cuda":torch.version.cuda,
             "torch_file":torch.__file__,"torch_threads":torch.get_num_threads(),
             "pytorch_device_context_initialized":torch.cuda.is_initialized()}
missing=[r for r,v in distributions.items() if v["needed"] and v["satisfied"] is False]
result={"scope":"CPU-only base DTK environment package and import qualification for T01–T05; no model or device operation",
        "python":sys.version.split()[0],"executable":sys.executable,"requirements":distributions,
        "requirements_wheel_SHA256":contract["wheel_SHA256"],
        "imports":imports,"runtime":runtime,"missing_or_incompatible":missing,
        "swift_SFT_import_available":imports["swift.pipelines"]["loaded"],
        "wall_seconds":time.monotonic()-started}
with (root/"inventory.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
