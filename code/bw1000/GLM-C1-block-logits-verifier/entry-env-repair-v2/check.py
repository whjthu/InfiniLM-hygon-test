"""Check only the repaired original Python path and the actual janus import."""
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from capture_original_environment import capture

R=Path("/refactor")
D=Path(__file__).resolve().parent
claim_raw=(D/"CPU_RESOURCE_CLAIM.json").read_bytes()
assert sha256(claim_raw).hexdigest()==os.environ["GLM_PREFIX_ENV_CPU_CLAIM_SHA"]
claim=json.loads(claim_raw)
assert os.environ["SLURM_JOB_ID"]==claim["parent"]
step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]
for name,pin in claim["files_SHA256"].items():
    assert sha256((D/name).read_bytes()).hexdigest()==pin
inherited=dict(os.environ)
env=capture(R/"incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh",inherited)
assert all(env[key]==inherited[key]=="-1" for key in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
probe=r'''
import importlib,json,sys
from pathlib import Path
from hashlib import sha256
janus=importlib.import_module("janus")
path=Path(janus.__file__).resolve(strict=True)
assert path.is_relative_to("/old/offline-deps")
assert callable(janus.Queue)
assert not any(k in sys.modules for k in ("torch","infinilm","_infinilm"))
print(json.dumps(dict(path=str(path),SHA256=sha256(path.read_bytes()).hexdigest(),Queue=True)))
'''
p=subprocess.run(["/usr/bin/python3","-B","-c",probe],env=env,capture_output=True,text=True,timeout=20,check=True)
result=dict(passed=True,actual_CPU=step,original_source_SHA256=claim["original_source_SHA256"],
    recovered_PYTHONPATH=env["PYTHONPATH"],janus=json.loads(p.stdout),
    GPU=False,candidate_Native_dlopen=False,explicit_SDK_calls=0,Model=False)
out=R/"results/GLM952-prefix-continuation-env-CPU-v2"
out.mkdir()
with (out/"result.json").open("x") as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
