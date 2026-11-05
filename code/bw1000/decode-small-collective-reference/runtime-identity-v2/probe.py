"""Run the actual mapped-code identity function with devices masked, without CAPI calls."""
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

R=Path("/refactor")
D=Path(__file__).resolve().parent
raw=(D/"CPU_RESOURCE_CLAIM.json").read_bytes()
claim=json.loads(raw)
assert sha256(raw).hexdigest()==os.environ["DEEP_CHAIN_IDENTITY_CPU_CLAIM_SHA"]
assert os.environ["SLURM_JOB_ID"]==claim["parent"]
for name,pin in claim["files_SHA256"].items():assert sha256((D/name).read_bytes()).hexdigest()==pin
receipt=R/"candidates/Deep-decode-chain-reference-cpu-loader-v2/CPU_BUILD_RESULT.json"
b=receipt.read_bytes()
assert sha256(b).hexdigest()=="4e5ca7afaccf88a701dd40d5b7846ca386baa521c55b2cc08479421138efdab6"
r=json.loads(b)
env=dict(os.environ,LD_LIBRARY_PATH=r["resolved_search"])
inner=r'''
import ctypes,json,os
from hashlib import sha256
from pathlib import Path
from runtime_identity import verify_runtime_maps
r=json.loads(Path("/refactor/candidates/Deep-decode-chain-reference-cpu-loader-v2/CPU_BUILD_RESULT.json").read_bytes())
path=Path("/refactor/candidates/Deep-decode-chain-reference-cpu-v1/libDecodeSmallCollective.so")
assert sha256(path.read_bytes()).hexdigest()==r["native_sha256"]=="3bbcc833031caafa9a245ffe0eb832f9e92df9febef0344cf25d125389e4dbf7"
assert all(os.environ[k]=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
lib=ctypes.CDLL(str(path))
result=verify_runtime_maps(r["required_runtime_libraries"],Path("/proc/self/maps").read_text())
assert verify_runtime_maps(r["required_runtime_libraries"],Path("/proc/self/maps").read_text())==result
print(json.dumps(dict(passed=True,actual_CPU=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
    runtime_bindings=result,GPU=False,Native_CAPI_calls=0,candidate_dlopen=True)))
'''
env["PYTHONPATH"]=str(D)
p=subprocess.run(["/usr/bin/python3","-B","-c",inner],env=env,capture_output=True,text=True,check=True,timeout=35)
out=R/"results/Deep-chain-runtime-identity-CPU-v2"
out.mkdir()
data=json.loads(p.stdout)
with (out/"result.json").open("x") as stream:json.dump(data,stream,indent=2)
print(json.dumps(data))
