"""Inspect the unmodified DTK Python on a T01-owned node in one CPU Slurm step."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
work=root/"results/training-vlm-runtime-inventory-v1"
node="a01r4n16"
label="training-vlm-runtime-inventory-v1"

def query(*args):
    return subprocess.check_output(args,text=True,timeout=12).strip()

def accounting():
    return [line.split("|") for line in query("sacct","-n","-P","-j","1087656",
            "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw").splitlines() if "|"+label+"|" in line]

assert not work.exists() and not accounting()
parent=query("squeue","-h","-j","1087656","-o","%T|%D|%N").split("|")
assert parent[:2]==["RUNNING","40"]
assert node in query("scontrol","show","hostnames",parent[-1]).splitlines()
for line in query("squeue","--steps","-h","-j","1087656","-o","%i|%N").splitlines():
    handle,names=line.split("|",1)
    if handle not in ("1087656.batch","1087656.extern") and names not in ("","(null)"):
        assert node not in query("scontrol","show","hostnames",names).splitlines()
epoch=time.time()
deadline=epoch+150
work.mkdir(mode=0o700)
for name,spec in FILES.items():
    assert name in ("inventory_runtime.py","SWIFT_REQUIREMENTS.json")
    raw=base64.b64decode(spec["b64"],validate=True)
    assert hashlib.sha256(raw).hexdigest()==spec["SHA256"]
    with (work/name).open("xb") as stream:
        stream.write(raw)
with (work/"intent.json").open("x") as stream:
    json.dump({"claim_commit":CLAIM,"node":node,"epoch":epoch,"deadline":deadline,"GPU_allocation":0},stream)
shell=("set -euo pipefail; source /opt/dtk/env.sh; "
       "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "
       "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
       "exec python3 -B -u /replay/inventory_runtime.py")
args=["srun","--jobid=1087656","--job-name="+label,"--exact","--nodes=1","--ntasks=1","--nodelist="+node,
      "--cpus-per-task=4","--mem=8192M","--gres=none","--immediate=5","--time=00:02:00",
      "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/",
      "--bind",str(work)+":/replay:rw","--bind","/opt/hyhal:/opt/hyhal:ro",
      str(root/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",shell]
with (work/"runtime.log").open("xb") as log:
    result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-5))
rows=[]
for _ in range(20):
    rows=accounting()
    if len(rows)==1 and rows[0][2] in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","OUT_OF_MEMORY","NODE_FAIL"):
        break
    time.sleep(.5)
raw=(work/"runtime.log").read_bytes()
assert len(raw)<131072
answer={"claim_commit":CLAIM,"returncode":result.returncode,"accounting":rows,
        "node":node,"epoch":epoch,"deadline":deadline,"wall_seconds":time.time()-epoch,
        "log_SHA256":hashlib.sha256(raw).hexdigest(),
        "error_tail":[line[-240:] for line in raw.decode(errors="replace").splitlines()
                      if line.startswith(("srun:","ModuleNotFoundError","RuntimeError","AssertionError","TypeError"))][-5:]}
path=work/"inventory.json"
if path.is_file():
    data=path.read_bytes()
    assert len(data)<65536
    answer["inventory"]={"SHA256":hashlib.sha256(data).hexdigest(),"value":json.loads(data)}
with (work/"outcome.json").open("x") as stream:
    json.dump(answer,stream,indent=2)
print(json.dumps(answer))
