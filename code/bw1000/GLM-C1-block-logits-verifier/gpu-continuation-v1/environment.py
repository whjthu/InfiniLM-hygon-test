"""Use the production GLM environment and the claimed private stage entry."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
raw=(root/"GPU_MANIFEST.json").read_bytes()
assert sha256(raw).hexdigest()==os.environ["GLM_PREFIX_MANIFEST_SHA"]
claim=json.loads(raw)
original="/refactor/incoming/GLM-W4-remapped-final-slots-r469/deployment-v1/inside.sh"
env=dict(os.environ,GLM_R444_ENV_ONLY="1",GLM_EXACT_BANKS_R4188="1",GLM_CHANNELS_R4363="0",
    GLM_BOUNDED_ROPE_R4279="1",GLM_SHARED_CONTROL_R4279="1",GLM_NETWORK_TRIAL="shca",
    INFINILM_OWNER_CPU_WIDTH_R2392="4",GLM_PIPELINE_TAIL_NODE=claim["nodes"][1])
run=subprocess.run(["/bin/bash","--noprofile","--norc",original],env=env,capture_output=True,check=True,timeout=20)
marker=b"GLM_R444_ENV_BEGIN\0"
assert run.stdout.count(marker)==1
items=run.stdout.split(marker,1)[1].split(b"\0")
assert items[-1]==b"" and len(items)%2==1
for key,val in zip(items[:-1:2],items[1:-1:2],strict=True):env[key.decode()]=val.decode()
package="/refactor/incoming/GLM952-prefix-continuation/provider"
env["PYTHONPATH"]=package+":"+env.get("PYTHONPATH","")
env["PYTHONDONTWRITEBYTECODE"]="1"
stage=os.environ["SLURM_PROCID"]
assert stage in ("0","1")
step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]
argv=["/usr/bin/python3","-B","-m","public_glm_native.stage_compare",
    "--config",str(root/"config.json"),"--config-sha",claim["provider_config_SHA256"],
    "--source-sha",claim["source_manifest_SHA256"],"--claim",str(root/"GPU_MANIFEST.json"),
    "--claim-sha",os.environ["GLM_PREFIX_MANIFEST_SHA"],"--stage",stage,
    "--run",claim["run"],"--actual-step",step,"--gpu-start-epoch",os.environ["GLM_PREFIX_EPOCH"],"--execute"]
os.chdir(package)
os.execve(argv[0],argv,env)
