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
from capture_original_environment import capture
env=capture(original,dict(os.environ,GLM_PIPELINE_TAIL_NODE=claim["nodes"][1]))
package="/refactor/incoming/GLM952-prefix-continuation/native-admission-v2/provider"
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
