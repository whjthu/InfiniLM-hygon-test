"""Independently validate the captured A100 trajectory in its original workspace."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
reference = json.loads((ROOT / "hunyuan-sage-A100-capture-independent.json").read_bytes())
remote = r"""
from pathlib import Path
import hashlib,json,os
import torch

assert not torch.cuda.is_initialized()
work=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sage-first-policy-A100-v1")
intent=json.loads((work/"intent.json").read_bytes())
assert intent["request_id"]==REF["request_id"]
out=work/"results/hunyuanvideo-1.5/rl/first-policy"

def verified(path,pin):
    raw=path.read_bytes()
    assert len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["sha256"]
    return raw

import io
pins=REF["original_outcome"]["files"]
policy=json.loads(verified(out/"first-policy-update.json",pins["first-policy-update.json"]))
payload=torch.load(io.BytesIO(verified(out/"trajectories/step-0001.pt",pins["trajectories/step-0001.pt"])),
                   map_location="cpu",weights_only=True)
launch=json.loads((work/"launch.json").read_bytes())
assert hashlib.sha256((work/"launch.json").read_bytes()).hexdigest()=="6b230bb6ac909a4599faa37fc9a20e85d425c97e8c643ca111b3f124ef16c5a3"
config=launch["config"]
assert payload["global_step"]==1 and payload["adapter_sha"]==policy["adapter_before"]
assert payload["seed"]==[config["seed"]+100+i for i in range(4)]
assert payload["reward"]==policy["reward"]
assert payload["input_identity"]=={
    key+"_sha256":hashlib.sha256(Path(config[field]).read_bytes()).hexdigest()
    for key,field in (("cache","training_cache"),("target_video","target_video"))
}
assert hashlib.sha256(Path(config["initial_adapter"]).read_bytes()).hexdigest()==config["initial_adapter_file_sha"]
expected_old=torch.tensor([row["old_log_probs"] for row in policy["rollouts"]],dtype=torch.float32)
assert torch.equal(payload["sampled_log_prob"],expected_old)
assert payload["timestep"].tolist()==[1000.0,937.5,833.3333129882812,625.0]
assert payload["sigmas"].tolist()==[1.0,0.9375,0.8333333134651184,0.625,0.0]
tensors={}
for key,expected in (("latent",(4,5,32,3,30,53)),("noise",(4,4,32,3,30,53)),
                     ("sampled_log_prob",(4,4)),("timestep",(4,)),("sigmas",(5,))):
    value=payload[key]
    assert value.device.type=="cpu" and tuple(value.shape)==expected and value.dtype==torch.float32
    assert bool(torch.isfinite(value).all())
    tensors[key]={"shape":list(value.shape),"dtype":str(value.dtype),"numel":value.numel(),
                  "sha256":hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
assert all(not torch.equal(payload["latent"][:,i],payload["latent"][:,i+1]) for i in range(4))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(request_id=REF["request_id"],tensors=tensors,input_identity=payload["input_identity"],
    tensor_contents_verified=True,old_log_prob_matches_policy=True,launch_and_adapter_file_verified=True,
    reference_sha256=pins["trajectories/step-0001.pt"]["sha256"],cuda_initialized=False,
    original_launcher_passed=False,actual_replay_executed=False)))
"""
cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-F", str(A100 / "ssh_config"),
       "a100-26", "timeout 60 env CUDA_VISIBLE_DEVICES= /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
program = f"REF={reference!r}\n" + remote
try:
    call = subprocess.run(cmd, input=program.encode(), capture_output=True, timeout=70)
except subprocess.TimeoutExpired:
    raise SystemExit("SSH read timeout") from None
receipt = dict(rc=call.returncode, stdout_SHA256=hashlib.sha256(call.stdout).hexdigest(),
               stderr_SHA256=hashlib.sha256(call.stderr).hexdigest())
if call.returncode == 0:
    receipt["value"] = json.loads(call.stdout)
else:
    receipt["failure_tail"] = call.stderr.decode(errors="replace").splitlines()[-3:]
with (ROOT / "hunyuan-sage-A100-trajectory-values-cache-shape.json").open("x", encoding="utf8") as out:
    json.dump(receipt, out, indent=2)
print(json.dumps(receipt))
